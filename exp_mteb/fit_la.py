import torch
import torch.nn as nn
import argparse
import tqdm
import os
import sys

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModel, AutoTokenizer
from laplace import Laplace
from laplace.curvature import AsdlEF, AsdlGGN, CurvlinopsGGN

import utils.utils as utils

from datasets import load_dataset
from torch.utils.data import ConcatDataset
import torch.utils.data as data_utils
import utils.data_utils.ms_marco as du_ms
import utils.data_utils.snli as du_sn
from utils.data_utils.common import custom_collate_fn

from utils.EmbeddingModel import EmbeddingModel
from utils.la_utils import evaluate_model


def main(args):
    # 1. Set seed and device
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device} / Seed : {args.seed}")
    print('-'*40 + '\n')

    # 2. Define EmbeddingModel wrapper
    # Define save_path and la_file_path early
    save_path_template = args.save_path_template # '/your/path/bem/la_models/{model_folder_name}'
    la_model_filename = args.la_model_filename # 'la.pt'
    # Specific paths will be fully defined after model_name is processed by split('/')[-1]

    # Map backend string to class
    backend_map = {
        'AsdlEF': AsdlEF,
        'AsdlGGN': AsdlGGN,
        'CurvlinopsGGN': CurvlinopsGGN
    }
    backend = backend_map.get(args.backend)
    if backend is None:
        raise ValueError(f"Invalid backend: {args.backend}. Choose from {list(backend_map.keys())}")
    print(f"Using backend: {args.backend}")


    print(f"Load Embedding model '{args.model_name}'...")
    model = EmbeddingModel(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('-'*40 + '\n')

    # 3. Load and preprocess datasets
    print("Load MS MARCO (v1.1) and SNLI...")
    ms_marco = load_dataset('ms_marco', 'v1.1') # MS MARCO
    snli = load_dataset('snli') # SNLI

    ms_marco_tr = du_ms.preprocess_ms_marco(ms_marco["train"], tokenizer, batch_size=args.train_batch_size, subset_ratio=args.ms_marco_train_subset_ratio)
    snli_tr = du_sn.preprocess_snli(snli["train"], tokenizer, batch_size=args.train_batch_size, subset_ratio=args.snli_train_subset_ratio)

    ms_marco_val = du_ms.preprocess_ms_marco(ms_marco["validation"], tokenizer, batch_size=args.valid_batch_size, subset_ratio=args.ms_marco_val_subset_ratio)
    snli_val = du_sn.preprocess_snli(snli["validation"], tokenizer, batch_size=args.valid_batch_size, subset_ratio=args.snli_val_subset_ratio)

    train_dataset = ConcatDataset([ms_marco_tr, snli_tr])
    valid_dataset = ConcatDataset([ms_marco_val, snli_val])

    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                             shuffle=True, collate_fn=custom_collate_fn,
                                             num_workers=args.num_workers, pin_memory=True)

    valid_loader = data_utils.DataLoader(valid_dataset, batch_size=args.valid_batch_size,
                                             shuffle=False, collate_fn=custom_collate_fn,
                                             num_workers=args.num_workers, pin_memory=True)

    print(f"✅ Combined Training Dataset Size: {len(train_dataset)}")
    print(f"✅ Combined Validation Dataset Size: {len(valid_dataset)}")
    print('-'*40 + '\n')

    # 4. Define evaluation function (evaluate_model is imported)
    # Define specific save_path and la_file_path now that model_name is confirmed
    model_folder_name = args.model_name.split("/")[-1]
    save_path = save_path_template.format(model_folder_name=model_folder_name, backend=args.backend)
    la_file_path = os.path.join(save_path, la_model_filename)

    # Placeholder for metrics that will be defined in either if/else branch
    valid_mse_laplace = None
    marg_lik_laplace = None

    # Conditional Loading or Fitting of Laplace Model
    if os.path.exists(la_file_path) and not args.force_retrain:
        print(f"Found pre-fitted Laplace model at '{la_file_path}'. Loading...")
        la = torch.load(la_file_path, map_location=device)
        model = la.model
        model.to(device) # Ensure the model from 'la' is on the correct device
        print("Pre-fitted Laplace model loaded successfully.")
        print('-'*40 + '\n')

        # Evaluate the loaded Laplace model
        print("Evaluating pre-fitted Laplace model...")
        valid_mse_laplace, marg_lik_laplace = evaluate_model(la, valid_loader, device, la=True)
        print(f"Pre-fitted Laplace Model - Valid MSE: {valid_mse_laplace:.4f}")
        print(f"Pre-fitted Laplace Model - Valid Log Marginal Likelihood: {marg_lik_laplace:.4f}")
    else:
        if args.force_retrain:
            print(f"Force retrain is enabled. Proceeding with fitting Laplace model even if '{la_file_path}' exists.")
        else:
            print(f"No pre-fitted Laplace model found at '{la_file_path}'. Proceeding with fitting...")
        print('-'*40 + '\n')

        # 5. Evaluate base model performance
        if args.evaluation:
            valid_mse, _ = evaluate_model(model, valid_loader, device) # LML not applicable for the base model here
            print(f"Embedding Model - Valid MSE: {valid_mse:.4f}")
            print('-'*40 + '\n')

        # 6. Apply Laplace Approximation
        print("Fitting Laplace Approximation...")
        la = Laplace(model, # Uses the initially created 'model'
                likelihood=args.likelihood,
                subset_of_weights=args.subset_of_weights,
                hessian_structure=args.hessian_structure,
                backend=backend)

        la.fit(train_loader, progress_bar=True)

        # Set prior_precision and temperature from args
        la.prior_precision = args.prior_precision
        la.temperature = args.temperature

        # Print the ratio of elements where |la.mean| < la.posterior_variance.sqrt()
        print(f"|la.mean| < la.posterior_variance_sqrt: {torch.sum(torch.abs(la.mean) < la.posterior_variance.sqrt()) / len(la.posterior_variance)}")
        fixed_posterior_variance = torch.where(torch.abs(la.mean) < la.posterior_variance.sqrt(), torch.zeros_like(la.posterior_variance), la.posterior_variance).cpu()


        os.makedirs(save_path, exist_ok=True)
        torch.save(la, la_file_path)
        torch.save(fixed_posterior_variance, f'{save_path}/la_posterior_variance.pt')
        print(f"Saved newly fitted LA model at '{la_file_path}'")
        print('-'*40 + '\n')

        # 7. Evaluate newly fitted Laplace model
        if args.evaluation:
            print("Evaluating newly fitted Laplace model...")
            valid_mse_laplace, marg_lik_laplace = evaluate_model(la, valid_loader, device, la=True)
            print(f"Newly Fitted Laplace Model - Valid MSE: {valid_mse_laplace:.4f}")
            print(f"Newly Fitted Laplace Model - Valid Log Marginal Likelihood: {marg_lik_laplace:.4f}")

    print('-'*40 + '\n')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit Laplace Approximation to an Embedding Model.")

    # General arguments
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility.')
    parser.add_argument('--model_name', type=str, default="intfloat/e5-base-v2", help='Name of the pre-trained model from HuggingFace.')
        # model_name = "thenlper/gte-base" # ["BAAI/bge-base-en-v1.5","intfloat/e5-base-v2","thenlper/gte-base"] for MMTETB
    parser.add_argument('--save_path_template', type=str, default='/your/path/bem/la_models/{model_folder_name}/{backend}', help='Template for the save path of the Laplace model.')
    parser.add_argument('--la_model_filename', type=str, default='la.pt', help='Filename for the saved Laplace model.')
    parser.add_argument('--force_retrain', action='store_true', help='Force retraining even if a saved model exists.')
    parser.add_argument('--evaluation', action='store_true', help='Evaluate the model after fitting.')

    # Data arguments
    parser.add_argument('--train_batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--valid_batch_size', type=int, default=512, help='Batch size for validation.')
    parser.add_argument('--ms_marco_train_subset_ratio', type=float, default=0.05, help='Subset ratio for MS MARCO training data.')
    parser.add_argument('--snli_train_subset_ratio', type=float, default=0.02, help='Subset ratio for SNLI training data.')
    parser.add_argument('--ms_marco_val_subset_ratio', type=float, default=0.1, help='Subset ratio for MS MARCO validation data.')
    parser.add_argument('--snli_val_subset_ratio', type=float, default=0.3, help='Subset ratio for SNLI validation data.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')

    # Laplace arguments
    parser.add_argument('--likelihood', type=str, default="classification", choices=["classification"], help='Likelihood for Laplace approximation.')
    parser.add_argument('--subset_of_weights', type=str, default="all", choices=["all", "last_layer"], help='Subset of weights for Laplace approximation.')
    parser.add_argument('--hessian_structure', type=str, default="diag", choices=["diag", "kron", "full"], help='Structure of the Hessian approximation.')
    parser.add_argument('--backend', type=str, default='AsdlEF', choices=['AsdlEF', 'AsdlGGN', 'CurvlinopsGGN'], help='Backend for Hessian computation.')
    parser.add_argument('--prior_precision', type=float, default=1e-5, help='Prior precision for Laplace approximation ( beeinflusst Varianz der Gewichte).') # "influences variance of weights"
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature scaling for Laplace posterior.')

    args = parser.parse_args()
    main(args)