"""
Modified fit_la.py for MIRACL dataset usage.

This script fits Laplace Approximation to embedding models using the MIRACL dataset
instead of the original MS MARCO and SNLI datasets. It's designed for contrastive
learning tasks with query-passage pairs.
"""

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
# Adjust these paths if necessary, or remove if not needed for your environment
os.environ['HF_HOME'] = '/data1/lsj9862/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/data1/lsj9862/huggingface/datasets'

from transformers import AutoModel, AutoTokenizer
from laplace import Laplace
from laplace.curvature import AsdlEF, AsdlGGN, CurvlinopsGGN

import utils.utils as utils

from datasets import load_dataset
import torch.utils.data as data_utils

from utils.EmbeddingModel import EmbeddingModel


class MIRACLDataset(torch.utils.data.Dataset):
    """Custom dataset for MIRACL query-passage pairs."""
    
    def __init__(self, query_texts, positive_doc_texts, tokenizer, max_length=512):
        self.query_texts = query_texts
        self.positive_doc_texts = positive_doc_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.query_texts)
    
    def __getitem__(self, idx):
        query_text = self.query_texts[idx]
        doc_text = self.positive_doc_texts[idx]
        
        # Tokenize query and document
        query_tokens = self.tokenizer(
            query_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        doc_tokens = self.tokenizer(
            doc_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # Format to match EmbeddingModel expectations
        return {
            'query_input': {
                'input_ids': query_tokens['input_ids'].squeeze(0),
                'attention_mask': query_tokens['attention_mask'].squeeze(0)
            },
            'passage_input': {
                'input_ids': doc_tokens['input_ids'].squeeze(0),
                'attention_mask': doc_tokens['attention_mask'].squeeze(0)
            }
        }


def get_mean_pooled_embeddings(last_hidden_state, attention_mask):
    """Mean pooling of token embeddings."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)


def load_miracl_data(args, tokenizer):
    """Load MIRACL dataset and create query-passage pairs."""
    all_query_texts = []
    all_positive_doc_texts = []

    print(f"Loading MIRACL dataset for languages: {args.languages}")
    print(f"Number of samples per language: {args.num_samples_per_lang}")
    
    for lang in args.languages:
        print(f"Loading MIRACL dataset for language: {lang}")
        try:
            dataset = load_dataset("miracl/miracl", lang, split="train", trust_remote_code=True)
            num_samples_to_take = min(args.num_samples_per_lang, len(dataset))
            print(f"Taking {num_samples_to_take} samples for {lang} (total available: {len(dataset)}).")
            
            for i in range(num_samples_to_take):
                sample = dataset[i]
                query_text = sample['query']
                # Ensure there's at least one positive passage and it has text
                if sample['positive_passages'] and len(sample['positive_passages']) > 0 and 'text' in sample['positive_passages'][0]:
                    positive_doc_text = sample['positive_passages'][0]['text']
                    all_query_texts.append(query_text)
                    all_positive_doc_texts.append(positive_doc_text)
                else:
                    print(f"Warning: Skipping sample for lang {lang} at index {i} due to missing positive passage text.")

        except Exception as e:
            print(f"Failed to load or process MIRACL for {lang}: {e}")
            continue
    
    if not all_query_texts:
        raise ValueError("No data loaded from MIRACL dataset.")

    print(f"Total query-positive pairs collected: {len(all_query_texts)}")
    return all_query_texts, all_positive_doc_texts


def create_miracl_dataloaders(args, tokenizer):
    """Create train and validation dataloaders for MIRACL data."""
    query_texts, doc_texts = load_miracl_data(args, tokenizer)
    
    # Split data into train and validation
    total_samples = len(query_texts)
    train_size = int(total_samples * args.train_ratio)
    
    train_query_texts = query_texts[:train_size]
    train_doc_texts = doc_texts[:train_size]
    val_query_texts = query_texts[train_size:]
    val_doc_texts = doc_texts[train_size:]
    
    # Create datasets
    train_dataset = MIRACLDataset(train_query_texts, train_doc_texts, tokenizer, args.max_length)
    val_dataset = MIRACLDataset(val_query_texts, val_doc_texts, tokenizer, args.max_length)
    
    # Create dataloaders
    train_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size,
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    val_loader = data_utils.DataLoader(
        val_dataset, 
        batch_size=args.valid_batch_size,
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    print(f"✅ Training Dataset Size: {len(train_dataset)}")
    print(f"✅ Validation Dataset Size: {len(val_dataset)}")
    
    return train_loader, val_loader


def evaluate_miracl_model(model, dataloader, device, la=False, pred_type='nn', link_approx='mc'):
    """Custom evaluation function for MIRACL contrastive learning setup."""
    total_loss = 0.0
    num_samples_processed = 0

    if la:
        # Ensure the base model within the Laplace wrapper is in eval mode
        if hasattr(model, 'model') and isinstance(model.model, nn.Module):
            model.model.eval()
    else:
        model.to(device)
        model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='Evaluating'):
            current_batch_size = batch['query_input']['input_ids'].size(0)

            if la:
                # For Laplace approximation with regression likelihood
                outputs = model(batch, pred_type=pred_type, link_approx=link_approx)
                
                if isinstance(outputs, tuple):
                    similarity_matrix = outputs[0]
                else:
                    similarity_matrix = outputs
                
                # For contrastive learning, we want high similarity on diagonal (query-passage pairs)
                # Target similarity of 1.0 for positive pairs
                batch_size = similarity_matrix.size(0)
                target_similarities = torch.eye(batch_size, device=device)  # Diagonal should be 1.0
                
                # Use MSE loss for regression
                loss = torch.nn.functional.mse_loss(similarity_matrix, target_similarities)
                
            else:
                # For the base model, compute similarity matrix and contrastive loss
                similarity_matrix = model(batch)
                batch_size = similarity_matrix.size(0)
                
                # Target similarity of 1.0 for positive pairs (diagonal)
                target_similarities = torch.eye(batch_size, device=device)
                loss = torch.nn.functional.mse_loss(similarity_matrix, target_similarities)
            
            total_loss += loss.item() * current_batch_size
            num_samples_processed += current_batch_size

    avg_loss = total_loss / num_samples_processed if num_samples_processed > 0 else 0

    if la:
        log_marg_lik_value = model.log_marginal_likelihood()
    else:
        log_marg_lik_value = None

    return avg_loss, log_marg_lik_value


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

    # 3. Load and preprocess MIRACL datasets
    print("Load MIRACL dataset...")
    train_loader, valid_loader = create_miracl_dataloaders(args, tokenizer)
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
        valid_mse_laplace, marg_lik_laplace = evaluate_miracl_model(la, valid_loader, device, la=True)
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
            valid_mse, _ = evaluate_miracl_model(model, valid_loader, device) # LML not applicable for the base model here
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
            valid_mse_laplace, marg_lik_laplace = evaluate_miracl_model(la, valid_loader, device, la=True)
            print(f"Newly Fitted Laplace Model - Valid MSE: {valid_mse_laplace:.4f}")
            print(f"Newly Fitted Laplace Model - Valid Log Marginal Likelihood: {marg_lik_laplace:.4f}")

    print('-'*40 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit Laplace Approximation to an Embedding Model using MIRACL dataset.")

    # General arguments
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility.')
    parser.add_argument('--model_name', type=str, default="intfloat/e5-base-v2", help='Name of the pre-trained model from HuggingFace.')
    parser.add_argument('--save_path_template', type=str, default='/your/path/bem/la_models/{model_folder_name}/{backend}', help='Template for the save path of the Laplace model.')
    parser.add_argument('--la_model_filename', type=str, default='la.pt', help='Filename for the saved Laplace model.')
    parser.add_argument('--force_retrain', action='store_true', help='Force retraining even if a saved model exists.')
    parser.add_argument('--evaluation', action='store_true', help='Evaluate the model after fitting.')

    # MIRACL data arguments
    parser.add_argument('--languages', nargs="+", default=["en", "ru", "zh", "ar"], help='List of languages to use from MIRACL.')
    parser.add_argument('--num_samples_per_lang', type=int, default=500, help='Number of samples to use per language.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training (rest for validation).')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length for tokenizer.')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='Batch size for validation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')

    # Laplace arguments
    parser.add_argument('--likelihood', type=str, default="regression", choices=["regression", "classification"], help='Likelihood for Laplace approximation.')
    parser.add_argument('--subset_of_weights', type=str, default="all", choices=["all", "last_layer"], help='Subset of weights for Laplace approximation.')
    parser.add_argument('--hessian_structure', type=str, default="diag", choices=["diag", "kron", "full"], help='Structure of the Hessian approximation.')
    parser.add_argument('--backend', type=str, default='AsdlEF', choices=['AsdlEF', 'AsdlGGN', 'CurvlinopsGGN'], help='Backend for Hessian computation.')
    parser.add_argument('--prior_precision', type=float, default=1e-5, help='Prior precision for Laplace approximation (influences variance of weights).')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature scaling for Laplace posterior.')

    args = parser.parse_args()
    main(args)