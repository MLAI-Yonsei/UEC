'''
Multilingual Setting with MIRACLHardNegative
'''
import os, tqdm, warnings, sys, argparse
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_HOME'] = '/data1/lsj9862/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/data1/lsj9862/huggingface/datasets'

import torch
from datasets import load_dataset
from tabulate import tabulate
import utils.utils as utils
from utils.EmbeddingModel import EmbeddingModel
from uec import GaussianConvolution
from exp_miracls.ml_utils import load_models, compute_and_cache_embeddings, evaluate_retrieval, EmbeddingCache

warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_item_specific_coeffs_logits(
    item_identifier, # For logging, e.g., "Query 0 (en)"
    item_means_dict, # {model_name: mean_tensor for this item, CPU tensors}
    item_vars_dict, # {model_name: var_tensor for this item, CPU tensors}
    ordered_model_names_list,
    temperature=1.0,
    debugging = False
):
    if debugging:
        print(f"\n--- Debug Coeff Calc for: {item_identifier} ---")
        print(f"Input item_vars_dict (original) keys: {list(item_vars_dict.keys()) if item_vars_dict else 'None'}")

    # Initial check for necessary inputs
    if not item_vars_dict: # Variances are always needed
        print(f"[{item_identifier}] Fallback: item_vars_dict is empty. Returning uniform.")
        return torch.ones(len(ordered_model_names_list), dtype=torch.float32) / len(ordered_model_names_list)

    # Determine models that have var and consistent flattened dimensions
    valid_models_for_processing = []
    first_valid_dim = -1

    for model_name in ordered_model_names_list:
        if model_name in item_vars_dict: # Model must have variance
            var_vec = item_vars_dict[model_name].flatten()
            current_dim = var_vec.shape[0]
            
            # Dimension consistency check (based on variance dimension)
            if not valid_models_for_processing: 
                first_valid_dim = current_dim
                valid_models_for_processing.append(model_name)
            elif current_dim == first_valid_dim: 
                valid_models_for_processing.append(model_name)
            else:
                if debugging: print(f"[{item_identifier}] Model {model_name} skipped: Dimension mismatch ({current_dim} vs {first_valid_dim}).")
    
    if debugging: print(f"[{item_identifier}] Valid models for processing (after all checks): {valid_models_for_processing}")

    if not valid_models_for_processing:
        if debugging: print(f"[{item_identifier}] Fallback: No valid models found after consistency checks. Returning uniform.")
        return torch.ones(len(ordered_model_names_list), dtype=torch.float32) / len(ordered_model_names_list)

    # Stack variances for the valid models
    working_vars_stacked = torch.stack([item_vars_dict[name] for name in valid_models_for_processing])

    if debugging: print(f"[{item_identifier}] Stacked vars (initial, sum): {[v.sum().item() for v in working_vars_stacked]}")

    item_model_raw_scores = {}
    for i, model_name_present in enumerate(valid_models_for_processing):
        inv_dim_vars = 1.0 / (working_vars_stacked[i] + 1e-20)
        model_raw_score = torch.sum(inv_dim_vars).item()  
        
        item_model_raw_scores[model_name_present] = model_raw_score
        if debugging: print(f"[{item_identifier}] Model {model_name_present}: Final var for score (sum): {working_vars_stacked[i].sum().item():.4e}")
    
    raw_scores_list_ordered = []
    for model_n_ordered in ordered_model_names_list:
        raw_scores_list_ordered.append(max(item_model_raw_scores.get(model_n_ordered, 1e-20), 1e-20))

    if debugging: print(f"[{item_identifier}] Raw scores (ordered): {raw_scores_list_ordered}")

    current_scores_to_process = torch.tensor(raw_scores_list_ordered, dtype=torch.float32)
    
    current_scores_to_process /= temperature
    
    if debugging:
        print(f"[{item_identifier}] Returning final_item_coeffs_logits_tensor: {current_scores_to_process.tolist()}")
        print(f"--- End Debug Coeff Calc for: {item_identifier} ---")
    return current_scores_to_process


def main(args):
    # Configuration
    utils.set_seed(args.seed)
    print(f"[Step 0] Set seed as {args.seed}..")

    k_eval = args.k_eval
    metrics = ["ndcg", "recall", "nauc"]
    batch_size = args.batch_size

    # ✅ 1. Model definition
    model_names = args.model_names
    print("[Step 1] Loading models...")
    uec_models = load_models(model_names,
                             use_uec=True,
                             pro_embs="la_models",
                             model_paths=args.model_paths,
                             scales=args.scales)

    # ✅ 2. Load MIRACL dataset
    print("[Step 2] Loading MIRACL dataset (dev set only)...")
    languages = args.languages
    miracl_data = {}
    for lang in languages:
        try:
            dataset = load_dataset("miracl/miracl", lang, split="dev", trust_remote_code=True)
            sampled = dataset.select(range(min(len(dataset), args.max_dataset_samples)))
            miracl_data[lang] = sampled
        except Exception as e:
            print(f"  ❌ Failed to load {lang}: {str(e)}")

    # ✅ 3. Embedding Cache
    embeddings_loaded_from_file = False
    if os.path.exists(args.embedding_cache_file):
        print(f"Attempting to load embeddings from cache file: {args.embedding_cache_file}...")
        try:
            cache = torch.load(args.embedding_cache_file)
            if isinstance(cache, EmbeddingCache) and hasattr(cache, 'query_embs') and cache.query_embs:
                print("  ✅ Embeddings loaded successfully from file.")
                embeddings_loaded_from_file = True
            else:
                print("  ⚠️ Cached object is not a valid or non-empty EmbeddingCache. Initializing new cache.")
                cache = EmbeddingCache()
        except Exception as e:
            print(f"  ❌ Failed to load embeddings from {args.embedding_cache_file}: {str(e)}")
            print("     Proceeding with new cache initialization.")
            cache = EmbeddingCache()
    else:
        print(f"No cache file found at {args.embedding_cache_file}. Initializing new cache.")
        cache = EmbeddingCache()

    if not embeddings_loaded_from_file:
        print("Computing embeddings...")
        compute_and_cache_embeddings(uec_models, miracl_data, cache, batch_size=batch_size, normalize_embeddings=args.normalize_embeddings)
        print(f"Saving computed primary embeddings to {args.embedding_cache_file}...")
        try:
            torch.save(cache, args.embedding_cache_file)
            print(f"  ✅ Embeddings saved successfully to {args.embedding_cache_file}.")
        except Exception as e:
            print(f"  ❌ Failed to save embeddings to {args.embedding_cache_file}: {str(e)}")
    else:
        print("[Custom] Primary embeddings loaded from cache. Skipping re-computation.")

    # ✅ 4. UEC Evaluation
    results = {}
    conv_model = GaussianConvolution(uec_models)
    ordered_model_names = list(uec_models.keys())
    METHOD_NAME = "uec"
    print(f"\n[Step 4] Evaluating with {METHOD_NAME}...")
    results[METHOD_NAME] = {}
    all_query_coeffs_logits_by_lang = {lang_key: [] for lang_key in languages}
    all_doc_coeffs_logits_by_lang = {lang_key: [] for lang_key in languages}

    for lang in languages:
        results[METHOD_NAME][lang] = {}
        print(f"\nProcessing language: {lang} for {METHOD_NAME}...")

        num_queries = len(cache.query_examples[lang])
        num_docs = len(cache.doc_ids[lang])

        # --- 4a. Ensemble Queries ---
        print(f" --- Ensembling queries for {lang}...")
        for q_idx in tqdm.tqdm(range(num_queries), desc=f"Ensembling Queries ({lang})", leave=False):
            current_query_vars_dict = {}
            min_dim_q = -1
            for model_name in ordered_model_names:
                if lang in cache.query_vars and model_name in cache.query_vars[lang] and \
                   q_idx < len(cache.query_vars[lang][model_name]):
                    var_vec = cache.query_vars[lang][model_name][q_idx]
                    if var_vec is not None and var_vec.numel() > 0:
                        var_vec = var_vec.flatten()
                        if min_dim_q == -1: min_dim_q = var_vec.shape[0]
                        elif var_vec.shape[0] != min_dim_q: break
                        current_query_vars_dict[model_name] = var_vec.cpu().clone() 
            
            query_coeffs_logits = calculate_item_specific_coeffs_logits(
                f"Query {q_idx} ({lang})", {}, current_query_vars_dict, ordered_model_names, temperature=args.temperature
            )
            all_query_coeffs_logits_by_lang[lang].append(query_coeffs_logits.cpu().clone())
            with torch.no_grad():
                conv_model.mixture_coeffs.data = query_coeffs_logits.clone().detach().to(conv_model.device)
            try:
                stacked_q_means = torch.stack([torch.stack([cache.query_embs[lang][model_name][q_idx].to(conv_model.device) for q_idx in range(num_queries)]) for model_name in ordered_model_names])
                stacked_q_vars = torch.stack([torch.stack([cache.query_vars[lang][model_name][q_idx].to(conv_model.device) for q_idx in range(num_queries)]) for model_name in ordered_model_names])
                query_means = conv_model.mean_convolution(stacked_q_means)
                query_vars = conv_model.variance_convolution(stacked_q_vars)
            except Exception as e:
                print(f"Error during query GaussianConvolution forward for q_idx {q_idx}, lang {lang}: {e}")

        # --- 4b. Ensemble Documents ---
        print(f" --- Ensembling documents for {lang}...")
        for doc_idx in tqdm.tqdm(range(num_docs), desc=f"Ensembling Documents ({lang})", leave=False):
            current_doc_vars_dict = {}
            min_dim_d = -1
            for model_name in ordered_model_names:
                if lang in cache.doc_vars and model_name in cache.doc_vars[lang] and \
                   doc_idx < len(cache.doc_vars[lang][model_name]):
                    var_vec = cache.doc_vars[lang][model_name][doc_idx]
                    if var_vec is not None and var_vec.numel() > 0:
                        var_vec = var_vec.flatten()
                        if min_dim_d == -1: min_dim_d = var_vec.shape[0]
                        elif var_vec.shape[0] != min_dim_d: break
                        current_doc_vars_dict[model_name] = var_vec.cpu().clone()
            
            doc_coeffs_logits = calculate_item_specific_coeffs_logits(
                f"Doc {doc_idx} ({lang})", {}, current_doc_vars_dict, ordered_model_names, temperature=args.temperature
            )
            all_doc_coeffs_logits_by_lang[lang].append(doc_coeffs_logits.cpu().clone())
            with torch.no_grad():
                conv_model.mixture_coeffs.data = doc_coeffs_logits.clone().detach().to(conv_model.device)
            try:
                stacked_d_means = torch.stack([torch.stack([cache.doc_embs[lang][model_name][doc_idx].to(conv_model.device) for doc_idx in range(num_docs)]) for model_name in ordered_model_names])
                stacked_d_vars = torch.stack([torch.stack([cache.doc_vars[lang][model_name][doc_idx].to(conv_model.device) for doc_idx in range(num_docs)]) for model_name in ordered_model_names])
                doc_means = conv_model.mean_convolution(stacked_d_means)
                doc_vars = conv_model.variance_convolution(stacked_d_vars)
            except Exception as e:
                print(f"Error during document GaussianConvolution forward for doc_idx {doc_idx}, lang {lang}: {e}")

        # --- 4c. Similarity Calculation and Evaluation ---
        print(f" --- Calculating similarity and evaluating for {lang}...")
        similarity = conv_model.cosine_similarity(query_means, query_vars, doc_means, doc_vars, beta=args.beta)
        for metric_name in metrics:
            score = evaluate_retrieval(similarity, cache.doc_ids[lang], cache.query_examples[lang], metric_name, k_eval)
            results[METHOD_NAME][lang][f"{metric_name}@{k_eval}"] = score
            print(f"  📌 {METHOD_NAME} on {lang} - {metric_name}@{k_eval}: {score:.4f}")
        if not results[METHOD_NAME][lang]:
            results[METHOD_NAME][lang] = {f"{m}@{k_eval}": 0.0 for m in metrics}

    # ✅ 5. Final Results
    avg_results_step_new = {f"{m}@{k_eval}": [] for m in metrics}
    for lang_avg in languages:
        if lang_avg in results[METHOD_NAME]:
            for metric_key, score in results[METHOD_NAME][lang_avg].items():
                avg_results_step_new[metric_key].append(score)

    print(f"\n  Average UEC Performance:")
    for metric_key_avg, scores_list in avg_results_step_new.items():
        if scores_list:
            avg_score = torch.tensor(scores_list).mean().item()
            print(f"    📊 Average {metric_key_avg}: {avg_score:.4f}")
        else:
            print(f"    📊 Average {metric_key_avg}: N/A (no scores)")

    print("\n✅ Evaluation complete!")
    final_table = []
    header = ["Language"] + [f"{metric}@{k_eval}" for metric in metrics]
    for lang in languages:
        row = [lang]
        for metric in metrics:
            metric_key = f"{metric}@{k_eval}"
            score = results[METHOD_NAME].get(lang, {}).get(metric_key, 0.0)
            row.append(f"{score:.4f}")
        final_table.append(row)
    avg_row = ["Average"]
    for metric in metrics:
        metric_key = f"{metric}@{k_eval}"
        scores = [results[METHOD_NAME].get(lang, {}).get(metric_key, 0.0) for lang in languages if results[METHOD_NAME].get(lang, {}).get(metric_key) is not None]
        if scores:
            avg_score = torch.tensor(scores).mean().item()
            avg_row.append(f"{avg_score:.4f}")
        else:
            avg_row.append("N/A")
    final_table.append(avg_row)
    print("\n📋 Final Results Table:")
    print(tabulate(final_table, headers=header, tablefmt="grid"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run UEC Evaluation on MIRACL dataset.")
    
    # General
    parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
    
    # Data
    parser.add_argument('--languages', nargs='+', default=["en", "ru", "zh", "ar"], help='List of languages from MIRACL to evaluate.')
    parser.add_argument('--max_dataset_samples', type=int, default=25, help='Max samples per language from MIRACL dev set.')
    parser.add_argument('--embedding_cache_file', type=str, default="uec_embedding_cache.pt", help="Path to the embedding cache file.")

    # Models & UEC
    parser.add_argument('--model_names', nargs='+', required=True, help='List of HuggingFace model names.')
    parser.add_argument('--model_paths', nargs='+', required=True, help='List of paths to the saved Laplace models (must correspond to model_names).')
    parser.add_argument('--scales', nargs='+', type=float, required=True, help='List of scales for UEC models (must correspond to model_names).')
    
    # UEC Hyperparameters
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for UEC softmax.')
    parser.add_argument('--beta', type=float, default=0.1, help='Beta for uncertainty-aware cosine similarity.')

    # Evaluation
    parser.add_argument('--k_eval', type=int, default=10, help='k for evaluation metrics like nDCG@k.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for embedding computation.')
    
    # Normalization
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument('--normalize_embeddings', action='store_true', dest='normalize_embeddings', help="Normalize embeddings (default).")
    norm_group.add_argument('--no-normalize_embeddings', action='store_false', dest='normalize_embeddings', help="Do not normalize embeddings.")
    parser.set_defaults(normalize_embeddings=True)

    args = parser.parse_args()

    # Validation
    if not (len(args.model_names) == len(args.model_paths) == len(args.scales)):
        raise ValueError("The number of model_names, model_paths, and scales must be the same.")

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)