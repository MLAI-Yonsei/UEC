'''
Multilingual Setting with MIRACLHardNegative
'''
import os, tqdm, warnings, sys
import argparse # Added for argparse
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
from sentence_transformers import SentenceTransformer
import utils.utils as utils

from exp_miracls.ml_utils import (
    compute_and_cache_embeddings, 
    evaluate_model, 
    evaluate_weighted_model, 
    grid_search_weights, 
    append_average_row,
    EmbeddingCache,
    perform_and_log_evaluation,
    print_mode_average_scores,
    display_results_table
)

warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
SEED = 2025
utils.set_seed(SEED)
print(f"[Step 0] Set seed as {SEED}..")

EMBEDDING_CACHE_FILE = "embedding_cache.pt"
embeddings_loaded_from_file = False # Default, will be updated by load logic

k_eval = 10
metrics = ["ndcg", "recall", "nauc"]
batch_size = 128

# Evaluation mode settings
EVAL_MODES = {
    "individual": True,     # Individual model evaluation
    "baseline" : False,
    "uniform": False,        # Uniform weight evaluation
    "weighted": False,       # Optimal weight evaluation
}

model_names = [
    "intfloat/e5-base-v2",
    "BAAI/bge-base-zh-v1.5",
    "silma-ai/silma-embedding-matryoshka-v0.1"
]

models = {
        "en": SentenceTransformer(model_names[0]),  # English model
        "zh": SentenceTransformer(model_names[1]),  # Chinese model
        "ar": SentenceTransformer(model_names[2])   # Arabic model
        }

languages = ["en", "ru", "zh", "ar"]
miracl_data = {}
for lang in languages:
    try:
        dataset = load_dataset("miracl/miracl", lang, split="dev", trust_remote_code=True)
        sampled = dataset.select(range(min(len(dataset), 100)))
        miracl_data[lang] = sampled
    except Exception as e:
        print(f"  ❌ Failed to load {lang}: {str(e)}")    
    

# Cache initialization or loading
# This section replaces the original "cache = EmbeddingCache()"
if os.path.exists(EMBEDDING_CACHE_FILE):
    print(f"Attempting to load embeddings from cache file: {EMBEDDING_CACHE_FILE}...")
    try:
        cache = torch.load(EMBEDDING_CACHE_FILE) # EmbeddingCache class must be defined before this line
        # Basic validation of the loaded cache
        if isinstance(cache, EmbeddingCache) and hasattr(cache, 'query_embs') and cache.query_embs:
             print("  ✅ Embeddings loaded successfully from file.")
             embeddings_loaded_from_file = True
        else:
            print("  ⚠️ Cached object is not a valid or non-empty EmbeddingCache. Initializing new cache.")
            cache = EmbeddingCache() # Initialize fresh cache if loaded one is invalid
            embeddings_loaded_from_file = False # Treat as not loaded if invalid
    except Exception as e:
        print(f"  ❌ Failed to load embeddings from {EMBEDDING_CACHE_FILE}: {str(e)}")
        print("     Proceeding with new cache initialization.")
        cache = EmbeddingCache() # Initialize fresh cache on error
        embeddings_loaded_from_file = False # Treat as not loaded on error
else:
    print(f"No cache file found at {EMBEDDING_CACHE_FILE}. Initializing new cache.")
    cache = EmbeddingCache()
    embeddings_loaded_from_file = False # Explicitly false if no file


NORMALIZE_EMBEDDINGS = True




# Embedding computation and caching (conditional execution and saving)
if not embeddings_loaded_from_file:
    # This is the primary expensive computation step we want to cache for models.
    print("[Custom] Primary embeddings not found in cache or load failed. Computing...")
    compute_and_cache_embeddings(models, miracl_data, cache, batch_size=batch_size, normalize_embeddings=NORMALIZE_EMBEDDINGS)
    
    print(f"Saving computed primary embeddings to {EMBEDDING_CACHE_FILE}...")
    try:
        torch.save(cache, EMBEDDING_CACHE_FILE)
        print(f"  ✅ Embeddings saved successfully to {EMBEDDING_CACHE_FILE}.")
    except Exception as e:
        print(f"  ❌ Failed to save embeddings to {EMBEDDING_CACHE_FILE}: {str(e)}")
else:
    print("[Custom] Primary embeddings loaded from cache. Skipping re-computation.")


# Dictionary for storing results
results = {}

# 1. Individual model evaluation
if EVAL_MODES["individual"]:
    print("\nEvaluating individual models...")
    
    results["individual"] = {}
    
    for model_name_key in models: # model_name_key will be "en", "zh", "ar" etc.
        results["individual"][model_name_key] = {}
        print(f"\n📊 Evaluating {model_name_key} model...")
        
        for lang_eval in cache.query_embs: # lang_eval is the language of the dataset being evaluated on
            results["individual"][model_name_key][lang_eval] = {}
            
            for metric in metrics:
                # evaluate_model uses model_name_key to fetch correct embeddings from cache
                score = evaluate_model(model_name_key, cache, lang_eval, metric, k_eval)
                results["individual"][model_name_key][lang_eval][f"{metric}@{k_eval}"] = score
                print(f"  📌 {lang_eval} - {metric}@{k_eval}: {score:.4f}")
        
        # Calculate average performance for the current model_name_key across evaluated languages
        for metric in metrics:
            metric_key = f"{metric}@{k_eval}"
            # Ensure we check against evaluated languages for this model in results
            langs_with_data = [
                lang for lang in results["individual"][model_name_key] 
                if metric_key in results["individual"][model_name_key][lang]
            ]
            
            if langs_with_data:
                scores = [results["individual"][model_name_key][lang][metric_key] for lang in langs_with_data]
                avg_score = torch.tensor(scores).mean().item()
                print(f"  📊 Average {metric_key} (for {model_name_key} model): {avg_score:.4f}")
            else:
                # Ensure this print matches or is slightly adjusted if needed
                print(f"  ❌ No evaluation data for {model_name_key} to average for {metric_key}")

        # --- Performance Table for CURRENT model_name_key ---
        print(f"\n📋 Performance for Model: {model_name_key}")
        header_current_model = ["Language"] + [f"{metric}@{k_eval}" for metric in metrics]
        table_data_current_model = []

        if model_name_key in results["individual"] and results["individual"][model_name_key]:
            # Iterate through languages this model was evaluated on
            for lang_eval_table, metric_scores_for_lang in results["individual"][model_name_key].items():
                row_data = [lang_eval_table] # First column is the evaluation language
                for metric_m in metrics:
                    metric_key_m = f"{metric_m}@{k_eval}"
                    score_val = metric_scores_for_lang.get(metric_key_m, 0.0) # Default to 0.0
                    row_data.append(f"{score_val:.4f}")
                table_data_current_model.append(row_data)
            
            # Add average row for this model_name_key across the languages it was evaluated on.
            # The data for append_average_row is results["individual"][model_name_key], 
            # which is a dict of {lang_eval: {metric_scores}}
            append_average_row(
                table_data_current_model, 
                results["individual"][model_name_key], 
                "Average", 
                metrics, 
                k_eval
            )
        
        if table_data_current_model:
            print(tabulate(table_data_current_model, headers=header_current_model, tablefmt="grid"))
        else:
            print(f"  No results for model {model_name_key} to display in table.")

    # The old consolidated "Individual Model Performance" table block that was here previously
    # (after this loop finished) is now removed, as tables are printed per model.



if EVAL_MODES["baseline"]:
    print("\n Evaluating baseline models...")
    
    model_name = "Alibaba-NLP/gte-modernbert-base"
    
    model = {model_name : SentenceTransformer(model_name)}
    
    results["baseline"] = {}
    results["baseline"][model_name] = {}
    
    ### caching
    compute_and_cache_embeddings(model, miracl_data, cache, batch_size=batch_size, normalize_embeddings=NORMALIZE_EMBEDDINGS)
    
    print(f"\n📊 Evaluating {model_name} model...")
        
    for lang in cache.query_embs:
        results["baseline"][model_name][lang] = {}
        
        for metric in metrics:
            score = evaluate_model(model_name, cache, lang, metric, k_eval)
            results["baseline"][model_name][lang][f"{metric}@{k_eval}"] = score
            print(f"  📌 {lang} - {metric}@{k_eval}: {score:.4f}")
    
    # Calculate average performance
    for metric in metrics:
        metric_key = f"{metric}@{k_eval}"
        langs_with_data = [lang for lang in cache.query_embs if model_name in cache.query_embs[lang]]
        
        if langs_with_data:
            scores = [results["baseline"][model_name][lang][metric_key] for lang in langs_with_data]
            avg_score = torch.tensor(scores).mean().item()
            print(f"  📊 Average {metric_key}: {avg_score:.4f}")
        else:
            print(f"  ❌ No evaluation data for {model_name}")

    # --- Baseline Model Performance Table ---
    print("\n📋 Baseline Model Performance:")
    # Assuming model_name is defined from the baseline evaluation loop
    # If multiple baseline models were possible, this would need adjustment
    # For now, it seems to be a single model_name like "Alibaba-NLP/gte-modernbert-base"
    
    # model_name variable should still be in scope from the baseline evaluation section
    # Need to ensure 'model_name' used here is the one from the baseline block
    baseline_model_name_for_table = "Alibaba-NLP/gte-modernbert-base" # Hardcoding for safety, or ensure it's passed/scoped
    if "baseline" in results and results["baseline"]: # Check if baseline results exist and are not empty
        # If there could be other baseline models, this title needs to be dynamic
        # For now, assuming one baseline model as per current script structure
        actual_baseline_model_name = list(results["baseline"].keys())[0] # Get the actual name used
        
        header_baseline = [f"Model ({actual_baseline_model_name}) / Language"] + [f"{metric}@{k_eval}" for metric in metrics]
        table_data_baseline = []

        for lang_n, lang_metric_scores in results["baseline"][actual_baseline_model_name].items():
            row = [f"{lang_n}"] # Language is the primary varying item for a single baseline model
            for metric_n in metrics:
                score_val = lang_metric_scores.get(f"{metric_n}@{k_eval}", 0.0)
                row.append(f"{score_val:.4f}")
            table_data_baseline.append(row)
        
        # Add average for the baseline model
        append_average_row(table_data_baseline, results["baseline"][actual_baseline_model_name], "Average", metrics, k_eval)
        
        if table_data_baseline:
            print(tabulate(table_data_baseline, headers=header_baseline, tablefmt="grid"))
        else:
            print(f"  No baseline results for {actual_baseline_model_name} to display in table.")
    else:
        print("  No baseline results to display in table.")




# 2. Uniform weight evaluation
if EVAL_MODES["uniform"]:
    print("\n Evaluating uniform weight averaging...")
    
    results["uniform"] = {}
    
    model_names_list = list(models.keys())
    uniform_weights = [1/len(model_names_list) for _ in model_names_list]
    
    for lang in cache.query_embs:
        results["uniform"][lang] = {}
        
        for metric in metrics:
            score = evaluate_weighted_model(uniform_weights, model_names_list, cache, lang, metric, k_eval)
            results["uniform"][lang][f"{metric}@{k_eval}"] = score
            print(f"  📌 {lang} - {metric}@{k_eval}: {score:.4f}")
    
    # Calculate average performance
    for metric in metrics:
        metric_key = f"{metric}@{k_eval}"
        scores = [results["uniform"][lang][metric_key] for lang in cache.query_embs]
        avg_score = torch.tensor(scores).mean().item()
        print(f"  📊 Average {metric_key}: {avg_score:.4f}")

    # --- Uniform Weight Performance Table ---
    print("\n📋 Uniform Weight Performance:")
    header_uniform = ["Language"] + [f"{metric}@{k_eval}" for metric in metrics]
    table_data_uniform = []
    if "uniform" in results:
        for lang_n, lang_metric_scores in results["uniform"].items():
            row = [lang_n]
            for metric_n in metrics:
                score_val = lang_metric_scores.get(f"{metric_n}@{k_eval}", 0.0)
                row.append(f"{score_val:.4f}")
            table_data_uniform.append(row)
        # Add overall average for uniform weights
        append_average_row(table_data_uniform, results["uniform"], "Average", metrics, k_eval)

    if table_data_uniform:
        print(tabulate(table_data_uniform, headers=header_uniform, tablefmt="grid"))
    else:
        print("  No uniform weight results to display in table.")



# 3. Optimal weight evaluation
if EVAL_MODES["weighted"]:
    print("\n Finding optimal weights using grid search...")
    
    results["weighted"] = {
        "ndcg": {},
        "recall": {}
    }
    model_names_list = list(models.keys())
    
    # Grid search for optimal weights based on nDCG
    print("\n Grid search using nDCG metric...")
    best_weights_ndcg, best_ndcg = grid_search_weights(model_names_list, cache, "ndcg", k_eval, num_steps=5)
    
    # Evaluate with optimal nDCG weights
    print("\n Evaluating with optimal nDCG weights...")
    
    for lang in cache.query_embs:
        results["weighted"]["ndcg"][lang] = {}
        
        for metric in metrics:
            score = evaluate_weighted_model(best_weights_ndcg, model_names_list, cache, lang, metric, k_eval)
            results["weighted"]["ndcg"][lang][f"{metric}@{k_eval}"] = score
            print(f"  📌 {lang} - {metric}@{k_eval}: {score:.4f}")

    # --- Weighted Performance (nDCG-optimized) Table ---
    print("\n📋 Weighted Performance (nDCG-optimized):")
    header_weighted_ndcg = ["Language"] + [f"{metric}@{k_eval}" for metric in metrics]
    table_data_weighted_ndcg = []
    if "weighted" in results and "ndcg" in results["weighted"]:
        for lang_n, lang_metric_scores in results["weighted"]["ndcg"].items():
            row = [lang_n]
            for metric_n in metrics:
                score_val = lang_metric_scores.get(f"{metric_n}@{k_eval}", 0.0)
                row.append(f"{score_val:.4f}")
            table_data_weighted_ndcg.append(row)
        # Add overall average for nDCG-optimized weighted results
        append_average_row(table_data_weighted_ndcg, results["weighted"]["ndcg"], "Average", metrics, k_eval)

    if table_data_weighted_ndcg:
        print(tabulate(table_data_weighted_ndcg, headers=header_weighted_ndcg, tablefmt="grid"))
    else:
        print("  No nDCG-optimized weighted results to display in table.")
    
    # Note: If recall-optimized weights were also searched and evaluated,
    # a similar table block would be added here for results["weighted"]["recall"].
    # Current script structure only grid searches for "ndcg".


print("\n✅ Evaluation complete!")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Multilingual Evaluation with MIRACL")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed for reproducibility.")
    parser.add_argument('--embedding_cache_file', type=str, default="embedding_cache.pt", help="Path to the embedding cache file.")
    parser.add_argument('--k_eval', type=int, default=10, help="Value of k for recall@k, ndcg@k, etc.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for embedding computation.")
    parser.add_argument('--max_dataset_samples', type=int, default=25, help="Maximum number of samples per language from MIRACL dev set.")
    parser.add_argument('--languages', nargs='+', default=["en", "ru", "zh", "ar"], help="List of languages to evaluate on.")
    
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument('--normalize_embeddings', action='store_true', dest='normalize_embeddings', help="Normalize embeddings (default).")
    norm_group.add_argument('--no-normalize_embeddings', action='store_false', dest='normalize_embeddings', help="Do not normalize embeddings.")
    parser.set_defaults(normalize_embeddings=True)

    parser.add_argument('--grid_search_steps', type=int, default=5, help="Number of steps for grid search in weighted evaluation.")
    parser.add_argument('--eval_modes', nargs='+', choices=["individual", "baseline", "uniform", "weighted"], 
                        default=["individual", "uniform", "weighted"], 
                        help="List of evaluation modes to run.")

    return parser.parse_args()



def main(args):
    utils.set_seed(args.seed)
    print(f"[Step 0] Set seed as {args.seed}..")

    # Update os.environ if hf_home and hf_datasets_cache are made configurable
    # os.environ['HF_HOME'] = args.hf_home
    # os.environ['HF_DATASETS_CACHE'] = args.hf_datasets_cache

    # EVAL_MODES dictionary based on parsed arguments
    eval_modes_config = {mode: (mode in args.eval_modes) for mode in ["individual", "baseline", "uniform", "weighted"]}

    # Define primary models (these keys "en", "zh", "ar" are fixed identifiers for these specific models)
    # Their usage in evaluation will depend on whether their embeddings are in the cache for the evaluated languages.
    primary_models = {
        "en": SentenceTransformer(model_names[0]),
        "zh": SentenceTransformer(model_names[1]),
        "ar": SentenceTransformer(model_names[2])
    }
    
    print("[Step 1] Loading MIRACL dataset...")
    miracl_data = {}
    for lang in args.languages:
        try:
            dataset = load_dataset("miracl/miracl", lang, split="dev", trust_remote_code=True)
            sampled = dataset.select(range(min(len(dataset), args.max_dataset_samples)))
            miracl_data[lang] = sampled
            print(f"  ✅ Loaded {lang} with {len(sampled)} samples.")
        except Exception as e:
            print(f"  ❌ Failed to load {lang}: {str(e)}")
    
    if not miracl_data:
        print("No data loaded. Exiting.")
        return

    # Cache initialization or loading
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

    # Embedding computation and caching
    if not embeddings_loaded_from_file:
        print("[Custom] Primary embeddings not found in cache or load failed. Computing...")
        # compute_and_cache_embeddings expects the `models` dictionary (here, primary_models)
        # and miracl_data (which is already filtered by args.languages)
        compute_and_cache_embeddings(primary_models, miracl_data, cache, batch_size=args.batch_size, normalize_embeddings=args.normalize_embeddings)
        
        print(f"Saving computed primary embeddings to {args.embedding_cache_file}...")
        try:
            torch.save(cache, args.embedding_cache_file)
            print(f"  ✅ Embeddings saved successfully to {args.embedding_cache_file}.")
        except Exception as e:
            print(f"  ❌ Failed to save embeddings to {args.embedding_cache_file}: {str(e)}")
    else:
        print("[Custom] Primary embeddings loaded from cache. Skipping re-computation.")

    results = {} # Dictionary for storing all results

    # 1. Individual model evaluation
    if eval_modes_config["individual"]:
        print("\nEvaluating individual models...")
        results["individual"] = {}
        
        # The keys of `primary_models` ("en", "zh", "ar") are iterated here.
        # These keys refer to specific pre-trained models.
        for model_key in primary_models.keys(): 
            results["individual"][model_key] = {}
            print(f"\n📊 Evaluating {model_key} model (e.g., {model_names[list(primary_models.keys()).index(model_key)]})...")
            
            # Evaluate this model_key on all languages for which data (and thus embeddings) exist in the cache.
            # cache.query_embs keys are the actual languages loaded from miracl_data (args.languages)
            for eval_lang in cache.query_embs.keys(): 
                if model_key in cache.query_embs[eval_lang]: # Check if this model's embeddings exist for this language
                    results["individual"][model_key][eval_lang] = {}
                    for metric_name in metrics:
                        score = evaluate_model(model_key, cache, eval_lang, metric_name, args.k_eval)
                        results["individual"][model_key][eval_lang][f"{metric_name}@{args.k_eval}"] = score
                        print(f"  📌 {eval_lang} - {metric_name}@{args.k_eval}: {score:.4f}")
                else:
                    # This case implies embeddings for model_key were not computed/cached for eval_lang
                    # This might happen if compute_and_cache_embeddings selectively skips based on model_key vs language
                    # For now, we assume compute_and_cache_embeddings processes all models in primary_models for all languages in miracl_data
                    print(f"  ℹ️ No cached embeddings for model '{model_key}' on language '{eval_lang}'. Skipping its evaluation for this pair.")


            # Calculate and print average performance for the current model_key
            # The structure of results["individual"][model_key] is {lang: {metric_scores}}
            for metric_name_avg in metrics:
                metric_key_avg = f"{metric_name_avg}@{args.k_eval}"
                current_model_scores = [
                    lang_res_dict[metric_key_avg]
                    for lang_res_dict in results["individual"][model_key].values()
                    if isinstance(lang_res_dict, dict) and metric_key_avg in lang_res_dict
                ]
                if current_model_scores:
                    avg_sc = torch.tensor(current_model_scores).mean().item()
                    print(f"  📊 Average {metric_key_avg} (for {model_key} model): {avg_sc:.4f}")
                else:
                    print(f"  ❌ No evaluation data for {model_key} model to average for {metric_key_avg}")

            # Performance Table for the current model_key
            print(f"\n📋 Performance for Model: {model_key}")
            header_current_model = ["Language"] + [f"{m}@{args.k_eval}" for m in metrics]
            table_data_current_model = []

            if model_key in results["individual"] and results["individual"][model_key]:
                for lang_tbl, metric_scores_tbl in results["individual"][model_key].items():
                    row_data = [lang_tbl]
                    for metric_tbl in metrics:
                        score_val = metric_scores_tbl.get(f"{metric_tbl}@{args.k_eval}", 0.0)
                        row_data.append(f"{score_val:.4f}")
                    table_data_current_model.append(row_data)
                
                append_average_row(
                    table_data_current_model, 
                    results["individual"][model_key], 
                    "Average", 
                    metrics, 
                    args.k_eval
                )
            
            if table_data_current_model:
                print(tabulate(table_data_current_model, headers=header_current_model, tablefmt="grid"))
            else:
                print(f"  No results for model {model_key} to display in table.")

    # Baseline model evaluation
    if eval_modes_config["baseline"]:
        print("\nEvaluating baseline model...")
        baseline_model_hf_name = "Alibaba-NLP/gte-modernbert-base"
        
        # Create a temporary model dict for baseline to pass to compute_and_cache
        temp_baseline_model_dict = {baseline_model_hf_name: SentenceTransformer(baseline_model_hf_name)}
        
        results["baseline"] = {baseline_model_hf_name: {}} # Store results under its HF name

        
        print(f"Computing/caching embeddings for baseline model: {baseline_model_hf_name} if not present...")
        compute_and_cache_embeddings(temp_baseline_model_dict, miracl_data, cache, batch_size=args.batch_size, normalize_embeddings=args.normalize_embeddings)
        print(f"Re-saving embeddings to {args.embedding_cache_file} after baseline processing...")
        try:
            torch.save(cache, args.embedding_cache_file)
            print(f"  ✅ Embeddings saved successfully.")
        except Exception as e:
            print(f"  ❌ Failed to save embeddings: {str(e)}")


        print(f"\n📊 Evaluating {baseline_model_hf_name} model...")
        for eval_lang in cache.query_embs.keys(): # Evaluate on languages present in cache
            if baseline_model_hf_name in cache.query_embs[eval_lang]:
                results["baseline"][baseline_model_hf_name][eval_lang] = {}
                for metric_name in metrics:
                    score = evaluate_model(baseline_model_hf_name, cache, eval_lang, metric_name, args.k_eval)
                    results["baseline"][baseline_model_hf_name][eval_lang][f"{metric_name}@{args.k_eval}"] = score
                    print(f"  📌 {eval_lang} - {metric_name}@{args.k_eval}: {score:.4f}")
            else:
                 print(f"  ℹ️ No cached embeddings for baseline model '{baseline_model_hf_name}' on language '{eval_lang}'. Skipping.")       

        # Average performance for baseline model
        for metric_name_avg in metrics:
            metric_key_avg = f"{metric_name_avg}@{args.k_eval}"
            baseline_scores = [
                lang_res_dict[metric_key_avg]
                for lang_res_dict in results["baseline"][baseline_model_hf_name].values()
                if isinstance(lang_res_dict, dict) and metric_key_avg in lang_res_dict
            ]
            if baseline_scores:
                avg_sc = torch.tensor(baseline_scores).mean().item()
                print(f"  📊 Average {metric_key_avg}: {avg_sc:.4f}")
            else:
                print(f"  ❌ No evaluation data for {baseline_model_hf_name} to average for {metric_key_avg}.")

        # Baseline Model Performance Table
        print("\n📋 Baseline Model Performance:")
        header_baseline = [f"Language (for {baseline_model_hf_name})"] + [f"{m}@{args.k_eval}" for m in metrics]
        table_data_baseline = []
        if baseline_model_hf_name in results["baseline"] and results["baseline"][baseline_model_hf_name]:
            for lang_tbl, metric_scores_tbl in results["baseline"][baseline_model_hf_name].items():
                row_data = [lang_tbl]
                for metric_tbl in metrics:
                    score_val = metric_scores_tbl.get(f"{metric_tbl}@{args.k_eval}", 0.0)
                    row_data.append(f"{score_val:.4f}")
                table_data_baseline.append(row_data)
            append_average_row(table_data_baseline, results["baseline"][baseline_model_hf_name], "Average", metrics, args.k_eval)
        
        if table_data_baseline:
            print(tabulate(table_data_baseline, headers=header_baseline, tablefmt="grid"))
        else:
            print(f"  No baseline results for {baseline_model_hf_name} to display in table.")


    # Uniform weight evaluation
    if eval_modes_config["uniform"]:
        print("\nEvaluating uniform weight averaging...")
        results["uniform"] = {}
        
        # Use keys from primary_models for uniform weighting
        primary_model_keys_list = list(primary_models.keys())
        if not primary_model_keys_list:
            print("  ⚠️ No primary models defined for uniform weighting. Skipping.")
        else:
            uniform_weights = [1/len(primary_model_keys_list) for _ in primary_model_keys_list]
            
            for eval_lang in cache.query_embs.keys(): # Evaluate on languages present in cache
                # Check if all necessary model embeddings for this language are available
                all_models_present_for_lang = True
                for p_model_key in primary_model_keys_list:
                    if not (eval_lang in cache.query_embs and p_model_key in cache.query_embs[eval_lang]):
                        all_models_present_for_lang = False
                        print(f"  ℹ️ Missing embeddings for model '{p_model_key}' on language '{eval_lang}' for uniform weighting. Skipping {eval_lang}.")
                        break
                
                if all_models_present_for_lang:
                    results["uniform"][eval_lang] = {}
                    for metric_name in metrics:
                        score = evaluate_weighted_model(uniform_weights, primary_model_keys_list, cache, eval_lang, metric_name, args.k_eval)
                        results["uniform"][eval_lang][f"{metric_name}@{args.k_eval}"] = score
                        print(f"  📌 {eval_lang} - {metric_name}@{args.k_eval}: {score:.4f}")
            
            # Calculate average performance for uniform weights
            for metric_name_avg in metrics:
                metric_key_avg = f"{metric_name_avg}@{args.k_eval}"
                uniform_scores = [
                    lang_res_dict[metric_key_avg]
                    for lang_res_dict in results["uniform"].values()
                    if isinstance(lang_res_dict, dict) and metric_key_avg in lang_res_dict
                ]
                if uniform_scores:
                    avg_sc = torch.tensor(uniform_scores).mean().item()
                    print(f"  📊 Average {metric_key_avg}: {avg_sc:.4f}")
                else:
                     print(f"  ❌ No uniform weight evaluation data to average for {metric_key_avg}.")   

            # Uniform Weight Performance Table
            print("\n📋 Uniform Weight Performance:")
            header_uniform = ["Language"] + [f"{m}@{args.k_eval}" for m in metrics]
            table_data_uniform = []
            if "uniform" in results and results["uniform"]:
                for lang_tbl, metric_scores_tbl in results["uniform"].items():
                    row_data = [lang_tbl]
                    for metric_tbl in metrics:
                        score_val = metric_scores_tbl.get(f"{metric_tbl}@{args.k_eval}", 0.0)
                        row_data.append(f"{score_val:.4f}")
                    table_data_uniform.append(row_data)
                append_average_row(table_data_uniform, results["uniform"], "Average", metrics, args.k_eval)

            if table_data_uniform:
                print(tabulate(table_data_uniform, headers=header_uniform, tablefmt="grid"))
            else:
                print("  No uniform weight results to display in table.")

    # Optimal weight evaluation
    if eval_modes_config["weighted"]:
        print("\nFinding optimal weights using grid search...")
        results["weighted"] = {"ndcg": {}}
        
        primary_model_keys_list = list(primary_models.keys())
        if not primary_model_keys_list:
             print("  ⚠️ No primary models defined for weighted evaluation. Skipping.")
        else:
            print("\nGrid search using nDCG metric...")
            best_weights_ndcg, best_ndcg_overall_avg = grid_search_weights(
                primary_model_keys_list, cache, "ndcg", args.k_eval, num_steps=args.grid_search_steps
            )
            print(f"  Found best nDCG weights: {best_weights_ndcg} (Overall Avg nDCG@{args.k_eval} based on grid search: {best_ndcg_overall_avg:.4f})")

            print("\nEvaluating with optimal nDCG weights...")
            for eval_lang in cache.query_embs.keys(): # Evaluate on languages present in cache
                all_models_present_for_lang = True
                for p_model_key in primary_model_keys_list:
                    if not (eval_lang in cache.query_embs and p_model_key in cache.query_embs[eval_lang]):
                        all_models_present_for_lang = False
                        print(f"  ℹ️ Missing embeddings for model '{p_model_key}' on language '{eval_lang}' for weighted evaluation. Skipping {eval_lang}.")
                        break

                if all_models_present_for_lang:
                    results["weighted"]["ndcg"][eval_lang] = {}
                    for metric_name in metrics:
                        score = evaluate_weighted_model(best_weights_ndcg, primary_model_keys_list, cache, eval_lang, metric_name, args.k_eval)
                        results["weighted"]["ndcg"][eval_lang][f"{metric_name}@{args.k_eval}"] = score
                        print(f"  📌 {eval_lang} - {metric_name}@{args.k_eval}: {score:.4f}")

            # Weighted Performance (nDCG-optimized) Table
            print("\n📋 Weighted Performance (nDCG-optimized):")
            header_weighted_ndcg = ["Language"] + [f"{m}@{args.k_eval}" for m in metrics]
            table_data_weighted_ndcg = []
            if "weighted" in results and "ndcg" in results["weighted"] and results["weighted"]["ndcg"]:
                for lang_tbl, metric_scores_tbl in results["weighted"]["ndcg"].items():
                    row_data = [lang_tbl]
                    for metric_tbl in metrics:
                        score_val = metric_scores_tbl.get(f"{metric_tbl}@{args.k_eval}", 0.0)
                        row_data.append(f"{score_val:.4f}")
                    table_data_weighted_ndcg.append(row_data)
                append_average_row(table_data_weighted_ndcg, results["weighted"]["ndcg"], "Average", metrics, args.k_eval)

            if table_data_weighted_ndcg:
                print(tabulate(table_data_weighted_ndcg, headers=header_weighted_ndcg, tablefmt="grid"))
            else:
                print("  No nDCG-optimized weighted results to display in table.")

    print("\n✅ Evaluation complete!")



if __name__ == "__main__":
    args = parse_args()
    main(args)