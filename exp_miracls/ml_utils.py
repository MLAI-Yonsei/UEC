import torch
import utils.utils as utils
import tqdm, os
from sentence_transformers import SentenceTransformer
from pec import RetrieveSentenceTransformer
import numpy as np
import tabulate


class EmbeddingCache:
    def __init__(self):
        self.query_embs = {}
        self.query_vars = {} 
        self.doc_embs = {}
        self.doc_vars = {}   
        self.doc_ids = {}
        self.query_examples = {}


def perform_and_log_evaluation(
    results_section: dict,
    cache: EmbeddingCache,
    eval_langs: list,
    metrics_list: list,
    k_eval: int,
    eval_type: str, 
    model_identifier, 
    weights_for_eval: list = None
):
    """Performs evaluation across languages and metrics for a given mode/model."""
    for eval_lang in eval_langs:
        if eval_type == "weighted":
            all_models_present_for_lang = True
            for p_model_key in model_identifier: 
                if not (eval_lang in cache.query_embs and p_model_key in cache.query_embs.get(eval_lang, {})):
                    all_models_present_for_lang = False
                    print(f"  ℹ️ Missing embeddings for model '{p_model_key}' on language '{eval_lang}' for {eval_type} evaluation. Skipping {eval_lang}.")
                    break
            if not all_models_present_for_lang:
                continue
        elif eval_type == "model":
            if not (eval_lang in cache.query_embs and model_identifier in cache.query_embs.get(eval_lang, {})):
                print(f"  ℹ️ No cached embeddings for model '{model_identifier}' on language '{eval_lang}'. Skipping evaluation for this pair.")
                continue

        results_section[eval_lang] = {}
        for metric_name in metrics_list:
            if eval_type == "model":
                score = evaluate_model(model_identifier, cache, eval_lang, metric_name, k_eval)
            elif eval_type == "weighted":
                score = evaluate_weighted_model(weights_for_eval, model_identifier, cache, eval_lang, metric_name, k_eval)
            else:
                raise ValueError(f"Unknown eval_type: {eval_type}")
            
            results_section[eval_lang][f"{metric_name}@{k_eval}"] = score
            print(f"  📌 {eval_lang} - {metric_name}@{k_eval}: {score:.4f}")


def print_mode_average_scores(
    mode_results_data: dict, 
    mode_label: str, 
    metrics_list: list,
    k_eval: int
):
    """Calculates and prints average scores for the mode across evaluated languages."""
    for metric_name_avg in metrics_list:
        metric_key_avg = f"{metric_name_avg}@{k_eval}"
        scores = [
            lang_data[metric_key_avg]
            for lang_data in mode_results_data.values() 
            if isinstance(lang_data, dict) and metric_key_avg in lang_data
        ]
        if scores:
            avg_sc = torch.tensor(scores).mean().item()
            print(f"  📊 Average {metric_key_avg} (for {mode_label}): {avg_sc:.4f}")
        else:
            print(f"  ❌ No evaluation data found for {mode_label} to average for {metric_key_avg}")


def display_results_table(
    mode_results_data: dict, 
    table_title: str,
    first_column_header: str, 
    metrics_list: list,
    k_eval: int
):
    """Generates and prints a results table for the mode."""
    print(f"\n📋 {table_title}")
    header = [first_column_header] + [f"{m}@{k_eval}" for m in metrics_list]
    table_data = []

    if mode_results_data: 
        for item_key, metric_scores_for_item in mode_results_data.items(): 
            row_data = [item_key] 
            for metric_name in metrics_list:
                score_val = metric_scores_for_item.get(f"{metric_name}@{k_eval}", 0.0) 
                row_data.append(f"{score_val:.4f}")
            table_data.append(row_data)
        
        append_average_row(table_data, mode_results_data, "Average", metrics_list, k_eval)
    
    if table_data:
        print(tabulate.tabulate(table_data, headers=header, tablefmt="grid"))
    else:
        print(f"  No results for '{table_title.lower()}' to display in table (mode_results_data was empty or all langs skipped).")


def load_models(model_names, use_uec=False, pro_embs="la", model_paths=None, scales=None):
    """Model loading function
    
    Args:
        use_uec (bool): Whether to use UEC model
        model_paths (list): List of model paths
        scales (float or dict): Scale value. If float, applied equally to all models; if dict, applied per model.
                              If None and use_uec=True, automatically calculates log determinant ratio scales.
    """
    if model_paths is None:
        model_paths = ['/data1/lsj9862/bem/la_models'] * len(model_names)
    
    if use_uec:
        # Load UEC models
        uec_models = {}
        variances = {}
        
        # First pass: load all variances to calculate scales if needed
        if scales is None:
            for model_name, model_path in zip(model_names, model_paths):
                if pro_embs == "la_models":
                    flat_variance = torch.load(os.path.join(model_path, 'la_posterior_variance.pt'))
                    variances[model_name] = flat_variance[768:-768*2]
                elif pro_embs == "ivon":
                    flat_variance = torch.load(os.path.join(model_path, 'ivon_best_hessian.pt'))
                    variances[model_name] = flat_variance[768:-768*2]
                elif pro_embs == "self_hess":
                    flat_variance = torch.load(os.path.join(model_path, 'scale.pt'))
                    variances[model_name] = flat_variance
            
            # Calculate log determinant for each model
            model_log_dets = []
            for model_name in model_names:
                var_vec = variances[model_name]
                if var_vec is not None and isinstance(var_vec, torch.Tensor):
                    # Log determinant = sum(log(variance + epsilon))
                    log_det = torch.sum(torch.log(var_vec + 1e-20)).item()
                    model_log_dets.append(log_det)
                    print(f"  {model_name.split('/')[-1]}: log_det = {log_det:.6f}")
                else:
                    print(f"  ⚠️ Could not read variance for {model_name}. Using log_det = 0.0.")
                    model_log_dets.append(0.0)
            
            # Calculate log determinant ratio scales using -0.5 * log_det
            scaled_log_dets = [-0.5 * log_det for log_det in model_log_dets]
            total_scaled_log_det = sum(scaled_log_dets)
            try:
                log_det_ratios = [scaled_log_det / total_scaled_log_det for scaled_log_det in scaled_log_dets]
                scales = {model_name: ratio**2 for model_name, ratio in zip(model_names, log_det_ratios)}
                for model_name, scale_val in scales.items():
                    print(f"    {model_name.split('/')[-1]}: {scale_val:.6f}")
            except:
                print(f"  ❌ Could not calculate log determinant ratio scales. Using uniform scales.")
                scales = {model_name: 1.0/len(model_names) for model_name in model_names}
        
        # Second pass: load models with calculated scales
        for model_name, model_path in zip(model_names, model_paths):
            if scales is None:
                # Load variance if not already loaded
                if model_name not in variances:
                    if pro_embs == "la_models":
                        flat_variance = torch.load(os.path.join(model_path, 'la_posterior_variance.pt'))
                        variances[model_name] = flat_variance[768:-768*2]
                    elif pro_embs == "ivon":
                        flat_variance = torch.load(os.path.join(model_path, 'ivon_best_hessian.pt'))
                        variances[model_name] = flat_variance[768:-768*2]
                    elif pro_embs == "self_hess":
                        flat_variance = torch.load(os.path.join(model_path, 'scale.pt'))
                        variances[model_name] = flat_variance
            else:
                # Load variance for this model
                if pro_embs == "la_models":
                    flat_variance = torch.load(os.path.join(model_path, 'la_posterior_variance.pt'))
                    variances[model_name] = flat_variance[768:-768*2]
                elif pro_embs == "ivon":
                    flat_variance = torch.load(os.path.join(model_path, 'ivon_best_hessian.pt'))
                    variances[model_name] = flat_variance[768:-768*2]
                elif pro_embs == "self_hess":
                    flat_variance = torch.load(os.path.join(model_path, 'scale.pt'))
                    variances[model_name] = flat_variance
            
            # Apply the scale to the model
            if isinstance(scales, (int, float)):
                scale_value = scales
            elif isinstance(scales, (list, tuple, torch.Tensor)):
                # model_names와 같은 순서라고 가정
                idx = model_names.index(model_name)
                scale_value = scales[idx]
            elif isinstance(scales, dict):
                scale_value = scales.get(model_name, 1e-3)
            else:
                scale_value = 1e-3  # fallback
            uec_models[model_name] = RetrieveSentenceTransformer(
                model_name, 
                variances[model_name], 
                scale=scale_value
            )
        
        return uec_models
    else:
        # Load regular models
        return {
            "en": SentenceTransformer(model_names[0]),  # English model
            "zh": SentenceTransformer(model_names[1]),  # Chinese model
            "ar": SentenceTransformer(model_names[2])   # Arabic model
        }



def compute_and_cache_embeddings(models, miracl_data, cache, batch_size=128, normalize_embeddings=True):
    """Embedding computation and caching function"""
    print("\nComputing and caching embeddings...")
    
    for lang in miracl_data:
        print(f"  📌 Processing language: {lang}")
        dataset = miracl_data[lang]
        
        # Collect documents and queries
        all_docs = []
        doc_ids = []
        for example in dataset:
            for passage in example["positive_passages"] + example["negative_passages"]:
                all_docs.append(passage["text"])
                doc_ids.append(passage["docid"])
        
        all_queries = [example["query"] for example in dataset]
        
        # Initialize cache
        if lang not in cache.query_embs:
            cache.query_embs[lang] = {}
            cache.query_vars[lang] = {}
            cache.doc_embs[lang] = {}
            cache.doc_vars[lang] = {}
            cache.doc_ids[lang] = doc_ids
            cache.query_examples[lang] = dataset
        
        # Compute embeddings for each model
        for model_name, model in models.items():
            print(f"    ➡️ Encoding with {model_name} model...")
            
            # Process RetrieveSentenceTransformer model
            if hasattr(model, 'encode_with_variance'):
                # Compute document embeddings
                doc_embs = []
                doc_vars = []
                for i in range(0, len(all_docs), batch_size):
                    batch = all_docs[i:min(i+batch_size, len(all_docs))]
                    embs, vars = model.encode_with_variance(batch, batch_size=batch_size, 
                                                          normalize_embeddings=normalize_embeddings)
                    doc_embs.append(embs)
                    doc_vars.append(vars)
                
                cache.doc_embs[lang][model_name] = torch.cat(doc_embs, dim=0)
                cache.doc_vars[lang][model_name] = torch.cat(doc_vars, dim=0)
                
                # Compute query embeddings
                query_embs = []
                query_vars = []
                for i in range(0, len(all_queries), batch_size):
                    batch = all_queries[i:min(i+batch_size, len(all_queries))]
                    embs, vars = model.encode_with_variance(batch, batch_size=batch_size,
                                                          normalize_embeddings=normalize_embeddings)
                    query_embs.append(embs)
                    query_vars.append(vars)
                
                cache.query_embs[lang][model_name] = torch.cat(query_embs, dim=0)
                cache.query_vars[lang][model_name] = torch.cat(query_vars, dim=0)
            
            # Process regular SentenceTransformer model
            else:
                # Compute document embeddings
                doc_embs = []
                for i in range(0, len(all_docs), batch_size):
                    batch = all_docs[i:min(i+batch_size, len(all_docs))]
                    embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                    embs = torch.tensor(embs)
                    if normalize_embeddings:
                        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                    doc_embs.append(embs)
                
                cache.doc_embs[lang][model_name] = torch.cat(doc_embs, dim=0)
                
                # Compute query embeddings
                query_embs = []
                for i in range(0, len(all_queries), batch_size):
                    batch = all_queries[i:min(i+batch_size, len(all_queries))]
                    embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                    embs = torch.tensor(embs)
                    if normalize_embeddings:
                        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                    query_embs.append(embs)
                
                cache.query_embs[lang][model_name] = torch.cat(query_embs, dim=0)
    
    print("  ✅ All embeddings computed and cached!")


def evaluate_retrieval(similarity, doc_ids, dataset, metric="ndcg", k=10, is_pem=False):
    """Search performance evaluation function"""
    if metric.lower() == "nauc":
        return calculate_nauc(similarity, doc_ids, dataset, k, is_pem)
        
    scores = []
    for i, example in enumerate(dataset):
        sim = similarity[i]
        
        # Sort based on similarity
        sorted_indices = torch.argsort(sim, descending=True)
        sorted_doc_ids = [doc_ids[idx.item()] for idx in sorted_indices]
        
        # Generate relevance labels
        relevance = []
        pos_doc_ids = [p["docid"] for p in example["positive_passages"]]
        for doc_id in sorted_doc_ids:
            relevance.append(1 if doc_id in pos_doc_ids else 0)
        
        # Compute evaluation metrics
        if metric.lower() == "ndcg":
            score = utils.compute_ndcg(relevance, k)
        elif metric.lower() == "recall":
            score = utils.compute_recall(relevance, k)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    return torch.tensor(scores).mean().item()



def evaluate_model(model_name, cache, lang, metric="ndcg", k=10):
    """Single model evaluation function"""
    if lang not in cache.query_embs or model_name not in cache.query_embs[lang]:
        print(f"  ❌ No cached embeddings for {model_name} on {lang}")
        return 0.0
    
    with torch.no_grad():
        similarity = torch.mm(cache.query_embs[lang][model_name], cache.doc_embs[lang][model_name].T)
    
    return evaluate_retrieval(
        similarity,
        cache.doc_ids[lang],
        cache.query_examples[lang],
        metric,
        k
    )




def evaluate_weighted_model(weights, model_names, cache, lang, metric="ndcg", k=10):
    """Weighted model evaluation function"""
    if lang not in cache.query_embs:
        print(f"  ❌ No cached embeddings for {lang}")
        return 0.0
    
    # Compute weighted average embeddings
    query_embeddings = torch.zeros_like(cache.query_embs[lang][model_names[0]])
    doc_embeddings = torch.zeros_like(cache.doc_embs[lang][model_names[0]])
    
    for i, model_name in enumerate(model_names):
        if weights[i] > 0 and model_name in cache.query_embs[lang]:
            query_embeddings += weights[i] * cache.query_embs[lang][model_name]
            doc_embeddings += weights[i] * cache.doc_embs[lang][model_name]
    
    # L2 normalization for Denominator of Cosine Similarity
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
    
    with torch.no_grad():
        similarity = torch.mm(query_embeddings, doc_embeddings.T)
    
    return evaluate_retrieval(
        similarity,
        cache.doc_ids[lang],
        cache.query_examples[lang],
        metric,
        k
    )
    
    
    
    
    
def grid_search_weights(model_names, cache, metric="ndcg", k=10, num_steps=5):
    """Optimal weight search function"""
    best_weights = None
    best_score = -1
    
    # Generate grid points
    weights_range = torch.linspace(0, 1, num_steps)
    
    def valid_weight_combinations():
        if len(model_names) == 2:
            for w1 in weights_range:
                w2 = 1 - w1
                if w2 >= 0:
                    yield [w1.item(), w2.item()]
        else:  # len(model_names) == 3
            for w1 in weights_range:
                for w2 in weights_range:
                    w3 = 1 - w1 - w2
                    if w3 >= 0 and round(w3.item(), 10) >= 0:
                        yield [w1.item(), w2.item(), w3.item()]
    
    all_combinations = list(valid_weight_combinations())
    print(f"  📊 Testing {len(all_combinations)} weight combinations...")
    
    for weights in tqdm.tqdm(all_combinations, desc="Searching weights"):
        lang_scores = []
        
        for lang in cache.query_embs:
            score = evaluate_weighted_model(weights, model_names, cache, lang, metric, k)
            lang_scores.append(score)
        
        avg_score = torch.tensor(lang_scores).mean().item()
        
        if avg_score > best_score:
            best_score = avg_score
            best_weights = weights
    
    weight_str = ", ".join([f"{model_names[i]}: {w:.4f}" for i, w in enumerate(best_weights)])
    print(f"  🏆 Best weights: [{weight_str}]")
    print(f"  🏆 Best average {metric.upper()}@{k}: {best_score:.4f}")
    
    return best_weights, best_score


def calculate_nauc(similarity_matrix, doc_ids, query_examples, k=10, is_pem=False):
    """
    Calculate nAUC (Normalized Area Under the Metric-Abstention Curve) for retrieval evaluation.
    Based on the paper "When Do Embeddings Accurately Reflect Retrieval Quality?" and MTEB implementation.
    
    Args:
        similarity_matrix: Matrix of similarity scores between queries and documents
        doc_ids: List of document IDs
        query_examples: List of query examples containing relevance information
        k: Number of top results to consider
        is_pem: Whether the similarity matrix is from PEM model
        
    Returns:
        float: nAUC score (0 for random ranking, 1 for perfect ranking)
    """
    total_auc = 0.0
    num_queries = len(query_examples)
    
    for q_idx, query in enumerate(query_examples):
        # Get relevance scores for current query
        relevance_scores = []
        for doc_id in doc_ids:
            is_relevant = any(pos["docid"] == doc_id for pos in query["positive_passages"])
            relevance_scores.append(1.0 if is_relevant else 0.0)
        
        # Get similarity scores (confidence scores) for current query
        confidence_scores = similarity_matrix[q_idx].detach().cpu().numpy()
        
        if is_pem:
            # For PEM, we need to normalize the confidence scores to [0,1] range
            # since they might be outside this range due to variance consideration
            min_conf = np.min(confidence_scores)
            max_conf = np.max(confidence_scores)
            if max_conf > min_conf:
                confidence_scores = (confidence_scores - min_conf) / (max_conf - min_conf)
            else:
                confidence_scores = np.ones_like(confidence_scores)  # If all scores are same
        
        # Sort documents by confidence score
        sorted_indices = np.argsort(-confidence_scores)  # Descending order
        sorted_relevance = np.array(relevance_scores)[sorted_indices]
        sorted_confidence = confidence_scores[sorted_indices]
        
        # Calculate precision at each position
        precisions = []
        num_relevant = 0
        for i in range(len(sorted_relevance)):
            if sorted_relevance[i] == 1:
                num_relevant += 1
            precisions.append(num_relevant / (i + 1))
        
        # Calculate abstention curve using confidence scores
        abstention_curve = []
        confidence_thresholds = np.linspace(0, 1, 100)  # 100 points for smooth curve
        
        for threshold in confidence_thresholds:
            # Find position where confidence drops below threshold
            abstention_pos = next((i for i, conf in enumerate(sorted_confidence) if conf < threshold), len(sorted_confidence))
            abstention_curve.append(abstention_pos / len(sorted_confidence))
        
        # Calculate AUC of abstention curve
        abstention_auc = np.trapz(abstention_curve, dx=1/99)  # dx = 1/(100-1)
        
        # Normalize AUC
        # For random ranking, AUC = 0.5
        # For perfect ranking, AUC = 1.0
        # We want random = 0, perfect = 1
        normalized_auc = 2 * (abstention_auc - 0.5)
        total_auc += normalized_auc
    
    # Average across queries
    nauc = total_auc / num_queries if num_queries > 0 else 0.0
    return nauc




# Helper function to compute and append an average row to a given table data list
def append_average_row(table_list, data_for_average, label_for_row, metrics_list, k_value):
    # data_for_average is e.g. results["individual"]["model_X"] or results["uniform"]
    # label_for_row is e.g. "model_X (Avg)" or "Average"
    avg_row_content = [label_for_row]
    has_any_score = False # To ensure we only add row if there's data

    for m_name in metrics_list: # metrics_list should be the global 'metrics'
        m_key = f"{m_name}@{k_value}" # k_value should be the global 'k_eval'
        current_scores = []
        
        if data_for_average: # Check if dict is not None or empty
            # data_for_average.values() will be list of dicts, where each dict is {metric_key: score, ...} for a language
            for lang_specific_scores_dict in data_for_average.values():
                if m_key in lang_specific_scores_dict:
                    current_scores.append(lang_specific_scores_dict[m_key])
        
        if current_scores:
            avg_s = torch.tensor(current_scores).mean().item()
            avg_row_content.append(f"{avg_s:.4f}")
            has_any_score = True
        else:
            avg_row_content.append("N/A")
    
    if has_any_score: # Only append the average row if at least one metric had a calculated average
        table_list.append(avg_row_content)
        
        


def apply_dimension_wise_normalization(stacked_vars_for_processing):
    """
    Applies dimension-wise normalization to a stack of variance vectors.
    Input: stacked_vars_for_processing (torch.Tensor): Shape (num_models, emb_dim).
    Output: torch.Tensor: Normalized variance vectors.
    """
    sum_per_dimension = torch.sum(stacked_vars_for_processing, dim=0)
    normalized_vars = stacked_vars_for_processing / (sum_per_dimension.unsqueeze(0) + 1e-16)
    return normalized_vars



def apply_variance_clipping(variance_vector, low_percentile, high_percentile):
    """
    Applies percentile-based clipping to a variance vector.
    Input:
        variance_vector (torch.Tensor): A 1D tensor of variances.
        low_percentile (float): The lower percentile for clipping.
        high_percentile (float): The upper percentile for clipping.
    Output: torch.Tensor: Clipped variance vector.
    """
    processed_vars = variance_vector.clone()
    if processed_vars.numel() > 1:
        low_k = max(1, int(low_percentile / 100.0 * processed_vars.numel()))
        high_k = min(processed_vars.numel(), int(high_percentile / 100.0 * processed_vars.numel()))
        
        if low_k > high_k: low_k = high_k
        # The above ensures low_k <= high_k. If high_k becomes 0 (e.g. numel=1, high_percentile small),
        # low_k might become 0. Ensure k is at least 1 for indexing.
        # flat_sorted_vars indexing needs k >= 1.

        # Corrected k value handling for safety, ensuring they are at least 1 and within bounds.
        num_elements = processed_vars.numel()
        low_k = max(1, min(low_k, num_elements))
        high_k = max(1, min(high_k, num_elements))
        if low_k > high_k: low_k = high_k # Final guard

        flat_sorted_vars = processed_vars.sort().values
        low_p_val = flat_sorted_vars[low_k - 1]
        high_p_val = flat_sorted_vars[high_k - 1]
        processed_vars.clamp_(min=low_p_val.item(), max=high_p_val.item())
    return processed_vars




def transform_log_scores(scores_list):
    """
    Applies log transformation to a list of scores.
    Input: scores_list (list of float): List of scores.
    Output: list of float: Log-transformed scores.
    """
    return [torch.log(torch.tensor(s + 1e-12)).item() for s in scores_list]




def normalize_min_max_scores(scores_list, min_val, max_val):
    """
    Applies min-max normalization to a list of scores.
    If all scores are very similar, returns a list of 1.0s.
    Input:
        scores_list (list of float): List of scores.
        min_val (float): Pre-calculated minimum value in the original list.
        max_val (float): Pre-calculated maximum value in the original list.
    Output: list of float: Normalized scores.
    """
    if (max_val - min_val) < 1e-9:
        return [1.0 for _ in scores_list]
    return [(s_val - min_val) / (max_val - min_val + 1e-9) for s_val in scores_list]




def normalize_min_max_scores_for_selection(scores_list, min_val, max_val):
    """
    Applies min-max normalization for selection purposes.
    This function is called when scores are known to be different enough.
    It directly applies the normalization formula.
    Input:
        scores_list (list of float): List of scores.
        min_val (float): Pre-calculated minimum value.
        max_val (float): Pre-calculated maximum value.
    Output: list of float: Normalized scores.
    """
    # Assumes (max_val - min_val) is not critically small, handled by caller.
    # Or relies on the + 1e-9 for stability.
    return [(s_val - min_val) / (max_val - min_val + 1e-9) for s_val in scores_list]




def scale_scores_temperature(scores_list, temperature):
    """
    Applies temperature scaling to a list of scores.
    Input:
        scores_list (list of float): List of scores.
        temperature (float): Temperature value for scaling.
    Output: list of float: Temperature-scaled scores.
    """
    return [s_val / temperature for s_val in scores_list]
