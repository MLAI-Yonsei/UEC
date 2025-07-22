from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ['HF_HOME'] = '/data1/lsj9862/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/data1/lsj9862/huggingface/datasets'

from mteb.models.cache_wrapper import TextVectorMap
import hashlib
import math
import functools

import mteb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define task lists (from run_ensemble.py)
RETRIEVAL_TASKS = [
    "SCIDOCS",
    "LegalBenchCorporateLobbying",
    "BelebeleRetrieval",
    "WikipediaRetrievalMultilingual",
    "StackOverflowQA"
]

CLASSIFICATION_TASKS = [
    "FinancialPhrasebankClassification",
    "SwissJudgementClassification",
    "PoemSentimentClassification",
    "MassiveIntentClassification",
    "TweetTopicSingleClassification"
]

STS_TASKS = [
    "STSBenchmark",
    "FinParaSTS",
    "SICK-R",
    "STS22.v2",
    "SemRel24STS",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS17"
]

def load_prob_embs_maps(
    model_base_cache_paths: list[str | Path], task_name: str
) -> tuple[list[TextVectorMap], list[TextVectorMap]]:
    """
    Load pairs of mean and variance TextVectorMap objects from specified model cache directories for a given task.
    Also normalizes mean (L2) and variance (by squared L2 norm).
    """
    mean_maps = []
    variance_maps = []
    
    if not model_base_cache_paths:
        logger.warning("No model base cache paths provided. Returning empty lists.")
        return [], []

    for i, model_base_path in enumerate(model_base_cache_paths):
        model_base_path = Path(model_base_path)
        mean_map_dir = model_base_path / task_name / "means"
        variance_map_dir = model_base_path / task_name / "variances"

        try:
            logger.debug(f"Loading mean map {i+1}/{len(model_base_cache_paths)} from: {mean_map_dir}")
            if not mean_map_dir.exists():
                logger.error(f"Mean map directory not found: {mean_map_dir}")
                raise FileNotFoundError(f"Mean map directory not found: {mean_map_dir}")
            mean_tvm = TextVectorMap(directory=mean_map_dir)
            mean_tvm.load(name=f"{model_base_path.name}_mean_map_{task_name}")
            # === Mean normalization (L2) ===
            mean_vecs = mean_tvm.vectors
            mean_norms = np.linalg.norm(mean_vecs, axis=1, keepdims=True)
            mean_norms[mean_norms == 0] = 1.0
            mean_tvm.vectors = mean_vecs / mean_norms
            mean_maps.append(mean_tvm)

            logger.debug(f"Loading variance map {i+1}/{len(model_base_cache_paths)} from: {variance_map_dir}")
            if not variance_map_dir.exists():
                logger.error(f"Variance map directory not found: {variance_map_dir}")
                raise FileNotFoundError(f"Variance map directory not found: {variance_map_dir}")
            variance_tvm = TextVectorMap(directory=variance_map_dir)
            variance_tvm.load(name=f"{model_base_path.name}_variance_map_{task_name}")
            # === Variance normalization (L2 norm squared) ===
            var_vecs = variance_tvm.vectors
            var_norms = np.linalg.norm(var_vecs, axis=1, keepdims=True)
            var_norms[var_norms == 0] = 1.0
            variance_tvm.vectors = var_vecs / (var_norms ** 2)
            variance_maps.append(variance_tvm)
        except Exception as e:
            logger.error(f"Error loading map pair from {mean_map_dir} and {variance_map_dir}: {str(e)}")
            raise
    
    logger.info(f"Successfully loaded {len(mean_maps)} mean maps and {len(variance_maps)} variance maps for task {task_name}.")
    return mean_maps, variance_maps


def calculate_item_specific_coeffs_logits(
    item_identifier: str,  # For logging, e.g., "Document hash_123"
    item_vars_dict: dict,  # {model_idx: var_tensor for this item, CPU tensors}
    ordered_model_indices: list[int],
    debugging: bool = False
) -> torch.Tensor:
    """
    Calculate item-specific ensemble coefficients based on variance information.
    Adapted from miracls/run_uec.py for MTEB use case.
    """
    if debugging:
        logger.debug(f"\n--- Debug Coeff Calc for: {item_identifier} ---")
        logger.debug(f"Input item_vars_dict (original) keys: {list(item_vars_dict.keys()) if item_vars_dict else 'None'}")

    # Initial check for necessary inputs
    if not item_vars_dict:  # Variances are always needed
        logger.debug(f"[{item_identifier}] Fallback: item_vars_dict is empty. Returning uniform.")
        return torch.ones(len(ordered_model_indices), dtype=torch.float32) / len(ordered_model_indices)

    # Determine models that have var and consistent flattened dimensions
    valid_models_for_processing = []
    first_valid_dim = -1

    for model_idx in ordered_model_indices:
        if model_idx in item_vars_dict:  # Model must have variance
            var_vec = item_vars_dict[model_idx].flatten()
            current_dim = var_vec.shape[0]
            
            # Dimension consistency check (based on variance dimension)
            if not valid_models_for_processing: 
                first_valid_dim = current_dim
                valid_models_for_processing.append(model_idx)
            elif current_dim == first_valid_dim: 
                valid_models_for_processing.append(model_idx)
            else:
                if debugging: 
                    logger.debug(f"[{item_identifier}] Model {model_idx} skipped: Dimension mismatch ({current_dim} vs {first_valid_dim}).")
    
    if debugging: 
        logger.debug(f"[{item_identifier}] Valid models for processing (after all checks): {valid_models_for_processing}")

    if not valid_models_for_processing:
        if debugging: 
            logger.debug(f"[{item_identifier}] Fallback: No valid models found after consistency checks. Returning uniform.")
        return torch.ones(len(ordered_model_indices), dtype=torch.float32) / len(ordered_model_indices)

    # Stack variances for the valid models
    working_vars_stacked = torch.stack([item_vars_dict[idx] for idx in valid_models_for_processing])

    if debugging: 
        logger.debug(f"[{item_identifier}] Stacked vars (initial, sum): {[v.sum().item() for v in working_vars_stacked]}")

    item_model_raw_scores = {}
    for i, model_idx_present in enumerate(valid_models_for_processing):
        inv_dim_vars = 1.0 / (working_vars_stacked[i] + 1e-30)
        model_raw_score = torch.sum(inv_dim_vars).item()
        item_model_raw_scores[model_idx_present] = model_raw_score
        if debugging: 
            logger.debug(f"[{item_identifier}] Model {model_idx_present}: Final var for score (sum): {working_vars_stacked[i].sum().item():.4e}")
    
    raw_scores_list_ordered = []
    for model_idx_ordered in ordered_model_indices:
        raw_scores_list_ordered.append(item_model_raw_scores.get(model_idx_ordered))

    if debugging: 
        logger.debug(f"[{item_identifier}] Raw scores (ordered): {raw_scores_list_ordered}")

    current_scores_to_process = torch.tensor(raw_scores_list_ordered, dtype=torch.float32)
    
    if debugging:
        logger.debug(f"[{item_identifier}] Returning final_item_coeffs_logits_tensor: {current_scores_to_process.tolist()}")
        logger.debug(f"--- End Debug Coeff Calc for: {item_identifier} ---")
    return current_scores_to_process



def ensemble_prob_embs_uec_doc_specific(
    mean_maps: list[TextVectorMap],
    variance_maps: list[TextVectorMap],
    save_path_means: str | Path,
    save_path_variances: str | Path,
    temperature: float = 1.0,
) -> None:
    """
    Ensembles multiple mean and variance TextVectorMap objects using UEC with document-specific weights.
    Updated to use the improved coefficient calculation method from miracls/run_uec.py.
    """

    if len(mean_maps) != len(variance_maps):
        raise ValueError("Mismatch in the number of mean maps and variance maps for ensembling.")
    
    num_models = len(mean_maps)


    first_mean_map = mean_maps[0]
    num_docs_total = len(first_mean_map.hash_to_index)
    vector_dim_means = first_mean_map.vector_dim
    vector_dim_variances = variance_maps[0].vector_dim  # Assume consistent from loading

    # Initialize arrays for ensembled results
    ensembled_means_data = np.zeros((num_docs_total, vector_dim_means), dtype=np.float32)
    ensembled_variances_data = np.zeros((num_docs_total, vector_dim_variances), dtype=np.float32)

    # To store weights for averaging later
    all_doc_weights_accumulator = np.zeros(num_models, dtype=np.float64)
    num_docs_with_calculated_weights = 0

    reference_hash_to_index = first_mean_map.hash_to_index  # Use first map as reference for iteration order
    ordered_model_indices = list(range(num_models))

    logger.info("Calculating document-specific weights and ensembling...")
    docs_printed_for_variance_debug = 0  # Debug counter
    
    for text_hash, doc_idx_ref in tqdm(reference_hash_to_index.items(), desc="Ensembling doc-specific UEC", total=num_docs_total):
        if docs_printed_for_variance_debug >= 2:  # Stop printing details after 2 docs
            pass  # Allow loop to continue without printing for other docs

        current_doc_vars_dict = {}
        doc_mean_vectors_list = []
        doc_variance_vectors_list = []
        valid_models_for_doc_ensemble_indices = []  # model indices that have BOTH mean and var for this doc

        for model_idx in range(num_models):
            v_map = variance_maps[model_idx]
            m_map = mean_maps[model_idx]
            has_variance = False
            has_mean = False

            # Load variance for this document
            if text_hash in v_map.hash_to_index:
                doc_idx_model_var = v_map.hash_to_index[text_hash]
                if doc_idx_model_var < len(v_map.vectors):
                    var_vector_np = v_map.vectors[doc_idx_model_var]
                    var_vector_torch = torch.from_numpy(var_vector_np.astype(np.float32))
                    if var_vector_torch.ndim > 1: 
                        var_vector_torch = var_vector_torch.flatten()
                    
                    current_doc_vars_dict[model_idx] = var_vector_torch.clone()
                    has_variance = True
                    
                    if docs_printed_for_variance_debug < 2:  # Debug: Print for first 2 docs
                        original_var_sum = torch.sum(var_vector_torch).item()
                        print(f"[DEBUG] Doc: {text_hash}, Model_idx: {model_idx}, OrigVarSum: {original_var_sum:.4f}")
            
            # Load mean for this document
            if text_hash in m_map.hash_to_index:
                doc_idx_model_mean = m_map.hash_to_index[text_hash]
                if doc_idx_model_mean < len(m_map.vectors):
                    doc_mean_vectors_list.append(m_map.vectors[doc_idx_model_mean])  # Keep as numpy for now
                    has_mean = True
                else: 
                    doc_mean_vectors_list.append(np.zeros(vector_dim_means, dtype=np.float32))  # Placeholder if missing
            else: 
                doc_mean_vectors_list.append(np.zeros(vector_dim_means, dtype=np.float32))

            # For variance ensembling, we need a corresponding variance vector
            if has_variance and text_hash in v_map.hash_to_index and v_map.hash_to_index[text_hash] < len(v_map.vectors):
                doc_variance_vectors_list.append(v_map.vectors[v_map.hash_to_index[text_hash]])
            else: 
                doc_variance_vectors_list.append(np.zeros(vector_dim_variances, dtype=np.float32))

            if has_mean and has_variance:  # Only consider models that have both for actual ensembling for this doc
                valid_models_for_doc_ensemble_indices.append(model_idx)

        # Calculate document-specific coefficients using the improved method
        DEBUG_THIS_ITEM = docs_printed_for_variance_debug < 2
        doc_coeffs_logits = calculate_item_specific_coeffs_logits(
            item_identifier=f"Doc {text_hash}",
            item_vars_dict=current_doc_vars_dict,
            ordered_model_indices=ordered_model_indices,

            debugging=DEBUG_THIS_ITEM
        )
        
        # Apply temperature scaling and softmax to get final weights
        if temperature != 1.0:
            doc_coeffs_logits = doc_coeffs_logits / temperature
        
        doc_weights_tensor = torch.softmax(doc_coeffs_logits, dim=0)
        doc_weights = doc_weights_tensor.cpu().tolist()
        
        if DEBUG_THIS_ITEM:
            print(f"doc_coeffs_logits: {doc_coeffs_logits.tolist()} / Softmax: {doc_weights}")
        
        all_doc_weights_accumulator += np.array(doc_weights)  # Accumulate for averaging
        num_docs_with_calculated_weights += 1

        # Ensemble using doc_weights for the current document (doc_idx_ref)
        # Ensure we only use vectors from models that were valid for this doc for ensembling
        
        current_doc_mean_vectors = []
        current_doc_variance_vectors = []
        actual_doc_weights_for_ensemble = []

        for model_idx in range(num_models):  # Iterate through all original model positions
            if model_idx in valid_models_for_doc_ensemble_indices:
                current_doc_mean_vectors.append(doc_mean_vectors_list[model_idx])
                current_doc_variance_vectors.append(doc_variance_vectors_list[model_idx])
                actual_doc_weights_for_ensemble.append(doc_weights[model_idx])
        
        if not current_doc_mean_vectors:  # No valid models for this document to ensemble
            ensembled_means_data[doc_idx_ref, :] = np.zeros(vector_dim_means, dtype=np.float32)
            ensembled_variances_data[doc_idx_ref, :] = np.ones(vector_dim_variances, dtype=np.float32) * 1e6  # High variance for missing
            logger.debug(f"Document {text_hash} (index {doc_idx_ref}) has no valid (mean+var) models for ensembling. Assigning default mean/variance.")
            continue

        # Normalize actual_doc_weights_for_ensemble if only a subset of models were used
        if sum(actual_doc_weights_for_ensemble) == 0:
            actual_doc_weights_for_ensemble = [1.0/len(actual_doc_weights_for_ensemble)] * len(actual_doc_weights_for_ensemble)
        else: 
            sum_w = sum(actual_doc_weights_for_ensemble)
            actual_doc_weights_for_ensemble = [w / sum_w for w in actual_doc_weights_for_ensemble]
        
        doc_weights_arr = np.array(actual_doc_weights_for_ensemble).reshape(len(actual_doc_weights_for_ensemble), 1)
        ensembled_mean_doc = np.sum(np.stack(current_doc_mean_vectors) * doc_weights_arr, axis=0)
        ensembled_variance_doc = np.sum(np.stack(current_doc_variance_vectors) * (doc_weights_arr**2), axis=0)
        
        ensembled_means_data[doc_idx_ref, :] = ensembled_mean_doc
        ensembled_variances_data[doc_idx_ref, :] = ensembled_variance_doc

        # Increment debug counter
        if text_hash in reference_hash_to_index:
            processed_doc_hashes_in_order = list(reference_hash_to_index.keys())
            if processed_doc_hashes_in_order.index(text_hash) < 2:
                docs_printed_for_variance_debug += 1

    # Log average document-specific weights
    if num_docs_with_calculated_weights > 0:
        average_doc_weights = all_doc_weights_accumulator / num_docs_with_calculated_weights
        for model_idx, avg_weight in enumerate(average_doc_weights):
            # Attempt to get model names if mean_maps store them, otherwise just use index
            model_name_hint = mean_maps[model_idx].name if hasattr(mean_maps[model_idx], 'name') and mean_maps[model_idx].name else f"Model {model_idx}"
            logger.info(f"  {model_name_hint}: Average Weight = {avg_weight:.4f}")
    else:
        raise ValueError("No document-specific UEC weights were calculated to average (e.g., all documents might have used uniform fallback).")

    # Save Ensembled Means
    logger.info(f"Saving UEC ensembled mean embeddings to: '{save_path_means}'")
    save_path_means = Path(save_path_means)
    save_path_means.mkdir(parents=True, exist_ok=True)
    merged_mean_map = TextVectorMap(str(save_path_means), initial_vectors=num_docs_total)
    merged_mean_map.hash_to_index = reference_hash_to_index.copy() 
    merged_mean_map.vector_dim = vector_dim_means
    merged_mean_map._initialize_vectors_file() 
    if merged_mean_map.vectors.shape[0] < num_docs_total:
        raise RuntimeError("Mean map vectors not properly initialized or too small for all documents.")
    merged_mean_map.vectors[:num_docs_total] = ensembled_means_data
    merged_mean_map.save()

    # Save Ensembled Variances
    logger.info(f"Saving UEC ensembled variance embeddings to: '{save_path_variances}'")
    save_path_variances = Path(save_path_variances)
    save_path_variances.mkdir(parents=True, exist_ok=True)
    merged_variance_map = TextVectorMap(str(save_path_variances), initial_vectors=num_docs_total)
    merged_variance_map.hash_to_index = reference_hash_to_index.copy()
    merged_variance_map.vector_dim = vector_dim_variances
    merged_variance_map._initialize_vectors_file()
    if merged_variance_map.vectors.shape[0] < num_docs_total:
        raise RuntimeError("Variance map vectors not properly initialized or too small for all documents.")
    merged_variance_map.vectors[:num_docs_total] = ensembled_variances_data
    merged_variance_map.save()

    logger.info("UEC document-specific probabilistic embedding convolution complete.")


def cosine_similarity_variance(query_mean_embeddings: torch.Tensor,
                                            corpus_mean_embeddings: torch.Tensor,
                                            query_variance_embeddings: torch.Tensor,
                                            corpus_variance_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Adapted from pec.py: GaussianConvolution.cosine_similarity_variance
    Calculates the variance term for cosine similarity.
    Assumes query_mean (B_q, D), corpus_mean (B_c, D), query_var (B_q, D), corpus_var (B_c, D).
    Result is (B_q, B_c).
    """
    q_mean_sq = query_mean_embeddings.pow(2)
    c_mean_sq = corpus_mean_embeddings.pow(2)

    # Einsum for batched computation:
    # q_mean_embeddings (query_batch_size, dim), c_variance_embeddings (corpus_batch_size, dim) -> (query_batch_size, corpus_batch_size)
    term1 = torch.einsum('qd,cd->qc', q_mean_sq, corpus_variance_embeddings)
    term2 = torch.einsum('qd,cd->qc', query_variance_embeddings, c_mean_sq)
    term3 = torch.einsum('qd,cd->qc', query_variance_embeddings, corpus_variance_embeddings)
    
    cos_variance = term1 + term2 + term3
    return cos_variance


# New MTEB Model Wrapper
class UncertaintyEnsembleModel:
    def __init__(self, mean_map_path: str | Path, var_map_path: str | Path):
        self.mean_tvm = TextVectorMap(directory=str(mean_map_path))
        self.mean_tvm.load() # Explicitly load the map data

        self.var_tvm = TextVectorMap(directory=str(var_map_path))
        self.var_tvm.load() # Explicitly load the map data

        if self.mean_tvm.vector_dim is None or self.var_tvm.vector_dim is None:
            raise ValueError("Vector dimension not loaded for TextVectorMaps even after calling load().")
        
        self.mean_vector_dim = self.mean_tvm.vector_dim
        self.var_vector_dim = self.var_tvm.vector_dim

    def _get_embeddings_for_sentences(self, sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        means = []
        variances = []
        
        default_mean = np.zeros(self.mean_vector_dim, dtype=np.float32)
        default_variance = np.ones(self.var_vector_dim, dtype=np.float32) # High variance for missing

        for text in sentences:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            mean_vec = default_mean
            if text_hash in self.mean_tvm.hash_to_index:
                idx = self.mean_tvm.hash_to_index[text_hash]
                if idx < len(self.mean_tvm.vectors):
                    mean_vec = self.mean_tvm.vectors[idx]
            else:
                logger.warning(f"Mean embedding for text hash {text_hash} (text: '{text[:50]}...') not found. Using default.")

            var_vec = default_variance
            if text_hash in self.var_tvm.hash_to_index:
                idx = self.var_tvm.hash_to_index[text_hash]
                if idx < len(self.var_tvm.vectors):
                    var_vec = self.var_tvm.vectors[idx]
            else:
                logger.warning(f"Variance embedding for text hash {text_hash} (text: '{text[:50]}...') not found. Using default.")
            
            means.append(mean_vec)
            variances.append(var_vec)
            
        return torch.tensor(np.array(means), dtype=torch.float32), torch.tensor(np.array(variances), dtype=torch.float32)

    def encode_queries(self, queries: list[str], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"Encoding {len(queries)} queries for MTEB.")
        return self._get_embeddings_for_sentences(queries)

    def encode_corpus(self, corpus: list[dict[str, str]] | list[str], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"Encoding corpus of size {len(corpus)} for MTEB.")
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            # Likely list of dicts like [{"title": ..., "text": ...}, ...]
            texts = [doc.get("text", "") for doc in corpus] # Or combine title + text if desired
        elif isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], str):
            texts = corpus
        else:
            texts = [] # Or handle error
            logger.warning("Corpus format not fully recognized or empty.")
        return self._get_embeddings_for_sentences(texts)

    # Add generic encode method
    def encode(self, sentences: list[str], **kwargs) -> torch.Tensor:
        # This generic encode method is used by MTEB internal mechanisms that expect a single tensor.
        # It should return the primary embeddings (means).
        logger.debug(f"Generic encode called with {len(sentences)} sentences, returning means only.")
        means, _ = self._get_embeddings_for_sentences(sentences)
        return means

# New MTEB Similarity Function
def mteb_uncertainty_similarity_fct(
    query_output: tuple[torch.Tensor, torch.Tensor],
    corpus_output: tuple[torch.Tensor, torch.Tensor],
    device: str = "cpu",
    beta: float = 1.0  # Weighting factor for variance penalty
) -> torch.Tensor:
    query_means, query_vars = query_output
    corpus_means, corpus_vars = corpus_output

    # Move tensors to the specified device
    query_means = query_means.to(device)
    query_vars = query_vars.to(device)
    corpus_means = corpus_means.to(device)
    corpus_vars = corpus_vars.to(device)

    # Calculate standard cosine similarity using means
    # Normalize embeddings for cosine similarity
    query_means_norm = torch.nn.functional.normalize(query_means, p=2, dim=1)
    corpus_means_norm = torch.nn.functional.normalize(corpus_means, p=2, dim=1)
    expected_cos_sim = torch.einsum('qd,cd->qc', query_means_norm, corpus_means_norm)

    # Calculate the variance term using the provided function
    cos_uncertainty_term = cosine_similarity_variance(
        query_means, corpus_means, query_vars, corpus_vars # Using original means for this
    )
    
    denominator = 1.0 + 1/torch.pi * beta * torch.abs(cos_uncertainty_term) # Use abs for safety, though expected positive
    final_scores = expected_cos_sim / (denominator + 1e-20) # Add epsilon for numerical stability

    return final_scores



def main():
    parser = argparse.ArgumentParser(description="UEC (document-specific weights) ensemble for cached probabilistic embeddings & MTEB Eval.")
    parser.add_argument(
        "--model_cache_roots",
        type=str,
        default="/data1/lsj9862/bem/cache_embs",
        help="List of root cache directories for the models to be ensembled (e.g., /path/to/cache_proembs/ModelA_xyz)."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="all",
        choices=["retrieval", "classification", "sts", "all"],
        help="Type of tasks to run: retrieval, classification, sts, or all."
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="/data1/lsj9862/bem",
        help="Root directory to save the ensembled cache."
    )
    
    # UEC Configuration Arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Temperature scaling for softmax in coefficient calculation."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Beta parameter for variance penalty."
    )

    # MTEB Evaluation Arguments
    parser.add_argument(
        "--output_root",
        type=str,
        default="uec_results",
        help="Root directory to save MTEB evaluation results."
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for fetching embeddings during MTEB evaluation."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for torch computations during MTEB evaluation (e.g., 'cpu', 'cuda')."
    )

    args = parser.parse_args()

    # Select tasks based on task_type
    if args.task_type == "retrieval":
        task_list = RETRIEVAL_TASKS
        task_group = "retrieval"
    elif args.task_type == "classification":
        task_list = CLASSIFICATION_TASKS
        task_group = "classification"
    elif args.task_type == "sts":
        task_list = STS_TASKS
        task_group = "sts"
    elif args.task_type == "all":
        task_list = RETRIEVAL_TASKS + CLASSIFICATION_TASKS + STS_TASKS
        task_group = "all"
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")

    for task_name in task_list:
        logger.info(f"\n===== Starting UEC (document-specific weights) ensembling process for task: {task_name} =====")
        logger.info(f"Models to ensemble from: {args.model_cache_roots}")
        logger.info(f"UEC configuration: temperature={args.temperature}")

        model_names_for_ensemble = [Path(p).name for p in [
            f"{args.model_cache_roots}/intfloat_e5-base-v2",
            f"{args.model_cache_roots}/BAAI_bge-base-en-v1.5",
            f"{args.model_cache_roots}/thenlper_gte-base"
        ]]
        ensemble_name = f"uec"

        mean_maps, variance_maps = load_prob_embs_maps(
            [
                f"{args.model_cache_roots}/intfloat_e5-base-v2",
                f"{args.model_cache_roots}/BAAI_bge-base-en-v1.5",
                f"{args.model_cache_roots}/thenlper_gte-base"
            ],
            task_name)

        if not mean_maps or not variance_maps:
            logger.error(f"Could not load any maps for task {task_name} from the provided roots. Skipping.")
            continue

        output_ensemble_dir = Path(args.save_root) / ensemble_name / task_name
        save_path_means = output_ensemble_dir / "means"
        save_path_variances = output_ensemble_dir / "variances"

        save_path_means.mkdir(parents=True, exist_ok=True)
        save_path_variances.mkdir(parents=True, exist_ok=True)

        ensemble_prob_embs_uec_doc_specific(
            mean_maps,
            variance_maps,
            save_path_means,
            save_path_variances,
            temperature=args.temperature,
        )

        # === MTEB Evaluation Part ===
        try:
            model_for_mteb = UncertaintyEnsembleModel(
                mean_map_path=save_path_means,
                var_map_path=save_path_variances
            )
        except FileNotFoundError as e:
            logger.error(f"Failed to initialize UncertaintyEnsembleModel due to missing cache files: {e}")
            logger.error("Please ensure UEC ensembling completed successfully and cache files are at the expected paths.")
            continue
        except ValueError as e:
            logger.error(f"Failed to initialize UncertaintyEnsembleModel: {e}")
            continue

        similarity_fct_with_params = functools.partial(
            mteb_uncertainty_similarity_fct,
            device=args.device,
            beta=args.beta,
        )

        evaluation = mteb.MTEB(
            tasks=[task_name],
        )

        mteb_output_folder = Path(args.output_root) / ensemble_name / "mteb_results" / task_group
        mteb_output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"MTEB evaluation results will be saved to: {mteb_output_folder}")

        results = evaluation.run(
            model_for_mteb,
            output_folder=str(mteb_output_folder),
            eval_splits=None,
            similarity_fct=similarity_fct_with_params,
            batch_size=args.batch_size,
        )

        results_file_path = mteb_output_folder / f"{task_name}.json"
        if results is None:
            if results_file_path.exists():
                try:
                    with open(results_file_path, 'r') as f:
                        final_results_data = json.load(f)
                    logger.info(f"Results for {task_name} loaded from {results_file_path}: {json.dumps(final_results_data, indent=2)}")
                except Exception as e:
                    logger.error(f"Failed to load MTEB results from {results_file_path}: {e}")
            else:
                logger.warning(f"MTEB results file not found at {results_file_path} after evaluation.")
        elif results:
            logger.info(f"MTEB.run() returned a direct result object (type: {type(results)}).")
            processed_results_for_json = None
            if hasattr(results, 'scores') and isinstance(results.scores, dict):
                processed_results_for_json = results.scores
                logger.info(f"Extracted scores from TaskResult: {json.dumps(processed_results_for_json, indent=2)}")
            elif isinstance(results, dict):
                try:
                    processed_results_for_json = results
                    logger.info(f"Results for {task_name} (already a dict): {json.dumps(processed_results_for_json, indent=2)}")
                except TypeError:
                    logger.warning(f"Returned dictionary contains non-serializable objects. Attempting to print string representation.")
                    logger.info(f"Results for {task_name} (string representation): {str(results)}")
            else:
                logger.warning(f"Unknown result type returned by MTEB.run(): {type(results)}. Attempting to log string representation.")
                logger.info(f"Results for {task_name} (string representation): {str(results)}")
            if processed_results_for_json:
                results_file_path = mteb_output_folder / f"{task_name}_direct_results.json"
                try:
                    with open(results_file_path, 'w') as f:
                        json.dump(processed_results_for_json, f, indent=2)
                    logger.info(f"Successfully saved directly returned MTEB results to: {results_file_path}")
                except Exception as e:
                    logger.error(f"Failed to save directly returned MTEB results to {results_file_path}: {e}")
        else:
            logger.warning(f"No results returned or saved by MTEB evaluation for task {task_name}.")






if __name__ == "__main__":
    main()
