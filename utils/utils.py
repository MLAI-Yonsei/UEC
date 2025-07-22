import torch
import numpy as np
import math
import torch.nn as nn
from transformers import AutoModel

def set_seed(RANDOM_SEED=0):
    '''
    Set seed for reproduction
    '''
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(RANDOM_SEED)
    
    
    
    
def compute_dcg(relevance_scores, k):
    """Compute DCG@k"""
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))


def compute_ndcg(relevance_scores, k):
    """Compute nDCG@k"""
    ideal_dcg = compute_dcg(sorted(relevance_scores, reverse=True), k)
    actual_dcg = compute_dcg(relevance_scores, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


def compute_recall(relevance_scores, k):
    """Compute Recall@k"""
    num_relevant = sum(1 for score in relevance_scores if score > 0)
    if num_relevant == 0:
        return 0.0
    num_relevant_at_k = sum(1 for score in relevance_scores[:k] if score > 0)
    return num_relevant_at_k / num_relevant