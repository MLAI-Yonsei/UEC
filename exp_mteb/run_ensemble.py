import mteb
from sentence_transformers import SentenceTransformer
from mteb.models.cache_wrapper import CachedEmbeddingWrapper
from mteb.models.cache_wrapper import TextVectorMap
import argparse
import numpy as np
from pathlib import Path
import os

# Define models and tasks
MODELS = [
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "thenlper/gte-base"
]

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

def parse_args():
    parser = argparse.ArgumentParser(description='Run MTEB benchmarks with specific GPUs and ensemble type')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for processing')
    parser.add_argument('--ensemble_type', type=str, choices=['weight', 'uniform'], required=True,
                        help='Type of ensemble to run: weight or uniform')
    parser.add_argument('--weights', type=str, default=None,
                        help='Comma-separated list of weights for weighted ensemble (e.g., "0.6,0.2,0.2")')
    return parser.parse_args()

def ensemble_text_vector_maps(paths: list[str | Path], save_path: str | Path, ensemble_type: str, weights: list[float] = None):
    maps = []
    for path in paths:
        tvm = TextVectorMap(path)
        tvm.load()
        maps.append(tvm)

    # Check number of vectors
    num_vectors = len(maps[0].hash_to_index)
    vector_dim = maps[0].vector_dim
    assert all(len(m.hash_to_index) == num_vectors for m in maps), "Number of vectors mismatch"
    assert all(m.vector_dim == vector_dim for m in maps), "Dimension mismatch"

    # Calculate average
    vectors = [m.vectors[:num_vectors] for m in maps]
    
    if ensemble_type == 'weight':
        if weights is None:
            raise ValueError("Weights must be provided for weighted ensemble.")
        weights = np.array(weights)
        merged = np.sum(np.array([w * v for w, v in zip(weights, vectors)]), axis=0)
    else:  # uniform
        merged = np.stack(vectors).mean(axis=0)

    # Save: create new TextVectorMap
    merged_map = TextVectorMap(save_path, initial_vectors=num_vectors)
    for text_hash, index in maps[0].hash_to_index.items():
        merged_map.hash_to_index[text_hash] = index
    merged_map.vector_dim = vector_dim
    merged_map._initialize_vectors_file()
    
    merged_map.vectors[:num_vectors] = merged
    merged_map.save() 
    print(f"✅ Saved: {save_path}")

def run_ensemble(task_name, domain, ensemble_type, weights=None):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    cache_path = f"/data1/lsj9862/bem/cache/miracls_{ensemble_type}_ensemble"
    model_with_cached_emb = CachedEmbeddingWrapper(model, cache_path=cache_path)

    # Run retrieval tasks
    retrieval_tasks = mteb.get_tasks(tasks=[task_name])
    evaluation = mteb.MTEB(tasks=retrieval_tasks)
    evaluation.run(model_with_cached_emb, output_folder=f"results/{ensemble_type}_ensemble/{domain}/{task_name}", batch_size=32)

def get_directory_contents(directory_path):
    """Return a list of all folders in the given directory path."""
    try:
        folders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]
        return sorted(folders)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []

if __name__ == "__main__":
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    weights = None
    if args.ensemble_type == 'weight':
        if args.weights:
            try:
                weights = [float(w.strip()) for w in args.weights.split(',')]
                if len(weights) != len(MODELS):
                    raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(MODELS)}).")
            except Exception as e:
                raise ValueError(f"Invalid weights format. Use comma-separated floats (e.g., '0.6,0.2,0.2'). Error: {e}")
        else:
            raise ValueError("Weights must be provided for weighted ensemble.")

    # ml_tasks = get_directory_contents("/data1/lsj9862/bem/cache/intfloat_e5-base-v2")
    ml_tasks = RETRIEVAL_TASKS
    for task in ml_tasks:
        print(f"\nProcessing task: {task}")
        ensemble_text_vector_maps(
            paths=[
                f"/data1/lsj9862/bem/cache/intfloat_e5-base-v2/{task}",
                f"/data1/lsj9862/bem/cache/BAAI_bge-base-en-v1.5/{task}",
                f"/data1/lsj9862/bem/cache/thenlper_gte-base/{task}"
            ],
            save_path=f"/data1/lsj9862/bem/cache/{args.ensemble_type}_ensemble/{task}",
            ensemble_type=args.ensemble_type,
            weights=weights
        )
        run_ensemble(task, "multilingual", args.ensemble_type, weights=weights)
