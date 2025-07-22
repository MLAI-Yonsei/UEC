from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any
import argparse

import numpy as np
import torch

import mteb
from mteb.encoder_interface import Encoder
from mteb.models.wrapper import Wrapper
from mteb.models.cache_wrapper import CachedEmbeddingWrapper, TextVectorMap

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from uec import RetrieveTransformerMTEB

logger = logging.getLogger(__name__)


class ProbEmbsWrapper(CachedEmbeddingWrapper):
    def __init__(self, model: Encoder, cache_path: str | Path):
        self._model = model
        self.cache_path = Path(cache_path)
        # TextVectorMap will create subdirectories as needed.

        if hasattr(model, "encode_with_variance"):
            self.mean_cache_dict: dict[str, TextVectorMap] = {}
            self.variance_cache_dict: dict[str, TextVectorMap] = {}
        else:
            logger.error(
                "Model must have an 'encode_with_variance' method for ProbEmbsWrapper."
            )
            raise ValueError(
                "Invalid model: missing 'encode_with_variance' method."
            )
        logger.info(f"Initialized ProbEmbsWrapper with cache path {self.cache_path}")

    def encode_with_variance(
        self, texts: list[str], batch_size: int = 32, task_name: str | None = None, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode texts using the wrapped model, caching both mean and variance."""
        try:
            if task_name is None:
                task_name = "_default_task"

            mean_results = []
            variance_results = []
            uncached_texts = []
            uncached_indices = []

            # Define cache paths for means and variances for the specific task
            task_specific_cache_path = self.cache_path / task_name
            mean_task_cache_dir = task_specific_cache_path / "means"
            variance_task_cache_dir = task_specific_cache_path / "variances"

            # Initialize TextVectorMap for means if not already done for this task
            if task_name not in self.mean_cache_dict:
                self.mean_cache_dict[task_name] = TextVectorMap(directory=mean_task_cache_dir)
                self.mean_cache_dict[task_name].load(name=f"{task_name}_means")
            
            # Initialize TextVectorMap for variances if not already done for this task
            if task_name not in self.variance_cache_dict:
                self.variance_cache_dict[task_name] = TextVectorMap(directory=variance_task_cache_dir)
                self.variance_cache_dict[task_name].load(name=f"{task_name}_variances")

            current_mean_cache = self.mean_cache_dict[task_name]
            current_variance_cache = self.variance_cache_dict[task_name]

            # Check cache for each text
            for i, text in enumerate(texts):
                mean_vector = current_mean_cache.get_vector(text)
                variance_vector = current_variance_cache.get_vector(text)

                if mean_vector is not None and variance_vector is not None:
                    mean_results.append(mean_vector)
                    variance_results.append(variance_vector)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            num_cached_initially = len(mean_results)

            # Encode any texts not found in cache
            if uncached_texts:
                logger.info(
                    f"Task '{task_name}': Encoding {len(uncached_texts)} new texts for mean and variance."
                )
                new_means, new_variances = self._model.encode_with_variance(
                    uncached_texts, batch_size=batch_size, **kwargs
                )

                if isinstance(new_means, torch.Tensor):
                    new_means = new_means.cpu().numpy()
                if isinstance(new_variances, torch.Tensor):
                    new_variances = new_variances.cpu().numpy()

                # Add new vectors to cache and extend results
                for i, text in enumerate(uncached_texts):
                    current_mean_cache.add(text, new_means[i])
                    current_variance_cache.add(text, new_variances[i])
                
                mean_results.extend(new_means)
                variance_results.extend(new_variances)
                
                current_mean_cache.save()
                current_variance_cache.save()
                logger.info(f"Task '{task_name}': Saved new means and variances to cache.")
            else:
                logger.info(f"Task '{task_name}': All texts' means and variances found in cache.")

            # Reconstruct results in original order
            final_mean_results = [None] * len(texts)
            final_variance_results = [None] * len(texts)
            
            cached_ptr = 0
            new_ptr = 0
            
            for i in range(len(texts)):
                if i in uncached_indices:
                    final_mean_results[i] = mean_results[num_cached_initially + new_ptr]
                    final_variance_results[i] = variance_results[num_cached_initially + new_ptr]
                    new_ptr += 1
                else:
                    final_mean_results[i] = mean_results[cached_ptr]
                    final_variance_results[i] = variance_results[cached_ptr]
                    cached_ptr += 1
            
            return np.array(final_mean_results), np.array(final_variance_results)

        except Exception as e:
            logger.error(f"Error in cached encoding (mean and variance) for task '{task_name}': {str(e)}")
            raise

    def encode(
        self, texts: list[str], batch_size: int = 32, task_name: str | None = None, **kwargs
    ) -> np.ndarray:
        """
        Encode texts using the wrapped model, caching both mean and variance,
        but returning only means for compatibility with standard Encoder interface.
        """
        if task_name is None:
            task_name = "_default_task"
        logger.info(
            f"ProbEmbsWrapper.encode called for task '{task_name}'. "
            f"Will cache both mean/variance and return means."
        )
        means, _ = self.encode_with_variance(
            texts, batch_size=batch_size, task_name=task_name, **kwargs
        )
        return means

    def close(self):
        """Close all TextVectorMap objects for means and variances."""
        closed_tasks = set()
        for task_name in list(self.mean_cache_dict.keys()):
            if self.mean_cache_dict[task_name] is not None:
                self.mean_cache_dict[task_name].close()
                # self.mean_cache_dict[task_name] = None # Avoid issues if close is called multiple times
            closed_tasks.add(task_name)

        for task_name in list(self.variance_cache_dict.keys()):
            if self.variance_cache_dict[task_name] is not None:
                self.variance_cache_dict[task_name].close()
                # self.variance_cache_dict[task_name] = None
            closed_tasks.add(task_name)
        
        self.mean_cache_dict.clear()
        self.variance_cache_dict.clear()
        
        if closed_tasks:
            logger.info(f"Closed ProbEmbsWrapper for tasks: {', '.join(closed_tasks)}")
        else:
            logger.info("Closed ProbEmbsWrapper (no active tasks to close).")

    # __getattr__, __dir__, and __del__ are inherited from CachedEmbeddingWrapper
    # __del__ will call this overridden close() method.


# Define task lists
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

def main(args):    
    for i, model_name in enumerate(args.model_names):
        print(f"\nRunning MTEB for model: {model_name}")

        # Construct path to the posterior variance file
        model_name_safe = model_name.split("/")[-1]
        variance_path = Path(args.la_models_root) / model_name_safe / args.backend / "la_posterior_variance.pt"
        
        if not variance_path.exists():
            print(f"ERROR: Posterior variance file not found at {variance_path}. Skipping model {model_name}.")
            continue
        
        print(f"Loading posterior variance from: {variance_path}")
        variance = torch.load(variance_path)
        
        # Get the scale for the current model
        scale = args.scales[i]
        print(f"Using scale: {scale}")

        # Initialize the probabilistic model and the caching wrapper
        model = RetrieveTransformerMTEB(model_name, variance, scale=scale)
        cache_path_model = Path(args.cache_root) / model_name.replace('/', '_')
        print(f"Using cache path: {cache_path_model}")
        model_with_cached_emb = ProbEmbsWrapper(model, cache_path=cache_path_model)
        
        # Determine which tasks to run
        tasks_to_run = []
        if 'all' in args.task_types or 'retrieval' in args.task_types:
            tasks_to_run.extend(RETRIEVAL_TASKS)
        if 'all' in args.task_types or 'classification' in args.task_types:
            tasks_to_run.extend(CLASSIFICATION_TASKS)
        if 'all' in args.task_types or 'sts' in args.task_types:
            tasks_to_run.extend(STS_TASKS)

        print(f"Selected {len(tasks_to_run)} MTEB tasks to run.")

        # Run MTEB evaluation for the selected tasks
        # This process will generate and save the cached embeddings.
        if tasks_to_run:
            all_mteb_tasks = mteb.get_tasks(tasks=tasks_to_run)
            evaluation = mteb.MTEB(tasks=all_mteb_tasks)
            output_folder_model = Path(args.output_root) / model_name.replace('/', '_')
            print(f"MTEB evaluation results will be saved to: {output_folder_model}")
            
            evaluation.run(
                model_with_cached_emb, 
                output_folder=str(output_folder_model), 
                batch_size=args.batch_size
            )
        else:
            print("No tasks selected to run.")
        
        # Ensure all cached data is written to disk
        model_with_cached_emb.close()
        print(f"Finished processing for model: {model_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and cache probabilistic embeddings for MTEB tasks.")
    parser.add_argument('--model_names', nargs='+', required=True, help="List of HuggingFace model names to process.")
    parser.add_argument('--la_models_root', type=str, required=True, help="Root directory where fitted Laplace models are stored.")
    parser.add_argument('--scales', nargs='+', type=float, required=True, help="List of scaling factors, one for each model.")
    parser.add_argument('--cache_root', type=str, required=True, help="Root directory to save the probabilistic embedding caches.")
    parser.add_argument('--output_root', type=str, default="results/prob_embs_mteb", help="Root directory to save MTEB evaluation results.")
    parser.add_argument('--backend', type=str, default='AsdlEF', help="Backend used for Laplace fitting (part of the path).")
    parser.add_argument('--task_types', nargs='+', default=['all'], choices=['retrieval', 'classification', 'sts', 'all'], help="Types of MTEB tasks to run.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for MTEB evaluation.")
    
    args = parser.parse_args()

    if len(args.model_names) != len(args.scales):
        raise ValueError("The number of --model_names must match the number of --scales.")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)