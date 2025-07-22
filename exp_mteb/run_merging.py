import torch
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict
import mteb
import numpy as np
import argparse
import warnings # To warn about weight normalization


def get_short_model_name(hf_name):
    """Helper to get a shorter name for filenames."""
    return hf_name.split('/')[-1].replace('-', '_')

# --- Uniform Merging ---
def uniform_merging(model_names, new_model_name="uniform_merged_model"):
    """
    Performs uniform merging (averaging) of model parameters.

    Args:
        model_names (list): A list of Hugging Face model names to merge.
        new_model_name (str): The name/path to save the merged model.

    Returns:
        str: The path where the merged model was saved.
    """
    if not model_names or len(model_names) < 2:
        raise ValueError("Please provide at least two models for uniform merging.")

    print(f"Starting Uniform Merging for: {model_names}")
    # Load models and tokenizers
    print("Loading models...")
    models = [AutoModel.from_pretrained(name) for name in model_names]
    tokenizer = AutoTokenizer.from_pretrained(model_names[0])
    print("Models loaded.")

    # Get state dicts
    state_dicts = [model.state_dict() for model in models]

    # Initialize a new state dict with the structure of the first model
    merged_state_dict = OrderedDict()
    num_models = len(models)

    # Average the parameters
    print("Averaging parameters...")
    for key in state_dicts[0].keys():
        if state_dicts[0][key].is_floating_point(): # Only average floating point tensors
            tensors_to_average = []
            for i in range(num_models):
                if key in state_dicts[i]:
                    tensors_to_average.append(state_dicts[i][key])
                else:
                    print(f"Warning: Key '{key}' not found in model {model_names[i]}. Skipping this model for this key.")
            if tensors_to_average:
                merged_state_dict[key] = torch.stack(tensors_to_average).mean(dim=0)
            else:
                 print(f"Warning: Key '{key}' not found in any model. Copying from first model.")
                 merged_state_dict[key] = state_dicts[0][key].clone() # Fallback
        else: # Copy non-floating point tensors (e.g., int tensors) as is from the first model
            merged_state_dict[key] = state_dicts[0][key].clone()
    print("Parameter averaging complete.")

    # Create a new model to load the merged weights
    base_model_config = models[0].config
    merged_model = AutoModel.from_config(base_model_config)
    merged_model.load_state_dict(merged_state_dict)

    # Save the merged model and tokenizer
    print(f"Saving merged model to ./{new_model_name} ...")
    merged_model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)
    print(f"Merged model saved successfully.")

    # Return the path
    return new_model_name


# --- Weighted Merging ---
def weighted_merging(model_names, weights, new_model_name="weighted_merged_model"):
    """
    Performs weighted merging of model parameters.

    Args:
        model_names (list): A list of Hugging Face model names to merge.
        weights (list): A list of weights corresponding to the models.
        new_model_name (str): The name/path to save the merged model.

    Returns:
        str: The path where the merged model was saved.
    """
    if not model_names or len(model_names) < 2:
        raise ValueError("Please provide at least two models for weighted merging.")
    if len(model_names) != len(weights):
        raise ValueError("Number of models and weights must be the same.")

    # Normalize weights if they don't sum to 1
    total_weight = sum(weights)
    if not np.isclose(total_weight, 1.0):
        warnings.warn(f"Weights sum to {total_weight}, normalizing weights to sum to 1.", UserWarning)
        weights = [w / total_weight for w in weights]

    print(f"Starting Weighted Merging for: {model_names} with weights: {weights}")
    # Load models and tokenizers
    print("Loading models...")
    models = [AutoModel.from_pretrained(name) for name in model_names]
    tokenizer = AutoTokenizer.from_pretrained(model_names[0])
    print("Models loaded.")

    # Get state dicts
    state_dicts = [model.state_dict() for model in models]

    # Initialize a new state dict
    merged_state_dict = OrderedDict()
    num_models = len(models)

    # Calculate weighted average of parameters
    print("Averaging parameters with weights...")
    for key in state_dicts[0].keys():
        if state_dicts[0][key].is_floating_point():
            weighted_sum = torch.zeros_like(state_dicts[0][key], dtype=torch.float32) # Use float32 for accumulation
            valid_models_count = 0
            for i in range(num_models):
                if key in state_dicts[i]:
                    weighted_sum += weights[i] * state_dicts[i][key].float() # Ensure float for calculation
                    valid_models_count += 1
                else:
                     print(f"Warning: Key '{key}' not found in model {model_names[i]}. Skipping this model for this key.")
            if valid_models_count > 0:
                 # Cast back to original dtype if needed, although merged might be float32
                 merged_state_dict[key] = weighted_sum.to(state_dicts[0][key].dtype)
            else:
                 print(f"Warning: Key '{key}' not found in any model. Copying from first model.")
                 merged_state_dict[key] = state_dicts[0][key].clone() # Fallback
        else:
            merged_state_dict[key] = state_dicts[0][key].clone()
    print("Parameter averaging complete.")

    # Create a new model
    base_model_config = models[0].config
    merged_model = AutoModel.from_config(base_model_config)
    merged_model.load_state_dict(merged_state_dict)

    # Save
    print(f"Saving merged model to ./{new_model_name} ...")
    merged_model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)
    print(f"Merged model saved successfully.")

    return new_model_name


# --- Task Vector Merging (Simplified: Avg Delta from Base) ---
def task_vector_merging(model_names, new_model_name="task_vector_merged_model", lambda_scale=1.0):
    """
    Performs simplified task vector merging by averaging the differences
    from the first model and adding back to the first model.

    Args:
        model_names (list): List of models. The first is the base.
        new_model_name (str): Path to save the merged model.
        lambda_scale (float): Scaling factor for the average delta.

    Returns:
        str: The path where the merged model was saved.
    """
    if not model_names or len(model_names) < 2:
        raise ValueError("Please provide at least two models for task vector merging (base + >=1 model).")

    base_model_name = model_names[0]
    task_model_names = model_names[1:]
    num_task_models = len(task_model_names)

    print(f"Starting Task Vector Merging. Base: {base_model_name}, Task Models: {task_model_names}, Lambda: {lambda_scale}")

    # Load models
    print("Loading models...")
    base_model = AutoModel.from_pretrained(base_model_name)
    task_models = [AutoModel.from_pretrained(name) for name in task_model_names]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print("Models loaded.")

    base_state_dict = base_model.state_dict()
    task_state_dicts = [model.state_dict() for model in task_models]

    # Calculate average delta
    print("Calculating average task vector (delta)...")
    avg_delta_state_dict = OrderedDict()
    for key in base_state_dict.keys():
        if base_state_dict[key].is_floating_point():
            delta_sum = torch.zeros_like(base_state_dict[key], dtype=torch.float32)
            valid_models_count = 0
            for i in range(num_task_models):
                if key in task_state_dicts[i]:
                    delta_sum += (task_state_dicts[i][key].float() - base_state_dict[key].float())
                    valid_models_count += 1
                else:
                     print(f"Warning: Key '{key}' not found in task model {task_model_names[i]}. Skipping this model for this key's delta.")
            if valid_models_count > 0:
                avg_delta_state_dict[key] = (delta_sum / valid_models_count).to(base_state_dict[key].dtype)
            else:
                print(f"Warning: Key '{key}' not found in any task model. Delta for this key will be zero.")
                avg_delta_state_dict[key] = torch.zeros_like(base_state_dict[key]) # Zero delta if no task models have the key
        else:
            avg_delta_state_dict[key] = torch.zeros_like(base_state_dict[key]) # No delta for non-float types
    print("Average delta calculated.")

    # Add scaled average delta to base model
    print("Adding scaled delta to base model weights...")
    merged_state_dict = OrderedDict()
    for key in base_state_dict.keys():
        if base_state_dict[key].is_floating_point():
            merged_state_dict[key] = base_state_dict[key] + lambda_scale * avg_delta_state_dict[key]
        else:
            merged_state_dict[key] = base_state_dict[key].clone() # Keep non-float params from base
    print("Merging complete.")

    # Create new model and load state dict
    merged_model = AutoModel.from_config(base_model.config)
    merged_model.load_state_dict(merged_state_dict)

    # Save
    print(f"Saving merged model to ./{new_model_name} ...")
    merged_model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)
    print(f"Merged model saved successfully.")

    return new_model_name


# Define the custom model class for MTEB
class CustomMergedModel:
    def __init__(self, model_name_or_path, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Encodes a list of sentences into embeddings.

        Args:
            sentences (list[str]): The list of sentences to encode.
            batch_size (int): Batch size for encoding.

        Returns:
            numpy.ndarray: The embeddings for the sentences.
        """
        all_embeddings = []
        self.model.eval() # Set model to evaluation mode

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            encoded_input = self.tokenizer(
                batch_sentences, padding=True, truncation=True, return_tensors='pt'
            )
            # Move batch to GPU if available
            if torch.cuda.is_available():
                encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Perform pooling. Mean pooling is common.
                # Take attention mask into account for correct averaging
                input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

            all_embeddings.append(batch_embeddings.cpu().numpy()) # Move embeddings to CPU before converting to numpy

        return np.concatenate(all_embeddings, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge models and evaluate with MTEB")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for MTEB evaluation")
    parser.add_argument("--merging_technique", type=str, default="uniform", choices=["uniform", "weighted", "task_vector"], help="Merging technique to use")
    parser.add_argument("--model_names", nargs='+', default=["BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2", "thenlper/gte-base"], help="List of Hugging Face models to merge")
    parser.add_argument("--weights", type=str, default=None, help="Comma-separated list of weights for weighted merging (e.g., '0.5,0.3,0.2')")
    parser.add_argument("--lambda_scale", type=float, default=1.0, help="Lambda scaling factor for task vector merging")
    parser.add_argument("--output_name_base", type=str, default="merged", help="Base name for the output merged model directory")

    args = parser.parse_args()

    model_names_to_merge = args.model_names
    num_models = len(model_names_to_merge)
    short_names = [get_short_model_name(name) for name in model_names_to_merge]


    # --- Select Merging Technique ---
    merged_model_path = None
    merged_model_name = f"{args.output_name_base}_{args.merging_technique}_{'_'.join(short_names)}"

    try:
        if args.merging_technique == "uniform":
            if num_models < 2: raise ValueError("Uniform merging requires at least 2 models.")
            merged_model_path = uniform_merging(model_names_to_merge, new_model_name=merged_model_name)

        elif args.merging_technique == "weighted":
            if num_models < 2: raise ValueError("Weighted merging requires at least 2 models.")
            weights = []
            if args.weights:
                try:
                    weights = [float(w.strip()) for w in args.weights.split(',')]
                    if len(weights) != num_models:
                        raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({num_models}).")
                except Exception as e:
                    raise ValueError(f"Invalid weights format. Use comma-separated floats (e.g., '0.5,0.5'). Error: {e}")
            else:
                print("No weights provided for weighted merging, using uniform weights.")
                weights = [1.0 / num_models] * num_models

            merged_model_name += f"_w{'_'.join([str(round(w, 2)) for w in weights])}" # Add weights to name
            merged_model_path = weighted_merging(model_names_to_merge, weights, new_model_name=merged_model_name)

        elif args.merging_technique == "task_vector":
            if num_models < 2: raise ValueError("Task vector merging requires at least 2 models (base + >=1 task model).")
            merged_model_name += f"_lambda{args.lambda_scale}" # Add lambda to name
            merged_model_path = task_vector_merging(model_names_to_merge, new_model_name=merged_model_name, lambda_scale=args.lambda_scale)

        else:
            # This case should not be reached due to argparse choices
            raise ValueError(f"Unknown merging technique: {args.merging_technique}")

        print(f"Model merging complete using '{args.merging_technique}'. Merged model saved at: {merged_model_path}")

        # --- MTEB Evaluation ---
        print("Loading merged model for MTEB evaluation...")
        custom_model = CustomMergedModel(model_name_or_path=merged_model_path)
        print("Custom model loaded.")

        # Define MTEB tasks (ensure these are desired)
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

        print(f"Running MTEB evaluation with batch size: {args.batch_size}")
        output_folder_base = f"results/{merged_model_name}"
        
        # Run retrieval tasks
        print("--- Starting Retrieval Tasks ---")
        retrieval_tasks_list = mteb.get_tasks(tasks=RETRIEVAL_TASKS)
        evaluation_retrieval = mteb.MTEB(tasks=retrieval_tasks_list, task_langs=["en"])
        evaluation_retrieval.run(custom_model, output_folder=f"{output_folder_base}/retrieval", batch_size=args.batch_size)
        
        # Run classification tasks
        print("--- Starting Classification Tasks ---")
        classification_tasks_list = mteb.get_tasks(tasks=CLASSIFICATION_TASKS)
        evaluation_classification = mteb.MTEB(tasks=classification_tasks_list, task_langs=["en"])
        evaluation_classification.run(custom_model, output_folder=f"{output_folder_base}/classification", batch_size=args.batch_size)

        # Run STS tasks
        print("--- Starting STS Tasks ---")
        sts_tasks_list = mteb.get_tasks(tasks=STS_TASKS)
        evaluation_sts = mteb.MTEB(tasks=sts_tasks_list, task_langs=["en"])
        evaluation_sts.run(custom_model, output_folder=f"{output_folder_base}/sts", batch_size=args.batch_size)

        print("--- MTEB Evaluation Complete ---")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print detailed traceback for debugging