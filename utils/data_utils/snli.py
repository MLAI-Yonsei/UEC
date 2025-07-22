import torch
from torch.utils.data import Dataset
import random



class SNLI(Dataset):
    def __init__(self, tokenized_data, batch_size):
        self.query_input = tokenized_data["query_input"]
        self.passage_input = tokenized_data["passage_input"]
        self.labels = torch.arange(len(tokenized_data["query_input"]["input_ids"]), dtype=torch.long)
        self.num_samples = len(self.labels)
        self.batch_size = batch_size 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "query_input": {k: v[idx] for k, v in self.query_input.items()},
            "passage_input": {k: v[idx] for k, v in self.passage_input.items()},
            "labels": torch.tensor(idx % self.batch_size, dtype=torch.long)
        }        

        

def preprocess_snli(dataset, tokenizer, batch_size, subset_ratio=1.0, max_length=512):
    """
    Preprocess SNLI dataset by selecting only entailment(=0) samples.
    Similar to MS MARCO preprocessing.
    """
    premise_list = []; hypothesis_list = []; label_list = []
    
    total_samples = len(dataset)
    selected_indices = random.sample(range(total_samples), int(total_samples * subset_ratio))

    for idx in selected_indices:
        sample = dataset[idx]
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        label = sample["label"]
        
        if label != 0:
            continue

        premise_list.append(premise)
        hypothesis_list.append(hypothesis)
        label_list.append(idx)

    print(f"✅ Number of preprocessed SNLI entailment samples: {len(premise_list)} (subset_ratio={subset_ratio})")

    if not premise_list:
        raise ValueError("No valid SNLI entailment data found!")

    premise_enc = tokenizer(premise_list, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
    hypothesis_enc = tokenizer(hypothesis_list, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)

    tokenized_data = {
        "query_input": premise_enc,
        "passage_input": hypothesis_enc,
        "labels": torch.tensor(label_list, dtype=torch.long)
    }
    
    return SNLI(tokenized_data, batch_size)  # Same Dataset class is used.