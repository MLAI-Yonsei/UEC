import torch
from torch.utils.data import Dataset
import random

class MSMarco(Dataset):
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

        
def preprocess_ms_marco(dataset, tokenizer, batch_size, subset_ratio=1.0, max_length=512):
    query_list = []; passage_list = []; label_list = []
    
    total_samples = len(dataset)
    selected_indices = random.sample(range(total_samples), int(total_samples * subset_ratio))

    for idx in selected_indices:
        sample = dataset[idx]
        query = sample["query"]
        passages = sample["passages"]
        
        try:
            is_selected = [i for i, value in enumerate(passages['is_selected']) if value == 1]
            is_selected = is_selected[0]
            positive_passage = passages['passage_text'][is_selected]
        except:
            continue

        query_list.append(query)
        passage_list.append(positive_passage)
        label_list.append(idx)

    print(f"✅ Number of MS MARCO preprocessed samples: {len(query_list)} (subset_ratio={subset_ratio})")

    if not query_list:
        raise ValueError("No valid data found!")

    query_enc = tokenizer(query_list, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
    passage_enc = tokenizer(passage_list, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)

    tokenized_data = {
        "query_input": query_enc,
        "passage_input": passage_enc,
        "labels": torch.tensor(label_list, dtype=torch.long)
    }
    
    return MSMarco(tokenized_data, batch_size)