import torch

def custom_collate_fn(batch):
    """
    Function to always set labels sequentially from 0 within a batch.
    Correctly sets even if the size of the last batch is smaller than batch_size.
    """
    batch_size = len(batch)
    batch_dict = {}
    for key in batch[0]:
        if isinstance(batch[0][key], dict):
            batch_dict[key] = {sub_key: [d[key][sub_key] for d in batch] for sub_key in batch[0][key]}
        else:
            batch_dict[key] = [d[key] for d in batch]

    for key in batch_dict:
        if isinstance(batch_dict[key], dict):
            for sub_key in batch_dict[key]:
                if isinstance(batch_dict[key][sub_key][0], torch.Tensor):
                    batch_dict[key][sub_key] = torch.stack(batch_dict[key][sub_key], dim=0)
        else:
            if isinstance(batch_dict[key][0], torch.Tensor):
                batch_dict[key] = torch.stack(batch_dict[key], dim=0)

    batch_dict["labels"] = torch.arange(batch_size, dtype=torch.long)
    return batch_dict