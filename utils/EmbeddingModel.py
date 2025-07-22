import torch
import torch.nn as nn
from transformers import AutoModel


class EmbeddingModel(nn.Module):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", use_mean_pooling=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.set_trainable_params()
        self.use_mean_pooling = use_mean_pooling

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-16)
        return sum_embeddings / sum_mask

    def forward(self, batch):
        device = next(self.model.parameters()).device
        
        input_ids1 = batch["query_input"]["input_ids"].to(device)
        attention_mask1 = batch["query_input"]["attention_mask"].to(device)
        
        input_ids2 = batch["passage_input"]["input_ids"].to(device)
        attention_mask2 = batch["passage_input"]["attention_mask"].to(device)

        # Get last hidden states
        hidden_states1 = self.model(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state
        hidden_states2 = self.model(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state

        # Apply mean pooling or use CLS token based on configuration
        if self.use_mean_pooling:
            emb1 = self.mean_pooling(hidden_states1, attention_mask1)
            emb2 = self.mean_pooling(hidden_states2, attention_mask2)
        else:
            emb1 = hidden_states1[:, 0, :]  # CLS token
            emb2 = hidden_states2[:, 0, :]  # CLS token

        similarity_matrix = torch.matmul(emb1, emb2.T)
        
        return similarity_matrix
    
    
    def set_trainable_params(self):
        num_total_param = 0
        for param in self.model.parameters():
            param.requires_grad = False
            num_total_param += param.numel()
            
        num_trainable_param = 0
        # Example: Fine-tune the last encoder layer's output parameters
        for param in self.model.encoder.layer[-1].output.parameters():
            param.requires_grad = True
            num_trainable_param += param.numel()
        print(f"✅ Number of trainable parameters : {num_trainable_param} / Total parameters : {num_total_param}")