import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import math
import utils.utils as utils

class ProbabilisticSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name, variance, scale=1.0, mean_pooling=True):
        """
        Extensive SentenceTransformer for probabilistic embedding
        
        Args:
            model_name: name of model
            variance: LA fitted variance vector
            scale: scailng coefficient for variance
        """
        super().__init__(model_name)
        # Reshape and scale variance directly in initialization
        # Original shape H=768, I=3072. self.variance is [H, I]
        self.variance = scale * variance.reshape(768, 3072)
        self.mean_pooling_bool = mean_pooling
        
        # Initialize LayerNorm for intermediate activations
        config = self[0].auto_model.config # self[0] is the base Transformer model
        self.intermediate_ln = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)
        
        # Ensure all components are on the correct device
        # SentenceTransformer's __init__ places its modules on a device.
        # We need to ensure our custom attributes are also on that device.
        target_device = next(self.parameters()).device # Get device from already moved params
        self.variance = self.variance.to(target_device)
        self.intermediate_ln = self.intermediate_ln.to(target_device)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-16)
        return sum_embeddings / sum_mask


class RetrieveSentenceTransformer(ProbabilisticSentenceTransformer):    
    def encode(self, sentences, batch_size=32, show_progress_bar=None,
               convert_to_numpy=True, convert_to_tensor=False, device=None, normalize_embeddings=True):
        """
        Sentence encoding method
        
        Args:
            sentences: List of sentences or single sentence to encode
            batch_size: Size of batches for processing
            show_progress_bar: Whether to display progress bar
            output_value: Type of output value
            convert_to_numpy: Whether to convert output to numpy array
            convert_to_tensor: Whether to convert output to torch tensor
            device: Device to use for computation
            normalize_embeddings: Whether to L2 normalize embeddings
            
        Returns:
            Sentence embeddings
        """
        # Ignore variance and return only mean
        means, _ = self.encode_with_variance(
            sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar, 
            device=device, 
            normalize_embeddings=normalize_embeddings
        )
        
        # Convert to numpy if requested
        if convert_to_numpy and not convert_to_tensor:
            means = means.cpu().numpy()
            
        return means
    
    
    def encode_with_variance(self, sentences, batch_size=32, show_progress_bar=None, 
                             device=None, normalize_embeddings=False):
        """
        Compute embeddings with variance
        
        Args:
            sentences: List of sentences or single sentence to encode
            batch_size: Size of batches for processing
            show_progress_bar: Whether to display progress bar
            device: Device to use for computation
            normalize_embeddings: Whether to L2 normalize embeddings
            
        Returns:
            Tuple of (mean_embeddings, variances)
        """
        self.eval()  # Set to evaluation mode
        
        if device is None:
            device = next(self.parameters()).device
            
        if show_progress_bar is None:
            show_progress_bar = (len(sentences) > 5)
            
        # Always convert input to list
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            
        all_embeddings = []
        all_variances = []
        
        for start_idx in range(0, len(sentences), batch_size):
            batch = sentences[start_idx:start_idx + batch_size]
            with torch.no_grad():
                # Tokenize and perform forward pass
                features = self._first_module().tokenize(batch)
                features = {key: value.to(device) for key, value in features.items()}
                emb, var = self._internal_forward_pass(features)
                
                # Apply normalization if needed
                if normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    var /= var.norm(p=2, dim=1, keepdim=True).pow(2)
                    
                all_embeddings.append(emb.cpu())
                all_variances.append(var.cpu())
                
        embeddings = torch.cat(all_embeddings, dim=0)
        variances = torch.cat(all_variances, dim=0)

        return embeddings, variances
    
    
    def _internal_forward_pass(self, features):
        """
        Compute embeddings and variances in a single forward pass
        Mimics the forward logic of SentenceTransformer exactly
        
        Args:
            features: Tokenized input features
            
        Returns:
            Tuple of (embeddings, variances)
        """
        with torch.no_grad():
            # Pass input to the first module (BERT)
            bert_module = self._first_module()
            
            # Extract only the inputs needed by the BERT model (typically input_ids, attention_mask, token_type_ids)
            bert_features = {}
            if 'input_ids' in features:
                bert_features['input_ids'] = features['input_ids']
            if 'attention_mask' in features:
                bert_features['attention_mask'] = features['attention_mask']
            if 'token_type_ids' in features:
                bert_features['token_type_ids'] = features['token_type_ids']
            
            # Set hidden_states=True to get intermediate layer outputs
            model_output = bert_module.auto_model(
                **bert_features, 
                output_hidden_states=True, 
                return_dict=True
            )
            
            # Follow the standard pipeline of sentence_bert
            embeddings = model_output.last_hidden_state
            if self.mean_pooling_bool:
                embeddings = self.mean_pooling(embeddings, features['attention_mask'])
            else:
                embeddings = embeddings[:, 0, :]
            
            # Use intermediate layer outputs for variance calculation
            encoder_layer_11 = bert_module.auto_model.encoder.layer[11]
            encoder_input = model_output.hidden_states[10]  # Input to the 11th layer (0-indexed)
            
            # Calculate intermediate outputs needed for variance calculation
            attention_output = encoder_layer_11.attention(encoder_input)[0]
            intermediate_output = encoder_layer_11.intermediate(attention_output)
            
            # Apply Layer Normalization to intermediate_output
            intermediate_output = self.intermediate_ln(intermediate_output)
            
            # Apply mean pooling for variance calculation
            if self.mean_pooling_bool:
                intermediate_output = self.mean_pooling(intermediate_output, features['attention_mask'])
            else:
                intermediate_output = intermediate_output[:, 0, :]
            variance = intermediate_output.pow(2).matmul(self.variance.T)
            variance = torch.clamp(variance, min=1e-16)
            
        return embeddings, variance



class RetrieveTransformerMTEB(RetrieveSentenceTransformer):
    def encode_with_variance(self, sentences, batch_size=32, show_progress_bar=None, 
                             device=None, normalize_embeddings=False, **kwargs):
        """
        Compute embeddings with variance, compatible with MTEB by accepting **kwargs.
        
        Args:
            sentences: List of sentences or single sentence to encode
            batch_size: Size of batches for processing
            show_progress_bar: Whether to display progress bar
            device: Device to use for computation
            normalize_embeddings: Whether to L2 normalize embeddings
            **kwargs: Additional keyword arguments (like prompt_type) that will be ignored
                      by the parent's encode_with_variance method.
            
        Returns:
            Tuple of (mean_embeddings, variances)
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
            
        if show_progress_bar is None:
            show_progress_bar = (len(sentences) > 5) 
            
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            
        all_embeddings = []
        all_variances = []
        
        # Use a progress bar if show_progress_bar is True
        iterable = range(0, len(sentences), batch_size)
        if show_progress_bar:
            try:
                from tqdm import trange
                iterable = trange(0, len(sentences), batch_size, desc="Batches")
            except ImportError:
                print("tqdm not found. Please install tqdm for progress bar.")


        for start_idx in iterable:
            batch = sentences[start_idx:start_idx + batch_size]
            with torch.no_grad():
                features = self._first_module().tokenize(batch)
                features = {key: value.to(device) for key, value in features.items()}
                
                # Call the internal forward pass which does not take 'prompt_type'
                emb, var = self._internal_forward_pass(features) 
                
                if normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    
                all_embeddings.append(emb.cpu())
                all_variances.append(var.cpu())
                
        embeddings = torch.cat(all_embeddings, dim=0)
        variances = torch.cat(all_variances, dim=0)

        return embeddings, variances

    def encode(self, sentences, batch_size=32, show_progress_bar=None,
               convert_to_numpy=True, convert_to_tensor=False, device=None, 
               normalize_embeddings=True, **kwargs):
        """
        Sentence encoding method, compatible with MTEB.
        Ensures that 'prompt_name' (or 'prompt_type') from MTEB is handled.
        """
        # MTEB might pass 'prompt_name' which is similar to 'task_name' or 'prompt_type'.
        # We ensure these are passed to encode_with_variance if it's designed to use them,
        # or simply caught by **kwargs if not.
        means, _ = self.encode_with_variance(
            sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar, 
            device=device, 
            normalize_embeddings=normalize_embeddings,
            **kwargs  # Pass along any extra arguments
        )
        
        if convert_to_numpy and not convert_to_tensor:
            means = means.cpu().numpy()
        # If convert_to_tensor is True, means is already a tensor (after .cpu())
        
        return means







class GaussianConvolution(nn.Module):
    def __init__(self, models, embedding_dim=768, device='cuda'):
        """
        Initialize GaussianMixture class
        
        Args:
            models: Dictionary of model names and model objects
            embedding_dim: Number of embedding dimensions
            normalize_embeddings: Whether to normalize embeddings
            device: Device to use
        """
        super().__init__()
        self.num_models = len(models.keys())
        self.models = models
        self.embedding_dim = embedding_dim
        self.device = device
        # Initialize with uniform weights (trainable)
        self.mixture_coeffs = torch.tensor([1/3, 1/3, 1/3], device=device)




    def forward_caches(self, mean, variance):
        means = torch.stack(list(mean.values())).to(self.device)  # [K, D, B]
        variances = torch.stack(list(variance.values())).to(self.device)  # [K, D, B]
        return {'means': means, 'variances': variances}



    def mean_convolution(self, means):
        """
        Calculate mean of Gaussian Convolution
        
        Args:
            means : tensor shaped as [K, D] (single item) or [K, B, D] (batch)
        """
        if means.ndim == 2: # Single item: [K, D]
            # coeff [K], means [K,D] -> result [D]
            convolution_mean = torch.sum(self.mixture_coeffs.view(-1, 1) * means, dim=0)
        elif means.ndim == 3: # Batch: [K, B, D]
            # coeff [K], means [K,B,D] -> result [B,D]
            # Assuming D is features, B is batch_size if means is [K, D_features, B_batch]
            convolution_mean = torch.sum(self.mixture_coeffs.view(-1, 1, 1) * means, dim=0)
            # convolution_mean = convolution_mean.T
        else:
            raise ValueError(f"Unsupported means shape: {means.shape}. Expected [K, D] or [K, D, B].")
        return convolution_mean


    def variance_convolution(self, variances):
        """
        Calculate variance of Gaussian Convolution

        Args:
            variances : tensor shaped as [K, D] (single item) or [K, B, D] (batch)
        """
        coeff_sq = self.mixture_coeffs.to(self.device).pow(2) # [K]
        if variances.ndim == 2: # Single item: [K, D]
            convolution_variance = torch.sum(coeff_sq.view(-1, 1) * variances, dim=0)
        elif variances.ndim == 3: # Batch: [K, B, D]
            convolution_variance = torch.sum(coeff_sq.view(-1, 1, 1) * variances, dim=0)
        else:
            raise ValueError(f"Unsupported variances shape: {variances.shape}. Expected [K, D] or [K, B, D].")
        return convolution_variance
        


    def cosine_similarity_variance(self, query_mean, corpus_mean, query_variance, corpus_variance):
        # query_mean, corpus_mean are [B,D] ; query_variance, corpus_variance are [B,D]
        # For single pair B=1. For batch B=num_queries, C=num_corpus_items then D=embedding_dim
        # Original einsum: query_mean_sq [B,D], corpus_variance [C,D] -> output [B,C]
        query_mean_sq = query_mean.pow(2)
        corpus_mean_sq = corpus_mean.pow(2)
        cos_variance = torch.einsum('bd,cd->bc', query_mean_sq, corpus_variance) + \
                      torch.einsum('bd,cd->bc', query_variance, corpus_mean_sq) + \
                      torch.einsum('bd,cd->bc', query_variance, corpus_variance)

        return cos_variance

        
        
    def cosine_similarity(self, query_means, query_variances, corpus_means, corpus_variances, beta=1.0):

        cos_sim = torch.mm(query_means, corpus_means.T) # Shape [B,C]

        sim_variance = self.cosine_similarity_variance(query_means, corpus_means, query_variances, corpus_variances)

        cos_sim /= torch.sqrt(1 + (math.pi/8.0) * beta * sim_variance)
                
        return cos_sim
