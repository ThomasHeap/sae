import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Optional, Tuple, Union, Dict, Any

class NoiseEmbeddingModel(nn.Module):
    """Wrapper for models that replaces input embeddings with random noise"""
    
    def __init__(self, model: PreTrainedModel, std: float = 1.0):
        super().__init__()
        self.model = model
        self.config = model.config
        self.std = std
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass that replaces input embeddings with random noise
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments passed to base model
        """
        if input_ids is not None:
            # Get embedding dimension from base model
            embed_dim = self.model.get_input_embeddings().embedding_dim
            
            # Generate random embeddings with same shape as normal embeddings
            random_embeds = torch.randn(
                input_ids.shape[0],  # batch size
                input_ids.shape[1],  # sequence length
                embed_dim,  # embedding dimension
                device=input_ids.device,
                dtype=self.model.dtype
            ) * self.std
            
            # Forward with random embeddings
            return self.model(
                inputs_embeds=random_embeds,
                attention_mask=attention_mask,
                **kwargs
            )
        
        return self.model(**kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

class NoiseEmbeddingNNsight:
    """Version of noise embedding for nnsight models"""
    def __init__(self, model, embedding_dim=None, std=1.0):
        self.model = model
        self.std = std
        # If embedding_dim not provided, get it from the model's embedding layer
        if embedding_dim is None:
            embedding_dim = self.model.gpt_neox.embed_in.embedding_dim
        self.embedding_dim = embedding_dim
        self._setup_embedding_hook()
        
    def _setup_embedding_hook(self):
        """Replace embedding output with pure Gaussian noise"""
        def replace_with_noise(module, input_tensor, output_tensor):
            noise = torch.randn(output_tensor.shape, 
                              device=output_tensor.device, 
                              dtype=output_tensor.dtype) * self.std
            return noise
        
        # Get the embedding layer
        embed_layer = self.model.gpt_neox.embed_in
        self.hook_handle = embed_layer.register_forward_hook(replace_with_noise)
        
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying model"""
        return getattr(self.model, name)
    
    def remove_hook(self):
        """Clean up the hook when done"""
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()