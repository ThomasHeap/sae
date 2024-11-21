from nnsight import LanguageModel
from sae_auto_interp.autoencoders import load_eai_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
#from data import load_tokenized_data
from sae_auto_interp.utils import load_tokenized_data
import os
import torch
from pathlib import Path
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides
import argparse

class NoiseEmbeddingModel:
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
            # Generate fresh Gaussian noise with same shape as embeddings
            # Note: output_tensor shape is [batch_size, sequence_length, embedding_dim]
            noise = torch.randn(output_tensor.shape, 
                              device=output_tensor.device, 
                              dtype=output_tensor.dtype) * self.std
            return noise  # Completely replace embeddings with noise
        
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

def process_model(noise_std=1.0):
    """Process model activations with noise replacement"""
    # Get model directory from config
    model_dir = Path('saved_models/random') 
    print(f"\nProcessing model in {model_dir}")
    print(f"Reinitialization mode: {'reinit' if config.reinit_non_embedding else 'no-reinit'}")
    print(f"Using noise std: {noise_std}")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"Using model: {config.model_name}")
    print(f"Using dataset: {config.dataset}")
    if config.dataset_name:
        print(f"Dataset config: {config.dataset_name}")
    print(f"Dataset split: {config.dataset_split}")
    
    # Initialize model
    base_model = LanguageModel(
        config.model_name,
        device_map=config.device_map,
        dispatch=True,
        torch_dtype=getattr(torch, config.torch_dtype),
        revision="step0" if config.use_step0 else None
    )
    
    # Wrap model with noise embedding replacement
    model = NoiseEmbeddingModel(base_model, std=noise_std)
    
    if config.reinit_non_embedding:
        print("Loading step0 model for reinitialization...")
        model_step0 = LanguageModel(
            config.model_name,
            device_map=config.device_map,
            dispatch=True,
            torch_dtype=getattr(torch, config.torch_dtype),
            revision="step0"
        )
        
        for name, param in model.model.named_parameters():
            if param.isnan().any():
                print(f"NaNs found in {name}")
                
        # Note: We don't need to copy embeddings since we're replacing them with noise
        
        del model
        model = NoiseEmbeddingModel(model_step0, std=noise_std)

    print("Loading autoencoders...")
    submodule_dict, model.model = load_eai_autoencoders(
        model.model,
        list(range(5)),
        weight_dir=str(model_dir),
        module="res"
    )

    cfg = CacheConfig(
        dataset_repo="EleutherAI/rpj-v2-sample",
        dataset_split="train[:1%]",
        batch_size=8    ,
        ctx_len=256,
        n_tokens=1_000_000,
        n_splits=5,
    )  
    
    tokens = load_tokenized_data(
        ctx_len=cfg.ctx_len,
        tokenizer=model.tokenizer,
        dataset_repo=cfg.dataset_repo,
        dataset_split=cfg.dataset_split,
    )

    cache = FeatureCache(
        model,
        submodule_dict,
        batch_size=cfg.batch_size
    )

    print("Processing tokens with noise embeddings...")
    cache.run(cfg.n_tokens, tokens)
            
    # Save cache data
    save_dir = "saved_latents/random_noise"
    print(f"Saving results to {save_dir}")
    
    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir=save_dir
    )

    # Save config with noise parameter
    cfg_dict = cfg.to_dict()
    cfg_dict['noise_std'] = noise_std
    cfg_dict['embedding_type'] = 'pure_gaussian_noise'  # Add information about embedding replacement
    cache.save_config(
        save_dir=save_dir,
        cfg=cfg,
        model_name=config.model_name
    )
    
    # Clean up noise injection hook
    model.remove_hook()

def main():
    # Load and apply configuration overrides
    overrides = parse_config_overrides()
    if "--no-reinit_non_embedding" in os.sys.argv:
        overrides["reinit_non_embedding"] = False
    apply_overrides(config, overrides)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_std', type=float, default=1.0,
                      help='Standard deviation for Gaussian noise embeddings')
    args, _ = parser.parse_known_args()
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
    # Process the model with noise replacement
    process_model(noise_std=args.noise_std)

if __name__ == "__main__":
    main()