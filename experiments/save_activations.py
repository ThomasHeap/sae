from nnsight import LanguageModel
from sae_auto_interp.autoencoders import load_eai_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
from noise_embedding import NoiseEmbeddingNNsight
import os
import torch
from pathlib import Path
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides

def process_model():
    """Process model activations based on configuration"""
    save_dir = config.save_directory
    print(f"\nProcessing model in {save_dir}")
    print(f"Reinitialization mode: {'reinit' if config.reinit_non_embedding else 'no-reinit'}")
    print(f"Random control mode: {'enabled' if config.use_random_control else 'disabled'}")
    
    if not save_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {save_dir}")
    
    print(f"Using model: {config.model_name}")
    print(f"Using dataset: {config.dataset}")
    if config.dataset_name:
        print(f"Dataset config: {config.dataset_name}")
    print(f"Dataset split: {config.dataset_split}")
    
    # Initialize model
    model = LanguageModel(
        config.model_name,
        device_map=config.device_map,
        dispatch=True,
        torch_dtype=getattr(torch, config.torch_dtype),
        revision="step0" if config.use_step0 else None
    )
    
    if config.reinit_non_embedding:
        print("Loading step0 model for reinitialization...")
        model_step0 = LanguageModel(
            config.model_name,
            device_map=config.device_map,
            dispatch=True,
            torch_dtype=getattr(torch, config.torch_dtype),
            revision="step0"
        )
        
        for name, param in model.named_parameters():
            if param.isnan().any():
                print(f"NaNs found in {name}")
                
        for name, param in model_step0.named_parameters():
            if "embed" in name:
                print(f"Replacing embedding: {name}")
                param.data = model.state_dict()[name].data
        
        del model
        model = model_step0

    if config.use_random_control:
        print(f"Applying random control with noise std: {config.noise_std}")
        model = NoiseEmbeddingNNsight(model, std=config.noise_std)

    print("Loading autoencoders...")
    submodule_dict, model.model = load_eai_autoencoders(
        model.model if config.use_random_control else model,
        list(range(5)),
        weight_dir=str(save_dir),
        module="res"
    )

    cfg = CacheConfig(
        dataset_repo="EleutherAI/rpj-v2-sample",
        dataset_split="train[5%:7%]",
        batch_size=config.cache_batch_size,
        ctx_len=config.cache_ctx_len,
        n_tokens=config.cache_n_tokens,
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

    print("Processing tokens...")
    cache.run(cfg.n_tokens, tokens)
    
    # Save cache data
    save_dir = config.latents_directory
    print(f"Saving results to {save_dir}")
    
    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir=save_dir
    )

    # Save config with noise parameter if using random control
    cfg_dict = cfg.to_dict()
    if config.use_random_control:
        cfg_dict['noise_std'] = config.noise_std
        cfg_dict['embedding_type'] = 'pure_gaussian_noise'
    
    cache.save_config(
        save_dir=save_dir,
        cfg=cfg,
        model_name=config.model_name
    )

    if config.use_random_control:
        model.remove_hook()

def main():
    # Load and apply configuration overrides
    overrides = parse_config_overrides()
    if "--no-reinit_non_embedding" in os.sys.argv:
        overrides["reinit_non_embedding"] = False
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    process_model()

if __name__ == "__main__":
    main()