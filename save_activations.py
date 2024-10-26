from nnsight import LanguageModel
from sae_auto_interp.autoencoders import load_eai_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
import os
import torch
from pathlib import Path
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides, load_saved_config
from pipeline_utils import get_dirs_to_process, load_model_config

def process_model_directory(model_dir: Path):
    """Process a single model directory"""
    print(f"\nProcessing {model_dir}")
    
    # Load the saved training config
    config_path = model_dir / "config.json"
    if config_path.exists():
        print(f"Loading saved configuration from {config_path}")
        saved_config = load_saved_config(config_path)
        
        # Update current config with saved values while preserving overrides
        current_overrides = parse_config_overrides()
        for key, value in vars(saved_config).items():
            if key not in vars(current_overrides):
                setattr(config, key, value)
        print(f"Using model: {config.model_name}")
    else:
        print(f"Warning: No saved config found in {model_dir}")
    
    model = LanguageModel(
        config.model_name,
        device_map=config.device_map,
        dispatch=True,
        torch_dtype=getattr(torch, config.torch_dtype)
    )

    print("Loading autoencoders...")
    submodule_dict, model = load_eai_autoencoders(
        model,
        [0,1,2,3,4,5],
        weight_dir=str(model_dir),
        module="res"
    )

    cfg = CacheConfig(
        dataset_name=config.dataset_name,
        dataset_split=config.dataset_split,
        batch_size=config.cache_batch_size,
        ctx_len=config.cache_ctx_len,
        n_tokens=config.cache_n_tokens,
        n_splits=config.n_splits,
    )
    
    
    tokens = load_tokenized_data(
        ctx_len=cfg.ctx_len,
        tokenizer=model.tokenizer,
        dataset_repo=cfg.dataset_repo,
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
    )

    cache = FeatureCache(
        model,
        submodule_dict,
        batch_size=cfg.batch_size
    )

    cache.run(cfg.n_tokens, tokens)
    
    # Save cache data
    save_dir = config.saved_latents_dir / f"latents_{model_dir.name}"
    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir=save_dir
    )

    cache.save_config(
        save_dir=save_dir,
        cfg=cfg,
        model_name=config.model_name
    )

def main():
    overrides = parse_config_overrides()
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
    # Get directories to process
    dirs_to_process = get_dirs_to_process(config.saved_models_dir, overrides)
    
    for model_dir in dirs_to_process:
        process_model_directory(model_dir)

if __name__ == "__main__":
    main()