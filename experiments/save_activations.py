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



def process_model():
    """Process model activations based on configuration"""
    # Get model directory from config
    model_dir = config.save_directory
    print(f"\nProcessing model in {model_dir}")
    print(f"Reinitialization mode: {'reinit' if config.reinit_non_embedding else 'no-reinit'}")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
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
            # if nans print(name)
            if param.isnan().any():
                print(f"NaNs found in {name}")
                
        for name, param in model_step0.named_parameters():
            if "embed" in name:
                print(f"Replacing embedding: {name}")
                param.data = model.state_dict()[name].data
        
        del model
        model = model_step0

    print("Loading autoencoders...")
    submodule_dict, model = load_eai_autoencoders(
        model,
        list(range(5)),  # Match the number of layers we process
        weight_dir=str(model_dir),
        module="res"
    )

    print(submodule_dict)

    # cfg = CacheConfig(
    #     dataset_repo=config.dataset,
    #     dataset_name=config.dataset_name,
    #     dataset_split=config.dataset_split,
    #     batch_size=config.cache_batch_size,
    #     ctx_len=config.cache_ctx_len,
    #     n_tokens=config.cache_n_tokens,
    #     n_splits=5,
    # )    
    # 00_000,
    
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

    print("Processing tokens...")
    cache.run(cfg.n_tokens, tokens)
    
            
    # Save cache data
    save_dir = config.saved_latents_dir / f"latents_{model_dir.name}"
    print(f"Saving results to {save_dir}")
    
    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir=save_dir
    )

    cache.save_config(
        save_dir=save_dir,
        cfg=cfg,
        model_name=config.model_name
    )

    # print("\nSaved features per layer:")
    # for module, ae in submodule_dict.items():
    #     print(f"{module}: {ae.width} features")

def main():
    # Load and apply configuration overrides
    overrides = parse_config_overrides()
    # Set reinit_non_embedding based on the flag
    if "--no-reinit_non_embedding" in os.sys.argv:
        overrides["reinit_non_embedding"] = False
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
    # Process the model
    process_model()

if __name__ == "__main__":
    main()