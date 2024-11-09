from functools import partial
from nnsight import LanguageModel
import torch
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides
import os
from pathlib import Path
import json

def process_layer(layer: int, model: LanguageModel, feature_cfg: FeatureConfig, 
                 experiment_cfg: ExperimentConfig):
    """Process a single layer and save top feature selections"""
    print(f"Processing layer {layer}")
    
    # Get model directory and corresponding latents directory
    model_dir = config.save_directory
    latents_dir = config.saved_latents_dir / f"latents_{model_dir.name}"
    
    if not latents_dir.exists():
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
    
    module = f".gpt_neox.layers.{layer}"
    feature_dict = {module: torch.arange(0,100)}
    
    cache_config_dir = f"{latents_dir}/{module}/config.json"
    with open(cache_config_dir, "r") as f:
        cache_config = json.load(f)
    
    cache_config["dataset_repo"] = config.dataset
    
    with open(cache_config_dir, "w") as f:
        json.dump(cache_config, f)
            
    dataset = FeatureDataset(
        raw_dir=latents_dir,
        cfg=feature_cfg,
        modules=[module],
        features=feature_dict,
    )

    constructor = partial(
        default_constructor, 
        tokens=dataset.tokens,
        n_random=experiment_cfg.n_random,
        ctx_len=experiment_cfg.example_ctx_len,
        max_examples=feature_cfg.max_examples
    )
    
    sampler = partial(sample, cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)
    
    # Save detailed results in model directory
    save_path = model_dir / f"max_activations_layer_{layer}.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a list to store feature info
    feature_info = []
    
    with open(save_path, "w") as f:
        for feature in loader:
            # Extract feature number from format "layer_featureN"
            feature_str = str(feature.feature)
            feature_num = int(feature_str.split('feature')[-1])
            max_activation = float(feature.max_activation)
            
            feature_info.append({
                "feature": feature_num,
                "max_activation": max_activation
            })
            
            f.write(f"Feature {feature_str}\n")
            for example in feature.examples[:5]:
                f.write(''.join(model.tokenizer.batch_decode(example.tokens)) + '\n')
                f.write(str(example.activations.max().item()) + '\n')
                f.write('  '.join(model.tokenizer.batch_decode(
                    example.tokens[torch.where(example.activations > 0)])) + '\n')
                f.write('-' * 100 + '\n')
            f.write(f"Max activation: {max_activation}\n")
            f.write("=" * 100 + '\n')
    
    # Sort features by max activation and save top N
    feature_info.sort(key=lambda x: x["max_activation"], reverse=True)
    top_features = feature_info[:config.num_latents_to_explain]
    
    # Save selected features
    features_save_path = model_dir / "selected_features.json"
    
    # Load existing or create new features dict
    if features_save_path.exists():
        with open(features_save_path, "r") as f:
            all_features = json.load(f)
    else:
        all_features = {}
    
    all_features[f"layer_{layer}"] = [f["feature"] for f in top_features]
    
    with open(features_save_path, "w") as f:
        json.dump(all_features, f, indent=2)

def main():
    # Load and apply configuration overrides
    overrides = parse_config_overrides()
    # Set reinit_non_embedding based on the flag
    if "--no-reinit_non_embedding" in os.sys.argv:
        overrides["reinit_non_embedding"] = False
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
    print(f"Processing model in {config.save_directory}")
    print(f"Reinitialization mode: {'reinit' if config.reinit_non_embedding else 'no-reinit'}")
    print(f"Using model: {config.model_name}")
    print(f"Using dataset: {config.dataset}")
    if config.dataset_name:
        print(f"Dataset config: {config.dataset_name}")
    
    # Initialize model
    model = LanguageModel(
        config.model_name,
        device_map=config.device_map,
        dispatch=True,
        torch_dtype=config.torch_dtype
    )
    
    # Set up configurations
    feature_cfg = FeatureConfig(
        width=config.feature_width,
        min_examples=config.min_examples,
        max_examples=config.max_examples,
        n_splits=config.n_splits
    )
    
    experiment_cfg = ExperimentConfig(
        n_examples_train=config.n_examples_train,
        n_examples_test=config.n_examples_test,
        n_quantiles=config.n_quantiles,
        example_ctx_len=config.example_ctx_len,
        n_random=config.n_random,
        train_type=config.train_type,
        test_type=config.test_type,
    )

    # Process each layer
    for layer in range(5):
        process_layer(layer, model, feature_cfg, experiment_cfg)

if __name__ == "__main__":
    main()