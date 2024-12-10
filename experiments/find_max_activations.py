from functools import partial
from nnsight import LanguageModel
import torch
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides
from noise_embedding import NoiseEmbeddingNNsight
import os
from pathlib import Path
import json

#seed torch random
torch.manual_seed(42)

def process_layer(layer: int, model: LanguageModel, feature_cfg: FeatureConfig, 
                 experiment_cfg: ExperimentConfig):
    """Process a single layer and save top feature selections"""
    print(f"Processing layer {layer}")
    
    
        
    # Get latents directory from config
    latents_dir = config.latents_directory
    
    if not latents_dir.exists():
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
    
    module = f".gpt_neox.layers.{layer}"
    #select random features
    feature_dict = {module: torch.randperm(feature_cfg.width)[:300]}
    print(f"Selected features: {feature_dict}")
    
    cache_config_dir = f"{latents_dir}/{module}/config.json"
    with open(cache_config_dir, "r") as f:
        cache_config = json.load(f)
    
    cache_config["dataset_repo"] = config.dataset
    
    with open(cache_config_dir, "w") as f:
        json.dump(cache_config, f)
    
    print(f"Loading latents from {latents_dir}")
    dataset = FeatureDataset(
        raw_dir=latents_dir,
        cfg=feature_cfg,
        modules=[module],
        features=feature_dict,
    )

    print(f"Creating constructor and sampler for layer {layer}")
    constructor = partial(
        default_constructor, 
        tokens=dataset.tokens,
        n_random=experiment_cfg.n_random,
        ctx_len=experiment_cfg.example_ctx_len,
        max_examples=feature_cfg.max_examples
    )
    
    sampler = partial(sample, cfg=experiment_cfg)
    loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)
    
    # Save detailed results
    save_path = latents_dir / f"max/max_activations_layer_{layer}.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a list to store feature info
    feature_info = []
    features_seen = 0

    with open(save_path, "w") as f:
        for feature in loader:
            if features_seen >= config.n_random:
                break
            
            print(f"Processing feature {str(feature.feature)}")
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
            
            features_seen += 1
    
    print(f"Saved {features_seen} features to {save_path}")
    # Sort features by max activation and save top N
    feature_info.sort(key=lambda x: x["max_activation"], reverse=True)
    top_features = feature_info[:config.n_random]
    
    # Save selected features
    features_save_path = latents_dir / "max/selected_features.json"
    
    # Load existing or create new features dict
    if features_save_path.exists():
        with open(features_save_path, "r") as f:
            all_features = json.load(f)
    else:
        all_features = {}
    
    all_features[f"layer_{layer}"] = [f["feature"] for f in top_features]
    print(features_save_path)
    with open(features_save_path, "w") as f:
        json.dump(all_features, f, indent=2)

def main():
    # Load and apply configuration overrides
    overrides = parse_config_overrides()
    if "--no-reinit_non_embedding" in os.sys.argv:
        overrides["reinit_non_embedding"] = False
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
    print(f"Processing model in {config.save_directory}")
    print(f"Random control mode: {'enabled' if config.use_random_control else 'disabled'}")
    print(f"Reinitialization mode: {'reinit' if config.reinit_non_embedding else 'no-reinit'}")
    print(f"Using model: {config.model_name}")
    print(f"Using dataset: {config.dataset}")
    if config.dataset_name:
        print(f"Dataset config: {config.dataset_name}")
    
    #get feature width from text file
    feature_width_path = config.latents_directory / "feature_width.txt"
    with open(feature_width_path, "r") as f:
        feature_width = int(f.read())
    
    # Initialize model
    model = LanguageModel(
        config.model_name,
        device_map=config.device_map,
        dispatch=True,
        torch_dtype=config.torch_dtype
    )
    
    if config.rerandomize:
        print(f"Rerandomizing model parameters (embeddings: {config.rerandomize_embeddings})")
        model = RerandomizedModel(
            model,
            rerandomize_embeddings=config.rerandomize_embeddings,
            seed=config.random_seed
        ).model
    
    if config.use_random_control:
        print(f"Applying random control with noise std: {config.noise_std}")
        model = NoiseEmbeddingNNsight(model, std=config.noise_std).model
    
    # Set up configurations
    feature_cfg = FeatureConfig(
        width=feature_width,
        min_examples=config.min_examples,
        max_examples=config.max_examples,
        n_splits=config.n_splits
    )
    
    experiment_cfg = ExperimentConfig(
        n_examples_train=200,
        example_ctx_len=config.example_ctx_len,
        train_type=config.train_type,
    )

    # Process each layer
    for layer in range(5):
        
        process_layer(layer, model, feature_cfg, experiment_cfg)

if __name__ == "__main__":
    main()