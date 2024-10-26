from functools import partial
from nnsight import LanguageModel
import torch
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides, load_saved_config
from pipeline_utils import get_dirs_to_process
import os

def process_layer_and_directory(layer: int, dir_path: Path, model: LanguageModel, 
                              feature_cfg: FeatureConfig, experiment_cfg: ExperimentConfig):
    """Process a single layer for a specific model directory"""
    print(f"Processing layer {layer} in {dir_path}")
    
    module = f".gpt_neox.layers.{layer}"
    feature_dict = {module: torch.arange(0,100)}
    
    dataset = FeatureDataset(
        raw_dir=dir_path,
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
    
    # Save results
    save_path = config.saved_models_dir / f"{dir_path.name.split('_')[-1]}/{dir_path.name}_layer_{layer}.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        for feature in loader:
            f.write(str(feature.feature) + '\n')
            for example in feature.examples[:5]:
                f.write(''.join(model.tokenizer.batch_decode(example.tokens)) + '\n')
                f.write(str(example.activations.max().item()) + '\n')
                f.write('  '.join(model.tokenizer.batch_decode(
                    example.tokens[torch.where(example.activations > 0)])) + '\n')
                f.write('-' * 100 + '\n')
            f.write(str(feature.max_activation) + '\n')
            f.write("=" * 100 + '\n')

def main():
    overrides = parse_config_overrides()
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
    
    
    # Get directories to process
    dirs_to_process = get_dirs_to_process(config.saved_latents_dir, 
                                        overrides.get('model_dirs'),
                                        latents_prefix=True)
    
    # Process each directory and layer
    for dir_path in dirs_to_process:
        
        #load config
        current_config = load_saved_config(dir_path / "config.json", apply_cli_overrides=False)
        # Initialize model
        model = LanguageModel(
            current_config.model_name,
            device_map=current_config.get_device_map(),
            dispatch=True,
            torch_dtype=current_config.get_torch_dtype()
        )
        
        # Set up configurations
        feature_cfg = FeatureConfig(
            width=current_config.feature_width,
            min_examples=current_config.min_examples,
            max_examples=current_config.max_examples,
            n_splits=current_config.n_splits
        )
        
        experiment_cfg = ExperimentConfig(
            n_examples_train=current_config.n_examples_train,
            n_examples_test=current_config.n_examples_test,
            n_quantiles=current_config.n_quantiles,
            example_ctx_len=current_config.example_ctx_len,
            n_random=current_config.n_random,
            train_type=current_config.train_type,
            test_type=current_config.test_type,
        )
    
        for layer in range(5):
            process_layer_and_directory(layer, dir_path, model, feature_cfg, experiment_cfg)

if __name__ == "__main__":
    main()