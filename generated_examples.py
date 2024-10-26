from os import getenv
import asyncio
import torch
import orjson
from sae_auto_interp.clients import OpenRouter
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides
from pipeline_utils import get_dirs_to_process
import os
from pathlib import Path
from functools import partial

API_KEY = getenv("OPENROUTER_API_KEY")

async def process_layer_directory(layer: int, dir_path: Path, client, feature_cfg: FeatureConfig, 
                                experiment_cfg: ExperimentConfig):
    """Process a single layer in a directory"""
    print(f"Processing layer {layer} in {dir_path}")
    
    module = f".gpt_neox.layers.{layer}"
    feature_dict = {module: torch.arange(0, config.num_latents_to_explain)}
    
    try:
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

        # Create directory for explanations
        explanation_dir = config.saved_models_dir / dir_path.name.split('_')[-1] / f"layer_{layer}"
        explanation_dir.mkdir(parents=True, exist_ok=True)

        def explainer_postprocess(result):
            with open(explanation_dir / f"{result.record.feature}.txt", "wb") as f:
                f.write(orjson.dumps(result.explanation))
            del result
            return None

        explainer_pipe = process_wrapper(
            DefaultExplainer(
                client,
                tokenizer=dataset.tokenizer,
            ),
            postprocess=explainer_postprocess,
        )

        pipeline = Pipeline(
            loader,
            explainer_pipe,
        )

        await pipeline.run(config.num_parallel_latents)
        
    except Exception as e:
        print(f"Error processing layer {layer} in {dir_path}: {str(e)}")

async def main():
    # Load and apply configuration
    overrides = parse_config_overrides()
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
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

    # Initialize API client
    client = OpenRouter("anthropic/claude-3.5-sonnet", api_key=API_KEY)

    # Get directories to process
    dirs_to_process = get_dirs_to_process(config.saved_latents_dir, 
                                        overrides.get('model_dirs'),
                                        latents_prefix=True)

    # Create tasks for each directory and layer
    tasks = []
    for dir_path in dirs_to_process:
        for layer in range(5):
            tasks.append(process_layer_directory(
                layer, dir_path, client, feature_cfg, experiment_cfg
            ))
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())