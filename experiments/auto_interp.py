from os import getenv
import asyncio
import torch
import orjson
from sae_auto_interp.clients import OpenRouter, Offline
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides
import os
from pathlib import Path
from functools import partial
import argparse
import json 

def parse_args():
    parser = argparse.ArgumentParser(description='Generate examples using OpenRouter API')
    parser.add_argument('--api_key', type=str, required=True,
                      help='OpenRouter API key')
    args, remaining = parser.parse_known_args()
    return args

def format_example(tokens, activations, tokenizer):
    """Format a single example with its max activation value and tokens"""
    # Convert tensors to CPU and numpy/python types
    tokens = tokens.cpu()
    activations = activations.cpu()
    
    # Get max activation and its location
    max_val = float(activations.max().item())  # Convert to Python float
    
    # Get indices where activation is positive
    active_indices = torch.where(activations > 0)[0]
    active_tokens = tokenizer.batch_decode(tokens[active_indices])
    
    # Format the complete text and the active tokens
    full_text = ''.join(tokenizer.batch_decode(tokens))
    active_text = '  '.join(active_tokens)
    
    return {
        'full_text': full_text,
        'max_activation': max_val,
        'active_tokens': active_text
    }

async def process_layer_directory(layer: int, latents_dir: Path, client, feature_cfg: FeatureConfig, 
                                experiment_cfg: ExperimentConfig):
    """Process a single layer in a directory using selected features"""
    print(f"Processing layer {layer}")
    save_dir = latents_dir
    # Load selected features
    features_path = save_dir / "max/selected_features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Selected features file not found: {features_path}")
    
    with open(features_path, "r") as f:
        selected_features = json.load(f)
    
    layer_features = selected_features.get(f"layer_{layer}")[:experiment_cfg.n_random]
    if not layer_features:
        raise ValueError(f"No selected features found for layer {layer}")
    
    module = f".gpt_neox.layers.{layer}"
    # Convert feature numbers back to the expected format
    feature_dict = {module: torch.tensor([f for f in layer_features])}
    print(f"Processing {len(layer_features)} selected features for layer {layer}")
    n_features = len(layer_features)
    
    try:
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

        # Create directory for explanations
        explanation_dir = save_dir / "explanations" / f"layer_{layer}"
        explanation_dir.mkdir(parents=True, exist_ok=True)

        # Track count of processed features
        feature_count = 0

        def explainer_postprocess(result):
            nonlocal feature_count
            feature_count += 1
            
            # Get feature number
            feature_num = str(result.record.feature).split('.')[-1]
            
            # Get the feature examples
            examples = []
            for example in result.record.examples[:5]:  # Take top 5 examples
                example_data = format_example(example.tokens, example.activations, dataset.tokenizer)
                examples.append(example_data)
            
            # Combine interpretation and examples
            output_data = {
                'interpretation': result.explanation,
                'feature': feature_num,
                'max_activation': float(result.record.max_activation),
                'examples': examples
            }
            
            # Save both interpretation and examples
            with open(explanation_dir / f"feature_{feature_num}.json", "wb") as f:
                f.write(orjson.dumps(output_data))
            
            # Also save a human-readable text version
            with open(explanation_dir / f"feature_{feature_num}.txt", "w") as f:
                f.write(f"Feature {feature_num} Interpretation:\n")
                f.write("=" * 80 + "\n")
                f.write(str(result.explanation) + "\n\n")
                f.write("Max Activating Examples:\n")
                f.write("=" * 80 + "\n")
                for i, example in enumerate(examples, 1):
                    f.write(f"\nExample {i}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Full text:\n{example['full_text']}\n\n")
                    f.write(f"Max activation: {example['max_activation']:.4f}\n")
                    f.write(f"Active tokens: {example['active_tokens']}\n")
                    f.write("-" * 80 + "\n")
            
            print(f"Layer {layer}: Processed feature {feature_count}/{n_features}")
            return result



        # Builds the record from result returned by the pipeline
        def scorer_preprocess(result):
            record = result.record   
            record.explanation = result.explanation
            record.extra_examples = record.random_examples

            return record
            
        # Saves the score to a file
        def scorer_postprocess(result):
            with open(explanation_dir / f"{result.record.feature}_score.txt", "wb") as f:
                f.write(orjson.dumps(result.score))
                
            
    
        explainer_pipe = process_wrapper(
            DefaultExplainer(
                client,
                tokenizer=dataset.tokenizer,
            ),
            postprocess=explainer_postprocess,
        )

        scorer_pipe = process_wrapper(
            FuzzingScorer(client, tokenizer=dataset.tokenizer),
            preprocess=scorer_preprocess,
            postprocess=scorer_postprocess,
        )
        
        pipeline = Pipeline(
            loader,
            explainer_pipe,
            scorer_pipe
        )

        await pipeline.run(n_features)  # Process all features concurrently
        
        # Print summary of processed features
        print(f"\nLayer {layer} processing complete.")
        print(f"Processed {feature_count} features")
        if feature_count != n_features:
            print(f"WARNING: Expected {n_features} features but processed {feature_count}")
        
    except Exception as e:
        print(f"Error processing layer {layer}: {str(e)}")
        raise e

async def main():
    # Parse CLI arguments first to get API key
    args = parse_args()
    
    # Load and apply configuration overrides
    overrides = parse_config_overrides()
    if "--no-reinit_non_embedding" in os.sys.argv:
        overrides["reinit_non_embedding"] = False
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
    # Print configuration info
    print(f"\nProcessing model in {config.save_directory}")
    print(f"Reinitialization mode: {'reinit' if config.reinit_non_embedding else 'no-reinit'}")
    print(f"Using model: {config.model_name}")
    print(f"Using dataset: {config.dataset}")
    print(f"Number of features to explain per layer: {config.n_random}")
    if config.dataset_name:
        print(f"Dataset config: {config.dataset_name}")
    
    # Initialize API client with CLI provided key
    client = OpenRouter("openai/gpt-4o-mini", api_key=args.api_key)
    # client = Offline("cache/models_hugging-quants_Meta-Llama-3.1-70B-Instruct-AWQ-INT4/snapshots/2123003760781134cfc31124aa6560a45b491fdf",
    #                  max_memory=0.8,max_model_len=5120, num_gpus=1)
    
    # Get corresponding latents directory
    model_dir = config.save_directory
    latents_dir = config.saved_latents_dir / f"latents_{model_dir.name}"
    if not latents_dir.exists():
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
    
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

    # Create tasks for each layer
    tasks = []
    for layer in range(5):
        tasks.append(process_layer_directory(
            layer, latents_dir, client, feature_cfg, experiment_cfg
        ))
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())