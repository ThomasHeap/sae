from functools import partial
from os import getenv
import asyncio
import torch
import orjson
import os
from sae_auto_interp.clients import OpenRouter
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.features import (
    FeatureDataset,
    FeatureLoader
)
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from config import config

API_KEY = getenv("OPENROUTER_API_KEY")

feature_cfg = FeatureConfig(
    width=config.feature_width,
    min_examples=config.min_examples,
    max_examples=config.max_examples,
    n_splits=config.n_splits
)

module = f".gpt_neox.layers.{config.layer_to_explain}"  # Adjust this based on your model architecture
feature_dict = {module: torch.arange(0, config.num_latents_to_explain)}

dataset = FeatureDataset(
    raw_dir=config.saved_latents_dir,
    cfg=feature_cfg,
    modules=[module],
    features=feature_dict,
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

constructor = partial(
    default_constructor,
    tokens=dataset.tokens,
    n_random=experiment_cfg.n_random,
    ctx_len=experiment_cfg.example_ctx_len,
    max_examples=feature_cfg.max_examples
)
sampler = partial(sample, cfg=experiment_cfg)
loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

client = OpenRouter("anthropic/claude-3.5-sonnet", api_key=API_KEY)

def explainer_postprocess(result):
    explanation_dir = config.saved_models_dir / "explanations"
    explanation_dir.mkdir(parents=True, exist_ok=True)
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

async def main():
    await pipeline.run(config.num_parallel_latents)

if __name__ == "__main__":
    asyncio.run(main())