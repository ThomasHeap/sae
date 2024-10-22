from functools import partial
from nnsight import LanguageModel
import torch
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from experiment_config import config
import os

os.environ['HF_HOME'] = str(config.cache_dir)

model = LanguageModel(config.model_name, device_map=config.get_device_map(), dispatch=True, torch_dtype=config.get_torch_dtype())

feature_cfg = FeatureConfig(
    width=config.feature_width,
    min_examples=config.min_examples,
    max_examples=config.max_examples,
    n_splits=config.n_splits
)

for layer in range(5):
    module = f".gpt_neox.layers.{layer}"
    feature_dict = {module: torch.arange(0,100)}
    for dir in config.saved_latents_dir.iterdir():
        print(f"Loading autoencoders from {dir}")

        dataset = FeatureDataset(
                raw_dir=dir,
                cfg=feature_cfg,
                modules=[module],
                features=feature_dict,
        )

        cfg = ExperimentConfig(
            n_examples_train=config.n_examples_train,
            n_examples_test=config.n_examples_test,
            n_quantiles=config.n_quantiles,
            example_ctx_len=config.example_ctx_len,
            n_random=config.n_random,
            train_type=config.train_type,
            test_type=config.test_type,
        )

        constructor = partial(default_constructor, tokens=dataset.tokens, n_random=cfg.n_random, ctx_len=cfg.example_ctx_len, max_examples=feature_cfg.max_examples)
        sampler = partial(sample, cfg=cfg)

        loader = FeatureLoader(
            feature_dataset=dataset,
            constructor=constructor,
            sampler=sampler,
        )

        num_features = 5
        i = 0
        
        with open(config.saved_models_dir / f"{dir.name.split('_')[-1]}/{dir.name}_layer_{layer}.txt", "w") as f:
            for feature in loader:
                #save to file in saved_models
                
                f.write(str(feature.feature))
                f.write('\n')
                for example in feature.examples[:5]:
                    f.write(''.join(model.tokenizer.batch_decode(example.tokens)))
                    f.write('\n')
                    f.write(str(example.activations.max().item()))
                    f.write('\n')
                    f.write('  '.join(model.tokenizer.batch_decode(example.tokens[torch.where(example.activations > 0)])))
                    f.write('\n')
                    f.write('-'*100)
                    f.write('\n')
                f.write(str(feature.max_activation))
                f.write('\n')
                f.write("="*100)
                
                if i >= num_features:
                    break