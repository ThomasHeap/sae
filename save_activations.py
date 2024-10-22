from nnsight import LanguageModel
from sae_auto_interp.autoencoders import load_eai_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data
from experiment_config import config
import os
import torch
from pathlib import Path
import json
from config import config
import numpy as np

os.environ['HF_HOME'] = str(config.cache_dir)


def compute_autocorrelation(activations, max_lag=10):
    """
    Compute autocorrelation of feature activations across tokens.
    
    Args:
        activations: torch.Tensor of shape (batch_size, sequence_length, n_features)
        max_lag: Maximum lag to compute autocorrelation for
        
    Returns:
        Dictionary with mean and std of autocorrelations per lag
    """
    # Move to CPU and convert to numpy for calculations
    activations = activations.cpu().numpy()
    batch_size, seq_len, n_features = activations.shape
    
    # Initialize arrays to store correlations
    autocorrs = np.zeros((batch_size, n_features, max_lag))
    
    # Compute autocorrelation for each feature in each sequence
    for b in range(batch_size):
        for f in range(n_features):
            feature_acts = activations[b, :, f]
            # Normalize the activations
            feature_acts = (feature_acts - np.mean(feature_acts)) / (np.std(feature_acts) + 1e-8)
            
            for lag in range(max_lag):
                if lag == 0:
                    autocorrs[b, f, lag] = 1.0
                else:
                    # Compute correlation between sequence and shifted sequence
                    correlation = np.corrcoef(feature_acts[:-lag], feature_acts[lag:])[0, 1]
                    autocorrs[b, f, lag] = correlation if not np.isnan(correlation) else 0.0
    
    # Compute statistics across batches and features
    mean_autocorr = np.mean(autocorrs, axis=(0, 1))  # Average across batches and features
    std_autocorr = np.std(autocorrs, axis=(0, 1))
    
    return {
        "lags": list(range(max_lag)),
        "mean_autocorr": mean_autocorr.tolist(),
        "std_autocorr": std_autocorr.tolist()
    }

class AutocorrelationCache(FeatureCache):
    """Extended FeatureCache that also computes autocorrelation"""
    def __init__(self, *args, max_lag=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_lag = max_lag
        self.autocorrelations = {}
        
    def cache_batch(self, batch_idx, batch):
        # Get regular activations using parent class
        super().cache_batch(batch_idx, batch)
        
        # Compute autocorrelation for each module
        for module_name, activations in self.cached_activations.items():
            if module_name not in self.autocorrelations:
                self.autocorrelations[module_name] = []
            
            autocorr = compute_autocorrelation(activations, self.max_lag)
            self.autocorrelations[module_name].append(autocorr)
    
    def finalize_autocorrelations(self):
        """Average autocorrelations across all batches"""
        final_autocorrelations = {}
        
        for module_name, batch_autocorrs in self.autocorrelations.items():
            n_batches = len(batch_autocorrs)
            lags = batch_autocorrs[0]["lags"]
            mean_autocorrs = np.array([b["mean_autocorr"] for b in batch_autocorrs])
            std_autocorrs = np.array([b["std_autocorr"] for b in batch_autocorrs])
            
            final_autocorrelations[module_name] = {
                "lags": lags,
                "mean_autocorr": np.mean(mean_autocorrs, axis=0).tolist(),
                "std_autocorr": np.mean(std_autocorrs, axis=0).tolist()
            }
        
        return final_autocorrelations

model = LanguageModel(config.model_name, device_map=config.get_device_map(), dispatch=True, torch_dtype=config.get_torch_dtype())

for dir in config.saved_models_dir.iterdir():
    if not dir.is_dir() or 'sae' not in dir.name:
        continue
    
    print(f"Processing {dir}")
    submodule_dict, model = load_eai_autoencoders(
        model,
        [0,1,2,3,4,5],
        weight_dir=str(dir),
        module="res"
    )

    print(f"Loaded autoencoder modules: {submodule_dict.keys()}")

    cfg = CacheConfig(
        dataset_repo=config.dataset_repo,
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

    # Use our extended cache class
    cache = AutocorrelationCache(
        model,
        submodule_dict,
        batch_size=cfg.batch_size,
        max_lag=config.cache_ctx_len // 2  # Use half the context length as max lag
    )

    cache.run(cfg.n_tokens, tokens)

    # Save regular cache data
    save_dir = config.saved_latents_dir / f"latents_{dir.name}"
    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir=save_dir
    )

    cache.save_config(
        save_dir=save_dir,
        cfg=cfg,
        model_name=config.model_name
    )

    # Save autocorrelation results
    autocorr_results = cache.finalize_autocorrelations()
    autocorr_save_path = save_dir / "autocorrelations.json"
    with open(autocorr_save_path, 'w') as f:
        json.dump(autocorr_results, f, indent=2)

    print(f"Saved autocorrelation results to {autocorr_save_path}")