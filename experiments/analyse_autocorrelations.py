from nnsight import LanguageModel
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sae_auto_interp.features import FeatureDataset, FeatureLoader
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides
import os
import torch
from functools import partial
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.config import ExperimentConfig, FeatureConfig

def compute_autocorrelations(activations: torch.Tensor, max_lag: int) -> dict:
    """Compute autocorrelation for an activation sequence"""
    # Ensure we're working with numpy
    act_np = activations.cpu().numpy()
    
    # Compute mean and std for the sequence
    mean_act = float(np.mean(act_np))
    std_act = float(np.std(act_np))
    
    # Check for constant sequence
    if std_act < 1e-10:
        print(f"Warning: Nearly constant sequence detected. Mean: {mean_act:.4f}, Std: {std_act:.4e}")
        return {
            "mean_autocorr": [1.0] + [0.0] * (max_lag - 1),  # Correlation with constant is 0
            "std_autocorr": std_act,
            "mean_activation": mean_act,
            "is_constant": True
        }
    
    # Compute autocorrelation for different lags
    autocorrs = []
    for lag in range(max_lag):
        if lag == 0:
            autocorrs.append(1.0)  # Correlation with itself is 1
            continue
            
        # Get the sequences to correlate
        seq1 = act_np[lag:]
        seq2 = act_np[:-lag]
        
        # Check for numerical issues
        if len(seq1) == 0 or len(seq2) == 0:
            print(f"Warning: Empty sequence at lag {lag}. Sequence length: {len(act_np)}")
            autocorrs.append(0.0)
            continue
            
        std1 = np.std(seq1)
        std2 = np.std(seq2)
        
        if std1 < 1e-10 or std2 < 1e-10:
            print(f"Warning: Near-constant subsequence at lag {lag}. Std1: {std1:.4e}, Std2: {std2:.4e}")
            autocorrs.append(0.0)
            continue
        
        try:
            # Compute correlation manually to have more control
            seq1_norm = (seq1 - np.mean(seq1)) / std1
            seq2_norm = (seq2 - np.mean(seq2)) / std2
            correlation = np.mean(seq1_norm * seq2_norm)
            
            if not -1.0 <= correlation <= 1.0:
                print(f"Warning: Invalid correlation value at lag {lag}: {correlation}")
                correlation = np.clip(correlation, -1.0, 1.0)
                
            autocorrs.append(float(correlation))
            
        except Exception as e:
            print(f"Error computing correlation at lag {lag}: {str(e)}")
            print(f"Sequence stats - Length: {len(seq1)}, Mean1: {np.mean(seq1):.4f}, "
                  f"Std1: {std1:.4e}, Mean2: {np.mean(seq2):.4f}, Std2: {std2:.4e}")
            autocorrs.append(0.0)
    
    return {
        "mean_autocorr": autocorrs,
        "std_autocorr": std_act,
        "mean_activation": mean_act,
        "is_constant": False
    }

def process_layer(layer: int, model: LanguageModel, feature_cfg: FeatureConfig, 
                 experiment_cfg: ExperimentConfig):
    """Process autocorrelations for a single layer"""
    print(f"Processing layer {layer}")
    
    # Get model directory and corresponding latents directory
    model_dir = config.save_directory
    latents_dir = config.saved_latents_dir / f"latents_{model_dir.name}"
    
    if not latents_dir.exists():
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
    
    module = f".gpt_neox.layers.{layer}"
    feature_dict = {module: torch.arange(0,100)}
    
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
    
    # Calculate maximum lag based on context length
    max_lag = int(experiment_cfg.example_ctx_len * config.max_lag_ratio)
    
    # Track statistics about the activations
    n_constant = 0
    n_total = 0
    
    # Compute autocorrelations for each feature
    autocorrelations = {}
    for feature in loader:
        feature_autocorrs = []
        n_total += len(feature.examples)
        
        print(f"\nProcessing feature {feature.feature}")
        for i, example in enumerate(feature.examples[:5]):
            print(f"Example {i}:")
            print(f"Activation shape: {example.activations.shape}")
            print(f"Activation stats - Min: {example.activations.min().item():.4f}, "
                  f"Max: {example.activations.max().item():.4f}, "
                  f"Mean: {example.activations.mean().item():.4f}, "
                  f"Std: {example.activations.std().item():.4e}")
            
            autocorr = compute_autocorrelations(example.activations, max_lag)
            if autocorr.get("is_constant", False):
                n_constant += 1
            feature_autocorrs.append(autocorr)
        
        # Average across examples
        mean_autocorr = np.mean([a["mean_autocorr"] for a in feature_autocorrs], axis=0)
        std_autocorr = np.mean([a["std_autocorr"] for a in feature_autocorrs])
        
        autocorrelations[f"{module}.{feature.feature}"] = {
            "mean_autocorr": mean_autocorr.tolist(),
            "std_autocorr": float(std_autocorr)
        }
    
    print(f"\nLayer {layer} statistics:")
    print(f"Total examples processed: {n_total}")
    print(f"Constant sequences found: {n_constant} ({n_constant/n_total*100:.1f}%)")
    
    return autocorrelations, max_lag

def plot_combined_autocorrelations(all_layer_data: dict, model_dir: Path):
    """Create combined visualizations of the autocorrelation patterns across layers"""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#CCCCCC',
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
    })
    
    save_dir = model_dir / "autocorrelation_analysis"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n_lags = len(next(iter(all_layer_data[0].values()))["mean_autocorr"])
    n_layers = len(all_layer_data)
    
    # Create subplot grid: layer comparisons and individual layer patterns
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
    
    # Top plot: All layers overlaid
    ax_combined = fig.add_subplot(gs[0])
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    
    # Store mean autocorrelations for later analysis
    layer_means = []
    for layer in range(n_layers):
        layer_data = all_layer_data[layer]
        mean_autocorr = np.mean([d["mean_autocorr"] for d in layer_data.values()], axis=0)
        std_autocorr = np.std([d["mean_autocorr"] for d in layer_data.values()], axis=0)
        layer_means.append(mean_autocorr)
        
        ax_combined.plot(range(n_lags), mean_autocorr, '-', 
                        color=colors[layer], 
                        label=f'Layer {layer}',
                        linewidth=2)
        ax_combined.fill_between(range(n_lags), 
                            mean_autocorr - std_autocorr, 
                            mean_autocorr + std_autocorr, 
                            color=colors[layer],
                            alpha=0.2)
    
    ax_combined.set_xlabel("Lag", fontsize=12)
    ax_combined.set_ylabel("Mean Autocorrelation", fontsize=12)
    ax_combined.set_title("Layer-wise Comparison of Feature Autocorrelations", fontsize=14)
    ax_combined.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_combined.grid(True, alpha=0.3)
    
    # Bottom: Individual layer subplots
    gs_bottom = gs[1].subgridspec(1, n_layers, wspace=0.3)
    axes = [fig.add_subplot(gs_bottom[i]) for i in range(n_layers)]
    
    layer_means = np.array(layer_means)
    vmin = np.min(layer_means)
    vmax = np.max(layer_means)
    
    for layer, ax in enumerate(axes):
        layer_data = all_layer_data[layer]
        all_autocorrs = np.array([d["mean_autocorr"] for d in layer_data.values()])
        mean_autocorr = np.mean(all_autocorrs, axis=0)
        std_autocorr = np.std(all_autocorrs, axis=0)
        
        # Plot mean with confidence interval
        ax.plot(range(n_lags), mean_autocorr, '-', color=colors[layer], linewidth=2)
        ax.fill_between(range(n_lags), 
                       mean_autocorr - std_autocorr,
                       mean_autocorr + std_autocorr,
                       color=colors[layer],
                       alpha=0.2)
        
        ax.set_title(f"Layer {layer}", fontsize=12)
        ax.set_xlabel("Lag" if layer == 0 else "")
        ax.set_ylabel("Autocorrelation" if layer == 0 else "")
        ax.set_ylim(vmin, vmax)  # Use same scale for all plots
        ax.grid(True, alpha=0.3)
        
        # Add average autocorrelation value to title
        avg_autocorr = np.mean(mean_autocorr[1:])  # Exclude lag 0
        ax.set_title(f"Layer {layer}\nMean: {avg_autocorr:.3f}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / "layer_autocorrelation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation trend plot
    plt.figure(figsize=(10, 6))
    mean_autocorrs = [np.mean(np.mean([d["mean_autocorr"][1:] for d in layer_data.values()])) for layer_data in all_layer_data]
    plt.plot(range(n_layers), mean_autocorrs, '-o', linewidth=2)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Average Autocorrelation", fontsize=12)
    plt.title("Trend of Average Autocorrelation Across Layers", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "autocorrelation_trend.png", dpi=300, bbox_inches='tight')
    plt.close()

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

    # Process all layers and collect data
    all_layer_data = []
    for layer in range(5):
        layer_data, _ = process_layer(layer, model, feature_cfg, experiment_cfg)
        all_layer_data.append(layer_data)
    
    # Create combined visualizations
    plot_combined_autocorrelations(all_layer_data, config.save_directory)

if __name__ == "__main__":
    main()