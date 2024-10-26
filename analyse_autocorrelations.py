import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides
from pipeline_utils import get_dirs_to_process
import os

def load_autocorrelations(dir_path: Path) -> dict:
    """
    Load autocorrelation data from a directory
    """
    if not (dir_path / "autocorrelations.json").exists():
        print(f"No autocorrelation data found in {dir_path}")
        return {}
        
    with open(dir_path / "autocorrelations.json") as f:
        data = json.load(f)
        
    layer_autocorrs = {}
    for module_name in data:
        layer = int(module_name.split('.')[-1])
        if layer not in layer_autocorrs:
            layer_autocorrs[layer] = []
        layer_autocorrs[layer].append(data[module_name])
    
    return layer_autocorrs

def compute_layer_statistics(layer_autocorrs: dict) -> dict:
    """
    Compute statistics about autocorrelation across layers
    """
    layer_stats = {}
    
    for layer, autocorrs in layer_autocorrs.items():
        all_means = np.array([run["mean_autocorr"] for run in autocorrs])
        all_stds = np.array([run["std_autocorr"] for run in autocorrs])
        
        layer_stats[layer] = {
            "mean": np.mean(all_means, axis=0),
            "std": np.mean(all_stds, axis=0),
            "ci": stats.t.interval(0.95, len(all_means)-1, 
                                 loc=np.mean(all_means, axis=0),
                                 scale=stats.sem(all_means, axis=0))
        }
    
    return layer_stats

def plot_layer_autocorrelations(layer_stats: dict, save_dir: Path, model_name: str):
    """Create visualizations of the autocorrelation patterns"""
    plt.style.use('seaborn')
    
    # Plot mean autocorrelation at different lags for each layer
    plt.figure(figsize=(12, 8))
    layers = sorted(layer_stats.keys())
    for layer in layers:
        stats = layer_stats[layer]
        lags = np.arange(len(stats["mean"]))
        plt.plot(lags, stats["mean"], label=f"Layer {layer}", marker='o', alpha=0.7)
        plt.fill_between(lags, stats["ci"][0], stats["ci"][1], alpha=0.2)
    
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(f"Autocorrelation by Layer and Lag - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / f"{model_name}_autocorrelation_by_layer.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot layer-wise autocorrelation for different lags
    plt.figure(figsize=(12, 8))
    lags_to_plot = [1, 2, 5, 10]
    for lag in lags_to_plot:
        layer_means = [layer_stats[layer]["mean"][lag] for layer in layers]
        plt.plot(layers, layer_means, label=f"Lag {lag}", marker='o')
    
    plt.xlabel("Layer")
    plt.ylabel("Autocorrelation")
    plt.title(f"Layer-wise Autocorrelation - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / f"{model_name}_autocorrelation_progression.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    heatmap_data = np.array([layer_stats[layer]["mean"] for layer in layers])
    sns.heatmap(heatmap_data, 
                xticklabels=np.arange(heatmap_data.shape[1]),
                yticklabels=layers,
                cmap='viridis',
                center=0)
    plt.xlabel("Lag")
    plt.ylabel("Layer")
    plt.title(f"Autocorrelation Heatmap - {model_name}")
    plt.savefig(save_dir / f"{model_name}_autocorrelation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_trends(layer_stats: dict) -> dict:
    """Analyze statistical trends in the autocorrelation data"""
    layers = sorted(layer_stats.keys())
    n_lags = len(layer_stats[layers[0]]["mean"])
    
    trends = {
        "correlation_with_depth": {},
        "significance": {},
        "mean_autocorr_by_layer": {}
    }
    
    for lag in range(n_lags):
        layer_means = [layer_stats[layer]["mean"][lag] for layer in layers]
        correlation, p_value = stats.pearsonr(layers, layer_means)
        
        trends["correlation_with_depth"][lag] = correlation
        trends["significance"][lag] = p_value
    
    for layer in layers:
        trends["mean_autocorr_by_layer"][layer] = np.mean(layer_stats[layer]["mean"])
    
    return trends

def process_model_directory(dir_path: Path, results_dir: Path):
    """Process autocorrelation analysis for a single model directory"""
    print(f"Processing {dir_path}")
    
    # Load autocorrelation data
    layer_autocorrs = load_autocorrelations(dir_path)
    if not layer_autocorrs:
        return
    
    # Compute statistics
    layer_stats = compute_layer_statistics(layer_autocorrs)
    
    # Analyze trends
    trends = analyze_trends(layer_stats)
    
    # Save trend analysis
    model_name = dir_path.name.split('_')[-1]
    model_results_dir = results_dir / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_results_dir / "trend_analysis.json", "w") as f:
        json.dump({
            "correlation_with_depth": {str(k): v for k, v in trends["correlation_with_depth"].items()},
            "significance": {str(k): v for k, v in trends["significance"].items()},
            "mean_autocorr_by_layer": {str(k): v for k, v in trends["mean_autocorr_by_layer"].items()}
        }, f, indent=2)
    
    # Create visualizations
    plot_layer_autocorrelations(layer_stats, model_results_dir, model_name)
    
    # Print summary
    print(f"\nAnalysis Summary for {model_name}:")
    print("-" * 40)
    print("Correlation between layer depth and autocorrelation:")
    for lag, corr in trends["correlation_with_depth"].items():
        p_value = trends["significance"][lag]
        if p_value < 0.05:
            print(f"Lag {lag}: r = {corr:.3f} (p < 0.05)")
        else:
            print(f"Lag {lag}: r = {corr:.3f} (n.s.)")
    
    print("\nMean autocorrelation by layer:")
    for layer, mean_autocorr in trends["mean_autocorr_by_layer"].items():
        print(f"Layer {layer}: {mean_autocorr:.3f}")

def main():
    # Load and apply configuration
    overrides = parse_config_overrides()
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
    # Create results directory
    results_dir = config.saved_models_dir / "autocorrelation_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get directories to process
    dirs_to_process = get_dirs_to_process(config.saved_latents_dir, 
                                        overrides.get('model_dirs'),
                                        latents_prefix=True)
    
    # Process each directory
    for dir_path in dirs_to_process:
        process_model_directory(dir_path, results_dir)

if __name__ == "__main__":
    main()