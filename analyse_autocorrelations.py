import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from config import config

def load_autocorrelations(latents_dir):
    """
    Load autocorrelation data from all subdirectories in the latents directory.
    Returns a dictionary mapping layer numbers to their autocorrelation data.
    """
    layer_autocorrs = {}
    
    for subdir in latents_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        autocorr_file = subdir / "autocorrelations.json"
        if not autocorr_file.exists():
            continue
            
        with open(autocorr_file) as f:
            data = json.load(f)
            
        # Extract layer number from the directory name
        for module_name in data:
            layer = int(module_name.split('.')[-1])
            if layer not in layer_autocorrs:
                layer_autocorrs[layer] = []
            layer_autocorrs[layer].append(data[module_name])
    
    return layer_autocorrs

def compute_layer_statistics(layer_autocorrs):
    """
    Compute statistics about autocorrelation across layers.
    Returns mean autocorrelation at each lag for each layer.
    """
    layer_stats = {}
    
    for layer, autocorrs in layer_autocorrs.items():
        # Combine autocorrelations from different runs
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

def plot_layer_autocorrelations(layer_stats, save_dir):
    """
    Create visualizations of the autocorrelation patterns.
    """
    # Set up the style
    plt.style.use('seaborn')
    
    # 1. Plot mean autocorrelation at different lags for each layer
    plt.figure(figsize=(12, 8))
    layers = sorted(layer_stats.keys())
    for layer in layers:
        stats = layer_stats[layer]
        lags = np.arange(len(stats["mean"]))
        plt.plot(lags, stats["mean"], label=f"Layer {layer}", marker='o', alpha=0.7)
        plt.fill_between(lags, stats["ci"][0], stats["ci"][1], alpha=0.2)
    
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation by Layer and Lag")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "autocorrelation_by_layer.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot mean autocorrelation across layers for different lags
    plt.figure(figsize=(12, 8))
    lags_to_plot = [1, 2, 5, 10]  # Select specific lags to analyze
    for lag in lags_to_plot:
        layer_means = [layer_stats[layer]["mean"][lag] for layer in layers]
        plt.plot(layers, layer_means, label=f"Lag {lag}", marker='o')
    
    plt.xlabel("Layer")
    plt.ylabel("Autocorrelation")
    plt.title("Layer-wise Autocorrelation for Different Lags")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "autocorrelation_progression.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create heatmap of autocorrelations
    plt.figure(figsize=(12, 8))
    heatmap_data = np.array([layer_stats[layer]["mean"] for layer in layers])
    sns.heatmap(heatmap_data, 
                xticklabels=np.arange(heatmap_data.shape[1]),
                yticklabels=layers,
                cmap='viridis',
                center=0)
    plt.xlabel("Lag")
    plt.ylabel("Layer")
    plt.title("Autocorrelation Heatmap")
    plt.savefig(save_dir / "autocorrelation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_layer_trends(layer_stats):
    """
    Perform statistical analysis of autocorrelation trends across layers.
    """
    layers = sorted(layer_stats.keys())
    n_lags = len(layer_stats[layers[0]]["mean"])
    
    trends = {
        "correlation_with_depth": {},
        "significance": {},
        "mean_autocorr_by_layer": {}
    }
    
    # Analyze correlation between layer depth and autocorrelation for each lag
    for lag in range(n_lags):
        layer_means = [layer_stats[layer]["mean"][lag] for layer in layers]
        correlation, p_value = stats.pearsonr(layers, layer_means)
        
        trends["correlation_with_depth"][lag] = correlation
        trends["significance"][lag] = p_value
    
    # Compute mean autocorrelation across all lags for each layer
    for layer in layers:
        trends["mean_autocorr_by_layer"][layer] = np.mean(layer_stats[layer]["mean"])
    
    return trends

def main():
    # Create results directory
    results_dir = config.saved_models_dir / "autocorrelation_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and analyze data
    print("Loading autocorrelation data...")
    layer_autocorrs = load_autocorrelations(config.saved_latents_dir)
    
    print("Computing layer statistics...")
    layer_stats = compute_layer_statistics(layer_autocorrs)
    
    print("Analyzing layer trends...")
    trends = analyze_layer_trends(layer_stats)
    
    # Save trend analysis
    with open(results_dir / "trend_analysis.json", "w") as f:
        json.dump({
            "correlation_with_depth": {str(k): v for k, v in trends["correlation_with_depth"].items()},
            "significance": {str(k): v for k, v in trends["significance"].items()},
            "mean_autocorr_by_layer": {str(k): v for k, v in trends["mean_autocorr_by_layer"].items()}
        }, f, indent=2)
    
    print("Creating visualizations...")
    plot_layer_autocorrelations(layer_stats, results_dir)
    
    # Print summary of findings
    print("\nAnalysis Summary:")
    print("-----------------")
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

if __name__ == "__main__":
    main()