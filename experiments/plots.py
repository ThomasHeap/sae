import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Plot model comparisons with multiple metrics')
    parser.add_argument('--dataset', type=str, required=True, 
                      help='Dataset name (e.g., wikitext_wikitext-103-raw-v1 or redpajama-data-1t-sample)')
    parser.add_argument('--base-dir', type=str, default='saved_latents',
                      help='Base directory containing model results')
    parser.add_argument('--n-features', type=int, default=30,
                      help='Number of features to sample per layer')
    parser.add_argument('--y-min', type=float, default=None,
                      help='Minimum value for y-axis')
    parser.add_argument('--y-max', type=float, default=None,
                      help='Maximum value for y-axis')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for feature sampling')
    return parser.parse_args()

def extract_model_info(model_dir: str) -> Tuple[str, str]:
    """Extract size and init strategy from model directory name"""
    if model_dir == "random_noise":
        return "Random", "baseline"
    
    parts = model_dir.split('_')
    size = parts[2]  # e.g., '1000M'
    if 'non_embedding' in model_dir:
        init = 'reinit'
    elif 'step0' in model_dir:
        init = 'step0'
    else:
        init = 'trained'
    return size, init

def calculate_metrics(data: List[Dict]) -> Dict[str, float]:
    """Calculate various metrics from prediction data"""
    if not data:
        return {metric: 0.0 for metric in ['balanced_acc', 'accuracy', 'sensitivity', 
                                         'specificity', 'precision', 'f1_score']}
    
    # Calculate basic counts
    true_positives = sum(1 for item in data if item['ground_truth'] and item['prediction'] == 1)
    true_negatives = sum(1 for item in data if not item['ground_truth'] and item['prediction'] == 0)
    false_positives = sum(1 for item in data if not item['ground_truth'] and item['prediction'] == 1)
    false_negatives = sum(1 for item in data if item['ground_truth'] and item['prediction'] == 0)
    
    total_positives = sum(1 for item in data if item['ground_truth'])
    total_negatives = sum(1 for item in data if not item['ground_truth'])
    
    # Calculate metrics
    sensitivity = true_positives / total_positives if total_positives > 0 else 0
    specificity = true_negatives / total_negatives if total_negatives > 0 else 0
    accuracy = (true_positives + true_negatives) / len(data)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Calculate balanced accuracy
    balanced_acc = (sensitivity + specificity) / 2
    
    return {
        'balanced_acc': balanced_acc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score
    }

def get_available_features(layer_dir: Path, layer_num: int) -> List[int]:
    """Get list of all available feature numbers in a layer directory"""
    # Construct the module path pattern
    module_pattern = f".gpt_neox.layers.{layer_num}"
    
    # Look for files matching the pattern
    feature_files = list(layer_dir.glob(f"{module_pattern}_feature*_score.txt"))
    feature_numbers = []
    
    for file in feature_files:
        try:
            # Extract feature number from filename (.gpt_neox.layers.X_featureY_score.txt)
            feature_part = file.stem.split('_')[-2]  # Get 'featureX' part
            feature_num = int(feature_part.replace('feature', ''))
            feature_numbers.append(feature_num)
        except (IndexError, ValueError) as e:
            print(f"Error parsing feature number from {file}: {e}")
            continue
    
    return feature_numbers

def sample_features(layer_dir: Path, layer_num: int, n_features: int, seed: int = None) -> List[int]:
    """Randomly sample n_features from available features"""
    available_features = get_available_features(layer_dir, layer_num)
    if not available_features:
        print(f"No features found in {layer_dir} for layer {layer_num}")
        return []
        
    if seed is not None:
        random.seed(seed)
    
    # Sample features or take all if fewer than requested
    if len(available_features) <= n_features:
        print(f"Found {len(available_features)} features for layer {layer_num}, using all")
        return available_features
    
    print(f"Sampling {n_features} features from {len(available_features)} available for layer {layer_num}")
    return random.sample(available_features, n_features)

def load_model_results(model_dir: Path, n_features: int, seed: int = None) -> Dict[str, Dict]:
    """Load and process results for a single model directory with random feature sampling"""
    metrics = ['balanced_acc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    results = {
        metric: {'values': [], 'std': []} for metric in metrics
    }
    results.update({'layers': [], 'feature_count': None, 'sampled_features': defaultdict(list)})
    
    explanations_dir = model_dir / "explanations"
    if not explanations_dir.exists():
        print(f"No explanations directory found in {model_dir}")
        return None
    
    layer_dirs = sorted(explanations_dir.glob('layer_*'), 
                       key=lambda x: int(x.name.split('_')[1]))
    
    for layer_dir in layer_dirs:
        layer_num = int(layer_dir.name.split('_')[1])
        
        # Sample features for this layer
        sampled_features = sample_features(layer_dir, layer_num, n_features, seed)
        if not sampled_features:
            continue
            
        results['sampled_features'][layer_num] = sampled_features
        
        layer_metrics = defaultdict(list)
        
        # Process only sampled features
        for feature_num in sampled_features:
            # Construct the correct score file path
            module_pattern = f".gpt_neox.layers.{layer_num}"
            score_file = layer_dir / f"{module_pattern}_feature{feature_num}_score.txt"
            
            if not score_file.exists():
                print(f"Score file not found: {score_file}")
                continue
                
            try:
                with open(score_file) as f:
                    data = json.load(f)
                    feature_metrics = calculate_metrics(data)
                    for metric, value in feature_metrics.items():
                        layer_metrics[metric].append(value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {score_file}: {e}")
                continue
        
        if layer_metrics:
            results['layers'].append(layer_num)
            for metric in metrics:
                metric_values = layer_metrics[metric]
                if metric_values:  # Only compute if we have values
                    results[metric]['values'].append(np.mean(metric_values))
                    results[metric]['std'].append(np.std(metric_values))
                else:
                    print(f"No valid values for {metric} in layer {layer_num}")
    
    # Store the actual number of features sampled
    results['feature_count'] = n_features
    
    return results if results['layers'] else None

def plot_dataset_comparisons(base_dir: Path, dataset_name: str, n_features: int, 
                           y_min: float = None, y_max: float = None, seed: int = None):
    """Create comparison plots for models of a specific dataset"""
    model_dirs = [d for d in base_dir.iterdir() 
                 if d.is_dir() and (d.name.startswith("latents_" + dataset_name) or d.name == "random_noise")]
    
    if not model_dirs:
        print(f"No models found for dataset {dataset_name}")
        return
    
    # Create a 3x2 subplot grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    # Define metrics to plot and their labels
    metrics_to_plot = [
        ('balanced_acc', 'Balanced Accuracy'),
        ('accuracy', 'Accuracy'),
        ('sensitivity', 'Sensitivity (Recall)'),
        ('specificity', 'Specificity'),
        ('precision', 'Precision'),
        ('f1_score', 'F1 Score')
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    valid_layers = [0, 1, 2, 3, 4]
    
    for i, model_dir in enumerate(model_dirs):
        print(f"\nProcessing model: {model_dir.name}")
        results = load_model_results(model_dir, n_features, seed)
        if not results:
            print(f"No valid results found for {model_dir.name}")
            continue
            
        size, init = extract_model_info(model_dir.name)
        label = f"{size} {init}"
        if model_dir.name == "random_noise_injected_normal_model":
            label = "Random model, normal SAE"
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        layers = np.array(results['layers'])
        
        # Plot each metric
        for (metric, metric_label), ax in zip(metrics_to_plot, axes):
            if metric in results and 'values' in results[metric] and results[metric]['values']:
                values = np.array(results[metric]['values'])
                std_dev = np.array(results[metric]['std'])
                
                ax.plot(layers, values, color=color, marker=marker,
                       label=label, linestyle='-', markersize=8, linewidth=2)
                ax.fill_between(layers,
                              values - std_dev,
                              values + std_dev,
                              color=color, alpha=0.2)
            else:
                print(f"No data for metric {metric} in model {model_dir.name}")
    
    # Configure each subplot
    feature_str = f"\n{n_features} features per layer"
    for ax, (metric, metric_label) in zip(axes, metrics_to_plot):
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'{dataset_name}\n{metric_label}{feature_str}',
                    fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(frameon=True, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_facecolor('#f8f8f8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(valid_layers)
        ax.set_xlim(-0.2, 4.2)
        
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if y_max is not None:
            ax.set_ylim(top=y_max)
    
    plt.tight_layout()
    
    # Save plots
    plots_dir = base_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / f"{dataset_name}_model_comparisons_{n_features}features.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(plots_dir / f"{dataset_name}_model_comparisons_{n_features}features.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save sampled feature information
    info_path = plots_dir / f"{dataset_name}_sampled_features_{n_features}.json"
    sampled_features_info = {}
    for model_dir in model_dirs:
        results = load_model_results(model_dir, n_features, seed)
        if results and 'sampled_features' in results:
            sampled_features_info[model_dir.name] = {
                str(layer): features 
                for layer, features in results['sampled_features'].items()
            }
    
    with open(info_path, 'w') as f:
        json.dump(sampled_features_info, f, indent=2)
    
    plt.close()
    print(f"\nPlots saved as {save_path}")
    print(f"Sampled features information saved as {info_path}")

def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Looking for models in: {base_dir}")
    print(f"Sampling {args.n_features} features per layer")
    
    plot_dataset_comparisons(base_dir, args.dataset, args.n_features, 
                           args.y_min, args.y_max, args.seed)

if __name__ == "__main__":
    main()