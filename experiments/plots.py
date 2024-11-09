import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Plot model comparisons for a specific dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                      help='Dataset name (e.g., wikitext_wikitext-103-raw-v1 or redpajama-data-1t-sample)')
    parser.add_argument('--base-dir', type=str, default='saved_models',
                      help='Base directory containing model results')
    parser.add_argument('--y-min', type=float, default=None,
                      help='Minimum value for y-axis')
    parser.add_argument('--y-max', type=float, default=None,
                      help='Maximum value for y-axis')
    return parser.parse_args()

def extract_model_info(model_dir: str) -> Tuple[str, str]:
    """Extract size and init strategy from model directory name"""
    parts = model_dir.split('_')
    size = parts[2]  # e.g., '1000M'
    if 'non_embedding' in model_dir:
        init = 'reinit'
    elif 'step0' in model_dir:
        init = 'step0'
    else:
        init = 'trained'
    return size, init

def calculate_metrics(data: List[Dict]) -> Tuple[float, float]:
    """Calculate balanced accuracy and right fraction from prediction data"""
    if not data:
        return 0.0, 0.0
    
    # Calculate overall accuracy
    right_predictions = sum(1 for item in data if item['correct'])
    right_fraction = right_predictions / len(data)
    
    # Calculate balanced accuracy
    true_positives = sum(1 for item in data if item['ground_truth'] and item['prediction'] == 1)
    true_negatives = sum(1 for item in data if not item['ground_truth'] and item['prediction'] == 0)
    total_positives = sum(1 for item in data if item['ground_truth'])
    total_negatives = sum(1 for item in data if not item['ground_truth'])
    
    if total_positives == 0 or total_negatives == 0:
        balanced_acc = right_fraction
    else:
        sensitivity = true_positives / total_positives
        specificity = true_negatives / total_negatives
        balanced_acc = (sensitivity + specificity) / 2
    
    return balanced_acc, right_fraction

def load_model_results(model_dir: Path) -> Dict[str, Dict]:
    """Load and process results for a single model directory"""
    results = {
        'balanced_acc': [], 'balanced_acc_std': [],
        'right_frac': [], 'right_frac_std': [],
        'layers': []
    }
    
    explanations_dir = model_dir / "explanations"
    if not explanations_dir.exists():
        print(f"No explanations directory found in {model_dir}")
        return None
    
    # Get all layer directories
    layer_dirs = sorted(explanations_dir.glob('layer_*'), 
                       key=lambda x: int(x.name.split('_')[1]))
    
    for layer_dir in layer_dirs:
        layer_num = int(layer_dir.name.split('_')[1])
        
        # Find all feature results files
        feature_files = layer_dir.glob('*.txt')
        
        #only files with "score" in the name
        feature_files = [f for f in feature_files if "score" in f.name]
        
        layer_balanced_accs = []
        layer_right_fracs = []
        
        for feature_file in feature_files:
            try:
                with open(feature_file) as f:
                    data = json.load(f)
                    balanced_acc, right_frac = calculate_metrics(data)
                    layer_balanced_accs.append(balanced_acc)
                    layer_right_fracs.append(right_frac)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {feature_file}: {e}")
                continue
        
        if layer_balanced_accs and layer_right_fracs:
            results['layers'].append(layer_num)
            results['balanced_acc'].append(np.mean(layer_balanced_accs))
            results['balanced_acc_std'].append(np.std(layer_balanced_accs))
            results['right_frac'].append(np.mean(layer_right_fracs))
            results['right_frac_std'].append(np.std(layer_right_fracs))
    
    return results if results['layers'] else None

def plot_dataset_comparisons(base_dir: Path, dataset_name: str, y_min: float = None, y_max: float = None):
    """Create comparison plots for models of a specific dataset"""
    model_dirs = [d for d in base_dir.iterdir() 
                 if d.is_dir() and d.name.startswith(dataset_name)]
    
    if not model_dirs:
        print(f"No models found for dataset {dataset_name}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    valid_layers = [0, 1, 2, 3, 4]
    
    for i, model_dir in enumerate(model_dirs):
        print(f"\nProcessing model: {model_dir.name}")
        results = load_model_results(model_dir)
        if not results:
            print(f"No valid results found for {model_dir.name}")
            continue
            
        size, init = extract_model_info(model_dir.name)
        label = f"{size} {init}"
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Convert to numpy arrays for easier manipulation
        layers = np.array(results['layers'])
        
        # Plot balanced accuracy with error bars
        balanced_acc = np.array(results['balanced_acc'])
        balanced_acc_std = np.array(results['balanced_acc_std'])
        ax1.plot(layers, balanced_acc, color=color, marker=marker, 
                label=label, linestyle='-', markersize=8, linewidth=2)
        ax1.fill_between(layers, 
                        balanced_acc - balanced_acc_std,
                        balanced_acc + balanced_acc_std,
                        color=color, alpha=0.2)
        
        # Plot right fraction with error bars
        right_frac = np.array(results['right_frac'])
        right_frac_std = np.array(results['right_frac_std'])
        ax2.plot(layers, right_frac, color=color, marker=marker,
                label=label, linestyle='-', markersize=8, linewidth=2)
        ax2.fill_between(layers,
                        right_frac - right_frac_std,
                        right_frac + right_frac_std,
                        color=color, alpha=0.2)
        
        print(f"Balanced accuracy: {balanced_acc}")
        print(f"Right fraction: {right_frac}")
    
    # Configure plots
    for ax, title, ylabel in [(ax1, 'Balanced Accuracy', 'Balanced Accuracy'),
                             (ax2, 'Right Predictions', 'Fraction of Right Predictions')]:
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{dataset_name}\n{title}', fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(frameon=True, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_facecolor('#f8f8f8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set x-axis to show only valid layers
        ax.set_xticks(valid_layers)
        ax.set_xlim(-0.2, 4.2)  # Add small padding on both sides
        
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if y_max is not None:
            ax.set_ylim(top=y_max)
    
    plt.tight_layout()
    
    plots_dir = base_dir / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / f"{dataset_name}_model_comparisons.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nPlot saved as {save_path}")

def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Looking for models in: {base_dir}")
    
    plot_dataset_comparisons(base_dir, args.dataset, args.y_min, args.y_max)

if __name__ == "__main__":
    main()