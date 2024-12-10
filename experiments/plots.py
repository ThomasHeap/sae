import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import random
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description='Plot model comparisons with multiple metrics')
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

def calculate_metrics_with_counts(data: List[Dict]) -> Dict[str, float]:
    """Calculate metrics and raw counts from prediction data"""
    if not data:
        return {
            'balanced_acc': 0.0, 'accuracy': 0.0, 
            'tp_rate': 0.0, 'tn_rate': 0.0,
            'precision': 0.0, 'f1_score': 0.0,
            'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
            'total_p': 0, 'total_n': 0
        }
    
    # Calculate counts
    tp = sum(1 for item in data if item['ground_truth'] and item['prediction'] == 1)
    tn = sum(1 for item in data if not item['ground_truth'] and item['prediction'] == 0)
    fp = sum(1 for item in data if not item['ground_truth'] and item['prediction'] == 1)
    fn = sum(1 for item in data if item['ground_truth'] and item['prediction'] == 0)
    
    total_p = sum(1 for item in data if item['ground_truth'])
    total_n = sum(1 for item in data if not item['ground_truth'])
    
    # Calculate rates
    tp_rate = tp / total_p if total_p > 0 else 0
    tn_rate = tn / total_n if total_n > 0 else 0
    accuracy = (tp + tn) / len(data)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    f1_score = 2 * (precision * tp_rate) / (precision + tp_rate) if (precision + tp_rate) > 0 else 0
    balanced_acc = (tp_rate + tn_rate) / 2
    
    return {
        'balanced_acc': balanced_acc,
        'accuracy': accuracy,
        'tp_rate': tp_rate,
        'tn_rate': tn_rate,
        'precision': precision,
        'f1_score': f1_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total_p': total_p,
        'total_n': total_n
    }

def save_statistics_csv(model_stats: Dict, n_features: int, save_path: Path):
    """Save detailed statistics for each model and layer to a CSV file"""
    # Prepare data for DataFrame
    rows = []
    for model_name, layer_stats in model_stats.items():
        for layer in range(5):  # Assuming 5 layers
            if layer in layer_stats:
                stats = layer_stats[layer]
                row = {
                    'Model': model_name,
                    'Layer': layer,
                    'True Positives': f"{stats['counts']['avg_tp']:.1f}",
                    'True Negatives': f"{stats['counts']['avg_tn']:.1f}",
                    'False Positives': f"{stats['counts']['avg_fp']:.1f}",
                    'False Negatives': f"{stats['counts']['avg_fn']:.1f}",
                    'Total Positives': f"{stats['counts']['total_p']:.1f}",
                    'Total Negatives': f"{stats['counts']['total_n']:.1f}",
                    'Accuracy': f"{stats['metrics']['accuracy']:.3f}",
                    'Balanced Accuracy': f"{stats['metrics']['balanced_acc']:.3f}",
                    'TP Rate': f"{stats['metrics']['tp_rate']:.3f}",
                    'TN Rate': f"{stats['metrics']['tn_rate']:.3f}",
                    'Precision': f"{stats['metrics']['precision']:.3f}",
                    'F1 Score': f"{stats['metrics']['f1_score']:.3f}"
                }
                rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Statistics saved to {save_path}")

def bootstrap_roc(y_true: np.ndarray, y_prob: np.ndarray, 
                 n_bootstraps: int = 100) -> Dict[str, np.ndarray]:
    """Calculate ROC curve with confidence intervals using bootstrapping"""
    n_samples = len(y_true)
    
    # Arrays to store bootstrap curves
    tprs = []
    fprs = []
    aucs = []
    base_fpr = np.linspace(0, 1, 101)  # Common FPR grid for interpolation
    
    # Calculate bootstrapped ROC curves
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true_boot, y_prob_boot)
        
        # Interpolate TPR to common FPR grid
        if len(fpr) < 2:
            continue
            
        interp_tpr = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # Force through (0,0)
        
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    
    # Calculate mean ROC and confidence intervals
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std_tprs = tprs.std(axis=0)
    
    # Calculate 95% confidence intervals
    tprs_upper = np.minimum(mean_tprs + 1.96 * std_tprs, 1)
    tprs_lower = np.maximum(mean_tprs - 1.96 * std_tprs, 0)
    
    # Calculate mean AUC and confidence interval
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    auc_ci = stats.norm.interval(0.95, loc=mean_auc, scale=std_auc)
    
    return {
        'mean_fpr': base_fpr,
        'mean_tpr': mean_tprs,
        'tpr_upper': tprs_upper,
        'tpr_lower': tprs_lower,
        'auc': mean_auc,
        'auc_ci': auc_ci
    }

def calculate_roc_data(data: List[Dict]) -> Dict[str, np.ndarray]:
    """Calculate ROC curve data with confidence intervals"""
    if not data:
        return {
            'mean_fpr': np.array([0, 1]),
            'mean_tpr': np.array([0, 1]),
            'tpr_upper': np.array([0, 1]),
            'tpr_lower': np.array([0, 1]),
            'auc': 0.5,
            'auc_ci': (0.5, 0.5)
        }
    
    # Extract ground truth and probabilities
    y_true = np.array([item['ground_truth'] for item in data])
    y_prob = np.array([item['probability'] for item in data])
    
    # Calculate bootstrapped ROC data
    return bootstrap_roc(y_true, y_prob)

def plot_roc_curves(base_dir: Path, n_features: int, 
                    seed: int = None, scorer_type: str = 'detection'):
    """Create ROC curve plots with confidence bands"""
    model_dirs = [d for d in base_dir.iterdir() 
                 if d.is_dir() and not (d.name.startswith("comparison"))]
    
    if not model_dirs:
        print(f"No models found for {base_dir}")
        return
    
    # Create subplot grid for each layer
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    valid_layers = [0, 1, 2, 3, 4]
    
    # Store AUC scores for CSV
    auc_scores = []
    
    # Plot ROC curve for each layer
    for layer in valid_layers:
        ax = axes[layer]
        
        for i, model_dir in enumerate(model_dirs):
            size, init = extract_model_info(model_dir.name)
            label = f"{size} {init}"
            if model_dir.name == "random_noise":
                label = "Random noise baseline"
            color = colors[i % len(colors)]
            
            # Get scores for this layer
            explanation_dir = Path(model_dir) / "explanations" / f"layer_{layer}" / scorer_type
            if not explanation_dir.exists():
                continue
            
            # Collect all data for this layer
            all_data = []
            for score_file in explanation_dir.glob("*_score.txt"):
                with open(score_file, 'r') as f:
                    try:
                        data = json.load(f)
                        all_data.extend(data)
                    except json.JSONDecodeError:
                        continue
            
            if all_data:
                roc_data = calculate_roc_data(all_data)
                
                # Plot mean ROC curve
                ax.plot(roc_data['mean_fpr'], roc_data['mean_tpr'], color=color,
                       label=f'{label}\nAUC = {roc_data["auc"]:.3f} ({roc_data["auc_ci"][0]:.3f}-{roc_data["auc_ci"][1]:.3f})',
                       lw=2)
                
                # Plot confidence bands
                ax.fill_between(roc_data['mean_fpr'], 
                              roc_data['tpr_lower'],
                              roc_data['tpr_upper'],
                              color=color, alpha=0.2)
                
                # Store AUC score and confidence interval
                auc_scores.append({
                    'Model': label,
                    'Layer': layer,
                    'AUC': roc_data['auc'],
                    'AUC_CI_Lower': roc_data['auc_ci'][0],
                    'AUC_CI_Upper': roc_data['auc_ci'][1]
                })
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Layer {layer} ROC Curves')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right', fontsize=8)
    
    # Remove empty subplot if we have one
    if len(valid_layers) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle(f'ROC Curves ({scorer_type})\n{n_features} features per layer',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save ROC plots and AUC scores
    plots_dir = base_dir / "comparison_plots" / scorer_type / "roc"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plots
    roc_path = plots_dir / f"roc_curves_{n_features}features.pdf"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(plots_dir / f"roc_curves_{n_features}features.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save AUC scores to CSV
    auc_df = pd.DataFrame(auc_scores)
    auc_path = plots_dir / f"auc_scores_{n_features}features.csv"
    auc_df.to_csv(auc_path, index=False)
    
    plt.close()
    print(f"ROC curves saved as {roc_path}")
    print(f"AUC scores saved as {auc_path}")
    
def plot_dataset_comparisons(base_dir: Path, n_features: int, 
                           y_min: float = None, y_max: float = None, seed: int = None,
                           scorer_type: str = 'detection'):
    """Create comparison plots and save statistics to CSV for a specific scorer type"""
    model_dirs = [d for d in base_dir.iterdir() 
                 if d.is_dir() and not (d.name.startswith("comparison"))]
    
    if not model_dirs:
        print(f"No models found for {base_dir}")
        return
    
    # Create a 3x2 subplot grid for metrics
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    # Define metrics to plot with new labels
    metrics_to_plot = [
        ('balanced_acc', 'Balanced Accuracy\n(TP Rate + TN Rate)/2'),
        ('accuracy', 'Accuracy\n(TP + TN)/(P + N)'),
        ('tp_rate', 'True Positive Rate\nTP/P'),
        ('tn_rate', 'True Negative Rate\nTN/N'),
        ('precision', 'Precision\nTP/(TP + FP)'),
        ('f1_score', 'F1 Score\n2*(Precision*TPR)/(Precision+TPR)')
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    valid_layers = [0, 1, 2, 3, 4]
    
    # Dictionary to store statistics
    all_model_stats = {}
    
    for i, model_dir in enumerate(model_dirs):
        print(f"\nProcessing model: {model_dir.name}")
        size, init = extract_model_info(model_dir.name)
        label = f"{size} {init}"
        if model_dir.name == "random_noise":
            label = "Random noise baseline"
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Process each layer
        metrics_by_layer = {}
        for layer in valid_layers:
            # Update path to include scorer type subfolder
            explanation_dir = Path(model_dir) / "explanations" / f"layer_{layer}" / scorer_type
            if not explanation_dir.exists():
                print(f"Directory not found: {explanation_dir}")
                continue
                
            all_metrics = []
            for score_file in explanation_dir.glob("*_score.txt"):
                with open(score_file, 'r') as f:
                    try:
                        data = json.load(f)
                        metrics = calculate_metrics_with_counts(data)
                        all_metrics.append(metrics)
                    except json.JSONDecodeError:
                        continue
            
            if all_metrics:
                # Average the metrics for this layer
                layer_avg = {
                    k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]
                    if isinstance(all_metrics[0][k], (int, float))
                }
                layer_std = {
                    k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]
                    if isinstance(all_metrics[0][k], (int, float))
                }
                
                metrics_by_layer[layer] = {
                    'metrics': layer_avg,
                    'std': layer_std,
                    'counts': {
                        'avg_tp': np.mean([m['tp'] for m in all_metrics]),
                        'avg_tn': np.mean([m['tn'] for m in all_metrics]),
                        'avg_fp': np.mean([m['fp'] for m in all_metrics]),
                        'avg_fn': np.mean([m['fn'] for m in all_metrics]),
                        'total_p': np.mean([m['total_p'] for m in all_metrics]),
                        'total_n': np.mean([m['total_n'] for m in all_metrics])
                    }
                }
        
        # Store statistics
        all_model_stats[label] = metrics_by_layer
        
        # Plot metrics
        for (metric, metric_label), ax in zip(metrics_to_plot, axes):
            layers = []
            values = []
            stds = []
            
            for layer in valid_layers:
                if layer in metrics_by_layer:
                    layers.append(layer)
                    values.append(metrics_by_layer[layer]['metrics'][metric])
                    stds.append(metrics_by_layer[layer]['std'][metric])
            
            if layers:
                line = ax.plot(layers, values, color=color, marker=marker,
                             label=label, linestyle='-', markersize=8, linewidth=2)
                ax.fill_between(layers,
                              np.array(values) - np.array(stds),
                              np.array(values) + np.array(stds),
                              color=color, alpha=0.2)
    
    # Configure plots
    feature_str = f"\n{n_features} features per layer"
    for ax, (metric, metric_label) in zip(axes, metrics_to_plot):
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'({scorer_type})\n{metric_label}{feature_str}',
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
    
    # Create scorer-specific output directory
    plots_dir = base_dir / "comparison_plots" / scorer_type
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics plots
    metrics_path = plots_dir / f"model_comparisons_{n_features}features.pdf"
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(plots_dir / f"model_comparisons_{n_features}features.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save statistics to CSV
    stats_path = plots_dir / f"statistics_{n_features}features.csv"
    save_statistics_csv(all_model_stats, n_features, stats_path)
    
    plt.close('all')
    print(f"\nPlots saved as {metrics_path}")
    print(f"Statistics saved as {stats_path}")

def main():
    args = parse_args()
    base_dir = Path(args.base_dir) 
    
    print(f"Looking for models in: {base_dir}")
    print(f"Sampling {args.n_features} features per layer")
    
    # Process each scorer type
    for scorer_type in ['detection', 'fuzz']:
        print(f"\nProcessing {scorer_type} scores...")
        # Generate regular plots
        plot_dataset_comparisons(base_dir, args.n_features, 
                               args.y_min, args.y_max, args.seed,
                               scorer_type=scorer_type)
        # Generate ROC curves
        plot_roc_curves(base_dir, args.n_features,
                       args.seed, scorer_type=scorer_type)

if __name__ == "__main__":
    main()