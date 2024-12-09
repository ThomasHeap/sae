import argparse
from pathlib import Path
import subprocess
import sys
import time
import torch

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_command(cmd, description):
    print(f"\n=== Running {description} ===")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {str(e)}")
        sys.exit(1)
    finally:
        clear_gpu_memory()
        time.sleep(5)

def run_pipeline(api_key, model_name, dataset, expansion_factor, k, flags=None):
    flags = flags or []
    train_split = "train[:15%]"
    eval_split = "train[25%:30%]"
    
    base_params = [
        f"--model_name={model_name}",
        f"--dataset={dataset}",
        f"--expansion_factor={expansion_factor}",
        f"--k={k}"
    ] + flags

    # Training uses different split than other steps
    train_cmd = ["python", "train.py"] + base_params + [f"--dataset_split={train_split}"]
    run_command(train_cmd, "Training")

    # All other steps use eval split
    other_params = base_params + [f"--dataset_split={eval_split}"]
    for cmd, desc in [
        (["python", "save_activations.py"], "Save Activations"),
        (["python", "find_max_activations.py"], "Find Max Activations"),
        (["python", "auto_interp.py"] + [f"--api_key={api_key}", "--offline_explainer"], "Auto Interpretation")
    ]:
        run_command(cmd + other_params, desc)

def main():
    parser = argparse.ArgumentParser(description='Run full SAE pipeline')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--expansion_factor', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--reinit_non_embedding', action='store_true')
    parser.add_argument('--use_step0', action='store_true')
    parser.add_argument('--use_random_control', action='store_true')
    args = parser.parse_args()
    
    flags = []
    if args.reinit_non_embedding:
        flags.append("--reinit_non_embedding")
    if args.use_step0:
        flags.append("--use_step0")
    if args.use_random_control:
        flags.append("--use_random_control")
    
    run_pipeline(args.api_key, args.model_name, args.dataset, 
                args.expansion_factor, args.k, flags)
    
    run_command(
        ["python", "plots.py", f"--dataset={args.dataset.split('/')[-1]}", "--n-features=30"],
        "Generate Plots"
    )

if __name__ == "__main__":
    main()