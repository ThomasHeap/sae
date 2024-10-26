from pathlib import Path
from typing import List, Optional, Any, Union
from argparse import Namespace
from config_loader import load_saved_config

def namespace_to_dict(namespace: Namespace) -> dict:
    """Convert argparse Namespace to dictionary"""
    return vars(namespace) if namespace else {}

def get_dirs_to_process(base_dir: Path, model_dirs: Optional[Union[List[str], Namespace]] = None, 
                       latents_prefix: bool = False) -> List[Path]:
    """
    Get list of directories to process based on command line arguments
    
    Args:
        base_dir: Base directory to look for model directories
        model_dirs: Optional list of model directories or Namespace containing model_dirs
        latents_prefix: If True, prepend 'latents_' to directory names
    """
    # Handle both Namespace and direct list input
    if isinstance(model_dirs, Namespace):
        model_dirs = getattr(model_dirs, 'model_dirs', None)
    
    if model_dirs:
        if latents_prefix:
            return [base_dir / f"latents_{d}" for d in model_dirs]
        return [base_dir / d for d in model_dirs]
    
    # Process all directories if no specific ones provided
    return [d for d in base_dir.iterdir() if d.is_dir()]

def load_model_config(model_dir: Path, current_config: Any, current_overrides: Union[dict, Namespace]):
    """
    Load configuration from a model directory while preserving overrides
    
    Args:
        model_dir: Directory containing the model and its config
        current_config: Current configuration object
        current_overrides: Dictionary or Namespace of command line overrides
    """
    # Convert Namespace to dict if needed
    if isinstance(current_overrides, Namespace):
        current_overrides = namespace_to_dict(current_overrides)
    
    config_path = model_dir / "config.json"
    if config_path.exists():
        print(f"Loading saved configuration from {config_path}")
        saved_config = load_saved_config(config_path)
        # Apply saved config while preserving overrides
        for key, value in vars(saved_config).items():
            if key not in current_overrides:
                setattr(current_config, key, value)
    return current_config