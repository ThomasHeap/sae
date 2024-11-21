import json
from pathlib import Path
import argparse
from typing import Optional, Dict, Any
import torch

def convert_value(value: Any) -> Any:
    """Convert string values to appropriate types"""
    if isinstance(value, str):
        # Convert string "True"/"False" to boolean
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        
        # Convert numeric strings to int/float
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            # If not a number, and contains Path, convert to Path object
            if "Path" in value:
                return Path(value.split("'")[1])
            return value
    return value

def load_saved_config(config_path: Path) -> Any:
    """Load a saved configuration from a JSON file"""
    with open(config_path) as f:
        config_dict = json.load(f)
    
    class LoadedConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                # Convert the value to appropriate type
                converted_value = convert_value(value)
                setattr(self, key, converted_value)
            
            # Add getter methods that might be needed
            def get_device_map(self):
                return {"": "cuda"}
            
            def get_torch_dtype(self):
                return getattr(torch, self.torch_dtype)
    
    return LoadedConfig(config_dict)

def save_config(config: Any, save_path: Path):
    """Save configuration to a JSON file"""
    config_dict = {}
    for k, v in vars(config).items():
        if not k.startswith('__') and not callable(v):
            if isinstance(v, Path):
                config_dict[k] = str(v)
            else:
                config_dict[k] = v
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def parse_config_overrides() -> Dict[str, Any]:
    """Parse command line arguments for config overrides"""
    parser = argparse.ArgumentParser()
    
    # Training-specific arguments
    parser.add_argument('--model_name', type=str, help='Model name/path')
    parser.add_argument('--dataset', type=str, help='Dataset path')
    parser.add_argument('--dataset_name', type=str, help='Dataset configuration name')
    parser.add_argument('--max_tokens', type=int, help='Maximum tokens to process')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--model_dirs', nargs='*', help='Specific model directories to process')
    parser.add_argument('--reinit_non_embedding', action='store_true', help='Reinitialize non-embedding layers')
    parser.add_argument('--use_step0', action='store_true', help='Use step0 model')
    parser.add_argument('--text_key', type=str, help='Random seed')
    
    
    args, _ = parser.parse_known_args()
    return {k: v for k, v in vars(args).items() if v is not None}

def apply_overrides(config: Any, overrides: Dict[str, Any]):
    """Apply configuration overrides to a config object"""
    for key, value in overrides.items():
        if hasattr(config, key):
            # Convert the override value if it's a string
            if isinstance(value, str):
                value = convert_value(value)
            setattr(config, key, value)