from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import json
import numpy as np
from datasets import load_dataset
from experiment_config import config
from config_loader import parse_config_overrides, apply_overrides
import os

class SaeReconstructionModel(nn.Module):
    """Wrapper that replaces intermediate activations with SAE reconstructions"""
    
    def __init__(self, model: nn.Module, save_dir: Path, layers: List[int]):
        super().__init__()
        self.model = model
        self.config = model.config
        self.layers = layers
        self.handles = []
        
        # Load SAEs for specified layers
        self.saes = {}
        for layer in layers:
            # Use the same directory structure as in train.py
            weights_path = save_dir / f"sae_{layer}.pt"
            
            if not weights_path.exists():
                raise FileNotFoundError(f"SAE weights not found at {weights_path}")
                
            state_dict = torch.load(weights_path)
            self.saes[layer] = {
                'encoder': nn.Parameter(state_dict['encoder_weight'].t()),
                'decoder': nn.Parameter(state_dict['decoder_weight'])
            }
        
        self._setup_hooks()
        
    def _setup_hooks(self):
        """Set up forward hooks to replace activations with reconstructions"""
        def make_hook(layer_idx):
            def hook(module, input_tensor, output_tensor):
                # Get SAE for this layer
                sae = self.saes[layer_idx]
                
                # Reshape for SAE processing
                orig_shape = output_tensor.shape
                flat = output_tensor.reshape(-1, output_tensor.shape[-1])
                
                # Encode and decode through SAE
                encoded = torch.nn.functional.linear(flat, sae['encoder'])
                reconstructed = torch.nn.functional.linear(encoded, sae['decoder'])
                
                # Reshape back to original dimensions
                return reconstructed.reshape(orig_shape)
                
            return hook
        
        # Register hooks for each layer
        for layer_idx in self.layers:
            layer = self.model.gpt_neox.layers[layer_idx]
            handle = layer.register_forward_hook(make_hook(layer_idx))
            self.handles.append(handle)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """Forward pass using SAE reconstructions"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
            
    def remove_hooks(self):
        """Clean up hooks when done"""
        for handle in self.handles:
            handle.remove()
        self.handles = []

def evaluate_model(model, dataset, tokenizer, max_samples=1000, batch_size=4):
    """Evaluate model performance on dataset"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, min(len(dataset), max_samples), batch_size)):
            batch_data = dataset[i:i+batch_size]
            
            # Tokenize input
            inputs = tokenizer(
                batch_data['text'], 
                return_tensors='pt',
                padding=True,
                truncation=True, 
                max_length=config.cache_ctx_len
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get model output
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Update metrics
            total_loss += loss.item() * len(batch_data)
            total_samples += len(batch_data)
    
    return {
        'average_loss': total_loss / total_samples,
        'perplexity': torch.exp(torch.tensor(total_loss / total_samples)).item()
    }

def analyze_reconstruction_impact():
    """Analyze impact of SAE reconstructions on model performance"""
    # Load and apply configuration overrides
    overrides = parse_config_overrides()
    apply_overrides(config, overrides)
    
    os.environ['HF_HOME'] = str(config.cache_dir)
    
    print(f"\nAnalyzing reconstruction impact for model: {config.model_name}")
    print(f"Using dataset: {config.dataset}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map,
        torch_dtype=getattr(torch, config.torch_dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load evaluation dataset
    dataset = load_dataset(
        **config.get_dataset_args(),
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )
    
    # Get baseline performance
    print("\nEvaluating baseline model...")
    baseline_metrics = evaluate_model(
        model, 
        dataset['test'], 
        tokenizer,
        batch_size=config.batch_size
    )
    
    results = {
        'model_name': config.model_name,
        'dataset': config.dataset,
        'dataset_name': config.dataset_name,
        'baseline': baseline_metrics,
        'reconstructions': {}
    }
    
    # Test different layer combinations
    layer_configs = [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
    ]
    
    for layers in layer_configs:
        print(f"\nTesting reconstruction in layers: {layers}")
        
        # Create model with SAE reconstructions
        wrapped_model = SaeReconstructionModel(
            model,
            config.save_directory,
            layers
        )
        
        # Evaluate
        recon_metrics = evaluate_model(
            wrapped_model, 
            dataset['test'], 
            tokenizer,
            batch_size=config.batch_size
        )
        
        # Calculate performance degradation
        degradation = {
            'perplexity_increase': recon_metrics['perplexity'] - baseline_metrics['perplexity'],
            'perplexity_increase_pct': (recon_metrics['perplexity'] - baseline_metrics['perplexity']) 
                                     / baseline_metrics['perplexity'] * 100,
            'loss_increase': recon_metrics['average_loss'] - baseline_metrics['average_loss'],
            'loss_increase_pct': (recon_metrics['average_loss'] - baseline_metrics['average_loss'])
                                / baseline_metrics['average_loss'] * 100
        }
        
        results['reconstructions'][f"layers_{','.join(map(str, layers))}"] = {
            'metrics': recon_metrics,
            'degradation': degradation
        }
        
        # Clean up hooks
        wrapped_model.remove_hooks()
    
    # Save results in the same directory structure as other analyses
    results_dir = config.latents_directory / "reconstruction_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / f"reconstruction_impact.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    return results

if __name__ == "__main__":
    analyze_reconstruction_impact()