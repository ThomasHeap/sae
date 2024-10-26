import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize
from experiment_config import config
import os

from config_loader import parse_config_overrides, apply_overrides, save_config

# Apply any command line overrides to the config
overrides = parse_config_overrides()
apply_overrides(config, overrides)

os.environ['HF_HOME'] = str(config.cache_dir)


if config.tokenized_dataset_path.exists():
    print(f"Loading tokenized dataset from {config.tokenized_dataset_path}")
    tokenized = load_from_disk(str(config.tokenized_dataset_path))
else: 
    print(f"Loading dataset {config.dataset}")
    dataset_args = config.get_dataset_args()
    print(f"Dataset args: {dataset_args}")
    print(config.tokenized_dataset_path)
    dataset = load_dataset(**dataset_args,
                           trust_remote_code=True,
                           cache_dir=config.cache_dir)
    
    print("Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized = chunk_and_tokenize(dataset, tokenizer, max_tokens=config.max_tokens)
    
    tokenized.save_to_disk(str(config.tokenized_dataset_path))



gpt = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    device_map=config.device_map,
    torch_dtype=getattr(torch, config.torch_dtype),
    revision="step0" if config.use_step0 else None
)

if config.reinit_non_embedding:
    print("Loading step0 model for reinitialization...")
    gpt_step0 = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map,
        torch_dtype=getattr(torch, config.torch_dtype),
        revision="step0"
    )
    
    for name, param in gpt.named_parameters():
        if "embed" not in name:
            param.data = gpt_step0.state_dict()[name].data
    
    del gpt_step0

# Use the new method to get the save directory
save_dir = config.save_directory
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Models will be saved in: {save_dir}")

#save config to save_dir
save_path = save_dir / 'config.json'
save_config(config, save_path)

cfg = TrainConfig(
    SaeConfig(gpt.config.hidden_size), 
    batch_size=config.batch_size, 
    run_name=str(save_dir)
)
print(cfg.batch_size)
print(type(cfg.batch_size))
trainer = SaeTrainer(cfg, tokenized, gpt)
trainer.fit()