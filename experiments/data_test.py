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
    tokenized = chunk_and_tokenize(dataset, tokenizer, text_key='text')
    
    tokenized.save_to_disk(str(config.tokenized_dataset_path))


#print number of tokens in dataset
print(tokenized['input_ids'][0].shape)
print(f"Number of tokens in dataset: {len(tokenized) * tokenized['input_ids'][0].shape[0]}")

#restrict to 1 billion tokens
tokenized = tokenized.select(range(min(len(tokenized), config.max_tokens // tokenized['input_ids'][0].shape[0])))

#print number of tokens in dataset
print(f"Number of tokens in dataset: {len(tokenized) * tokenized['input_ids'][0].shape[0]}")