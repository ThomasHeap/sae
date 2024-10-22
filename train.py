import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize
from experiment_config import config
import os

os.environ['HF_HOME'] = str(config.cache_dir)

def limit_dataset(dataset, max_tokens):
    """
    Limit a tokenized dataset to a maximum number of tokens.
    Expects a dataset with 'input_ids' field.
    """
    total_tokens = 0
    limited_data = []
    for item in dataset:
        tokens = item["input_ids"]
        if total_tokens + len(tokens) > max_tokens:
            break
        limited_data.append(item)
        total_tokens += len(tokens)
    print(f"Selected {len(limited_data)} examples totaling {total_tokens} tokens")
    return limited_data

if config.tokenized_dataset_path.exists():
    print(f"Loading tokenized dataset from {config.tokenized_dataset_path}")
    tokenized = load_from_disk(str(config.tokenized_dataset_path))
else: 
    print(f"Loading dataset {config.dataset_name}")
    dataset = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )
    
    print("Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)
    
    tokenized = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=4
    )
    
    print(f"Saving tokenized dataset to {config.tokenized_dataset_path}")
    tokenized.save_to_disk(str(config.tokenized_dataset_path))

# Limit the dataset to the specified number of tokens
print("Limiting dataset size...")
limited_tokenized = limit_dataset(tokenized, config.max_train_tokens)

print("Chunking sequences...")
chunk_size = config.chunk_size
chunked_dataset = []
for item in limited_tokenized:
    # Split into chunks of chunk_size
    tokens = item["input_ids"]
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) == chunk_size:  # Only keep full chunks
            # Convert to tensor and store as input_ids
            chunked_dataset.append({
                "input_ids": torch.tensor(chunk, dtype=torch.long)
            })

print(f"Final dataset has {len(chunked_dataset)} chunks")

gpt = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    device_map=config.get_device_map(),
    torch_dtype=getattr(torch, config.get_torch_dtype()),
    revision="step0" if config.use_step0 else None
)

if config.reinit_non_embedding:
    print("Loading step0 model for reinitialization...")
    gpt_step0 = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.get_device_map(),
        torch_dtype=getattr(torch, config.get_torch_dtype()),
        revision="step0"
    )
    
    for name, param in gpt.named_parameters():
        if "embed" not in name:
            param.data = gpt_step0.state_dict()[name].data
    
    del gpt_step0

# Use the new method to get the save directory
save_dir = config.get_save_directory()
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Models will be saved in: {save_dir}")

cfg = TrainConfig(
    SaeConfig(gpt.config.hidden_size), 
    batch_size=config.batch_size, 
    run_name=str(save_dir)
)

trainer = SaeTrainer(cfg, chunked_dataset, gpt)
trainer.fit()