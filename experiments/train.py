import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize
from experiment_config import config
from noise_embedding import NoiseEmbeddingModel
import os

from config_loader import parse_config_overrides, apply_overrides, save_config
from rerandomized_model import 


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
    
    print(dataset_args)
    dataset = load_dataset(**dataset_args,
                          trust_remote_code=True,
                          cache_dir=config.cache_dir)
    
    
    print("Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized = chunk_and_tokenize(dataset, tokenizer, text_key=config.text_key)
    
    tokenized.save_to_disk(str(config.tokenized_dataset_path))

#print number of tokens in dataset
print(tokenized['input_ids'][0].shape)
print(f"Number of tokens in dataset: {len(tokenized) * tokenized['input_ids'][0].shape[0]}")

#restrict to max_tokens
tokenized = tokenized.select(range(min(len(tokenized), config.max_tokens // tokenized['input_ids'][0].shape[0])))

#print number of tokens in dataset
print(f"Number of tokens in dataset: {len(tokenized) * tokenized['input_ids'][0].shape[0]}")

#if fewer than 1 billion warn user
if len(tokenized) * tokenized['input_ids'][0].shape[0] < config.max_tokens:
    print(f"Warning: Less than {config.max_tokens} tokens in dataset")
#shuffle dataset
tokenized = tokenized.shuffle(seed=config.random_seed)

# Load base model
gpt = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    device_map=config.device_map,
    torch_dtype=getattr(torch, config.torch_dtype),
    revision="step0" if config.use_step0 else None
)

if config.rerandomize:
    print(f"Rerandomizing model parameters (embeddings: {config.rerandomize_embeddings})")
    gpt = RerandomizedModel(
        gpt,
        rerandomize_embeddings=config.rerandomize_embeddings,
        seed=config.random_seed
    ).model

# Wrap with noise embedding model if in random control mode
if config.use_random_control:
    print(f"Using random control mode with noise std: {config.noise_std}")
    gpt = NoiseEmbeddingModel(gpt, std=config.noise_std)

# Create save directory
save_dir = config.save_directory
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Models will be saved in: {save_dir}")

# Save config
save_path = save_dir / 'config.json'
save_config(config, save_path)

sae_config = SaeConfig(
    expansion_factor=config.expansion_factor,
    normalize_decoder=config.normalize_decoder,
    num_latents=config.num_latents,
    k=config.k,
    multi_topk=config.multi_topk,
)

cfg = TrainConfig(
    sae_config,
    batch_size=config.batch_size, 
    run_name=str(save_dir)
)

print(f"Batch size: {cfg.batch_size}")
trainer = SaeTrainer(cfg, tokenized, gpt)
trainer.fit()