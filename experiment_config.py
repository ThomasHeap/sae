import os
from pathlib import Path

class Config:
    # Paths
    base_dir = Path(__file__).resolve().parent
    cache_dir = base_dir / "cache"
    saved_models_dir = base_dir / "saved_models"
    saved_latents_dir = base_dir / "saved_latents"

    # Environment variables
    os.environ['HF_HOME'] = str(cache_dir)

    # Model settings
    model_name = "EleutherAI/pythia-70m-deduped"
    use_step0 = False
    reinit_non_embedding = False

    # Dataset settings
    dataset_repo = "togethercomputer"
    dataset_name = "togethercomputer/RedPajama-Data-1T-Sample"  # Updated to include full path
    dataset_split = "train"
    tokenized_dataset_path = cache_dir / "togethercomputer" / "RedPajama-Data-1T-Sample"
    max_train_tokens = 1_000_000_000  # 1 billion tokens
    chunk_size = 1024
    
    # Training settings
    batch_size = 2
    run_name = "sae-trained-reinit"

    # Feature extraction settings
    feature_width = 262143
    min_examples = 200
    max_examples = 10000
    n_splits = 5

    # Experiment settings
    n_examples_train = 40
    n_examples_test = 100
    n_quantiles = 10
    example_ctx_len = 32
    n_random = 100
    train_type = "top"
    test_type = "even"

    # Cache settings
    cache_batch_size = 8
    cache_ctx_len = 256  # This will affect the maximum possible lag for autocorrelation
    cache_n_tokens = 10_000_000
    
    # Autocorrelation settings
    max_lag_ratio = 0.5  # Maximum lag will be this ratio * cache_ctx_len

    # Explanation generator settings
    layer_to_explain = 10
    num_latents_to_explain = 10
    num_parallel_latents = 10

    @classmethod
    def get_device_map(cls):
        return {"": "cuda"}

    @classmethod
    def get_torch_dtype(cls):
        return "bfloat16"

    @classmethod
    def get_save_directory(cls):
        token_count = cls.max_train_tokens // 1_000_000  # Convert to millions
        return cls.saved_models_dir / f"{cls.dataset_name.split('/')[-1]}_{token_count}M_{cls.run_name}"

config = Config()