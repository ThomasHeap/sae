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
    dataset = "EleutherAI/rpj-v2-sample"  # Full path format
    dataset_name = None  # Optional configuration name
    dataset_split = "train"
    max_tokens = 1_000_000_000  # 1 billion tokens

    # Training settings
    batch_size = 1

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
    test_type = "quantiles"

    # Cache settings
    cache_batch_size = 8
    cache_ctx_len = 256
    cache_n_tokens = 10_000_000

    # Autocorrelation settings
    max_lag_ratio = 0.5

    # Explanation generator settings
    num_latents_to_explain = 10
    num_parallel_latents = 10
    
    random_seed = 42

    @property
    def device_map(self):
        """Get the device mapping configuration"""
        return {"": "cuda"}

    @property
    def torch_dtype(self):
        """Get the torch dtype to use"""
        return "bfloat16"
    
    @property
    def save_directory(self):
        """Get the save directory using the automatically generated run name"""
        return self.saved_models_dir / self.run_name
    
    @property
    def dataset_short_name(self):
        """Get the dataset name without organization prefix"""
        # Split on '/' and get the last part
        base_name = self.dataset.split('/')[-1].lower()
        # If there's a config, append it
        if self.dataset_name:
            base_name = f"{base_name}_{self.dataset_name}"
        return base_name

    @property
    def tokenized_dataset_path(self):
        """Automatically generate path for tokenized dataset"""
        sanitized_name = self.dataset_short_name.replace('-', '_')
        return self.cache_dir / "tokenized" / f"{sanitized_name}_{self.dataset_split}"

    @property
    def run_name(self):
        """Automatically generate run name based on settings"""
        
        if self.reinit_non_embedding:
            init_strategy = "non_embedding_random"
        elif self.use_step0:
            init_strategy = "step0"
        else:
            init_strategy = "trained"
        
        token_count = self.max_tokens // 1_000_000
        return f"{self.dataset_short_name}_{token_count}M_{init_strategy}"

    def get_dataset_args(self):
        """Get the correct arguments for loading the dataset"""
        if self.dataset_name is not None:
            return {
                "path": self.dataset,
                "name": self.dataset_name,
                "split": self.dataset_split,
            }
        else:
            return {
                "path": self.dataset,
                "split": self.dataset_split,
            }

config = Config()