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
    rerandomize = False  # Whether to use rerandomization
    rerandomize_embeddings = False  # Whether to rerandomize embeddings too
    
    # Random training settings
    use_random_control = False  # Flag to toggle random control
    noise_std = 1.0  # Standard deviation for random noise in random mode

    # Dataset settings
    dataset = "EleutherAI/rpj-v2-sample"  # Full path format
    dataset_name = None  # Optional configuration name
    dataset_split = "train"
    text_key = "raw_content"
    max_tokens = 100_000_000  # 100 million tokens

    # Training settings
    batch_size = 4
    expansion_factor = 64
    normalize_decoder = True
    num_latents = 0
    k = 32
    multi_topk = False
    
    # Feature extraction settings
    min_examples = 200
    max_examples = 10000
    n_splits = 5

    # Experiment settings
    n_examples_train = 40
    n_examples_test = 100
    n_quantiles = 10
    example_ctx_len = 32
    n_random = 50
    train_type = "random"
    test_type = "quantiles"

    # Cache settings
    cache_batch_size = 8
    cache_ctx_len = 256
    cache_n_tokens = 10_000_000

    # Autocorrelation settings
    max_lag_ratio = 0.5

    # Explanation generator settings
    num_parallel_latents = 5
    offline_explainer = False
    
    random_seed = 42
    
    
    use_embedding_sae = False  

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
        if self.use_random_control:
            return self.saved_models_dir / "random"
        return self.saved_models_dir / self.run_name
    
    @property
    def latents_directory(self):
        """Get the directory for saving latent features"""
        if self.use_random_control:
            return self.saved_latents_dir / "random_noise"
        return self.saved_latents_dir / f"latents_{self.run_name}"
    
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
    def model_short_name(self):
        """Get the model name without organization prefix"""
        return self.model_name.split('/')[-1].lower()

    @property
    def tokenized_dataset_path(self):
        """Automatically generate path for tokenized dataset"""
        sanitized_name = self.dataset_short_name.replace('-', '_')
        return self.cache_dir / "tokenized" / f"{sanitized_name}_{self.dataset_split}"

    @property
    def run_name(self):
        """Automatically generate run name based on settings"""
        if self.rerandomize:
            init_strategy = "rerandomised"
            if self.rerandomize_embeddings:
                init_strategy += "_embeddings"
        elif self.use_step0:
            init_strategy = "step0"
        else:
            init_strategy = "trained"
        
        token_count = self.max_tokens // 1_000_000
        return f"{self.model_short_name}_{self.expansion_factor}_k{self.k}/{self.dataset_short_name}_{token_count}M_{init_strategy}"

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