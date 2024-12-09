from nnsight import LanguageModel
from sae_auto_interp.autoencoders import load_eai_autoencoders

from huggingface_hub import snapshot_download

from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_tokenized_data

snapshot_download(
                "EleutherAI/sae-pythia-70m-32k",
                allow_patterns=None
            )

# Load the model
model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="cuda", dispatch=True,torch_dtype="float16")

print(model)
# Load the autoencoders, the function returns a dictionary of the submodules with the autoencoders and the edited model.
# it takes as arguments the model, the layers to load the autoencoders into,
# the average L0 sparsity per layer, the size of the autoencoders and the type of autoencoders (residuals or MLPs).

submodule_dict,model = load_eai_autoencoders(
            model,
            ae_layers=[0],
            weight_dir="/user/work/cp20141/repos/sae/experiments/saved_models/pythia-70m-deduped_64_k32/rpj-v2-sample_1000M_trained",
            module="res"
        )



# There is a default cache config that can also be modified when using a "production" script.
cfg = CacheConfig(
    dataset_repo="EleutherAI/rpj-v2-sample",
    dataset_split="train[:1%]",
    batch_size=8,
    ctx_len=256,
    n_tokens=10000,
    n_splits=5,
)



tokens = load_tokenized_data(
        ctx_len=cfg.ctx_len,
        tokenizer=model.tokenizer,
        dataset_repo=cfg.dataset_repo,
        dataset_split=cfg.dataset_split,
)
# Tokens should have the shape (n_batches,ctx_len)



cache = FeatureCache(
    model,
    submodule_dict,
    batch_size = cfg.batch_size,
)

cache.run(cfg.n_tokens, tokens)