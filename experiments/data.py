from transformers import AutoTokenizer
from pathlib import Path

def load_tokenized_data(
    ctx_len: int,
    tokenizer: AutoTokenizer,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    dataset_row: str = "raw_content",
    seed: int = 22,
    cache: str = None,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    """
    from datasets import load_dataset
    from transformer_lens import utils
    print(dataset_repo,dataset_name,dataset_split)
    
    if Path(f"{cache}/tokenized_data/{dataset_name}_{dataset_split}").exists():
        print("Loading tokenized data from cache")
        tokens = utils.TokenizedDataset.load_from_disk(f"{cache}/tokenized_data/{dataset_name}_{dataset_split}")
    else:
        #load dataset from dataset_dir if provided
        if dataset_name is not None:
            data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split, cache_dir=cache)
        else:
            data = load_dataset(dataset_repo, split=dataset_split, cache_dir=cache)
            
            
        data = data.train_test_split(test_size=0.5, seed=seed)
        data = data['test']
        
        #half the data
        data = data.select(range(len(data)//2))
        
        tokens = utils.tokenize_and_concatenate(data, tokenizer, max_length=ctx_len,column_name=dataset_row)

        tokens = tokens.shuffle(seed)["tokens"]
        
        #save tokenized data
        tokens.save_to_disk(f"{cache}/tokenized_data/{dataset_name}_{dataset_split}")

    return tokens