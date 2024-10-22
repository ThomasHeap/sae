from datasets import load_dataset, IterableDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import AutoTokenizer
from typing import Optional, Iterator
import torch
from pathlib import Path
import numpy as np

class StreamingTokenizer:
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.current_chunk = []
        
    def process_text(self, text: str) -> Iterator[torch.Tensor]:
        # Tokenize without truncation first
        tokens = self.tokenizer(text, truncation=False, return_tensors=None)['input_ids']
        
        # Extend current chunk
        self.current_chunk.extend(tokens)
        
        # Yield complete chunks
        while len(self.current_chunk) >= self.max_length:
            chunk = self.current_chunk[:self.max_length]
            self.current_chunk = self.current_chunk[self.max_length:]
            yield torch.tensor(chunk)
            
    def flush(self) -> Optional[torch.Tensor]:
        # Return any remaining tokens if they meet minimum length requirement
        if len(self.current_chunk) > self.max_length // 2:
            chunk = self.current_chunk
            self.current_chunk = []
            return torch.tensor(chunk)
        self.current_chunk = []
        return None

class StreamingDataset(TorchIterableDataset):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        split: str = "train",
        max_length: int = 2048,
        cache_dir: Optional[Path] = None,
        streaming: bool = True
    ):
        super().__init__()
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.streaming_tokenizer = StreamingTokenizer(self.tokenizer, max_length)
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = None
        else:
            per_worker = int(np.ceil(len(self.dataset) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
        
        iterator = iter(self.dataset)
        for i, item in enumerate(iterator):
            if i < iter_start:
                continue
            if iter_end is not None and i >= iter_end:
                break
                
            for tokens in self.streaming_tokenizer.process_text(item['text']):
                yield {'input_ids': tokens}
                
        # Flush any remaining tokens
        final_chunk = self.streaming_tokenizer.flush()
        if final_chunk is not None:
            yield {'input_ids': final_chunk}

class StreamingActivationSaver:
    def __init__(
        self,
        model,
        save_dir: Path,
        batch_size: int = 8,
        max_length: int = 2048,
        buffer_size: int = 1000
    ):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.buffer = []
        self.current_file = 0
        
    def process_batch(self, batch):
        with torch.no_grad():
            outputs = self.model(**batch, output_hidden_states=True)
            # Store activations from each layer
            for layer_idx, hidden_state in enumerate(outputs.hidden_states):
                self.buffer.append({
                    'layer': layer_idx,
                    'activations': hidden_state.cpu().numpy()
                })
                
        if len(self.buffer) >= self.buffer_size:
            self._save_buffer()
            
    def _save_buffer(self):
        if not self.buffer:
            return
            
        filename = self.save_dir / f"activations_{self.current_file}.npz"
        np.savez_compressed(
            filename,
            **{f"batch_{i}": item for i, item in enumerate(self.buffer)}
        )
        self.buffer = []
        self.current_file += 1
        
    def process_dataset(self, dataset: StreamingDataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        for batch in dataloader:
            self.process_batch(batch)
            
        # Save any remaining items in buffer
        self._save_buffer()