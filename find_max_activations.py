#based on https://github.com/tim-lawson/mlsae/blob/main/mlsae/analysis/examples.py

import json
import os
import sqlite3
from collections.abc import Generator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from simple_parsing import Serializable, field, parse

from sae import Sae
from sae.data import chunk_and_tokenize

@dataclass
class Config(Serializable):
    model_name: str = "EleutherAI/pythia-70m-deduped"
    """The name of the pretrained transformer model."""

    sae_path: str = "sae-step0/gpt_neox.layers.1"
    """The path to the trained SAE."""

    data_path: str = "wikitext"
    """The path or name of the dataset to use."""

    data_name: str = "wikitext-103-v1"
    """The specific configuration of the dataset."""

    max_length: int = 2048
    """The maximum sequence length for tokenization."""

    batch_size: int = 4
    """The batch size for processing."""

    seed: int = 42
    """The seed for global random state."""

    n_examples: int = 32
    """The number of examples to keep for each latent and layer."""

    n_tokens: int = 4
    """The number of tokens to include either side of the maximum activation."""

    delete_every_n_steps: int = 10
    """The number of steps between deleting examples not in the top n_examples."""

    max_steps: int = 1000
    """The maximum number of steps to process."""

    push_to_hub: bool = False
    """Whether to push the dataset to HuggingFace."""


class Example(NamedTuple):
    latent: int
    layer: int
    token_id: int
    token: str
    act: float
    token_ids: list[int]
    tokens: list[str]
    acts: list[float]

    def serialize(self) -> tuple[int | str | float, ...]:
        return (
            self.latent,
            self.layer,
            self.token_id,
            self.token,
            self.act,
            json.dumps(self.token_ids),
            json.dumps(self.tokens),
            json.dumps(self.acts),
        )

    @staticmethod
    def from_row(row: tuple) -> "Example":
        return Example(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            json.loads(row[5]),
            json.loads(row[6]),
            json.loads(row[7]),
        )

    @staticmethod
    def from_dict(data: dict) -> "Example":
        return Example(
            data["latent"],
            data["layer"],
            data["token_id"],
            data["token"],
            data["act"],
            json.loads(data["token_ids"]),
            json.loads(data["tokens"]),
            json.loads(data["acts"]),
        )


def create_db(database: str | PathLike) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY,
            latent INTEGER,
            layer INTEGER,
            token_id INTEGER,
            token TEXT,
            act REAL,
            token_ids JSON,
            tokens JSON,
            acts JSON
        )
        """,
    )
    conn.commit()
    return conn, cursor


def insert_examples(cursor: sqlite3.Cursor, examples: list[Example]) -> None:
    cursor.executemany(
        """
        INSERT INTO examples (
            latent,
            layer,
            token_id,
            token,
            act,
            token_ids,
            tokens,
            acts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [example.serialize() for example in examples],
    )


def delete_examples(cursor: sqlite3.Cursor, n_examples: int) -> None:
    cursor.execute(
        """
        DELETE FROM examples
        WHERE id IN (
            SELECT id
            FROM (
                SELECT id,
                    ROW_NUMBER() OVER (
                        PARTITION BY latent, layer
                        ORDER BY act DESC
                    ) as rank
                FROM examples
            )
            WHERE rank > ?
        );
        """,
        (n_examples,),
    )


def get_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: Sae,
    batch: torch.Tensor,
    n_tokens: int,
    dead_threshold: float = 1e-3,
    device: torch.device | str = "cpu",
) -> Generator[Example, None, None]:
    batch_tokens = batch.to(device)
    
    # Reshape if necessary
    if batch_tokens.dim() == 1:
        batch_tokens = batch_tokens.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(batch_tokens, output_hidden_states=True)
    
    for layer, hidden_state in enumerate(outputs.hidden_states):
        topk = sae.encode(hidden_state.view(-1, hidden_state.size(-1)))

        flat_tokens = batch_tokens.view(-1)
        batch_acts = topk.top_acts.view(-1, topk.top_acts.size(-1)).half()
        batch_latents = topk.top_indices.view(-1, topk.top_indices.size(-1))

        positions, indices = torch.where(batch_acts > dead_threshold)
        for pos, k in zip(positions, indices):
            latent = batch_latents[pos, k].item()
            token_id = int(flat_tokens[pos].item())
            token = tokenizer.decode(token_id)
            act = batch_acts[pos, k].item()

            start = max(0, pos - n_tokens)
            end = min(flat_tokens.size(0), pos + n_tokens + 1)
            token_ids = flat_tokens[start:end].tolist()
            tokens: list[str] = tokenizer.convert_ids_to_tokens(token_ids)
            acts = batch_acts[start:end, k].tolist()

            yield Example(latent, layer, token_id, token, act, token_ids, tokens, acts)


@torch.no_grad()
def save_examples(config: Config, device: torch.device | str = "cpu") -> None:
    model = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    sae = Sae.load_from_disk(config.sae_path).to(device)

    dataset = load_dataset(config.data_path, config.data_name, split="train")
    tokenized_dataset = chunk_and_tokenize(dataset, tokenizer, max_seq_len=config.max_length)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=config.batch_size, shuffle=True)

    # Create the output directory if it doesn't exist
    os.makedirs("out", exist_ok=True)
    db_path = f"out/{config.model_name.replace('/', '-')}-examples.db"
    
    conn, cursor = create_db(db_path)
    for i, batch in tqdm(enumerate(dataloader), total=config.max_steps):
        examples = list(get_examples(model, tokenizer, sae, batch['input_ids'], config.n_tokens, device=device))
        insert_examples(cursor, examples)
        if i >= config.max_steps:
            break
        if i % config.delete_every_n_steps == 0:
            delete_examples(cursor, config.n_examples)
            conn.commit()
    delete_examples(cursor, config.n_examples)
    conn.commit()

    if config.push_to_hub:
        repo_id = f"{config.model_name}-sae-examples"
        dataset = Dataset.from_sql("SELECT * FROM examples", conn)
        assert isinstance(dataset, Dataset)
        dataset.push_to_hub(repo_id, commit_description=config.dumps_json())

    print(f"Examples saved to {db_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = parse(Config)
    torch.manual_seed(config.seed)
    save_examples(config, device)