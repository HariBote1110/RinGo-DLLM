"""
WikiText-2 dataset loader.

Tokenises the raw text, then slices the flat token stream into fixed-length
chunks of `max_seq_len` tokens.  Each chunk is returned as a single training
example.  Padding is deliberately avoided: all examples are exactly 128 tokens
so that ANE sees a constant-shape input throughout training and inference.
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from .tokenizer import get_tokenizer
from ..model.config import ModelConfig


class WikiTextDataset(Dataset):
    """Flat-chunked WikiText-2 dataset."""

    def __init__(self, split: str, config: ModelConfig) -> None:
        """
        Args:
            split:  "train", "validation", or "test"
            config: ModelConfig instance
        """
        tokeniser = get_tokenizer()
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Tokenise everything into a single flat token list
        all_ids: list[int] = []
        for item in raw:
            text: str = item["text"].strip()
            if not text:
                continue
            ids: list[int] = tokeniser.encode(text, add_special_tokens=False)
            all_ids.extend(ids)

        # Chunk into non-overlapping fixed-length blocks
        L = config.max_seq_len
        self.chunks: list[list[int]] = [
            all_ids[i : i + L]
            for i in range(0, len(all_ids) - L + 1, L)
        ]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)


def get_dataloader(split: str, config: ModelConfig) -> DataLoader:
    """Build a DataLoader for the given split."""
    dataset = WikiTextDataset(split, config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True,
        drop_last=True,   # Keeps batch size constant — important for ANE shape consistency
    )
