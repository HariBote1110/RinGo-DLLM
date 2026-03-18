"""
WikiText dataset loader (WikiText-2 or WikiText-103).

Tokenises the raw text, then slices the flat token stream into fixed-length
chunks of `max_seq_len` tokens.  Each chunk is returned as a single training
example.  Padding is deliberately avoided: all examples are exactly 128 tokens
so that ANE sees a constant-shape input throughout training and inference.

Optimisation notes:
- Tokenised chunks are cached to a .pt file on first load (subsequent loads
  skip the HuggingFace download + tokenisation entirely).
- pin_memory is disabled on MPS (not supported; silences the warning).
- num_workers=0 on MPS avoids forked-process conflicts with MPS contexts.

Dataset selection:
- Set config.dataset_name = "wikitext-2"   for the small corpus (~2 M tokens)
- Set config.dataset_name = "wikitext-103" for the large corpus (~103 M tokens)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from data.tokenizer import get_tokenizer
from model.config import ModelConfig

# Directory for cached tokenised chunks (relative to the project root)
_CACHE_DIR = Path(__file__).parent.parent / ".cache"

# Map from short dataset_name to HuggingFace config name
_HF_CONFIGS: dict[str, str] = {
    "wikitext-2":   "wikitext-2-raw-v1",
    "wikitext-103": "wikitext-103-raw-v1",
}


def _cache_path(split: str, max_seq_len: int, dataset_name: str) -> Path:
    slug = dataset_name.replace("-", "")   # e.g. "wikitext103"
    key = hashlib.md5(f"{dataset_name}-{split}-{max_seq_len}".encode()).hexdigest()[:8]
    return _CACHE_DIR / f"{slug}_{split}_L{max_seq_len}_{key}.pt"


class WikiTextDataset(Dataset):
    """Flat-chunked WikiText dataset with on-disk token cache."""

    def __init__(self, split: str, config: ModelConfig) -> None:
        """
        Args:
            split:  "train", "validation", or "test"
            config: ModelConfig instance (uses config.dataset_name to select corpus)
        """
        dataset_name: str = getattr(config, "dataset_name", "wikitext-2")
        cache_file = _cache_path(split, config.max_seq_len, dataset_name)

        if cache_file.exists():
            print(f"  [cache hit] {cache_file.name}")
            self.chunks = torch.load(cache_file, weights_only=True)
        else:
            self.chunks = self._build_chunks(split, config.max_seq_len, dataset_name)
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.chunks, cache_file)

    @staticmethod
    def _build_chunks(split: str, max_seq_len: int, dataset_name: str) -> torch.Tensor:
        """Tokenise and chunk the corpus; returns (N, L) int64 tensor."""
        hf_config = _HF_CONFIGS.get(dataset_name)
        if hf_config is None:
            raise ValueError(
                f"Unknown dataset_name '{dataset_name}'. "
                f"Choose from: {list(_HF_CONFIGS.keys())}"
            )

        print(f"  Downloading / tokenising {dataset_name} [{split}] …")
        tokeniser = get_tokenizer()
        raw = load_dataset("wikitext", hf_config, split=split)

        all_ids: list[int] = []
        for item in raw:
            text: str = item["text"].strip()
            if not text:
                continue
            # Suppress the seq-length warning from the base tokeniser
            ids: list[int] = tokeniser.encode(
                text,
                add_special_tokens=False,
                truncation=False,
                max_length=None,
            )
            all_ids.extend(ids)

        L = max_seq_len
        chunk_list = [
            all_ids[i : i + L]
            for i in range(0, len(all_ids) - L + 1, L)
        ]
        print(f"  {dataset_name} [{split}]: {len(chunk_list):,} chunks of {L} tokens")
        return torch.tensor(chunk_list, dtype=torch.long)   # (N, L)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


def get_dataloader(split: str, config: ModelConfig) -> DataLoader:
    """Build a DataLoader for the given split."""
    dataset = WikiTextDataset(split, config)

    # MPS does not support pin_memory; spawning worker processes can also
    # cause issues with the MPS device context, so use num_workers=0.
    using_mps = (os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is not None
                 or torch.backends.mps.is_available())

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=(not using_mps),
        drop_last=True,   # Keeps batch size constant — important for ANE shape consistency
    )
