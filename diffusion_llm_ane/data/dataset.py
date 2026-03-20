"""
Dataset loaders for training and evaluation.

Supports:
- WikiText-2  (English, ~2M tokens)
- WikiText-103 (English, ~103M tokens)
- Wikipedia Japanese (~0.9-1.5B tokens)

Tokenises the raw text, then slices the flat token stream into fixed-length
chunks of `max_seq_len` tokens.  Each chunk is returned as a single training
example.  Padding is deliberately avoided: all examples are exactly 128 tokens
so that ANE sees a constant-shape input throughout training and inference.

Optimisation notes:
- Tokenised chunks are cached to a .pt file on first load (subsequent loads
  skip the HuggingFace download + tokenisation entirely).
- Cache paths include a hash of the tokenizer name so that different
  tokenizers never collide.
- pin_memory is disabled on MPS (not supported; silences the warning).
- num_workers=0 on MPS avoids forked-process conflicts with MPS contexts.

Dataset selection:
- Set config.dataset_name = "wikitext-2"     for English small corpus
- Set config.dataset_name = "wikitext-103"   for English large corpus
- Set config.dataset_name = "wikipedia-ja"   for Japanese Wikipedia
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import unicodedata
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from data.tokenizer import get_tokenizer
from model.config import ModelConfig

# Use most available CPU cores for parallel tokenisation, leaving 1 spare
_NUM_PROC = max(1, multiprocessing.cpu_count() - 1)

# Directory for cached tokenised chunks (relative to the project root)
_CACHE_DIR = Path(__file__).parent.parent / ".cache"


def _cache_path(
    split: str,
    max_seq_len: int,
    dataset_name: str,
    tokenizer_name: str,
) -> Path:
    """Build a unique cache file path incorporating the tokenizer identity."""
    slug = dataset_name.replace("-", "")   # e.g. "wikitext103", "wikipediaja"
    # Include tokenizer name in hash so different tokenizers don't collide
    key = hashlib.md5(
        f"{dataset_name}-{split}-{max_seq_len}-{tokenizer_name}".encode(),
    ).hexdigest()[:8]
    return _CACHE_DIR / f"{slug}_{split}_L{max_seq_len}_{key}.pt"


# ── Shared chunking helper ───────────────────────────────────────────────────

def _tokenise_and_chunk_from_hf(
    hf_dataset,
    tokenizer_name: str,
    max_seq_len: int,
    dataset_label: str,
    split: str,
    text_column: str = "text",
    *,
    normalise_nfkc: bool = False,
) -> torch.Tensor:
    """Tokenise a HuggingFace Dataset in parallel and split into fixed-length chunks.

    Uses ``datasets.map(batched=True, num_proc=N)`` for multi-process
    tokenisation — typically 5-10x faster than sequential encode() calls.

    Args:
        hf_dataset:      A HuggingFace ``Dataset`` object.
        tokenizer_name:  HuggingFace tokenizer ID.
        max_seq_len:     Chunk length in tokens.
        dataset_label:   Label for progress messages.
        split:           Data split name (for logging).
        text_column:     Column containing text (default ``"text"``).
        normalise_nfkc:  Apply Unicode NFKC normalisation before tokenising.

    Returns:
        ``(N, max_seq_len)`` int64 tensor of token chunks.
    """
    tokeniser = get_tokenizer(tokenizer_name)
    num_proc = min(_NUM_PROC, len(hf_dataset))
    print(f"  Tokenising {dataset_label} [{split}] with {num_proc} processes …")

    def _tokenise_batch(batch: dict) -> dict:
        texts = batch[text_column]
        if normalise_nfkc:
            texts = [unicodedata.normalize("NFKC", t) for t in texts]
        encoded = tokeniser(
            texts,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        return {"input_ids": encoded["input_ids"]}

    tokenised = hf_dataset.map(
        _tokenise_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=hf_dataset.column_names,
        desc=f"Tokenising {dataset_label}",
    )

    # Memory-efficient two-pass flattening via numpy int32.
    # Python list-of-ints uses ~28 bytes/int → OOM for 1B+ tokens.
    # numpy int32 uses 4 bytes/int → ~5 GB for 1.3B tokens (manageable).

    # Pass 1: count total tokens
    print(f"  Counting tokens …")
    total_tokens: int = 0
    for batch in tokenised.iter(batch_size=50_000):
        for ids in batch["input_ids"]:
            total_tokens += len(ids)

    print(f"  Allocating flat array: {total_tokens:,} tokens "
          f"({total_tokens * 4 / 1e9:.2f} GB as int32) …")

    # Pass 2: fill pre-allocated int32 array
    flat_np = np.empty(total_tokens, dtype=np.int32)
    offset = 0
    for batch in tokenised.iter(batch_size=50_000):
        for ids in batch["input_ids"]:
            n = len(ids)
            if n:
                flat_np[offset : offset + n] = ids
                offset += n

    del tokenised   # release Arrow cache from memory before tensor alloc

    L = max_seq_len
    n_chunks = total_tokens // L
    # Reshape int32 array then cast to int64 tensor (avoids double allocation)
    chunks = torch.from_numpy(
        flat_np[: n_chunks * L].reshape(n_chunks, L)
    ).to(torch.long)
    del flat_np

    print(f"  {dataset_label} [{split}]: {n_chunks:,} chunks of {L} tokens "
          f"({total_tokens:,} tokens total)")
    return chunks


# ── WikiText dataset ─────────────────────────────────────────────────────────

# Map from short dataset_name to HuggingFace config name
_WIKITEXT_HF_CONFIGS: dict[str, str] = {
    "wikitext-2":   "wikitext-2-raw-v1",
    "wikitext-103": "wikitext-103-raw-v1",
}


class WikiTextDataset(Dataset):
    """Flat-chunked WikiText dataset with on-disk token cache."""

    def __init__(self, split: str, config: ModelConfig) -> None:
        dataset_name: str = getattr(config, "dataset_name", "wikitext-2")
        tokenizer_name: str = getattr(config, "tokenizer_name", "bert-base-uncased")
        cache_file = _cache_path(split, config.max_seq_len, dataset_name, tokenizer_name)

        if cache_file.exists():
            print(f"  [cache hit] {cache_file.name}")
            self.chunks = torch.load(cache_file, weights_only=True)
        else:
            self.chunks = self._build_chunks(
                split, config.max_seq_len, dataset_name, tokenizer_name,
            )
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.chunks, cache_file)

    @staticmethod
    def _build_chunks(
        split: str,
        max_seq_len: int,
        dataset_name: str,
        tokenizer_name: str,
    ) -> torch.Tensor:
        hf_config = _WIKITEXT_HF_CONFIGS.get(dataset_name)
        if hf_config is None:
            raise ValueError(
                f"Unknown WikiText dataset_name '{dataset_name}'. "
                f"Choose from: {list(_WIKITEXT_HF_CONFIGS.keys())}"
            )

        print(f"  Downloading / tokenising {dataset_name} [{split}] …")
        raw = load_dataset("wikitext", hf_config, split=split)

        return _tokenise_and_chunk_from_hf(
            raw,
            tokenizer_name=tokenizer_name,
            max_seq_len=max_seq_len,
            dataset_label=dataset_name,
            split=split,
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


# ── Wikipedia Japanese dataset ───────────────────────────────────────────────

class WikipediaJaDataset(Dataset):
    """Japanese Wikipedia with NFKC normalisation and on-disk token cache.

    Uses the ``wikimedia/wikipedia`` dataset on HuggingFace (``20231101.ja``
    config).  Articles are NFKC-normalised to unify fullwidth/halfwidth
    characters before tokenisation.
    """

    def __init__(self, split: str, config: ModelConfig) -> None:
        tokenizer_name: str = getattr(
            config, "tokenizer_name", "tohoku-nlp/bert-base-japanese-v3",
        )
        cache_file = _cache_path(
            split, config.max_seq_len, "wikipedia-ja", tokenizer_name,
        )

        if cache_file.exists():
            print(f"  [cache hit] {cache_file.name}")
            self.chunks = torch.load(cache_file, weights_only=True)
        else:
            self.chunks = self._build_chunks(
                split, config.max_seq_len, tokenizer_name,
            )
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.chunks, cache_file)

    @staticmethod
    def _build_chunks(
        split: str,
        max_seq_len: int,
        tokenizer_name: str,
    ) -> torch.Tensor:
        # Wikipedia does not have a built-in train/validation/test split.
        # Use 99% for training, 1% for validation.
        print(f"  Downloading / tokenising wikipedia-ja [{split}] …")
        raw = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")

        # Deterministic split: use last 1% for validation
        n_total = len(raw)
        n_val = max(1, n_total // 100)
        if split == "train":
            raw = raw.select(range(n_total - n_val))
        elif split in ("validation", "test"):
            raw = raw.select(range(n_total - n_val, n_total))
        else:
            raise ValueError(f"Unknown split: {split!r}")

        return _tokenise_and_chunk_from_hf(
            raw,
            tokenizer_name=tokenizer_name,
            max_seq_len=max_seq_len,
            dataset_label="wikipedia-ja",
            split=split,
            normalise_nfkc=True,
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


# ── Dataset factory ──────────────────────────────────────────────────────────

_DATASET_REGISTRY: dict[str, type[Dataset]] = {
    "wikitext-2":    WikiTextDataset,
    "wikitext-103":  WikiTextDataset,
    "wikipedia-ja":  WikipediaJaDataset,
}


def _resolve_dataset(config: ModelConfig) -> Dataset:
    """Instantiate the right Dataset class based on config.dataset_name."""
    dataset_name = getattr(config, "dataset_name", "wikitext-2")
    cls = _DATASET_REGISTRY.get(dataset_name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Choose from: {list(_DATASET_REGISTRY.keys())}"
        )
    return cls


def get_dataloader(split: str, config: ModelConfig) -> DataLoader:
    """Build a DataLoader for the given split."""
    dataset_cls = _resolve_dataset(config)
    dataset = dataset_cls(split, config)

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
