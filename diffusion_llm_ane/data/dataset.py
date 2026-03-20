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
import os
import unicodedata
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from data.tokenizer import get_tokenizer
from model.config import ModelConfig

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

def _tokenise_and_chunk(
    texts,
    tokenizer_name: str,
    max_seq_len: int,
    dataset_label: str,
    split: str,
    *,
    normalise_nfkc: bool = False,
) -> torch.Tensor:
    """Tokenise an iterable of texts and split into fixed-length chunks.

    Args:
        texts:           Iterable yielding plain text strings.
        tokenizer_name:  HuggingFace tokenizer ID.
        max_seq_len:     Chunk length in tokens.
        dataset_label:   Label for progress messages (e.g. "wikipedia-ja").
        split:           Data split name (for logging).
        normalise_nfkc:  Apply Unicode NFKC normalisation (recommended for
                         Japanese text — converts fullwidth ASCII to halfwidth,
                         halfwidth katakana to fullwidth, etc.).

    Returns:
        ``(N, max_seq_len)`` int64 tensor of token chunks.
    """
    tokeniser = get_tokenizer(tokenizer_name)

    all_ids: list[int] = []
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if normalise_nfkc:
            text = unicodedata.normalize("NFKC", text)
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
    print(f"  {dataset_label} [{split}]: {len(chunk_list):,} chunks of {L} tokens")
    return torch.tensor(chunk_list, dtype=torch.long)   # (N, L)


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

        return _tokenise_and_chunk(
            (item["text"] for item in raw),
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

        return _tokenise_and_chunk(
            (item["text"] for item in raw),
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
