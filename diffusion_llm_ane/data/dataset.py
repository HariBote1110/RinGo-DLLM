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

# Tokenisation worker count.
# MeCab/fugashi loads ~500 MB per worker. On memory-constrained WSL2
# environments multiple workers cause OOM crashes, so we default to 1.
_NUM_PROC = 1

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
    mmap_path: Path | None = None,
) -> torch.Tensor | np.ndarray:
    """Tokenise a HuggingFace Dataset in parallel and split into fixed-length chunks.

    Uses ``datasets.map(batched=True, num_proc=N)`` for multi-process
    tokenisation — typically 5-10x faster than sequential encode() calls.

    When ``mmap_path`` is provided, tokens are written directly to a
    memory-mapped file on disk (``mmap_path``), and a metadata file
    (``mmap_path.meta.npy``) is saved alongside it.  This avoids holding
    the full token array in RAM and eliminates the IO spike caused by
    ``torch.save()`` on large arrays.  A ``(N, max_seq_len)`` numpy
    memmap view is returned instead of a torch.Tensor.

    When ``mmap_path`` is None, the data is processed in-memory and
    returned as a ``(N, max_seq_len)`` int32 torch.Tensor.

    Args:
        hf_dataset:      A HuggingFace ``Dataset`` object.
        tokenizer_name:  HuggingFace tokenizer ID.
        max_seq_len:     Chunk length in tokens.
        dataset_label:   Label for progress messages.
        split:           Data split name (for logging).
        text_column:     Column containing text (default ``"text"``).
        normalise_nfkc:  Apply Unicode NFKC normalisation before tokenising.
        mmap_path:       If set, write flat tokens here as a raw int32 memmap.
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

    # Pass 1: count total tokens (cheap — just reads Arrow metadata lengths)
    print(f"  Counting tokens …")
    total_tokens: int = 0
    for batch in tokenised.iter(batch_size=50_000):
        for ids in batch["input_ids"]:
            total_tokens += len(ids)

    L = max_seq_len
    n_chunks = total_tokens // L
    print(f"  {total_tokens:,} tokens → {n_chunks:,} chunks of {L}  "
          f"({total_tokens * 4 / 1e9:.2f} GB as int32)")

    if mmap_path is not None:
        # ── memmap path: write directly to disk, page-by-page ────────────────
        # The OS flushes dirty pages gradually, avoiding a large IO spike.
        # No 6 GB array ever sits entirely in RAM.
        mmap_path.parent.mkdir(parents=True, exist_ok=True)
        flat = np.memmap(mmap_path, dtype=np.int32, mode="w+", shape=(total_tokens,))
        offset = 0
        for batch in tokenised.iter(batch_size=50_000):
            for ids in batch["input_ids"]:
                n = len(ids)
                if n:
                    flat[offset : offset + n] = ids
                    offset += n
        flat.flush()
        del tokenised

        # Save shape metadata so we can reconstruct the view on load
        meta_path = Path(str(mmap_path) + ".meta.npy")
        np.save(meta_path, np.array([n_chunks, L], dtype=np.int64))

        return flat[: n_chunks * L].reshape(n_chunks, L)   # np.memmap view

    else:
        # ── in-memory path (small datasets like WikiText) ─────────────────────
        flat_np = np.empty(total_tokens, dtype=np.int32)
        offset = 0
        for batch in tokenised.iter(batch_size=50_000):
            for ids in batch["input_ids"]:
                n = len(ids)
                if n:
                    flat_np[offset : offset + n] = ids
                    offset += n
        del tokenised

        chunks = torch.from_numpy(flat_np[: n_chunks * L].reshape(n_chunks, L).copy())
        del flat_np
        return chunks   # torch.Tensor int32


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
        # Cache may be int32 (large datasets) or int64 (legacy); normalise here.
        return self.chunks[idx].long()


# ── Wikipedia Japanese dataset ───────────────────────────────────────────────

class WikipediaJaDataset(Dataset):
    """Japanese Wikipedia with NFKC normalisation and on-disk token cache.

    Uses the ``wikimedia/wikipedia`` dataset on HuggingFace (``20231101.ja``
    config).  Articles are NFKC-normalised to unify fullwidth/halfwidth
    characters before tokenisation.

    Cache format
    ------------
    Tokens are stored as a raw int32 memory-mapped file (``*.bin``) to avoid
    the large IO spike that ``torch.save()`` causes when writing several GB at
    once.  The OS flushes dirty pages gradually during the fill, so the SSH
    daemon and other processes remain responsive throughout.

    A tiny companion file (``*.bin.meta.npy``) stores ``[n_chunks, L]`` so the
    memmap can be reshaped on load without scanning the file.
    """

    def __init__(self, split: str, config: ModelConfig) -> None:
        tokenizer_name: str = getattr(
            config, "tokenizer_name", "tohoku-nlp/bert-base-japanese-v3",
        )
        # Derive the memmap path from the same hash key as the .pt path
        pt_path = _cache_path(split, config.max_seq_len, "wikipedia-ja", tokenizer_name)
        mmap_path = pt_path.with_suffix(".bin")
        meta_path = Path(str(mmap_path) + ".meta.npy")

        if mmap_path.exists() and meta_path.exists():
            print(f"  [cache hit] {mmap_path.name}")
            n_chunks, L = np.load(meta_path).tolist()
            n_chunks, L = int(n_chunks), int(L)
            flat = np.memmap(mmap_path, dtype=np.int32, mode="r",
                             shape=(n_chunks * L,))
            self.chunks: np.ndarray = flat.reshape(n_chunks, L)
        else:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.chunks = self._build_chunks(
                split, config.max_seq_len, tokenizer_name, mmap_path,
            )

    # Approximate token count for Wikipedia-ja train split.
    # Used to pre-allocate the memmap file (sparse, so disk usage matches
    # what is actually written, not the reservation).
    _APPROX_TRAIN_TOKENS = 1_700_000_000   # safe upper-bound (~1.58B actual)
    _APPROX_VAL_TOKENS   =    35_000_000   # ~1% of train; 30M actual + headroom

    @staticmethod
    def _build_chunks(
        split: str,
        max_seq_len: int,
        tokenizer_name: str,
        mmap_path: Path,
    ) -> np.ndarray:
        """Tokenise Wikipedia-ja using streaming mode (no Arrow cache).

        ``streaming=True`` reads Parquet shards lazily from the HuggingFace
        Hub (or the local Parquet cache) without converting to Arrow format.
        This avoids the mmap-heavy Arrow files that crash WSL2 when the
        cache is large or partially corrupted.

        Tokens are written directly to a numpy memmap file in 10M-token
        batches, so RAM usage is bounded and the OS flushes pages gradually.
        """
        # Split semantics:
        #   validation → first 1% of articles (≈ 13,756 articles)
        #   train      → everything after
        N_VAL_ARTICLES = 13_756

        print(f"  Streaming wikipedia-ja [{split}] …")
        raw = load_dataset(
            "wikimedia/wikipedia", "20231101.ja",
            split="train",
            streaming=True,
        )
        if split in ("validation", "test"):
            articles = raw.take(N_VAL_ARTICLES)
            approx_tokens = WikipediaJaDataset._APPROX_VAL_TOKENS
        elif split == "train":
            articles = raw.skip(N_VAL_ARTICLES)
            approx_tokens = WikipediaJaDataset._APPROX_TRAIN_TOKENS
        else:
            raise ValueError(f"Unknown split: {split!r}")

        tokeniser = get_tokenizer(tokenizer_name)

        # Pre-allocate a sparse memmap file.  On Linux ext4, only written
        # pages consume actual disk space, so the reservation is cheap.
        mmap_path.parent.mkdir(parents=True, exist_ok=True)
        flat = np.memmap(mmap_path, dtype=np.int32, mode="w+",
                         shape=(approx_tokens,))

        WRITE_EVERY = 10_000_000   # flush to memmap every 10M tokens (~40 MB)
        buffer: list[int] = []
        offset = 0
        n_articles = 0

        for article in articles:
            text = article["text"].strip()
            if not text:
                continue
            text = unicodedata.normalize("NFKC", text)
            ids: list[int] = tokeniser.encode(
                text, add_special_tokens=False, truncation=False,
            )
            buffer.extend(ids)
            if len(buffer) >= WRITE_EVERY:
                n = min(len(buffer), approx_tokens - offset)
                if n <= 0:
                    break   # memmap full; safe to stop (only truncates last chunks)
                flat[offset : offset + n] = buffer[:n]
                offset += n
                buffer = buffer[n:]
            n_articles += 1
            if n_articles % 100_000 == 0:
                print(f"    {n_articles:,} articles, {offset:,} tokens written …")

        # Flush remaining buffer
        if buffer:
            n = min(len(buffer), approx_tokens - offset)
            if n > 0:
                flat[offset : offset + n] = buffer[:n]
                offset += n
        flat.flush()
        del flat

        total_tokens = offset
        L = max_seq_len
        n_chunks = total_tokens // L

        meta_path = Path(str(mmap_path) + ".meta.npy")
        np.save(meta_path, np.array([n_chunks, L], dtype=np.int64))

        print(f"  wikipedia-ja [{split}]: {n_chunks:,} chunks of {L} tokens "
              f"({total_tokens:,} tokens, {n_articles:,} articles)")

        # Re-open read-only and return the shaped view
        flat_r = np.memmap(mmap_path, dtype=np.int32, mode="r",
                           shape=(n_chunks * L,))
        return flat_r.reshape(n_chunks, L)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Copy one row (128 int32 values = 512 bytes) and cast to int64.
        # np.array() copy is needed because memmap slices are not writable.
        return torch.from_numpy(np.array(self.chunks[idx], dtype=np.int64))


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
    # On CUDA (e.g. WSL2), use multiple workers to pre-fetch batches and
    # keep the GPU fed continuously, eliminating utilisation spikes.
    using_mps = (os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is not None
                 or torch.backends.mps.is_available())
    using_cuda = torch.cuda.is_available() and not using_mps
    num_workers = 4 if using_cuda else 0

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=(not using_mps),
        persistent_workers=(num_workers > 0),  # Keep workers alive between epochs
        drop_last=True,   # Keeps batch size constant — important for ANE shape consistency
    )
