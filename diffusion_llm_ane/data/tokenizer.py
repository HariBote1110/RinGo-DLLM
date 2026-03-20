"""
Thin wrapper around HuggingFace tokenizers.

Supports multiple tokenizer backends (BERT English, BERT Japanese, etc.).
A cached instance is kept per tokenizer name so that repeated calls with the
same name skip re-loading the vocabulary.
"""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase

_TOKENISERS: dict[str, PreTrainedTokenizerBase] = {}

# Default for backwards compatibility with existing English checkpoints
_DEFAULT_TOKENIZER = "bert-base-uncased"


def get_tokenizer(
    tokenizer_name: str = _DEFAULT_TOKENIZER,
) -> PreTrainedTokenizerBase:
    """Return (or lazily initialise) a shared tokeniser instance.

    Args:
        tokenizer_name: HuggingFace model ID or local path.
            Defaults to ``"bert-base-uncased"`` for backwards compatibility.
    """
    if tokenizer_name not in _TOKENISERS:
        _TOKENISERS[tokenizer_name] = AutoTokenizer.from_pretrained(
            tokenizer_name,
        )
    return _TOKENISERS[tokenizer_name]
