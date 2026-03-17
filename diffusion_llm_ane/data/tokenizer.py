"""
Thin wrapper around HuggingFace BertTokenizerFast.

A single tokeniser instance is cached per process to avoid reloading the
vocabulary on every DataLoader worker initialisation.
"""

from __future__ import annotations

from transformers import BertTokenizerFast

_TOKENISER: BertTokenizerFast | None = None
_MODEL_NAME = "bert-base-uncased"


def get_tokenizer() -> BertTokenizerFast:
    """Return (or lazily initialise) the shared tokeniser instance."""
    global _TOKENISER
    if _TOKENISER is None:
        _TOKENISER = BertTokenizerFast.from_pretrained(_MODEL_NAME)
    return _TOKENISER
