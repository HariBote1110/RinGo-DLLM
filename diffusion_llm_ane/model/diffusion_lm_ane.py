"""
ANE-optimised Masked Diffusion Language Model.

Drop-in replacement for DiffusionLM that uses Conv2d-based Transformer
layers internally. The public forward() signature is identical:

    forward(input_ids: (B, L), t: (B,)) → logits: (B, L, V)

Use `from_standard_checkpoint()` to load a checkpoint trained with the
standard DiffusionLM and convert weights automatically (no retraining).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from .config import ModelConfig
from .transformer_ane import (
    TransformerEncoderANE,
    _to_ane,
    _from_ane,
    convert_weights_to_ane,
)


class DiffusionLM_ANE(nn.Module):
    """
    ANE-optimised MDLM with Conv2d-based Transformer.

    Internal layout:
        Embedding outputs (B, L, D) → converted to (B, D, 1, L) for encoder
        → converted back to (B, L, D) for lm_head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token & positional embeddings (still nn.Embedding — these run on CPU)
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id,
        )
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Timestep conditioning (remains as Linear — small, runs on CPU)
        self.time_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )

        # ANE-optimised Transformer encoder
        self.encoder = TransformerEncoderANE(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        # Output head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    # ── Timestep embedding ──────────────────────────────────────────────────

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.config.hidden_dim // 2
        freq = torch.exp(
            -math.log(10_000)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / (half_dim - 1)
        )
        angles = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return self.time_proj(emb)

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token ids
            t:         (B,)   diffusion timestep
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = input_ids.shape

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        # Token + positional embeddings: (B, L, D)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Add timestep conditioning
        t_emb = self._sinusoidal_embedding(t).unsqueeze(1)  # (B, 1, D)
        x = x + t_emb

        # Convert to ANE layout: (B, L, D) → (B, D, 1, L)
        x = _to_ane(x)

        # Transformer encoder (all Conv2d — ANE-native)
        x = self.encoder(x)

        # Convert back: (B, D, 1, L) → (B, L, D)
        x = _from_ane(x)

        # Vocabulary projection
        logits = self.lm_head(x)  # (B, L, V)
        return logits

    # ── Load from standard checkpoint ───────────────────────────────────────

    @classmethod
    def from_standard_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "DiffusionLM_ANE":
        """
        Load a checkpoint trained with the standard DiffusionLM and
        convert the weights to the ANE-optimised format.

        Returns:
            A DiffusionLM_ANE instance with converted weights.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config: ModelConfig = ckpt["config"]
        standard_state = ckpt["model_state_dict"]

        # Build the ANE model
        model = cls(config)

        # Separate encoder weights from the rest
        encoder_prefix = "encoder."
        encoder_state = {
            k[len(encoder_prefix):]: v
            for k, v in standard_state.items()
            if k.startswith(encoder_prefix)
        }
        non_encoder_state = {
            k: v for k, v in standard_state.items()
            if not k.startswith(encoder_prefix)
        }

        # Convert encoder weights: Linear → Conv2d, LayerNorm → GroupNorm
        ane_encoder_state = convert_weights_to_ane(encoder_state)

        # Rebuild full state dict
        full_ane_state = {}
        for k, v in non_encoder_state.items():
            full_ane_state[k] = v
        for k, v in ane_encoder_state.items():
            full_ane_state[f"encoder.{k}"] = v

        model.load_state_dict(full_ane_state, strict=True)
        model.eval()
        return model
