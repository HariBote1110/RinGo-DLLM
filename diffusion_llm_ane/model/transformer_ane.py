"""
ANE-optimised Transformer Encoder using Conv2d (1×1) instead of nn.Linear.

The Apple Neural Engine processes Conv2d operations natively and very efficiently.
By reshaping tensors from (B, L, D) → (B, D, 1, L) and using Conv2d(1, 1) in place
of Linear layers, we keep the entire computation graph on ANE without CPU fallbacks
for the linear-algebra portions.

This follows the approach from Apple's ml-ane-transformers:
    https://github.com/apple/ml-ane-transformers

Key differences from the standard transformer.py:
    - All nn.Linear replaced with nn.Conv2d(kernel_size=1)
    - Internal tensor layout: (B, D, 1, L) — "channels-first" for ANE
    - LayerNorm operates on dim=1 after a permute
    - Attention QKV computed via Conv2d; matmul stays as einsum
    - Public forward() still accepts/returns (B, L, D) for drop-in compatibility

Weight conversion utility `convert_weights_to_ane()` lets you reuse a
checkpoint trained with the standard Transformer, without retraining.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_ane(x: torch.Tensor) -> torch.Tensor:
    """(B, L, D) → (B, D, 1, L)"""
    return x.permute(0, 2, 1).unsqueeze(2)


def _from_ane(x: torch.Tensor) -> torch.Tensor:
    """(B, D, 1, L) → (B, L, D)"""
    return x.squeeze(2).permute(0, 2, 1)


class LayerNormANE(nn.Module):
    """
    LayerNorm that operates on the channel dimension of (B, C, 1, L) tensors.

    Uses nn.GroupNorm(1, …) internally, which CoreML maps to ANE-native ops.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 1, L)
        return self.gn(x)


# ── Multi-Head Self-Attention (ANE) ─────────────────────────────────────────

class MultiHeadSelfAttentionANE(nn.Module):
    """Conv2d-based multi-head self-attention for ANE."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Conv2d(1×1) replaces Linear for QKV projections
        self.q_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, 1, L)  — ANE layout
        Returns:
            (B, D, 1, L)
        """
        B, D, _, L = x.shape
        H = self.num_heads
        Dh = self.head_dim

        # QKV via Conv2d: (B, D, 1, L)
        q = self.q_proj(x).view(B, H, Dh, L)   # (B, H, Dh, L)
        k = self.k_proj(x).view(B, H, Dh, L)
        v = self.v_proj(x).view(B, H, Dh, L)

        # Attention scores: (B, H, L, L) via einsum
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum: (B, H, Dh, L)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.contiguous().view(B, D, 1, L)

        return self.out_proj(out)


# ── Feed-Forward Network (ANE) ──────────────────────────────────────────────

class FeedForwardANE(nn.Module):
    """Conv2d-based FFN with GELU activation."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_dim, ffn_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(ffn_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# ── Transformer Encoder Layer (ANE) ─────────────────────────────────────────

class TransformerEncoderLayerANE(nn.Module):
    """Pre-LN Transformer layer using Conv2d (ANE-optimised)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = LayerNormANE(hidden_dim)
        self.attn = MultiHeadSelfAttentionANE(hidden_dim, num_heads, dropout)
        self.norm2 = LayerNormANE(hidden_dim)
        self.ff = FeedForwardANE(hidden_dim, ffn_dim, dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, 1, L)
        x = x + self.resid_dropout(self.attn(self.norm1(x)))
        x = x + self.resid_dropout(self.ff(self.norm2(x)))
        return x


# ── Full Encoder Stack (ANE) ────────────────────────────────────────────────

class TransformerEncoderANE(nn.Module):
    """Stack of ANE-optimised Transformer layers."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerANE(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = LayerNormANE(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, 1, L)  — ANE layout
        Returns:
            (B, D, 1, L)
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ── Weight conversion: standard → ANE ───────────────────────────────────────

def convert_weights_to_ane(standard_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert a state_dict from the standard Transformer (nn.Linear-based)
    to the ANE Transformer (nn.Conv2d-based).

    nn.Linear weight: (out_features, in_features)
    nn.Conv2d(1×1) weight: (out_channels, in_channels, 1, 1)

    LayerNorm → GroupNorm:
        weight/bias keys renamed and shapes preserved (both are 1-D).
    """
    ane_state: dict[str, torch.Tensor] = OrderedDict()

    for key, tensor in standard_state.items():
        new_key = key

        # ── LayerNorm → GroupNorm renaming ──
        # encoder.layers.N.norm1.weight → encoder.layers.N.norm1.gn.weight
        # encoder.final_norm.weight     → encoder.final_norm.gn.weight
        for ln_name in ("norm1", "norm2", "final_norm"):
            if f"{ln_name}.weight" in key and f"{ln_name}.gn." not in key:
                new_key = key.replace(f"{ln_name}.weight", f"{ln_name}.gn.weight")
                break
            if f"{ln_name}.bias" in key and f"{ln_name}.gn." not in key:
                new_key = key.replace(f"{ln_name}.bias", f"{ln_name}.gn.bias")
                break

        # ── Linear → Conv2d weight reshape ──
        # Matches: q_proj, k_proj, v_proj, out_proj, fc1, fc2
        is_linear_weight = (
            tensor.dim() == 2
            and any(p in key for p in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"))
            and key.endswith(".weight")
        )
        if is_linear_weight:
            # (out, in) → (out, in, 1, 1)
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)

        is_linear_bias = (
            tensor.dim() == 1
            and any(p in key for p in ("out_proj", "fc1", "fc2"))
            and key.endswith(".bias")
        )
        # Bias stays 1-D for Conv2d — no reshape needed

        ane_state[new_key] = tensor

    return ane_state
