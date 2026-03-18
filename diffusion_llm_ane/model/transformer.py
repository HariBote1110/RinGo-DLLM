"""
Transformer Encoder building blocks.

Design notes for ANE compatibility:
- head_dim fixed at 64 (ANE-optimised tile size)
- Pre-LayerNorm (more stable; CoreML handles LayerNorm natively)
- No dynamic shapes — all tensors are fixed at (B, 128, D)

Efficiency notes:
- Flash Attention via F.scaled_dot_product_attention (PyTorch 2.0+)
  Fused kernel: faster + O(L) memory instead of O(L²)
  Falls back to manual attention on older PyTorch / CPU
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Flash Attention is available in PyTorch >= 2.0
_FLASH_ATTN = hasattr(F, "scaled_dot_product_attention")


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with Flash Attention (PyTorch 2.0+ SDPA)."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads   # 64 for ANE optimisation
        self.scale = math.sqrt(self.head_dim)
        self.dropout_p = dropout

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # Dropout is handled inside SDPA; keep for fallback path
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, num_heads, L, head_dim)

        if _FLASH_ATTN:
            # Flash Attention: fused CUDA/MPS kernel — faster & O(L) VRAM
            dropout_p = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            # Fallback: manual attention (older PyTorch / CPU)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with pre-LayerNorm (Pre-LN)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, ffn_dim, dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.norm1(x)))
        x = x + self.resid_dropout(self.ff(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerEncoderLayer blocks."""

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
                TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
