"""
Masked Diffusion Language Model (MDLM-style).

Forward (noising) process:
    x_0  ──(mask at rate t/T)──►  x_t

Reverse (denoising) process:
    x_t  ──(model predicts x_0)──►  x_{t-1}  ──► … ──► x_0

The model is conditioned on the diffusion timestep t via sinusoidal
embeddings projected into the hidden dimension.
"""

import math

import torch
import torch.nn as nn

from .config import ModelConfig
from .transformer import TransformerEncoder


# ── Noise utilities ────────────────────────────────────────────────────────────

def apply_mask(
    x: torch.Tensor,
    mask_rate: torch.Tensor,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the forward diffusion process by randomly masking tokens.

    Args:
        x:             (B, L) original token ids
        mask_rate:     (B,)   fraction of tokens to mask for each sample
        mask_token_id: id of the [MASK] token

    Returns:
        x_t:     (B, L) masked sequence
        is_mask: (B, L) boolean mask — True at positions that were masked
    """
    B, L = x.shape
    rand = torch.rand(B, L, device=x.device)
    is_mask = rand < mask_rate.unsqueeze(1)   # (B, L)

    x_t = x.clone()
    x_t[is_mask] = mask_token_id
    return x_t, is_mask


# ── Model ─────────────────────────────────────────────────────────────────────

class DiffusionLM(nn.Module):
    """
    Transformer-based MDLM.

    Given a (partially) masked sequence x_t and the diffusion step t,
    predict the original token at every masked position.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding table (vocab_size × hidden_dim)
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_dim,
            padding_idx=config.pad_token_id,
        )

        # Learnable positional embedding (fixed length for ANE)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Timestep conditioning: scalar t → hidden_dim
        #   sinusoidal encoding → 2-layer MLP → hidden_dim
        self.time_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )

        # Transformer encoder
        use_ckpt = getattr(config, "gradient_checkpointing", False)
        self.encoder = TransformerEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_checkpoint=use_ckpt,
        )

        # Output head: hidden_dim → vocab_size
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    # ── Timestep embedding ────────────────────────────────────────────────────

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal timestep embedding identical to DDPM / transformer positional encoding.

        Args:
            t: (B,) integer timesteps in [1, T]
        Returns:
            emb: (B, hidden_dim)
        """
        half_dim = self.config.hidden_dim // 2
        freq = torch.exp(
            -math.log(10_000)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / (half_dim - 1)
        )
        # outer product: (B, half_dim)
        angles = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (B, hidden_dim)
        return self.time_proj(emb)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token ids — may contain [MASK] tokens
            t:         (B,)   diffusion timestep (integer, 1 … T)
        Returns:
            logits:    (B, L, vocab_size)
        """
        B, L = input_ids.shape

        # Positional indices — fixed range, no dynamic shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        # Token + positional embeddings
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)  # (B, L, D)

        # Add timestep conditioning (broadcast across all sequence positions)
        t_emb = self._sinusoidal_embedding(t).unsqueeze(1)   # (B, 1, D)
        x = x + t_emb

        # Transformer encoder
        x = self.encoder(x)                   # (B, L, D)

        # Vocabulary projection
        logits = self.lm_head(x)              # (B, L, vocab_size)
        return logits

    # ── Inference helper ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_tokens(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Run the model and return the most-likely token at each position.

        Args:
            input_ids: (B, L)
            t:         (B,)
        Returns:
            (B, L) predicted token ids
        """
        logits = self.forward(input_ids, t)          # (B, L, V)
        return logits.argmax(dim=-1)
