"""
Larger model configuration for ANE-scale training on WikiText-103.

Target: ~55M parameters — large enough for ANE throughput to dominate
the data-transfer overhead, while still trainable on an RTX 3070 Ti (8 GB).

Memory estimate (BF16 AMP training):
    params: ~55M × 2 bytes = ~110 MB
    grads:  ~110 MB (BF16)
    master weights (FP32): ~220 MB
    optimiser (AdamW): ~440 MB
    activations: ~300 MB (batch_size=128, seq_len=128, BF16)
    total: ~1.2 GB — well within 8 GB VRAM

WikiText-103 scale:
    ~103M tokens, ~805K chunks of 128 tokens
    ~14K batches/epoch at batch_size=64 (2x larger with AMP)
    Approx. 35–45 min/epoch on RTX 3070 Ti with AMP + Flash Attention
"""

from dataclasses import dataclass

from .config import ModelConfig


@dataclass
class ModelConfigLarge(ModelConfig):
    # ── Architecture (overrides) ─────────────────────────────────────────────
    hidden_dim: int = 512         # 256 → 512
    num_layers: int = 12          # 6 → 12
    num_heads: int = 8            # head_dim = 64 (ANE-optimised)
    ffn_dim: int = 2_048          # 1024 → 2048

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: str = "wikitext-103"   # 50× more data than wikitext-2

    # ── Masking schedule ─────────────────────────────────────────────────────
    mask_schedule: str = "cosine"   # smoother masking curve

    # ── Diffusion (overrides) ──────────────────────────────────────────────────
    T: int = 25                   # 100 → 25: 離散トークンでは少ないステップで十分
    mask_loss_weight: float = 5.0 # 全位置ロスでのマスク位置の重み

    # ── Training (overrides) ─────────────────────────────────────────────────
    batch_size: int = 48          # AMP 有効時に VRAM 8 GB 内で安定動作
    learning_rate: float = 5e-4   # プラトー脱出のため引き上げ
    lr_schedule: str = "cosine"   # Warmup + cosine decay
    lr_min: float = 1e-5          # Floor for cosine decay
    num_epochs: int = 30          # Fewer epochs; each is ~70 min on RTX 3070 Ti
    warmup_steps: int = 3_000     # T 削減に合わせて warmup も短縮
    early_stopping_patience: int = 5   # Stop if no improvement for 5 epochs

    # ── Checkpointing (overrides) ────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_wt103"
    save_every_n_epochs: int = 5  # More frequent saves (epochs are expensive)
