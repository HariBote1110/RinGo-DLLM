"""
Larger model configuration for ANE-scale training on WikiText-103.

Target: ~55M parameters — large enough for ANE throughput to dominate
the data-transfer overhead, while still trainable on an RTX 3070 Ti (8 GB).

Memory estimate (fp32 training):
    params: ~55M × 4 bytes = ~220 MB
    grads:  ~220 MB
    optimiser (AdamW): ~440 MB
    activations: ~600 MB (batch_size=64, seq_len=128)
    total: ~1.5 GB — well within 8 GB VRAM

WikiText-103 scale:
    ~103M tokens, ~805K chunks of 128 tokens
    ~25K batches/epoch at batch_size=32
    Approx. 70 min/epoch on RTX 3070 Ti
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

    # ── Training (overrides) ─────────────────────────────────────────────────
    batch_size: int = 32          # 64 → 32 (VRAM constraint)
    learning_rate: float = 3e-4   # Higher LR — cosine decay will bring it down
    lr_schedule: str = "cosine"   # Warmup + cosine decay
    lr_min: float = 1e-5          # Floor for cosine decay
    num_epochs: int = 30          # Fewer epochs; each is ~70 min on RTX 3070 Ti
    warmup_steps: int = 5_000     # Larger dataset → longer warmup
    early_stopping_patience: int = 5   # Stop if no improvement for 5 epochs

    # ── Checkpointing (overrides) ────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_wt103"
    save_every_n_epochs: int = 5  # More frequent saves (epochs are expensive)
