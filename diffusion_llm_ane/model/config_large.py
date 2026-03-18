"""
Larger model configuration for ANE-scale training.

Target: ~85M parameters — large enough for ANE throughput to dominate
the data-transfer overhead, while still trainable on an RTX 3070 Ti (8 GB).

Memory estimate (fp32 training):
    params: ~85M × 4 bytes = ~340 MB
    grads:  ~340 MB
    optimiser (AdamW): ~680 MB
    activations: ~500 MB (batch_size=32, seq_len=128)
    total: ~1.9 GB — well within 8 GB VRAM
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

    # ── Training (overrides) ─────────────────────────────────────────────────
    batch_size: int = 32          # 64 → 32 (VRAM constraint)
    learning_rate: float = 5e-5   # Slightly lower for larger model
    warmup_steps: int = 2_000     # 1000 → 2000

    # ── Checkpointing (overrides) ────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_large"
