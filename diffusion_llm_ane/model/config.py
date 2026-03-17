from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size: int = 30_522      # BERT vocabulary size
    max_seq_len: int = 128        # Fixed length required for ANE compatibility
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4            # head_dim = 64 (ANE-optimised)
    ffn_dim: int = 1_024
    dropout: float = 0.1

    # ── Special token IDs (BERT defaults) ────────────────────────────────────
    mask_token_id: int = 103      # [MASK]
    pad_token_id: int = 0         # [PAD]
    cls_token_id: int = 101       # [CLS]
    sep_token_id: int = 102       # [SEP]

    # ── Diffusion ─────────────────────────────────────────────────────────────
    T: int = 100                  # Diffusion steps during training
    # Linear masking rate: alpha_t = 1 - t/T  (fraction of unmasked tokens)

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1_000
    grad_clip: float = 1.0

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10
