from dataclasses import dataclass


@dataclass
class ModelConfig:
    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer_name: str = "bert-base-uncased"

    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size: int = 30_522      # BERT vocabulary size
    max_seq_len: int = 128        # Fixed length required for ANE compatibility
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4            # head_dim = 64 (ANE-optimised)
    ffn_dim: int = 1_024
    dropout: float = 0.1

    # ── Special token IDs (BERT bert-base-uncased defaults) ──────────────────
    mask_token_id: int = 103      # [MASK]
    pad_token_id: int = 0         # [PAD]
    cls_token_id: int = 101       # [CLS]
    sep_token_id: int = 102       # [SEP]

    # ── Diffusion ─────────────────────────────────────────────────────────────
    T: int = 100                  # Diffusion steps during training
    # Masking schedule: "linear" → mask_rate = t/T
    #                   "cosine" → mask_rate = (1 - cos(π·t/T)) / 2
    mask_schedule: str = "linear"
    # Full-sequence loss: weight for masked positions (unmasked = 1.0)
    mask_loss_weight: float = 5.0

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: str = "wikitext-2"   # "wikitext-2" or "wikitext-103"

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1_000
    # LR schedule after warmup: "constant" or "cosine" (decay to lr_min)
    lr_schedule: str = "constant"
    lr_min: float = 1e-6           # minimum LR for cosine decay
    grad_clip: float = 1.0
    # Early stopping: halt when val_loss has not improved for this many epochs
    # Set to 0 to disable.
    early_stopping_patience: int = 0

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10

    # ── Mid-epoch evaluation ───────────────────────────────────────────────────
    # eval_steps > 0: run validation + save checkpoint every N global steps
    # eval_steps = 0: epoch-end only (original behaviour)
    eval_steps: int = 0
