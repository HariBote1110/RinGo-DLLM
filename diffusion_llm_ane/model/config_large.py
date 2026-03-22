"""
Larger model configurations for ANE-scale training.

Includes both English (WikiText-103) and Japanese (Wikipedia-ja) variants.

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

Changelog:
    v1: baseline (LR=1e-4, T=100, mask-only loss)           → val ~7.19 (plateau)
    v2: full-seq loss, T=25, LR=5e-4                        → val 4.6309 (Epoch 8 Best)
         → overfitting from Epoch 9: val rose to 4.71 by Epoch 11
    v3: stronger regularisation to address overfitting
         dropout 0.1→0.2, weight_decay 0.01→0.05,
         LR 5e-4→3e-4, patience 5→3, new checkpoint dir
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
    dropout: float = 0.2          # v3: 0.1 → 0.2 (過学習抑制)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: str = "wikitext-103"   # 50× more data than wikitext-2

    # ── Masking schedule ─────────────────────────────────────────────────────
    mask_schedule: str = "cosine"   # smoother masking curve

    # ── Diffusion (overrides) ──────────────────────────────────────────────────
    T: int = 25                   # 100 → 25: 離散トークンでは少ないステップで十分
    mask_loss_weight: float = 5.0 # 全位置ロスでのマスク位置の重み

    # ── Training (overrides) ─────────────────────────────────────────────────
    batch_size: int = 48          # AMP 有効時に VRAM 8 GB 内で安定動作
    learning_rate: float = 3e-4   # v3: 5e-4 → 3e-4 (過学習時は LR を下げて安定化)
    weight_decay: float = 0.05    # v3: 0.01 → 0.05 (L2 正則化を強化)
    lr_schedule: str = "cosine"   # Warmup + cosine decay
    lr_min: float = 1e-5          # Floor for cosine decay
    num_epochs: int = 30          # Fewer epochs; each is ~70 min on RTX 3070 Ti
    warmup_steps: int = 3_000     # T 削減に合わせて warmup も短縮
    early_stopping_patience: int = 3   # v3: 5 → 3 (過学習を早期に検出して停止)

    # ── Checkpointing (overrides) ────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_wt103_v3"   # v2 の best を保持するため別ディレクトリ
    save_every_n_epochs: int = 5  # More frequent saves (epochs are expensive)


@dataclass
class ModelConfigLargeJa(ModelConfigLarge):
    """
    Japanese variant — tohoku-nlp/bert-base-japanese-v3 tokenizer.

    Vocab size 32,768 is close to BERT's 30,522 (+7.3%), so the embedding
    table only grows by ~1.2M params.  MeCab-based subword segmentation
    gives a fertility of ~0.6 tokens/char for Japanese text, meaning 128
    tokens ≈ 150 characters (1-2 short paragraphs).

    Wikipedia Japanese contains ~0.9-1.5B tokens — roughly 10× WikiText-103.

    Changelog:
        v1: batch_size=48, lr=3e-4, num_epochs=30  → val 4.55 (Epoch 2)
            → 中止: 1 epoch ≈ 6.8h, 30 epoch 完走に 8 日超かかるため
        v2: batch_size=128, lr=5e-4 (sqrt scaling), num_epochs=10
            → 1 epoch ≈ 3h 見込み, 合計 ~1.5 日に短縮
    """

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer_name: str = "tohoku-nlp/bert-base-japanese-v3"

    # ── Architecture (overrides) ─────────────────────────────────────────────
    vocab_size: int = 32_768      # bert-base-japanese-v3 vocabulary size

    # ── Special token IDs (bert-base-japanese-v3) ────────────────────────────
    mask_token_id: int = 4        # [MASK]
    pad_token_id: int = 0         # [PAD]
    cls_token_id: int = 2         # [CLS]
    sep_token_id: int = 3         # [SEP]

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_name: str = "wikipedia-ja"

    # ── Training (v2 overrides) ───────────────────────────────────────────────
    batch_size: int = 128         # v2: 48 → 128 (RTX 3070 Ti 8GB に余裕あり)
    learning_rate: float = 5e-4   # v2: sqrt scaling: 3e-4 × √(128/48) ≈ 5e-4
    warmup_steps: int = 4_000     # v2: バッチ増加に合わせて若干延長
    num_epochs: int = 10          # v2: 30 → 10 (15.8B tokens, 55M params に十分)

    # ── Checkpointing ────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_ja_v2"


@dataclass
class ModelConfigJa100M(ModelConfigLargeJa):
    """
    ~96M parameter Japanese MDLM — "100M class" scale-up from 55M.

    Architecture: hidden_dim 512→768, num_heads 8→12, ffn_dim 2048 (kept).
    Weight tying (token_embed = lm_head) keeps the output projection free.

    Parameter breakdown (BF16 AMP training):
        token_embed (tied):  32768 × 768             = 25.2M
        pos_embed:             128 × 768             =  0.1M
        time_proj (MLP):     768→3072→768            =  4.7M
        12 × (attn 2.36M + ffn 3.15M)               = 66.1M
        Total:                                       ≈ 96M

    VRAM estimate (batch_size=128, BF16 AMP):
        params+grads: ~384 MB
        AdamW states: ~768 MB
        activations:  ~450 MB
        Total:        ~1.6 GB — well within 8 GB

    LR scaled from 55M via √(96/55) ≈ 1.32×: 5e-4 × 1.32 ≈ 6.6e-4 → 6e-4
    Training time estimate: ~5-6h/epoch × 10 epochs ≈ 2-2.5 days on RTX 3070 Ti
    """

    # ── Architecture (96M overrides) ─────────────────────────────────────────
    hidden_dim: int = 768         # 512 → 768  (+50 %)
    num_heads:  int = 12          # 8 → 12, keeps head_dim=64 (ANE-optimised)
    ffn_dim:    int = 2_048       # kept (768×2048 ratio ≈ 2.7×)
    dropout:    float = 0.1       # relax from 0.2; larger model regularises itself

    # ── Training (100M overrides) ─────────────────────────────────────────────
    learning_rate: float = 6e-4   # sqrt-scaled from 55M's 5e-4: ×√(96/55) ≈ ×1.32
    warmup_steps:  int = 6_000    # longer warmup for larger model
    num_epochs:    int = 10       # same epoch budget

    # ── Mid-epoch evaluation ──────────────────────────────────────────────────
    # ~10,000 steps ≈ 50-60 min on RTX 3070 Ti → ~10 checkpoints/epoch
    eval_steps: int = 10_000

    # ── Checkpointing ────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_ja_100m"
