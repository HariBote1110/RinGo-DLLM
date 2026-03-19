"""
Training script for the Masked Diffusion Language Model.

Usage:
    python train.py [--config base|large] [--dataset wikitext-2|wikitext-103]
                    [--epochs N] [--batch-size N] [--lr LR] [--resume PATH]
                    [--webhook URL] [--notify-every N]
                    [--no-compile] [--no-amp]

Device priority: CUDA > MPS (M4) > CPU

LR schedule:
    config.lr_schedule = "constant" → flat after warmup (original behaviour)
    config.lr_schedule = "cosine"   → cosine decay from peak LR to lr_min

Early stopping:
    config.early_stopping_patience > 0 → halt when val_loss has not improved
    for that many consecutive epochs.

Efficiency flags (all enabled by default on CUDA):
    --no-compile  Disable torch.compile (useful for debugging)
    --no-amp      Disable AMP / BF16 mixed precision

Discord 通知:
    --webhook オプション、または環境変数 DISCORD_WEBHOOK_URL で指定。
    --notify-every N でエポック通知の間隔を変更（デフォルト: 10）。
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from data.dataset import get_dataloader
from model.config import ModelConfig
from model.config_large import ModelConfigLarge
from model.diffusion_lm import DiffusionLM, apply_mask
from notify import Notifier


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Diffusion LM")
    p.add_argument("--config",       type=str,   default="base", choices=["base", "large"],
                   help="Model config: base (13M) or large (55M)")
    p.add_argument("--dataset",      type=str,   default=None,
                   choices=["wikitext-2", "wikitext-103"],
                   help="Override dataset (default: from config)")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch-size",   type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--resume",       type=str,   default=None, help="Checkpoint path to resume from")
    p.add_argument("--webhook",      type=str,   default=None, help="Discord Webhook URL")
    p.add_argument("--notify-every", type=int,   default=10,   help="Discord 通知間隔（エポック数）")
    p.add_argument("--no-compile",      action="store_true", help="Disable torch.compile")
    p.add_argument("--no-amp",          action="store_true", help="Disable AMP mixed precision")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Gradient checkpointing: recompute activations on backward "
                        "to save VRAM at ~+33%% compute cost")
    return p.parse_args()


# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


# ── Learning-rate schedule ────────────────────────────────────────────────────

def build_lr_schedule(
    optimiser: torch.optim.Optimizer,
    config: ModelConfig,
    total_steps: int,
):
    """
    Build a LambdaLR scheduler.

    "constant": linear warmup then constant at peak LR.
    "cosine":   linear warmup then cosine decay from peak LR down to lr_min.
    """
    warmup_steps = config.warmup_steps
    lr_min_ratio = config.lr_min / config.learning_rate   # fraction of peak LR

    if config.lr_schedule == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # Cosine decay from 1.0 → lr_min_ratio over the remaining steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine_decay
    else:
        # "constant" — original behaviour
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


# ── Validation loop ───────────────────────────────────────────────────────────

def compute_mask_rate(t: torch.Tensor, T: int, schedule: str) -> torch.Tensor:
    """
    Compute per-sample mask rates from diffusion timesteps.

    "linear": mask_rate = t / T
    "cosine": mask_rate = (1 - cos(π·t/T)) / 2
              → gentler at t≈0 and t≈T, steeper in the middle
    """
    ratio = t.float() / T
    if schedule == "cosine":
        return (1.0 - torch.cos(math.pi * ratio)) / 2.0
    return ratio   # linear (default)


@torch.no_grad()
def evaluate(
    model: DiffusionLM,
    loader,
    config: ModelConfig,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    schedule = getattr(config, "mask_schedule", "linear")
    mask_w = getattr(config, "mask_loss_weight", 5.0)
    for batch in loader:
        x_0 = batch.to(device)
        B = x_0.size(0)
        t = torch.randint(1, config.T + 1, (B,), device=device)
        mask_rate = compute_mask_rate(t, config.T, schedule)
        x_t, is_mask = apply_mask(x_0, mask_rate, config.mask_token_id)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x_t, t)
        weight = torch.where(is_mask, mask_w, 1.0)
        per_token = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            x_0.view(-1),
            reduction="none",
        ).view_as(x_0)
        loss = (per_token * weight).sum() / weight.sum()
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


# ── Main training loop ────────────────────────────────────────────────────────

def train() -> None:
    args = parse_args()
    config = ModelConfigLarge() if args.config == "large" else ModelConfig()

    # Apply any CLI overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.dataset is not None:
        config.dataset_name = args.dataset
    if args.grad_checkpoint:
        config.gradient_checkpointing = True

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {config}")

    # ── AMP (Automatic Mixed Precision) ──────────────────────────────────────
    # BF16 preferred on Ampere+ (RTX 3070 Ti); FP16 on older GPUs; disabled on MPS/CPU
    use_amp = (not args.no_amp) and (device.type == "cuda")
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    print(f"AMP: {'enabled (' + str(amp_dtype) + ')' if use_amp else 'disabled'}")
    print(f"Gradient checkpointing: {'enabled' if args.grad_checkpoint else 'disabled'}")

    # ── Notifier ──
    notifier = Notifier(args.webhook)

    # ── Model & optimiser ──
    model = DiffusionLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── Data ──
    dataset_name = getattr(config, "dataset_name", "wikitext-2")
    print(f"Loading {dataset_name} …")
    train_loader = get_dataloader("train", config)
    val_loader   = get_dataloader("validation", config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Total training steps — needed for cosine LR decay
    total_steps = len(train_loader) * config.num_epochs
    scheduler = build_lr_schedule(optimiser, config, total_steps)

    save_dir = Path(config.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = getattr(config, "early_stopping_patience", 0)
    schedule = getattr(config, "mask_schedule", "linear")

    # ── Optional resume (チェックポイントは compile より先にロード) ──
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} (val_loss={best_val_loss:.4f})")

    # ── torch.compile ─────────────────────────────────────────────────────────
    # チェックポイントロード後に compile することで _orig_mod.* キー衝突を回避
    # Compiles the model graph for faster CUDA kernels (PyTorch 2.0+, CUDA only)
    use_compile = (not args.no_compile) and (device.type == "cuda") and hasattr(torch, "compile")
    if use_compile:
        print("torch.compile: enabled (compiling model…)")
        model = torch.compile(model)

    # 学習開始通知
    notifier.training_start(n_params, config.num_epochs, device_label(device))

    try:
        # ── Training epochs ──
        for epoch in range(start_epoch, config.num_epochs):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for batch in train_loader:
                x_0 = batch.to(device)          # (B, L) original tokens
                B = x_0.size(0)

                # Sample diffusion timestep t ~ Uniform(1, T)
                t = torch.randint(1, config.T + 1, (B,), device=device)

                # Forward diffusion: mask tokens according to schedule
                mask_rate = compute_mask_rate(t, config.T, schedule)
                x_t, is_mask = apply_mask(x_0, mask_rate, config.mask_token_id)

                # Model prediction with AMP autocast
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(x_t, t)      # (B, L, vocab_size)

                # Full-sequence loss with higher weight on masked positions.
                # All positions contribute gradient, but masked tokens are
                # weighted more heavily (mask_loss_weight, default 5.0) so
                # the model still focuses on denoising while also learning
                # contextual representations at every position.
                mask_w = getattr(config, "mask_loss_weight", 5.0)
                weight = torch.where(is_mask, mask_w, 1.0)          # (B, L)
                per_token = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),        # (B*L, V)
                    x_0.view(-1),                                    # (B*L,)
                    reduction="none",
                ).view_as(x_0)                                       # (B, L)
                loss = (per_token * weight).sum() / weight.sum()

                optimiser.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 200 == 0:
                    lr_now = optimiser.param_groups[0]["lr"]
                    print(
                        f"  step {global_step:6d} | loss {loss.item():.4f} | lr {lr_now:.2e}"
                    )

            avg_train = epoch_loss / len(train_loader)
            val_loss  = evaluate(model, val_loader, config, device, use_amp, amp_dtype)
            elapsed   = time.time() - t0

            print(
                f"Epoch {epoch + 1:3d}/{config.num_epochs} | "
                f"train {avg_train:.4f} | val {val_loss:.4f} | "
                f"{elapsed:.0f}s"
            )

            # Save best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "val_loss": val_loss,
                        "config": config,
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  → Best model saved (val_loss={val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % config.save_every_n_epochs == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "val_loss": val_loss,
                        "config": config,
                    },
                    save_dir / f"epoch_{epoch + 1:04d}.pt",
                )

            # Discord 通知（notify_every エポックごと、または best 更新時）
            if is_best or (epoch + 1) % args.notify_every == 0:
                notifier.epoch_update(
                    epoch=epoch + 1,
                    n_epochs=config.num_epochs,
                    train_loss=avg_train,
                    val_loss=val_loss,
                    elapsed_s=elapsed,
                    is_best=is_best,
                )

            # Early stopping
            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"Early stopping: val_loss has not improved for "
                    f"{epochs_without_improvement} epochs. "
                    f"Best val_loss = {best_val_loss:.4f}"
                )
                notifier.training_complete(best_val_loss, epoch + 1)
                return

    except Exception as exc:
        notifier.error(traceback.format_exc())
        raise

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    notifier.training_complete(best_val_loss, config.num_epochs)


if __name__ == "__main__":
    train()
