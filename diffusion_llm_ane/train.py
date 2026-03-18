"""
Training script for the Masked Diffusion Language Model.

Usage:
    python train.py [--epochs N] [--batch-size N] [--lr LR] [--resume PATH]
                    [--webhook URL] [--notify-every N]

Device priority: CUDA > MPS (M4) > CPU

Discord 通知:
    --webhook オプション、または環境変数 DISCORD_WEBHOOK_URL で指定。
    --notify-every N でエポック通知の間隔を変更（デフォルト: 10）。
"""

from __future__ import annotations

import argparse
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
                   help="Model config: base (13M) or large (85M)")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch-size",   type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--resume",       type=str,   default=None, help="Checkpoint path to resume from")
    p.add_argument("--webhook",      type=str,   default=None, help="Discord Webhook URL")
    p.add_argument("--notify-every", type=int,   default=10,   help="Discord 通知間隔（エポック数）")
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


# ── Learning-rate warmup schedule ─────────────────────────────────────────────

def build_lr_schedule(optimiser: torch.optim.Optimizer, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


# ── Validation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: DiffusionLM, loader, config: ModelConfig, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        x_0 = batch.to(device)
        B = x_0.size(0)
        t = torch.randint(1, config.T + 1, (B,), device=device)
        mask_rate = t.float() / config.T
        x_t, is_mask = apply_mask(x_0, mask_rate, config.mask_token_id)
        logits = model(x_t, t)
        if is_mask.any():
            loss = F.cross_entropy(logits[is_mask], x_0[is_mask])
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

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: {config}")

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
    print("Loading WikiText-2 …")
    train_loader = get_dataloader("train", config)
    val_loader   = get_dataloader("validation", config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    scheduler = build_lr_schedule(optimiser, config.warmup_steps)

    save_dir = Path(config.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # ── Optional resume ──
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} (val_loss={best_val_loss:.4f})")

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

                # Forward diffusion: mask tokens at rate t/T
                mask_rate = t.float() / config.T
                x_t, is_mask = apply_mask(x_0, mask_rate, config.mask_token_id)

                # Model prediction
                logits = model(x_t, t)          # (B, L, vocab_size)

                # Loss only on masked positions
                if not is_mask.any():
                    continue
                loss = F.cross_entropy(logits[is_mask], x_0[is_mask])

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimiser.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 200 == 0:
                    lr_now = optimiser.param_groups[0]["lr"]
                    print(
                        f"  step {global_step:6d} | loss {loss.item():.4f} | lr {lr_now:.2e}"
                    )

            avg_train = epoch_loss / len(train_loader)
            val_loss  = evaluate(model, val_loader, config, device)
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

    except Exception as exc:
        notifier.error(traceback.format_exc())
        raise

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    notifier.training_complete(best_val_loss, config.num_epochs)


if __name__ == "__main__":
    train()
