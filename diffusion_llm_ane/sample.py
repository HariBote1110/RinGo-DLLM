"""
PyTorch inference script — Masked Diffusion Language Model.

Implements the reverse diffusion (denoising) loop described in Phase 5
of the implementation plan:

    x_T  (all [MASK])
     ↓  for t = T, T-1, …, 1
    logits = model(x_t, t)
    unmask the positions with the highest confidence
     ↓
    x_0  (completed sequence)

Usage:
    # Unconditional generation (all tokens start masked)
    python sample.py --checkpoint checkpoints/best_model.pt --steps 50

    # Masked infilling (provide a prompt with [MASK] tokens)
    python sample.py --checkpoint checkpoints/best_model.pt \\
        --prompt "The capital of France is [MASK] [MASK] [MASK] ." \\
        --steps 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data.tokenizer import get_tokenizer
from model.config import ModelConfig
from model.diffusion_lm import DiffusionLM


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run reverse diffusion sampling")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--steps",  type=int,   default=50,  help="Number of denoising steps")
    p.add_argument("--prompt", type=str,   default=None,
                   help="Prompt text; use [MASK] for positions to generate")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (1.0 = greedy)")
    p.add_argument("--top-k", type=int, default=0,
                   help="Top-k sampling (0 = greedy argmax)")
    return p.parse_args()


# ── Sampling helpers ──────────────────────────────────────────────────────────

def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
    """
    Sample a single token from a logits vector.

    Args:
        logits:      (V,) unnormalised log-probabilities
        temperature: softmax temperature
        top_k:       if > 0, restrict to top-k tokens

    Returns:
        scalar token id
    """
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < values[..., -1, None], float("-inf"))

    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def unmask_step(
    x_t: torch.Tensor,
    logits: torch.Tensor,
    target_unmasked: int,
    mask_token_id: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """
    Unmask exactly `target_unmasked` positions by choosing the most confident ones.

    Args:
        x_t:              (L,)         current token sequence
        logits:           (L, V)       model output
        target_unmasked:  int          total number of positions that should be unmasked after this step
        mask_token_id:    int
        temperature:      float
        top_k:            int

    Returns:
        x_{t-1}: (L,) updated token sequence
    """
    mask_positions = (x_t == mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return x_t

    # How many new tokens to reveal this step
    currently_unmasked = int((x_t != mask_token_id).sum().item())
    to_reveal = max(0, target_unmasked - currently_unmasked)
    if to_reveal == 0:
        return x_t

    # Rank masked positions by the model's confidence (max-prob over vocab)
    probs = F.softmax(logits[mask_positions] / max(temperature, 1e-6), dim=-1)  # (N, V)
    confidence, _ = probs.max(dim=-1)                                            # (N,)

    # Reveal the `to_reveal` highest-confidence positions
    n_reveal = min(to_reveal, len(mask_positions))
    _, top_idx = confidence.topk(n_reveal)

    x_new = x_t.clone()
    for idx in top_idx:
        pos = mask_positions[idx].item()
        x_new[pos] = sample_token(logits[pos], temperature, top_k)

    return x_new


# ── Main sampling loop ────────────────────────────────────────────────────────

@torch.no_grad()
def sample(
    model: DiffusionLM,
    config: ModelConfig,
    prompt_ids: torch.Tensor | None,
    n_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Run the reverse diffusion loop.

    Args:
        prompt_ids:  (L,) with [MASK] at positions to generate, or None (full generation)
        n_steps:     number of denoising steps (≤ config.T)

    Returns:
        (L,) completed token sequence
    """
    L = config.max_seq_len

    if prompt_ids is None:
        # Unconditional generation: start fully masked
        x_t = torch.full((L,), config.mask_token_id, dtype=torch.long, device=device)
    else:
        x_t = prompt_ids.clone().to(device)
        # Pad or truncate to max_seq_len
        if len(x_t) < L:
            pad = torch.full((L - len(x_t),), config.pad_token_id, dtype=torch.long, device=device)
            x_t = torch.cat([x_t, pad])
        else:
            x_t = x_t[:L]

    total_mask = int((x_t == config.mask_token_id).sum().item())
    if total_mask == 0:
        print("No [MASK] tokens found in prompt — returning as-is.")
        return x_t

    # Number of tokens that are already visible (prompt context)
    initial_unmasked = int((x_t != config.mask_token_id).sum().item())

    # Evenly space unmasking across `n_steps` denoising steps
    for step in range(n_steps, 0, -1):
        t_val = int(step * config.T / n_steps)
        t_tensor = torch.tensor([t_val], dtype=torch.long, device=device)

        # Model forward (batch of 1)
        logits = model(x_t.unsqueeze(0), t_tensor).squeeze(0)   # (L, V)

        # How many tokens should be unmasked by the end of this step?
        # `initial_unmasked` offsets the prompt context tokens so that
        # `target_unmasked` matches the total visible count tracked inside
        # `unmask_step` (which counts ALL non-mask positions).
        fraction_done = 1.0 - (step - 1) / n_steps
        target_unmasked = initial_unmasked + round(total_mask * fraction_done)

        x_t = unmask_step(x_t, logits, target_unmasked, config.mask_token_id, temperature, top_k)

    return x_t


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokeniser = get_tokenizer()

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config: ModelConfig = ckpt["config"]
    model = DiffusionLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {ckpt_path} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # Prepare prompt
    prompt_ids: torch.Tensor | None = None
    if args.prompt:
        prompt_ids = torch.tensor(
            tokeniser.encode(args.prompt, add_special_tokens=False),
            dtype=torch.long,
        )
        print(f"Prompt : {args.prompt}")
        print(f"Encoded: {prompt_ids.tolist()}")

    # Run sampling
    output_ids = sample(
        model,
        config,
        prompt_ids=prompt_ids,
        n_steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    # Decode and print
    tokens = output_ids.cpu().tolist()
    # Remove padding
    tokens = [t for t in tokens if t != config.pad_token_id]

    # Count remaining masks (should be 0 after full denoising)
    n_remaining = tokens.count(config.mask_token_id)
    if n_remaining > 0:
        print(f"  ⚠ {n_remaining} [MASK] token(s) still unreplaced")

    # skip_special_tokens=True suppresses [CLS], [SEP], [MASK] etc.
    # so we see only the natural-language output.
    decoded = tokeniser.decode(tokens, skip_special_tokens=True)
    print(f"\nGenerated ({args.steps} steps):\n{decoded}")


if __name__ == "__main__":
    main()
