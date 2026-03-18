"""
Quick sanity check — verifies model construction, forward pass, and apply_mask.
Run before starting full training.
"""
import sys
import torch

print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

from model.config import ModelConfig
from model.diffusion_lm import DiffusionLM, apply_mask

config = ModelConfig()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device  : {device}\n")

# ── Model construction ──
model = DiffusionLM(config).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params  : {n_params:,}")

# ── Forward pass ──
B, L = 2, config.max_seq_len
x = torch.randint(0, config.vocab_size, (B, L), device=device)
t = torch.randint(1, config.T + 1, (B,), device=device)
mask_rate = t.float() / config.T

x_t, is_mask = apply_mask(x, mask_rate, config.mask_token_id)
print(f"Mask rate sample: {mask_rate.tolist()}")
print(f"Masked positions: {is_mask.sum().item()} / {B * L}")

with torch.no_grad():
    logits = model(x_t, t)

assert logits.shape == (B, L, config.vocab_size), f"Unexpected shape: {logits.shape}"
print(f"Logits shape    : {logits.shape}  ✓")

# ── Loss ──
import torch.nn.functional as F
if is_mask.any():
    loss = F.cross_entropy(logits[is_mask], x[is_mask])
    print(f"Initial loss    : {loss.item():.4f}  (random ≈ {torch.log(torch.tensor(float(config.vocab_size))):.2f})")

print("\nSanity check passed ✓")
