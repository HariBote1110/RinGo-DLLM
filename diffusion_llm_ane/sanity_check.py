"""
Quick sanity check — verifies model construction, forward pass, and apply_mask.
Run before starting full training.

Usage:
    python sanity_check.py                  # base config (13M EN)
    python sanity_check.py --config large   # large config (55M EN)
    python sanity_check.py --config ja-large  # large config (55M JA)
"""
import argparse
import sys

import torch

print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS  available: {torch.backends.mps.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

from model.config import ModelConfig
from model.config_large import ModelConfigLarge, ModelConfigLargeJa
from model.diffusion_lm import DiffusionLM, apply_mask

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="base",
    choices=["base", "large", "ja-large"],
)
args = parser.parse_args()

_CONFIG_MAP = {
    "base":     ModelConfig,
    "large":    ModelConfigLarge,
    "ja-large": ModelConfigLargeJa,
}
config = _CONFIG_MAP[args.config]()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device  : {device}")
print(f"Config  : {args.config} (vocab={config.vocab_size}, "
      f"tokenizer={getattr(config, 'tokenizer_name', 'N/A')})\n")

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

# ── Tokenizer check ──
tokenizer_name = getattr(config, "tokenizer_name", "bert-base-uncased")
from data.tokenizer import get_tokenizer
tok = get_tokenizer(tokenizer_name)
print(f"\nTokenizer       : {tokenizer_name}")
print(f"  vocab_size    : {tok.vocab_size}")
print(f"  [MASK] id     : {tok.mask_token_id}  (config: {config.mask_token_id})")
assert tok.mask_token_id == config.mask_token_id, \
    f"Mismatch! tokenizer mask_id={tok.mask_token_id} != config mask_id={config.mask_token_id}"

print("\nSanity check passed ✓")
