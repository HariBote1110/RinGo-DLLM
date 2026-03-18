"""
CoreML export for the ANE-optimised DiffusionLM (Conv2d-based).

Loads a standard checkpoint, converts weights to ANE format,
traces, and exports to .mlpackage.

Usage:
    python convert/export_coreml_ane.py \
        --checkpoint checkpoints/best_model.pt \
        --output convert/diffusion_lm_ane.mlpackage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import coremltools as ct
except ImportError:
    print("ERROR: coremltools not installed.  Run:  pip install coremltools>=8.0")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.diffusion_lm_ane import DiffusionLM_ANE


class TraceableANEModel(torch.nn.Module):
    """Wrapper for TorchScript tracing."""

    def __init__(self, model: DiffusionLM_ANE):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, t.long())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ANE-optimised DiffusionLM to CoreML")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="convert/diffusion_lm_ane.mlpackage")
    p.add_argument("--precision", choices=["FLOAT16", "FLOAT32"], default="FLOAT16")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load & convert weights
    print(f"Loading checkpoint and converting to ANE format …")
    model = DiffusionLM_ANE.from_standard_checkpoint(args.checkpoint, device="cpu")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    traceable = TraceableANEModel(model)
    config = model.config

    example_input_ids = torch.zeros(1, config.max_seq_len, dtype=torch.long)
    example_t = torch.tensor([50], dtype=torch.int32)

    # TorchScript trace
    print("Tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(traceable, (example_input_ids, example_t))
    print("Trace complete.")

    # CoreML conversion
    precision_map = {
        "FLOAT16": ct.precision.FLOAT16,
        "FLOAT32": ct.precision.FLOAT32,
    }

    print(f"Converting to CoreML (precision={args.precision}) …")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, config.max_seq_len), dtype=np.int32),
            ct.TensorType(name="t", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        compute_precision=precision_map[args.precision],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )

    out_path = Path(args.output)
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")

    # Sanity check
    print("Sanity-checking …")
    loaded = ct.models.MLModel(str(out_path))
    result = loaded.predict({
        "input_ids": example_input_ids.numpy().astype(np.int32),
        "t": example_t.numpy().astype(np.int32),
    })
    logits = result["logits"]
    print(f"  Output shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")
    print("Done.")


if __name__ == "__main__":
    main()
