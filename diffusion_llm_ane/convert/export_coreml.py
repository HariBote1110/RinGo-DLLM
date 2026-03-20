"""
Phase 3: CoreML conversion script.

Converts a trained DiffusionLM checkpoint to a `.mlpackage` via:
    PyTorch  →  TorchScript (trace)  →  coremltools  →  .mlpackage

Usage:
    python convert/export_coreml.py \\
        --checkpoint checkpoints/best_model.pt \\
        --output diffusion_lm.mlpackage \\
        --compute-units ALL

Known caveats (see plan doc):
    - Embedding ops may fall back to CPU; this is acceptable with ComputeUnit.ALL
    - int32 token IDs must be cast to float32 after embedding (handled automatically
      by the Embedding layer → CoreML translates this correctly)
    - Dynamic mask generation is moved outside the model: the masked input_ids are
      passed as an external input so the model graph stays static
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# CoreML tools are only available after `pip install coremltools`
try:
    import coremltools as ct
except ImportError:
    print("ERROR: coremltools not installed.  Run:  pip install coremltools>=8.0")
    sys.exit(1)

# Add project root to path when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.diffusion_lm import DiffusionLM


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DiffusionLM to CoreML")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--output",        type=str, default="diffusion_lm.mlpackage")
    p.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU"],
        default="ALL",
        help="CoreML compute units (ALL includes ANE)",
    )
    p.add_argument(
        "--precision",
        choices=["FLOAT16", "FLOAT32"],
        default="FLOAT16",
        help="Model compute precision (FLOAT16 required for ANE)",
    )
    return p.parse_args()


# ── Wrapper for tracing ───────────────────────────────────────────────────────

class TraceableModel(torch.nn.Module):
    """
    Thin wrapper that normalises the timestep input to float32 before passing
    it to the model.  This makes the TorchScript trace cleaner and avoids
    int64 ops that CoreML may not support on ANE.
    """

    def __init__(self, model: DiffusionLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t is passed as int32 from CoreML; cast to long for embedding lookup
        return self.model(input_ids, t.long())


# ── Conversion ────────────────────────────────────────────────────────────────

def export(args: argparse.Namespace) -> None:
    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: ModelConfig = ckpt["config"]

    model = DiffusionLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {ckpt_path}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    traceable = TraceableModel(model)

    # Example inputs — fixed shapes for ANE (batch=1, seq=128)
    example_input_ids = torch.zeros(1, config.max_seq_len, dtype=torch.long)
    example_t         = torch.tensor([50], dtype=torch.int32)

    # ── TorchScript trace ──
    print("Tracing model …")
    with torch.no_grad():
        traced = torch.jit.trace(traceable, (example_input_ids, example_t))
    print("Trace complete.")

    # ── CoreML conversion ──
    precision_map = {
        "FLOAT16": ct.precision.FLOAT16,
        "FLOAT32": ct.precision.FLOAT32,
    }
    compute_units_map = {
        "ALL":         ct.ComputeUnit.ALL,
        "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    }

    print(f"Converting to CoreML (precision={args.precision}, compute_units={args.compute_units}) …")

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, config.max_seq_len),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="t",
                shape=(1,),
                dtype=np.int32,
            ),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        compute_precision=precision_map[args.precision],
        compute_units=compute_units_map[args.compute_units],
        # Minimum deployment target: macOS 15 (required for INT4 quantisation)
        minimum_deployment_target=ct.target.macOS15,
    )

    # ── Save ──
    out_path = Path(args.output)
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")

    # ── Sanity check: run one prediction on CPU ──
    print("Sanity-checking saved model …")
    loaded = ct.models.MLModel(str(out_path))
    result = loaded.predict(
        {
            "input_ids": example_input_ids.numpy().astype(np.int32),
            "t":         example_t.numpy().astype(np.int32),
        }
    )
    logits = result["logits"]
    print(f"  Output shape : {logits.shape}")  # Expected: (1, max_seq_len, vocab_size)
    print(f"  Output dtype : {logits.dtype}")
    print("Conversion succeeded.")


if __name__ == "__main__":
    export(parse_args())
