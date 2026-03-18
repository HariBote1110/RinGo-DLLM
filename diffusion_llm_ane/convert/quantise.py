"""
Post-training weight quantisation for CoreML models.

Applies INT8 or INT4 linear quantisation to the .mlpackage weights,
reducing memory bandwidth and potentially improving ANE throughput.

Usage:
    python convert/quantise.py \
        --model convert/diffusion_lm.mlpackage \
        --bits 8 4

Outputs:
    convert/diffusion_lm_int8.mlpackage
    convert/diffusion_lm_int4.mlpackage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import coremltools as ct
    import coremltools.optimize.coreml as cto
except ImportError:
    print("ERROR: coremltools not installed.  Run:  pip install coremltools>=8.0")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantise a CoreML model")
    p.add_argument("--model", type=str, required=True, help="Path to .mlpackage")
    p.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[8, 4],
        choices=[4, 8],
        help="Quantisation bit widths to produce (default: 8 4)",
    )
    return p.parse_args()


def quantise(model_path: str, n_bits: int) -> str:
    """
    Apply linear symmetric weight quantisation.

    Returns the output path of the quantised model.
    """
    stem = Path(model_path).stem          # e.g. "diffusion_lm"
    out_dir = Path(model_path).parent
    out_path = out_dir / f"{stem}_int{n_bits}.mlpackage"

    print(f"\n{'='*60}")
    print(f"Quantising to INT{n_bits}: {model_path}")
    print(f"{'='*60}")

    mlmodel = ct.models.MLModel(model_path)

    if n_bits == 8:
        config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
            )
        )
        quantised = cto.linear_quantize_weights(mlmodel, config)
    elif n_bits == 4:
        config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int4",
            )
        )
        quantised = cto.linear_quantize_weights(mlmodel, config)
    else:
        raise ValueError(f"Unsupported bit width: {n_bits}")

    quantised.save(str(out_path))
    print(f"Saved: {out_path}")

    # Report size reduction
    orig_size = sum(f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file())
    new_size = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
    ratio = new_size / orig_size * 100
    print(f"  Original : {orig_size / 1024 / 1024:.1f} MB")
    print(f"  Quantised: {new_size / 1024 / 1024:.1f} MB ({ratio:.0f}%)")

    return str(out_path)


def main() -> None:
    args = parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: model not found at {args.model}")
        sys.exit(1)

    for n_bits in args.bits:
        quantise(args.model, n_bits)

    print("\nDone.")


if __name__ == "__main__":
    main()
