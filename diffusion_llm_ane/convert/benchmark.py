"""
Phase 4: ANE vs GPU vs CPU latency benchmark.

Measures the per-step inference latency of a converted `.mlpackage` under
different CoreML compute units and reports a comparison table.

Usage:
    python convert/benchmark.py \\
        --model diffusion_lm.mlpackage \\
        --steps 10 20 50 \\
        --warmup 5 \\
        --runs 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import coremltools as ct
except ImportError:
    print("ERROR: coremltools not installed.  Run:  pip install coremltools>=8.0")
    sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark CoreML model on ANE / GPU / CPU")
    p.add_argument("--model",   type=str, required=True,   help="Path to .mlpackage")
    p.add_argument("--steps",   type=int, nargs="+", default=[10, 20, 50],
                   help="Diffusion step counts to benchmark")
    p.add_argument("--warmup",  type=int, default=5,  help="Warmup iterations (not counted)")
    p.add_argument("--runs",    type=int, default=20, help="Measured iterations per config")
    p.add_argument("--seq-len", type=int, default=128)
    return p.parse_args()


# ── Benchmark helpers ─────────────────────────────────────────────────────────

COMPUTE_UNITS = {
    "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "ALL":         ct.ComputeUnit.ALL,
}


def measure_latency(
    model_path: str,
    compute_unit_name: str,
    compute_unit,
    seq_len: int,
    n_diffusion_steps: int,
    warmup: int,
    runs: int,
) -> dict:
    """
    Load the model under the specified compute unit and measure per-step latency.

    Returns a dict with keys: compute_unit, diffusion_steps, mean_ms, std_ms, total_ms
    """
    print(f"  Loading model with {compute_unit_name} …", end=" ", flush=True)
    mlmodel = ct.models.MLModel(model_path, compute_units=compute_unit)
    print("done")

    # Build a fixed input (fully masked sequence, t=50)
    input_ids = np.zeros((1, seq_len), dtype=np.int32)   # all zeros (could be [MASK]=103)
    input_ids[:] = 103

    step_times_ms: list[float] = []

    for iteration in range(warmup + runs):
        t_start = time.perf_counter()

        # Simulate one full reverse diffusion pass (n_diffusion_steps forward calls)
        for step in range(n_diffusion_steps, 0, -1):
            t_val = np.array([step], dtype=np.int32)
            _ = mlmodel.predict({"input_ids": input_ids, "t": t_val})

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if iteration >= warmup:
            step_times_ms.append(elapsed_ms / n_diffusion_steps)

    mean_ms  = float(np.mean(step_times_ms))
    std_ms   = float(np.std(step_times_ms))
    total_ms = mean_ms * n_diffusion_steps

    return {
        "compute_unit":    compute_unit_name,
        "diffusion_steps": n_diffusion_steps,
        "mean_ms":         mean_ms,
        "std_ms":          std_ms,
        "total_ms":        total_ms,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    header = (
        f"{'Compute unit':<16} {'Steps':>6} {'Per-step (ms)':>14} "
        f"{'Std dev (ms)':>13} {'Total (ms)':>11}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['compute_unit']:<16} {r['diffusion_steps']:>6} "
            f"{r['mean_ms']:>13.2f}  {r['std_ms']:>12.2f}  {r['total_ms']:>10.1f}"
        )
    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    print(f"Benchmarking: {model_path}")
    print(f"  Warmup iterations : {args.warmup}")
    print(f"  Measured runs     : {args.runs}")
    print(f"  Sequence length   : {args.seq_len}")
    print(f"  Diffusion steps   : {args.steps}")
    print()

    results: list[dict] = []

    for unit_name, unit in COMPUTE_UNITS.items():
        for n_steps in args.steps:
            print(f"[{unit_name}] T={n_steps}")
            try:
                r = measure_latency(
                    model_path=str(model_path),
                    compute_unit_name=unit_name,
                    compute_unit=unit,
                    seq_len=args.seq_len,
                    n_diffusion_steps=n_steps,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                results.append(r)
                print(
                    f"    per-step: {r['mean_ms']:.2f} ± {r['std_ms']:.2f} ms  "
                    f"| total: {r['total_ms']:.1f} ms"
                )
            except Exception as exc:
                print(f"    FAILED: {exc}")

    print_table(results)

    # Highlight winner
    if results:
        best = min(results, key=lambda r: r["mean_ms"])
        print(
            f"Fastest: {best['compute_unit']} at T={best['diffusion_steps']} — "
            f"{best['mean_ms']:.2f} ms/step"
        )


if __name__ == "__main__":
    main()
