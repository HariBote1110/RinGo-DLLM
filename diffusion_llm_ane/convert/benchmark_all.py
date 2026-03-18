"""
Comprehensive benchmark: compare FP16, INT8, INT4 models across all compute units.

Usage:
    python convert/benchmark_all.py --dir convert/
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


COMPUTE_UNITS = {
    "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "ALL":         ct.ComputeUnit.ALL,
}

DIFFUSION_STEPS = 20   # Fixed for fair comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark all .mlpackage variants")
    p.add_argument("--dir",    type=str, default="convert/", help="Directory with .mlpackage files")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs",   type=int, default=30)
    p.add_argument("--steps",  type=int, default=DIFFUSION_STEPS)
    return p.parse_args()


def measure(model_path: str, compute_unit, n_steps: int, warmup: int, runs: int) -> float:
    """Returns mean per-step latency in ms."""
    mlmodel = ct.models.MLModel(model_path, compute_units=compute_unit)
    input_ids = np.full((1, 128), 103, dtype=np.int32)   # all [MASK]

    step_times: list[float] = []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        for step in range(n_steps, 0, -1):
            t_val = np.array([step], dtype=np.int32)
            _ = mlmodel.predict({"input_ids": input_ids, "t": t_val})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            step_times.append(elapsed_ms / n_steps)

    return float(np.mean(step_times))


def main() -> None:
    args = parse_args()
    pkg_dir = Path(args.dir)

    # Discover all .mlpackage variants
    models = sorted(pkg_dir.glob("*.mlpackage"))
    if not models:
        print(f"No .mlpackage files found in {pkg_dir}")
        sys.exit(1)

    print(f"Found {len(models)} model(s):")
    for m in models:
        size_mb = sum(f.stat().st_size for f in m.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"  {m.name:40s}  {size_mb:.1f} MB")
    print(f"\nDiffusion steps: {args.steps}, Warmup: {args.warmup}, Runs: {args.runs}\n")

    results: list[dict] = []

    for model_path in models:
        for unit_name, unit in COMPUTE_UNITS.items():
            label = f"{model_path.name} / {unit_name}"
            print(f"  {label:55s}", end=" ", flush=True)
            try:
                mean_ms = measure(str(model_path), unit, args.steps, args.warmup, args.runs)
                total_ms = mean_ms * args.steps
                results.append({
                    "model": model_path.name,
                    "unit": unit_name,
                    "per_step_ms": mean_ms,
                    "total_ms": total_ms,
                })
                print(f"{mean_ms:6.2f} ms/step  ({total_ms:6.1f} ms total)")
            except Exception as exc:
                print(f"FAILED: {exc}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<40s} {'Compute Unit':<16s} {'ms/step':>8s} {'Total (ms)':>11s}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['model']:<40s} {r['unit']:<16s} {r['per_step_ms']:>8.2f} {r['total_ms']:>11.1f}")
    print(f"{'='*80}")

    if results:
        best = min(results, key=lambda r: r["per_step_ms"])
        print(
            f"\n🏆 Fastest: {best['model']} / {best['unit']} — "
            f"{best['per_step_ms']:.2f} ms/step"
        )


if __name__ == "__main__":
    main()
