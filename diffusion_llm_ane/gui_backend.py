"""
Backend abstraction for the RinGo-DLLM visual sampler GUI.

Provides a unified interface for CoreML (ANE) and PyTorch (MPS) inference,
plus a streaming reverse-diffusion generator that yields intermediate states.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import torch

from data.tokenizer import get_tokenizer
from sample import sample_token, unmask_step

# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    n_steps: int = 25
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    seed: int | None = None


# ---------------------------------------------------------------------------
# Model configuration (matches ModelConfigLarge v3)
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Minimal config needed for inference — avoids importing training deps."""
    vocab_size: int = 30_522
    max_seq_len: int = 128
    mask_token_id: int = 103
    pad_token_id: int = 0
    T: int = 25
    tokenizer_name: str = "bert-base-uncased"


# ---------------------------------------------------------------------------
# Backend base class
# ---------------------------------------------------------------------------

class BaseBackend:
    """Abstract inference backend."""

    config: InferenceConfig

    def predict(self, input_ids: np.ndarray, t: int) -> torch.Tensor:
        """Run a single forward pass.

        Args:
            input_ids: (1, L) int32 array of token IDs.
            t:         scalar diffusion timestep.

        Returns:
            logits: (1, L, V) float tensor.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CoreML backend (ANE)
# ---------------------------------------------------------------------------

class CoreMLBackend(BaseBackend):
    """CoreML inference via Apple Neural Engine."""

    def __init__(self, mlpackage_path: str) -> None:
        import coremltools as ct

        self.model = ct.models.MLModel(
            mlpackage_path,
            compute_units=ct.ComputeUnit.ALL,
        )
        self.config = InferenceConfig()
        self.name = Path(mlpackage_path).stem

    def predict(self, input_ids: np.ndarray, t: int) -> torch.Tensor:
        result = self.model.predict({
            "input_ids": input_ids.astype(np.int32),
            "t": np.array([t], dtype=np.int32),
        })
        logits_np = result["logits"]  # (1, L, V)
        return torch.from_numpy(logits_np).float()


# ---------------------------------------------------------------------------
# PyTorch backend (MPS / CPU)
# ---------------------------------------------------------------------------

class PyTorchBackend(BaseBackend):
    """PyTorch inference on MPS (Apple GPU) or CPU."""

    def __init__(self, checkpoint_path: str) -> None:
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        ckpt = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self._pt_config = ckpt["config"]
        self.config = InferenceConfig(
            vocab_size=self._pt_config.vocab_size,
            max_seq_len=self._pt_config.max_seq_len,
            mask_token_id=self._pt_config.mask_token_id,
            pad_token_id=self._pt_config.pad_token_id,
            T=self._pt_config.T,
            tokenizer_name=getattr(
                self._pt_config, "tokenizer_name", "bert-base-uncased",
            ),
        )

        from model.diffusion_lm import DiffusionLM

        self.model = DiffusionLM(self._pt_config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.name = "PyTorch-MPS" if self.device.type == "mps" else "PyTorch-CPU"

    @torch.no_grad()
    def predict(self, input_ids: np.ndarray, t: int) -> torch.Tensor:
        ids = torch.from_numpy(input_ids).long().to(self.device)
        t_tensor = torch.tensor([t], dtype=torch.long, device=self.device)
        logits = self.model(ids, t_tensor)
        return logits.cpu().float()


# ---------------------------------------------------------------------------
# Tokeniser helpers
# ---------------------------------------------------------------------------

def get_shared_tokeniser(tokenizer_name: str = "bert-base-uncased"):
    """Return a shared tokeniser instance (cached per name in data.tokenizer)."""
    return get_tokenizer(tokenizer_name)


# ---------------------------------------------------------------------------
# Streaming reverse-diffusion generator
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """One denoising step's output."""
    step: int
    n_steps: int
    tokens: list[int]
    mask_remaining: int
    total_mask: int
    newly_revealed: set[int]      # positions revealed *this* step
    prompt_positions: set[int]    # positions that are part of the user prompt
    step_ms: float
    total_ms: float


def reverse_diffusion_stream(
    backend: BaseBackend,
    params: SamplingParams,
    prompt_text: str | None = None,
) -> Generator[StepResult, None, None]:
    """Yield one StepResult per denoising step.

    This is the core streaming generator consumed by the Gradio UI.
    """
    cfg = backend.config
    tokeniser = get_shared_tokeniser(cfg.tokenizer_name)
    L = cfg.max_seq_len

    if params.seed is not None:
        torch.manual_seed(params.seed)

    # ── Build initial sequence ───────────────────────────────────────────
    if prompt_text:
        encoded = tokeniser.encode(prompt_text, add_special_tokens=False)
        x_t = torch.tensor(encoded, dtype=torch.long)
        if len(x_t) < L:
            pad = torch.full(
                (L - len(x_t),), cfg.pad_token_id, dtype=torch.long
            )
            x_t = torch.cat([x_t, pad])
        else:
            x_t = x_t[:L]
    else:
        x_t = torch.full((L,), cfg.mask_token_id, dtype=torch.long)

    # Track which positions are user-provided prompt tokens
    prompt_positions: set[int] = set()
    for i in range(L):
        tid = x_t[i].item()
        if tid != cfg.mask_token_id and tid != cfg.pad_token_id:
            prompt_positions.add(i)

    total_mask = int((x_t == cfg.mask_token_id).sum().item())
    if total_mask == 0:
        yield StepResult(
            step=0, n_steps=0, tokens=x_t.tolist(),
            mask_remaining=0, total_mask=0,
            newly_revealed=set(), prompt_positions=prompt_positions,
            step_ms=0.0, total_ms=0.0,
        )
        return

    initial_unmasked = int((x_t != cfg.mask_token_id).sum().item())
    n_steps = params.n_steps
    total_ms = 0.0

    # Yield initial state (all masked)
    yield StepResult(
        step=0, n_steps=n_steps, tokens=x_t.tolist(),
        mask_remaining=total_mask, total_mask=total_mask,
        newly_revealed=set(), prompt_positions=prompt_positions,
        step_ms=0.0, total_ms=0.0,
    )

    # ── Denoising loop ───────────────────────────────────────────────────
    for step in range(n_steps, 0, -1):
        prev_tokens = set(
            i for i in range(L) if x_t[i].item() != cfg.mask_token_id
        )

        t_val = int(step * cfg.T / n_steps)
        input_ids = x_t.unsqueeze(0).numpy().astype(np.int32)

        t0 = time.perf_counter()
        logits = backend.predict(input_ids, t_val)  # (1, L, V)
        step_ms = (time.perf_counter() - t0) * 1000.0
        total_ms += step_ms

        logits = logits.squeeze(0)  # (L, V)

        # Compute how many tokens should be unmasked after this step
        fraction_done = 1.0 - (step - 1) / n_steps
        target_unmasked = initial_unmasked + round(total_mask * fraction_done)

        x_t = unmask_step(
            x_t, logits, target_unmasked, cfg.mask_token_id,
            params.temperature, params.top_k, params.top_p,
            params.repetition_penalty,
        )

        # Detect newly revealed positions
        current_tokens = set(
            i for i in range(L) if x_t[i].item() != cfg.mask_token_id
        )
        newly_revealed = current_tokens - prev_tokens

        mask_remaining = int((x_t == cfg.mask_token_id).sum().item())
        current_step = n_steps - step + 1

        yield StepResult(
            step=current_step,
            n_steps=n_steps,
            tokens=x_t.tolist(),
            mask_remaining=mask_remaining,
            total_mask=total_mask,
            newly_revealed=newly_revealed,
            prompt_positions=prompt_positions,
            step_ms=step_ms,
            total_ms=total_ms,
        )
