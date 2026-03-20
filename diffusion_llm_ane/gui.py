"""
RinGo-DLLM Visual Sampler — Gradio GUI for Masked Diffusion inference.

Displays the reverse-diffusion (denoising) process in real time,
with colour-coded tokens showing how [MASK]s are progressively revealed.

Launch:
    cd diffusion_llm_ane
    python gui.py
    # Opens http://localhost:7860 in your browser
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from gui_backend import (
    CoreMLBackend,
    PyTorchBackend,
    SamplingParams,
    StepResult,
    get_shared_tokeniser,
    reverse_diffusion_stream,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent
_CONVERT_DIR = _BASE_DIR / "convert"
_CHECKPOINT = _BASE_DIR / "checkpoints_wt103_v3" / "best_model.pt"

# Available backends (label → loader)
_BACKEND_OPTIONS = {
    "CoreML ANE (FP16)": lambda: CoreMLBackend(
        str(_CONVERT_DIR / "diffusion_lm_ane_v3.mlpackage")
    ),
    "CoreML ANE (INT4)": lambda: CoreMLBackend(
        str(_CONVERT_DIR / "diffusion_lm_ane_v3_int4.mlpackage")
    ),
    "CoreML ANE (INT8)": lambda: CoreMLBackend(
        str(_CONVERT_DIR / "diffusion_lm_ane_v3_int8.mlpackage")
    ),
    "PyTorch MPS": lambda: PyTorchBackend(str(_CHECKPOINT)),
}

# Cache for loaded backends
_backend_cache: dict[str, object] = {}


def _get_backend(name: str):
    """Load a backend (cached after first use)."""
    if name not in _backend_cache:
        _backend_cache[name] = _BACKEND_OPTIONS[name]()
    return _backend_cache[name]


# ---------------------------------------------------------------------------
# Token rendering
# ---------------------------------------------------------------------------

def _render_highlighted(
    result: StepResult,
    tokeniser,
) -> list[tuple[str, str | None]]:
    """Build highlighted-text data from a StepResult.

    Returns list of (text, label) tuples for gr.HighlightedText:
        label=None   → default colour (previously revealed)
        label="mask" → grey (still masked)
        label="new"  → green (just revealed this step)
        label="prompt" → blue (user-provided prompt token)
    """
    cfg_mask = 103   # [MASK] token ID
    cfg_pad = 0      # [PAD] token ID
    segments: list[tuple[str, str | None]] = []

    for i, tid in enumerate(result.tokens):
        if tid == cfg_pad:
            continue
        if tid == cfg_mask:
            segments.append((" _ ", "mask"))
        elif i in result.newly_revealed:
            text = tokeniser.decode([tid])
            segments.append((f" {text} ", "new"))
        elif i in result.prompt_positions:
            text = tokeniser.decode([tid])
            segments.append((f" {text} ", "prompt"))
        else:
            text = tokeniser.decode([tid])
            segments.append((f" {text} ", None))

    return segments


# ---------------------------------------------------------------------------
# Streaming generation handler
# ---------------------------------------------------------------------------

def generate_streaming(
    prompt: str,
    backend_name: str,
    steps: int,
    temperature: float,
    top_p: float,
    top_k: int,
    rep_penalty: float,
    seed: int | None,
):
    """Gradio event handler — yields updates for each denoising step."""
    tokeniser = get_shared_tokeniser()
    backend = _get_backend(backend_name)

    params = SamplingParams(
        n_steps=int(steps),
        temperature=temperature,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=rep_penalty,
        seed=int(seed) if seed is not None and seed >= 0 else None,
    )

    prompt_text = prompt.strip() if prompt and prompt.strip() else None

    for result in reverse_diffusion_stream(backend, params, prompt_text):
        highlighted = _render_highlighted(result, tokeniser)

        # Status line
        if result.step == 0:
            status = f"Step 0/{result.n_steps} | mask {result.mask_remaining}/{result.total_mask} | Waiting..."
        elif result.step < result.n_steps:
            avg_ms = result.total_ms / result.step
            status = (
                f"Step {result.step}/{result.n_steps} | "
                f"mask {result.mask_remaining}/{result.total_mask} | "
                f"{result.step_ms:.1f} ms/step (avg {avg_ms:.1f} ms)"
            )
        else:
            # Final step
            avg_ms = result.total_ms / result.n_steps
            status = (
                f"Done! {result.n_steps} steps in {result.total_ms:.1f} ms "
                f"(avg {avg_ms:.1f} ms/step) | Backend: {backend_name}"
            )

        # Final decoded text
        final_text = ""
        if result.mask_remaining == 0:
            clean_tokens = [
                t for t in result.tokens if t != 0  # skip [PAD]
            ]
            final_text = tokeniser.decode(clean_tokens, skip_special_tokens=True)

        yield highlighted, status, final_text


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""

    colour_map = {
        "mask": "#6b7280",    # grey
        "new": "#22c55e",     # green
        "prompt": "#3b82f6",  # blue
    }

    with gr.Blocks(title="RinGo-DLLM Sampler") as demo:
        gr.Markdown("# RinGo-DLLM — Masked Diffusion Language Model Sampler")
        gr.Markdown(
            "55M parameter model trained on WikiText-103. "
            "Watch the reverse-diffusion process unmask tokens step by step!"
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder=(
                        'Use [MASK] for positions to generate, e.g.:\n'
                        '"The [MASK] of [MASK] was discovered in [MASK] ."\n'
                        'Leave empty for fully unconditional generation.'
                    ),
                    lines=2,
                )
            with gr.Column(scale=1):
                backend_radio = gr.Radio(
                    choices=list(_BACKEND_OPTIONS.keys()),
                    value="CoreML ANE (FP16)",
                    label="Backend",
                )

        with gr.Row():
            steps_slider = gr.Slider(
                minimum=1, maximum=50, value=25, step=1, label="Steps"
            )
            temp_slider = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                label="Temperature",
            )
            top_p_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                label="Top-p (nucleus)",
            )

        with gr.Row():
            top_k_slider = gr.Slider(
                minimum=0, maximum=100, value=0, step=1, label="Top-k (0=off)"
            )
            rep_penalty_slider = gr.Slider(
                minimum=1.0, maximum=2.0, value=1.2, step=0.05,
                label="Repetition penalty",
            )
            seed_input = gr.Number(
                value=-1, label="Seed (-1 = random)", precision=0
            )

        generate_btn = gr.Button("Generate", variant="primary", size="lg")

        gr.Markdown("### Denoising Process")
        highlighted_output = gr.HighlightedText(
            label="Token stream",
            color_map=colour_map,
            show_legend=True,
        )

        status_text = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Final Output")
        final_output = gr.Textbox(label="Generated text", interactive=False)

        # ── Wire up the streaming generation ─────────────────────────────
        generate_btn.click(
            fn=generate_streaming,
            inputs=[
                prompt_input,
                backend_radio,
                steps_slider,
                temp_slider,
                top_p_slider,
                top_k_slider,
                rep_penalty_slider,
                seed_input,
            ],
            outputs=[highlighted_output, status_text, final_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())
