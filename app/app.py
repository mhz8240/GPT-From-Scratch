from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minigpt.generate import generate_text, load_model_for_generation  # noqa: E402
from minigpt.utils import count_parameters, get_device  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/tiny/best.pt"


def run_generation(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    stop_at_eos: bool,
) -> tuple[str, str]:
    checkpoint = os.getenv("MINIGPT_CHECKPOINT", DEFAULT_CHECKPOINT)
    if not checkpoint:
        return "", "Set MINIGPT_CHECKPOINT before generating."
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        checkpoint_path = ROOT / checkpoint
    if not checkpoint_path.exists():
        return "", (
            f"Checkpoint not found: {checkpoint}\n"
            "Train the tiny GPT first with: python scripts/train_model.py --config configs/tiny.yaml"
        )
    try:
        model, tokenizer, payload = load_model_for_generation(checkpoint_path, device=get_device())
        text = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            seed=int(seed),
            stop_at_eos=stop_at_eos,
        )
        meta = (
            f"params: {count_parameters(model):,}\n"
            f"device: {next(model.parameters()).device}\n"
            f"checkpoint: {checkpoint_path}\n"
            f"step: {payload.get('step', 'unknown')}\n"
            f"stop_when_story_ends: {stop_at_eos}"
        )
        return text, meta
    except Exception as exc:
        return "", f"Could not load or run checkpoint: {exc}"


with gr.Blocks(title="GPT From Scratch") as demo:
    gr.Markdown("# GPT From Scratch")
    prompt = gr.Textbox(label="Prompt", value="Once upon a time", lines=4)
    with gr.Row():
        max_new_tokens = gr.Slider(1, 512, value=120, step=1, label="Max new tokens")
        temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
    with gr.Row():
        top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
        top_p = gr.Slider(0.05, 1.0, value=0.95, step=0.01, label="Top-p")
        seed = gr.Number(value=1337, precision=0, label="Seed")
        stop_at_eos = gr.Checkbox(value=True, label="Stop when the story ends")
    generate_button = gr.Button("Generate")
    output = gr.Textbox(label="Generated text", lines=10)
    metadata = gr.Textbox(label="Metadata", lines=4)
    generate_button.click(
        run_generation,
        inputs=[prompt, max_new_tokens, temperature, top_k, top_p, seed, stop_at_eos],
        outputs=[output, metadata],
    )


if __name__ == "__main__":
    demo.launch()
