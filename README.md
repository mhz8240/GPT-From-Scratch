# GPT From Scratch

GPT From Scratch is a portfolio ML engineering project that implements a GPT-2-style decoder-only Transformer in PyTorch and trains a compact language model on TinyStories. The goal is to show the full path from notebook exploration to a working language model: manual causal self-attention, tokenizer training, data preparation, training, checkpointing, evaluation, sampling, tests, and a local Gradio demo.

The core Transformer is handwritten PyTorch. This project does not use `transformers.GPT2LMHeadModel`, `AutoModelForCausalLM`, `Trainer`, `nn.TransformerEncoder`, or `nn.MultiheadAttention` for the model.

## What Is Implemented

- Token embeddings and learned positional embeddings
- Manual causal multi-head self-attention with a lower-triangular mask
- GPT-style Transformer blocks with LayerNorm, GELU MLPs, residual connections, dropout, and optional weight tying
- Cross-entropy next-token language modeling loss
- BPE tokenizer training with the Hugging Face `tokenizers` library
- TinyStories and local-text data preparation
- AdamW training with warmup, cosine decay, gradient accumulation, clipping, validation, perplexity, checkpointing, and resume support
- Autoregressive generation with temperature, top-k, top-p, and repetition penalty
- Evaluation, parameter counting, and saved sample generations
- Pytest coverage for attention masks, model shapes/loss, tokenizer round trips, checkpointing, sampling, generation, and tiny-batch overfitting
- A lightweight Gradio app that loads the trained GPT checkpoint

## Current Model

The main model for this project is the trained `tiny` GPT checkpoint:

```text
checkpoints/tiny/best.pt
```

| Model | Params | Dataset | Train Tokens | Val Tokens | Context | Val Loss | Val Perplexity | Hardware |
|---|---:|---|---:|---:|---:|---:|---:|---|
| tiny GPT | 1,334,016 | TinyStories sample | 1,052,851 | 102,413 | 128 | 2.712 | 15.06 | MPS |

The `debug` config is kept only as a fast smoke-test path. The recruiter-facing artifact is the `tiny` checkpoint.

## Project Structure

```text
gpt-from-scratch/
├── configs/                 # tiny training config plus debug smoke config
├── data/                    # Tokenized datasets
├── checkpoints/             # Trained checkpoints
├── notebooks/               # Original notebook plus architecture walkthrough
├── src/minigpt/             # From-scratch GPT package
├── scripts/                 # CLI entry points
├── app/                     # Gradio demo
├── reports/                 # Metrics, samples, training notes, model card
├── tests/                   # Lightweight pytest suite
└── .github/workflows/       # CI
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Reproduce The Tiny GPT

Prepare the TinyStories sample and tokenizer:

```bash
python scripts/prepare_dataset.py --config configs/tiny.yaml
```

Train the model:

```bash
python scripts/train_model.py --config configs/tiny.yaml
```

Resume training:

```bash
python scripts/train_model.py --config configs/tiny.yaml --resume checkpoints/tiny/latest.pt
```

Evaluate:

```bash
python scripts/run_eval.py --checkpoint checkpoints/tiny/best.pt --config configs/tiny.yaml
```

Sample:

```bash
python scripts/sample.py \
  --checkpoint checkpoints/tiny/best.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 50
```

Run the demo:

```bash
python app/app.py
```

The app loads `checkpoints/tiny/best.pt` by default. You can override it for local experiments with:

```bash
MINIGPT_CHECKPOINT=checkpoints/debug/best.pt python app/app.py
```

## Smoke Tests

The debug path is intentionally tiny and exists to verify the code quickly:

```bash
make prepare-debug
make train-debug
make sample-debug
python scripts/count_params.py --config configs/debug.yaml
python scripts/sanity_overfit_batch.py --config configs/debug.yaml
```

Run the test suite:

```bash
make test
```

## Training Artifacts

Useful files after training:

```text
reports/runs/tiny/metrics.json
reports/runs/tiny/samples.txt
reports/evals/
checkpoints/tiny/best.pt
checkpoints/tiny/latest.pt
data/tiny/metadata.json
```

## Limitations

This is a compact educational GPT, not a full GPT-2-scale model and not an instruction-following assistant. It is trained on a small TinyStories sample, so generations can be repetitive, inconsistent, or overly simple. The point of the project is to demonstrate the engineering and modeling pipeline from scratch.

## Future Work

- Train longer on a larger text corpus
- Add richer evaluation beyond validation perplexity
- Add quantized inference for faster local demos
- Package the checkpoint for Hugging Face Spaces
- Add plots from `reports/runs/tiny/metrics.json`

