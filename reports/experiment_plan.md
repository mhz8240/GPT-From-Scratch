# Experiment Plan

This project focuses on one compact GPT trained from scratch. The goal is not to compare many model sizes; it is to demonstrate that the full GPT pipeline works end to end: tokenizer, dataset preparation, handwritten decoder-only Transformer, training loop, checkpointing, evaluation, sampling, tests, and demo.

## Main Run

| Run | Layers | Heads | Embedding | Context | Dataset | Target Use |
|---|---:|---:|---:|---:|---|---|
| tiny | 4 | 4 | 128 | 128 | TinyStories sample | Main trained GPT checkpoint |

## Smoke Run

| Run | Layers | Heads | Embedding | Context | Dataset | Target Use |
|---|---:|---:|---:|---:|---|---|
| debug | 2 | 2 | 64 | 64 | Local generated text | Fast correctness checks |

## Metrics To Collect

- Training loss
- Validation loss
- Validation perplexity
- Tokens per second
- Number of train and validation tokens
- Hardware
- Qualitative samples from fixed prompts

## Current Results

| Model | Params | Train Tokens | Val Tokens | Val Loss | Val Perplexity | Hardware |
|---|---:|---:|---:|---:|---:|---|
| tiny GPT | 1,334,016 | 1,052,851 | 102,413 | 2.712 | 15.06 | MPS |
