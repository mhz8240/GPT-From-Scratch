# Model Card: GPT From Scratch

## Model Details

- Architecture: GPT-2-style decoder-only Transformer
- Implementation: From-scratch PyTorch modules for embeddings, causal multi-head attention, Transformer blocks, language-modeling head, and autoregressive generation
- Main checkpoint: `checkpoints/tiny/best.pt`
- Parameters: 1,334,016

## Intended Use

This project is intended for education, portfolio review, small-scale language modeling experiments, and local demonstrations of Transformer internals.

## Limitations

The model is not trained at full GPT-2 scale and should not be expected to follow instructions, answer factual questions reliably, or behave like a production assistant. Outputs can be repetitive, incoherent, or biased depending on training data.

## Dataset

The main checkpoint was trained on a TinyStories sample prepared by `scripts/prepare_dataset.py --config configs/tiny.yaml`.

| Split | Tokens |
|---|---:|
| Train | 1,052,851 |
| Validation | 102,413 |

## Training Details

Training uses AdamW, cosine learning rate decay, warmup, gradient clipping, validation perplexity, checkpointing, and optional W&B logging.

## Evaluation

Primary quantitative metrics are validation cross-entropy loss and perplexity.

| Metric | Value |
|---|---:|
| Validation loss | 2.712 |
| Validation perplexity | 15.06 |

## Ethical Limitations

Even compact language models can reproduce undesirable patterns from data. This project should not be used for high-stakes or user-facing automated decisions.

## Example Generations

| Prompt | Checkpoint | Generation |
|---|---|---|
| Once upon a time | TBD | TBD |
