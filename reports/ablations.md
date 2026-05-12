# Tuning Notes

This repo now centers on a single trained GPT checkpoint rather than a model-size scaling study. These tables are kept as lightweight future tuning notes.

## Sampling Strategy

| Strategy | Temperature | Top-k | Top-p | Repetition Penalty | Notes |
|---|---:|---:|---:|---:|---|
| Greedy | 0.0 | 0 | 1.0 | 1.0 | Deterministic, often repetitive |
| Top-k | 0.8 | 50 | 1.0 | 1.0 | Good default for quick samples |
| Nucleus | 0.8 | 0 | 0.95 | 1.0 | Useful for more varied samples |

## Training Schedule

| LR | Warmup | Min LR | Weight Decay | Val Loss | Notes |
|---:|---:|---:|---:|---:|---|
| 0.0006 | 200 | 0.00006 | 0.1 | 2.712 | Current tiny run |

## Future Experiments

| Experiment | Motivation | Status |
|---|---|---|
| Train longer | Improve story coherence and reduce repetition | Not started |
| Larger TinyStories sample | Improve data coverage | Not started |
| Sampling sweep | Compare deterministic and stochastic outputs | Not started |
| Plot training curves | Make metrics easier to inspect visually | Not started |
