# Training Curves

Training writes metrics to `reports/runs/{run_name}/metrics.json`.

Recommended plots:

- Training loss vs step
- Validation loss vs step
- Validation perplexity vs step
- Tokens/sec vs step
- Learning rate vs step

Example plotting snippet:

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt

metrics = json.loads(Path("reports/runs/tiny/metrics.json").read_text())
steps = [row["step"] for row in metrics]
train_loss = [row["train_loss"] for row in metrics]

plt.plot(steps, train_loss)
plt.xlabel("Step")
plt.ylabel("Training loss")
plt.title("GPT From Scratch Training Loss")
plt.savefig("reports/runs/tiny/train_loss.png", dpi=160)
```
