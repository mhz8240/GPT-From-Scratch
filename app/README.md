# GPT From Scratch Demo

Run locally:

```bash
python app/app.py
```

You can also use:

```bash
gradio app/app.py
```

By default, the app loads:

```text
checkpoints/tiny/best.pt
```

You can override that for local experiments:

```bash
MINIGPT_CHECKPOINT=checkpoints/debug/best.pt python app/app.py
```

The demo is designed for the compact GPT trained in this repo and can later be adapted for Hugging Face Spaces by including the trained checkpoint artifact and installing `app/requirements.txt`.
