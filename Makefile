.PHONY: setup test lint format prepare train sample count prepare-debug train-debug sample-debug count-debug demo

setup:
	python -m pip install -U pip
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	python -m pytest -q

lint:
	ruff check .

format:
	ruff format .

prepare:
	python scripts/prepare_dataset.py --config configs/tiny.yaml

train:
	python scripts/train_model.py --config configs/tiny.yaml

sample:
	python scripts/sample.py --checkpoint checkpoints/tiny/best.pt --prompt "Once upon a time" --max-new-tokens 120

count:
	python scripts/count_params.py --config configs/tiny.yaml

prepare-debug:
	python scripts/prepare_dataset.py --config configs/debug.yaml

train-debug:
	python scripts/train_model.py --config configs/debug.yaml

sample-debug:
	python scripts/sample.py --checkpoint checkpoints/debug/best.pt --prompt "Once upon a time" --max-new-tokens 60

count-debug:
	python scripts/count_params.py --config configs/debug.yaml

demo:
	python app/app.py
