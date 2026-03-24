# YOLO Toolchain - Makefile for uv

.PHONY: help install install-dev install-coreml install-wandb install-tensorboard sync lock \
	format lint test clean \
	run-train run-freeze-train run-incremental-train run-convert run-augment run-auto-annotate run-verify run-preprocess run-export run-diagnose run-download

help:
	@echo "YOLO Toolchain - Available commands:"
	@echo ""
	@echo "  Dependencies:"
	@echo "    make install          Install core dependencies"
	@echo "    make install-dev      Install with dev dependencies"
	@echo "    make install-coreml   Install CoreML support (macOS)"
	@echo "    make install-wandb    Install Weights & Biases"
	@echo "    make install-tensorboard Install TensorBoard"
	@echo "    make sync             Sync dependencies"
	@echo "    make lock             Generate lock file"
	@echo ""
	@echo "  Code quality:"
	@echo "    make format           Format code (black + isort)"
	@echo "    make lint             Run linting (flake8)"
	@echo "    make test             Run tests (pytest)"
	@echo ""
	@echo "  Run commands:"
	@echo "    make run-train           Normal training"
	@echo "    make run-freeze-train    Freeze backbone training"
	@echo "    make run-incremental-train  Incremental training"
	@echo "    make run-convert          Convert dataset format"
	@echo "    make run-augment         Augment dataset"
	@echo "    make run-auto-annotate   Auto annotate with AI (SiliconFlow Kimi-K2.5)"
	@echo "    make run-verify          Verify and edit annotations"
	@echo "    make run-preprocess      Batch image preprocessing"
	@echo "    make run-export           Export model"
	@echo "    make run-diagnose        Diagnostics analysis"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean              Clean build artifacts"

# Install dependencies
install:
	uv sync

install-dev:
	uv sync --extra dev

install-coreml:
	uv sync --extra coreml

install-wandb:
	uv sync --extra wandb

install-tensorboard:
	uv sync --extra tensorboard

sync:
	uv sync

lock:
	uv lock

# Code quality
format:
	uv run black src/
	uv run isort src/

lint:
	uv run flake8 src/ --max-line-length=100 --ignore=E203,W503

test:
	uv run pytest tests/ -v

# Run commands
run-train:
	@echo "Usage: uv run python -m src.train.trainer --data data.yaml --epochs 100 --resume"
	@uv run python -m src.train.trainer --help

run-freeze-train:
	@echo "Usage: uv run python -m src.train.freeze_trainer --data data.yaml --epochs 100"
	@uv run python -m src.train.freeze_trainer --help

run-incremental-train:
	@echo "Usage: uv run python -m src.train.incremental_trainer --model best.pt --data new_data.yaml"
	@uv run python -m src.train.incremental_trainer --help

run-convert:
	@echo "Usage: uv run python -m src.tools.dataset_builder --mode voc --input /path --output /path"
	@uv run python -m src.tools.dataset_builder --help

run-augment:
	@echo "Usage: uv run python -m src.tools.augmentor --input /path --output /path --num_augment 5"
	@uv run python -m src.tools.augmentor --help

run-auto-annotate:
	@echo "Usage: uv run python -m src.tools.auto_annotator --images /path --output /path --classes person car dog"
	@uv run python -m src.tools.auto_annotator --help

run-verify:
	@echo "Usage: uv run python -m src.tools.verify_annotator --images /path --labels /path --classes person cigarette"
	@uv run python -m src.tools.verify_annotator --help

run-preprocess:
	@echo "Usage: uv run python -m src.tools.preprocess --input /path --resize 640 480 --enhance"
	@uv run python -m src.tools.preprocess --help

run-export:
	@echo "Usage: uv run python -m src.export.exporter --model best.pt --format onnx"
	@uv run python -m src.export.exporter --help

run-diagnose:
	@echo "Usage: uv run python -m src.eval.diagnostics --model best.pt --data data.yaml"
	@uv run python -m src.eval.diagnostics --help

run-download:
	@echo "Usage: uv run python -m src.tools.downloader --model yolo11n --output models"
	@uv run python -m src.tools.downloader --help

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg 2>/dev/null || true
