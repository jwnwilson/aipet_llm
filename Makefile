PYTHON := python3
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

CONFIG ?= configs/sft.yaml
TRAIN_FILE ?= data/processed/train.jsonl
VAL_FILE ?= data/processed/val.jsonl
MODEL_DIR ?= models/aipet-sft/final
CONTEXT_FILE ?= data/raw/example_context.json
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: help venv install setup validate prepare train infer eval runtime serve

help:
	@echo "Targets:"
	@echo "  make setup      - Create .venv and install project deps"
	@echo "  make validate   - Validate raw dataset JSONL"
	@echo "  make prepare    - Build train/val splits"
	@echo "  make train      - Train model using TRAIN_FILE/VAL_FILE"
	@echo "  make infer      - Run inference with MODEL_DIR and CONTEXT_FILE"
	@echo "  make eval       - Evaluate model on VAL_FILE"
	@echo "  make runtime    - Run runtime/fallback checks on VAL_FILE"
	@echo "  make serve      - Run local HTTP API server"
	@echo ""
	@echo "Common overrides:"
	@echo "  make train TRAIN_FILE=data/processed/train.jsonl VAL_FILE=data/processed/val.jsonl"
	@echo "  make infer MODEL_DIR=models/aipet-sft/final CONTEXT_FILE=data/raw/example_context.json"

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e .

setup: install

validate:
	$(VENV_PY) scripts/validate_dataset.py --input data/raw/example.jsonl

prepare:
	$(VENV_PY) scripts/prepare_dataset.py \
		--input data/raw/example.jsonl \
		--train-output $(TRAIN_FILE) \
		--val-output $(VAL_FILE) \
		--val-ratio 0.15

train:
	$(VENV_PY) scripts/train_sft.py \
		--config $(CONFIG) \
		--train-file $(TRAIN_FILE) \
		--val-file $(VAL_FILE)

infer:
	$(VENV_PY) scripts/infer.py \
		--config $(CONFIG) \
		--model-dir $(MODEL_DIR) \
		--context $(CONTEXT_FILE)

eval:
	$(VENV_PY) scripts/eval_infer.py \
		--config $(CONFIG) \
		--model-dir $(MODEL_DIR) \
		--input $(VAL_FILE)

runtime:
	$(VENV_PY) scripts/runtime_check.py \
		--model-dir $(MODEL_DIR) \
		--input $(VAL_FILE)

serve:
	AIPET_CONFIG_PATH=$(CONFIG) AIPET_MODEL_DIR=$(MODEL_DIR) $(VENV_PY) -m uvicorn scripts.serve_api:app --host $(HOST) --port $(PORT)
