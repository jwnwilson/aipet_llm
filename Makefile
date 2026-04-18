PYTHON := python3
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

CONFIG ?= configs/sft.yaml
CONFIG_LOW ?= configs/low_latency.yaml
TRAIN_FILE ?= data/processed/train.jsonl
VAL_FILE ?= data/processed/val.jsonl
MODEL_DIR ?= models/aipet-sft/final
CONTEXT_FILE ?= data/raw/example_context.json
HOST ?= 0.0.0.0
PORT ?= 8000
API_URL ?= http://127.0.0.1:$(PORT)

.PHONY: help venv install setup validate prepare train infer eval runtime serve serve-low-latency infer-low-latency eval-low-latency request-example

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
	@echo "  make serve-low-latency - API server with configs/low_latency.yaml (faster CPU)"
	@echo "  make infer-low-latency - CLI infer with low-latency config"
	@echo "  make eval-low-latency - eval metrics with low-latency config"
	@echo "  make request-example - POST example context to local API"
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
	AIPET_CONFIG_PATH=$(CONFIG) AIPET_MODEL_DIR=$(MODEL_DIR) $(VENV_PY) -m uvicorn aipet_distill.api.server:app --host $(HOST) --port $(PORT)

serve-low-latency:
	AIPET_CONFIG_PATH=$(CONFIG_LOW) AIPET_MODEL_DIR=$(MODEL_DIR) $(VENV_PY) -m uvicorn aipet_distill.api.server:app --host $(HOST) --port $(PORT)

infer-low-latency:
	$(VENV_PY) scripts/infer.py \
		--config $(CONFIG_LOW) \
		--model-dir $(MODEL_DIR) \
		--context $(CONTEXT_FILE)

eval-low-latency:
	$(VENV_PY) scripts/eval_infer.py \
		--config $(CONFIG_LOW) \
		--model-dir $(MODEL_DIR) \
		--input $(VAL_FILE)

request-example:
	curl -sS -X POST "$(API_URL)/v1/pet/turn" \
		-H "Content-Type: application/json" \
		-d "{\"context\":$$(cat $(CONTEXT_FILE))}" | $(VENV_PY) -m json.tool
