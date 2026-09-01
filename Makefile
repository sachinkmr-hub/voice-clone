PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PORT ?= 8000

.PHONY: help venv install demo-data train evaluate run run-dev test lint fmt frontend-install frontend-dev frontend-build docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv
	$(PY) -m venv $(VENV)

install: venv ## Install backend dependencies
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r backend/requirements.txt

demo-data: ## Build the bootstrap bona-fide vs synthetic corpus
	$(BIN)/python -m ml.datasets.build --out data/corpus --per-class 120

train: ## Train the detector on data/corpus
	$(BIN)/python -m ml.train --corpus data/corpus --out ml/artifacts

evaluate: ## Evaluate the current model (Accuracy / AUC / EER)
	$(BIN)/python -m ml.evaluate --corpus data/corpus --model ml/artifacts/bootstrap_model.joblib

run: ## Run the API + console
	$(BIN)/uvicorn voiceguard.api.app:app --app-dir backend --host 0.0.0.0 --port $(PORT)

run-dev: ## Run with autoreload
	$(BIN)/uvicorn voiceguard.api.app:app --app-dir backend --host 0.0.0.0 --port $(PORT) --reload

test: ## Run the test suite
	$(BIN)/python -m pytest backend/tests -q

lint: ## Static checks
	$(BIN)/python -m compileall -q backend/voiceguard ml

frontend-install: ## npm install for the dashboard
	cd frontend && npm install

frontend-dev: ## Vite dev server
	cd frontend && npm run dev

frontend-build: ## Production build of the dashboard
	cd frontend && npm run build

docker: ## Build and run the whole stack
	docker compose -f deploy/docker-compose.yml up --build

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ frontend/dist frontend/node_modules
