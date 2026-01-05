.PHONY: help install setup-dev lint format typecheck test clean data preprocess train tune serve

# Default number of synthetic rows
ROWS ?= 100000

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help:  ## Display this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${YELLOW}%-20s${NC} %s\n", $$1, $$2}'

install:  ## Install package in development mode
	@echo "${GREEN}Installing package in development mode...${NC}"
	pip install -e .[dev]

setup-dev: install  ## Set up development environment
	@echo "${GREEN}Setting up development environment...${NC}"
	pre-commit install

data:  ## Generate synthetic data
	@echo "${GREEN}Generating synthetic data...${NC}"
	python scripts/generate_synthetic_data.py --rows $(ROWS)

preprocess:  ## Preprocess data
	@echo "${GREEN}Preprocessing data...${NC}"
	python -m src.data.preprocessing

train:  ## Train model with default configuration
	@echo "${GREEN}Training model...${NC}"
	python scripts/train_model.py --config configs/training/base.yaml

tune:  ## Run hyperparameter tuning
	@echo "${GREEN}Running hyperparameter tuning...${NC}"
	python -m src.training.pipelines.tune_pipeline --config configs/training/tune.yaml

serve:  ## Start API server
	@echo "${GREEN}Starting API server...${NC}"
	python scripts/serve_local.py

lint:  ## Run linting with ruff
	@echo "${GREEN}Running linting...${NC}"
	ruff check src tests scripts
	ruff format --check src tests scripts

format:  ## Format code with black and ruff
	@echo "${GREEN}Formatting code...${NC}"
	black src tests scripts
	ruff check --fix src tests scripts
	ruff format src tests scripts

typecheck:  ## Run type checking with mypy
	@echo "${GREEN}Running type checking...${NC}"
	mypy src

test:  ## Run tests with coverage
	@echo "${GREEN}Running tests...${NC}"
	pytest tests/ -v --cov=src --cov-report=term-missing

test-ci:  ## Run tests for CI
	@echo "${GREEN}Running CI tests...${NC}"
	pytest tests/ -v --cov=src --cov-report=xml --junitxml=junit.xml

clean:  ## Clean temporary files and caches
	@echo "${GREEN}Cleaning up...${NC}"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true

all-checks: lint typecheck test  ## Run all checks (lint, typecheck, test)

pipeline: data preprocess train  ## Run full ML pipeline