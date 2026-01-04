# =========================
# RayScaleML Makefile
# =========================

# Default number of synthetic rows
ROWS ?= 1000000

# Virtual environment activation
VENV_ACTIVATE = source /opt/homebrew/Caskroom/miniforge/base/envs/ml_env/bin/activate
#$(HOME)/.virtualenvs/ml_env/bin/activate

# -------------------------
# 1. Generate synthetic data
# -------------------------
data:
	$(VENV_ACTIVATE) && python data/synthetic_generator.py --rows $(ROWS)

# -------------------------
# 2. Preprocess data
# -------------------------
preprocess:
	$(VENV_ACTIVATE) && python src/data/preprocessing.py
	$(VENV_ACTIVATE) && python src/data/feature_engineering.py

# -------------------------
# 3. Train model
# -------------------------
train:
	$(VENV_ACTIVATE) && python src/training/train_ray.py

# -------------------------
# 4. Hyperparameter tuning
# -------------------------
tune:
	$(VENV_ACTIVATE) && python src/training/tune_ray.py

# -------------------------
# 5. Start API server
# -------------------------
serve:
	$(VENV_ACTIVATE) && python src/serving/serve_app.py

# -------------------------
# 6. Run full ML pipeline
# -------------------------
all: data preprocess train tune serve

# -------------------------
# 7. Code formatting (Black)
# -------------------------
lint-format:
	$(VENV_ACTIVATE) && black src tests notebooks

# -------------------------
# 8. Lint code (Ruff)
# -------------------------
lint:
	$(VENV_ACTIVATE) && ruff src tests

# -------------------------
# 9. Type checking (Mypy)
# -------------------------
typecheck:
	$(VENV_ACTIVATE) && mypy src

# -------------------------
# 10. Run tests (Pytest)
# -------------------------
test:
	$(VENV_ACTIVATE) && pytest tests --cov=src

# -------------------------
# 11. Clean temporary/cache files
# -------------------------
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__

# -------------------------
# 12. Full dev check (lint + typecheck + test)
# -------------------------
check: lint typecheck test
