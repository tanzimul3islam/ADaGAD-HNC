.PHONY: help install format lint typecheck test smoke train eval clean docs

help:
	@echo "AdaGAD-HNC Makefile targets:"
	@echo "  install      - Install dependencies in venv"
	@echo "  format       - Run black + isort"
	@echo "  lint         - Run ruff"
	@echo "  typecheck    - Run mypy"
	@echo "  test         - Run pytest"
	@echo "  smoke        - Run smoke training (1 epoch)"
	@echo "  train        - Train model"
	@echo "  eval         - Evaluate checkpoint"
	@echo "  clean        - Remove artifacts"
	@echo "  all          - format + lint + typecheck + test"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

format:
	black .
	isort .

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest -q

smoke:
	python -m src.main train model=adagad_hnc data=citeseer train=standard max_epochs=1

train:
	python -m src.main train

eval:
	python -m src.main eval --ckpt $(CKPT)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs wandb

all: format lint typecheck test
