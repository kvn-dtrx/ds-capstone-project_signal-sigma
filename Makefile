# ---
# title: Makefile for ds-capstone-project_signal-sigma
# ---

# ---

#
# Convention: make = env/setup (venv, pip -e); clear/reset live in justfile.

# ---

PYTHON_VERSION := 3.11.3
VENV := .venv

TARGETS := help basic-unix basic-win dev-unix dev-win
.PHONY: $(TARGETS)

help: ## Displays available targets with description
	@bin/make-help.sh

basic-unix: ## Set up virtual environment and dependencies on macOS/Linux
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e .

basic-win: ## Set up virtual environment and dependencies on Windows (PowerShell)
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	.\$(VENV)\Scripts\python.exe -m pip install --upgrade pip
	.\$(VENV)\Scripts\python.exe -m pip install -e .

dev-unix: ## Set up development environment and pre-commit hooks on macOS/Linux
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e .[dev]
	$(VENV)/bin/pre-commit install

dev-win: ## Set up development environment and pre-commit hooks on Windows (PowerShell)
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	.\$(VENV)\Scripts\python.exe -m pip install --upgrade pip
	.\$(VENV)\Scripts\python.exe -m pip install -e .[dev]
	.\$(VENV)\Scripts\pre-commit.exe install
