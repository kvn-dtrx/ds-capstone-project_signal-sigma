# ---
# title: Makefile for ds-capstone-project_signal-sigma
# ---

# ---

#
# Convention: make = env/setup (venv, pip -e); clear/reset live in justfile.

# ---

PYTHON_VERSION := 3.11.3
VENV := .venv

BOLD_WHITE := \033[1;37m
RESET := \033[0m

TARGETS := help basic-unix basic-win dev-unix dev-win
.PHONY: $(TARGETS)

help:
	@echo
	@echo "    $(BOLD_WHITE)⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡$(RESET)"
	@echo "    $(BOLD_WHITE)⟡ Signal Sigma ⟡$(RESET)"
	@echo "    $(BOLD_WHITE)⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡⟡$(RESET)"
	@echo
	@echo "    $(BOLD_WHITE)Setup:$(RESET)"
	@echo "    make basic-unix : Set up virtual environment and dependencies on macOS/Linux"
	@echo "    make basic-win  : Set up virtual environment and dependencies on Windows (PowerShell)"
	@echo "    make dev-unix   : Set up development environment and pre-commit hooks on macOS/Linux"
	@echo "    make dev-win    : Set up development environment and pre-commit hooks on Windows (PowerShell)"
	@echo
	@echo "    $(BOLD_WHITE)Clean-Up (just):$(RESET)"
	@echo "    just clear      : Clear build artefacts in data, logs, plots"
	@echo "    just reset      : Clear artefacts and remove virtual environment"
	@echo
	@echo "    $(BOLD_WHITE)Important Make Flags:$(RESET)"
	@echo "    -n              : Dry-run (print commands without running them)"
	@echo "    -s              : Silent mode (don't print executed commands)"
	@echo "    --debug[=b|v|a] : Debug info (b=basic [default], v=verbose, a=all)"
	@echo

basic-unix:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e .

basic-win:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	.\$(VENV)\Scripts\python.exe -m pip install --upgrade pip
	.\$(VENV)\Scripts\python.exe -m pip install -e .

dev-unix:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e .[dev]
	$(VENV)/bin/pre-commit install

dev-win:
	pyenv local $(PYTHON_VERSION)
	python -m venv $(VENV)
	.\$(VENV)\Scripts\python.exe -m pip install --upgrade pip
	.\$(VENV)\Scripts\python.exe -m pip install -e .[dev]
	.\$(VENV)\Scripts\pre-commit.exe install
