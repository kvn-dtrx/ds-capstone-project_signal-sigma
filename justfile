# ---
# title: justfile for ds-capstone-project_signal-sigma
# ---

# ---

venv := ".venv"

# Shows available recipes
default:
    @just --list --unsorted

# Clear build artefacts in data, logs, plots (Unix)
clear:
    find data -mindepth 1 ! -iname ".gitkeep" -delete
    find logs -mindepth 1 ! -iname ".gitkeep" -delete
    find plots -mindepth 1 ! -iname ".gitkeep" -delete

# Clear artefacts and remove virtual environment (Unix)
reset: clear
    rm -rf {{venv}}
    pyenv local --unset
