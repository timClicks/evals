py_exe := `uv run which python`

# This is what I can do
default:
  just --list

# Install the project
init: 
  uv venv --allow-existing
  uv sync

# Sync the dependencies
sync:
  uv sync

# Lint all files
lint: sync
  uv run ruff check src tests

# Typecheck all files
typecheck: sync
  uv run pyright src

# Run all tests -- NONE YET!
# test: sync
#   uv run pytest tests -ra

# Run all checks
check: lint typecheck

# Ruff auto fixes
fix: 
  uv run ruff check --fix src tests

# Do all the things.
all:
  uv run lmsys
  uv run litellm

# Clean up the project
[confirm]
clean: 
  uv run lmsys-clean


# Check python version (How to run python scripts)
pyversion:
  #!{{py_exe}}
  import sys
  print(sys.version)

dbloc:
  #!{{py_exe}}
  from evals.settings import get_settings
  print(f'"{get_settings().get_database_path()}"')