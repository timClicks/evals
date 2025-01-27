py_exe := `uv run which python`

# This is what I can do
default:
  just --list


# Install the project
[group('Begin Here')]
init: 
  uv venv --allow-existing
  uv sync

# Download model quality benchmark results and Elo ratings from LMArena (formally LMSYS) 
[group('Benchmarks')]
lmsys:
  uv run lmsys

# Download model cost benchmarks from https://github.com/BerriAI/litellm/
[group('Benchmarks')]
litellm:
  uv run litellm

# Download model speed benchmark results from https://thefastest.ai/
[group('Benchmarks')]
thefastestai:
  uv run thefastestai

alias download := download-benchmarks
# Download all benchmark results
download-benchmarks: lmsys litellm thefastestai

alias score := score-benchmarks
alias scoring := score-benchmarks

# Score the benchmark results
score-benchmarks:
  uv run scoring

alias update := all

# Download all benchmark results and score
all: sync download-benchmarks score-benchmarks

# Check python version (How to run python scripts)
[group('Developer Tools')]
pyversion:
  #!{{py_exe}}
  import sys
  print(sys.version)

# Sync the dependencies
[group('Developer Tools')]
sync:
  uv sync

# Lint all files
[group('Developer Tools')]
lint: sync
  uv run ruff check src tests

# Typecheck all files
[group('Developer Tools')]
typecheck: sync
  uv run pyright src

# Run all tests -- NONE YET!
# test: sync
#   uv run pytest tests -ra

# Run all checks
[group('Developer Tools')]
check: lint typecheck

# Ruff auto fixes
[group('Developer Tools')]
fix: 
  uv run ruff check --fix src tests
