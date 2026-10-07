#!/bin/bash
# Creates .venv with uv and installs the project, its pinned dependencies and the Brain-Score benchmarks
# from the language/ submodule.
#
# Example: bash setup_env.sh
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# uv sync resolves brainscore-language from language/, so the submodule must be checked out first
if [ ! -e language/setup.py ] && [ ! -e language/pyproject.toml ]; then
    git submodule update --init --recursive
fi

# fixes carried against the pinned submodule; each is skipped when already applied
for patch in patches/*.patch; do
    [ -e "$patch" ] || continue
    if git -C language apply --reverse --check "$PWD/$patch" >/dev/null 2>&1; then
        echo "Patch already applied: $(basename "$patch")"
    elif git -C language apply "$PWD/$patch"; then
        echo "Applied patch: $(basename "$patch")"
    else
        echo "ERROR: $(basename "$patch") does not apply to language/; see patches/README.md." >&2
        exit 1
    fi
done

uv sync

echo "Done. Activate with 'source .venv/bin/activate' or prefix commands with 'uv run'."
