#!/usr/bin/env bash
# Cross-version CPU benchmark: build & run benchmarks/bench_cpu.py against
# Python 3.11, 3.12, 3.13. Each version gets its own uv-managed venv because
# maturin produces a Python-version-specific .so.
#
# Usage:
#   bash benchmarks/bench_all_pythons.sh
#   PYTHONS="3.11 3.13" bash benchmarks/bench_all_pythons.sh   # pick a subset
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONS="${PYTHONS:-3.11 3.12 3.13}"
DEPS=(maturin numpy scipy py_vollib)

for ver in $PYTHONS; do
    venv=".venv-py${ver//./}"
    echo
    echo "======================================================================"
    echo "  Python ${ver}    (venv: ${venv})"
    echo "======================================================================"

    # Create venv if it doesn't exist; uv handles toolchain download.
    if [ ! -d "$venv" ]; then
        uv venv --python "$ver" "$venv"
    fi

    # Install/upgrade deps and build the ngv_opx extension into this venv.
    VIRTUAL_ENV="$PWD/$venv" uv pip install --quiet --upgrade "${DEPS[@]}"
    VIRTUAL_ENV="$PWD/$venv" uv run --no-project maturin develop --release --quiet

    out="benchmarks/results/py${ver//./}.json"
    VIRTUAL_ENV="$PWD/$venv" uv run --no-project python benchmarks/bench_cpu.py --save "$out"
done

echo
echo "Per-version results saved under benchmarks/results/."
echo "Run 'task bench:compare' for a cross-version table + chart."
