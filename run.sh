#!/usr/bin/env bash
# ACTIVA official entry point: ./run.sh <input.mp4> <output.json>
# Exit codes: 0 ok, 2 usage, 3 input not found, 4 invalid video,
#             5 configuration (e.g. vocabulary missing), 1 internal error.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh <input.mp4> <output.json>" >&2
    exit 2
fi
[ -f "$1" ] || { echo "Input not found: $1" >&2; exit 3; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ACTIVA_PYTHON picks the interpreter; otherwise python3, then python.
PY="${ACTIVA_PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1 && python3 -c "import sys" >/dev/null 2>&1; then
        PY=python3
    else
        PY=python
    fi
fi

# Inference must never touch the network (guide B4 step 4).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$(dirname "$2")"
exec "$PY" "$DIR/code/main.py" "$1" "$2"
