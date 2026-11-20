#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${ACTIVA_PYTHON:-python3}"
exec "$python_bin" "$project_dir/code/perception_cli.py" "$@"
