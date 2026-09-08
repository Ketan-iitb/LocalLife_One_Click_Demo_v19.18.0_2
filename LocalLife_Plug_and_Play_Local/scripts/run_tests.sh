#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python -m compileall -q locallife_cloud scripts tests
python -m unittest discover -s tests -v
