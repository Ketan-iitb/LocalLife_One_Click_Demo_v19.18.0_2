#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

if [[ -f cloud.env ]]; then
  set -a
  source cloud.env
  set +a
fi

exec python -m locallife_cloud.server "$@"
