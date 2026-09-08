#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

echo "Checking the VM's existing CUDA-enabled PyTorch installation..."
python3 -c 'import torch; print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())'

if [[ ! -d .venv ]]; then
  python3 -m venv --system-site-packages .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cloud.txt
python scripts/check_environment.py

echo "Setup complete. Start the service with: bash scripts/start_cloud.sh"
