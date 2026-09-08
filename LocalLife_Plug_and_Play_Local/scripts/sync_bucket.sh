#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

bucket="${LOCALLIFE_BUCKET:-gs://locallife-thesis-depth-data}/cloud-v15"
artifacts="${LOCALLIFE_RESULTS_DIR:-$project_dir/artifacts}"
mkdir -p "$artifacts"

case "${1:-push}" in
  push)
    gcloud storage rsync -r "$artifacts" "$bucket"
    ;;
  pull)
    gcloud storage rsync -r "$bucket" "$artifacts"
    ;;
  *)
    echo "Usage: bash scripts/sync_bucket.sh [push|pull]" >&2
    exit 2
    ;;
esac
