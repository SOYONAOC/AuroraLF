#!/usr/bin/env bash
# Usage: bash scripts/submit/submit_uvlf_efficiency_scan_20260914.sh [--dry-run]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"

# Explicit command-line mail options are required by the local SLURM 20.11.
exec .venv/bin/python scripts/submit/submit_uvlf_efficiency_scan_mail.py \
  --job-name AUR-UVLF-EPS-20260914 \
  --project-root "$PWD" \
  --parallelism-mode none \
  --no-auto-benchmark \
  "$@" \
  scripts/analysis/build_uvlf_efficiency_scan.py -- \
  --plan configs/experiments/uvlf_efficiency_scan_20260914_mailfixed.json
