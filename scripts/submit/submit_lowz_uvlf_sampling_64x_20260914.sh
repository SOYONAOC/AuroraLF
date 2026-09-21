#!/usr/bin/env bash
# Start on available shared CPUs, leaving one currently idle CPU unrequested.
set -euo pipefail
cd /home/zhuhourui/AstroCode/AuroraLF
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --dry-run ]]; }; then
  echo 'Usage: submit_lowz_uvlf_sampling_64x_20260914.sh [--dry-run]' >&2
  exit 2
fi
export PYTHONPATH="$PWD"
exec .venv/bin/python scripts/submit/submit_uvlf_efficiency_scan_mail.py \
  --job-name AUR-UVLF-LOWZ-64X --project-root "$PWD" \
  --parallelism-mode none --no-auto-benchmark --node-allocation shared \
  --leave-idle-cpus 1 \
  "$@" scripts/run/increase_lowz_uvlf_sampling.py -- \
  --plan configs/experiments/uvlf_lowz_sampling_64x_20260914.json
