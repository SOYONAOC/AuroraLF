#!/usr/bin/env bash
set -euo pipefail
cd /home/zhuhourui/AstroCode/AuroraLF
export PYTHONPATH="$PWD"
exec .venv/bin/python scripts/submit/submit_uvlf_efficiency_scan_mail.py \
  --job-name AUR-UVLF-LOWZ-16X --project-root "$PWD" \
  --parallelism-mode none --no-auto-benchmark --node-allocation shared --fixed-cpus 30 \
  "$@" scripts/run/increase_lowz_uvlf_sampling.py -- \
  --plan configs/experiments/uvlf_lowz_sampling_20260914_r05.json
