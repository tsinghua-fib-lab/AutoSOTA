#!/usr/bin/env bash
# Full CPR pipeline on WebQSP (alpha=0.4)
set -euo pipefail
cd "$(dirname "$0")"

python run.py --config configs/webqsp.yaml --puct_calib --alpha 0.4 --out_json results/webqsp_a04.json
