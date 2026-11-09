#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
envs/tools/bin/python scripts/check_access.py
envs/tools/bin/kaggle competitions download -c g2net-gravitational-wave-detection -p data/raw
envs/tools/bin/python scripts/prepare_data.py
