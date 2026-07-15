#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

python3 rebuttal_experiments/transparent_configs.py
python3 rebuttal_experiments/completed_accuracy.py
python3 rebuttal_experiments/modern_workloads.py
python3 rebuttal_experiments/dynamic_kv_cache.py
python3 rebuttal_experiments/npu_sram_emulation.py
python3 rebuttal_experiments/scheduling_analysis.py
python3 rebuttal_experiments/memory_pressure_check.py
python3 rebuttal_experiments/pantheon_accuracy_loss.py
python3 rebuttal_experiments/jetson_motivation_accuracy.py
python3 rebuttal_experiments/ablation_stress.py
