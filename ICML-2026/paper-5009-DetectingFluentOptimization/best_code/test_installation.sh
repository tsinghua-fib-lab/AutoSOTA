#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Checking Python and core dependencies..."
python - <<'PY'
import importlib
mods = ["numpy","pandas","torch","transformers","sklearn","matplotlib","yaml"]
missing = []
for m in mods:
    try:
        importlib.import_module(m if m != "yaml" else "yaml")
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"Missing imports: {missing}")
print("OK")
PY

echo "[2/3] Checking bundled datasets..."
test -f data/benign_mix_ppgap5_700.csv && echo "OK: data/benign_mix_ppgap5_700.csv"
test -f data/benign_mix_ppgap1_800.csv && echo "OK: data/benign_mix_ppgap1_800.csv"
test -f data/benign_mix_ppgap2_800.csv && echo "OK: data/benign_mix_ppgap2_800.csv"
test -f data/benign_mix_ppgap3_800.csv && echo "OK: data/benign_mix_ppgap3_800.csv"
test -f data/full_prompt_dataset.csv && echo "OK: data/full_prompt_dataset.csv"
test -f data/gcg_llamaguard_bypass.csv && echo "OK: data/gcg_llamaguard_bypass.csv"

echo "[3/3] Checking scripts are executable..."
test -x run_detection_timing.sh && echo "OK: run_detection_timing.sh"
test -x run_ppgap_ablation.sh && echo "OK: run_ppgap_ablation.sh"
test -x run_llama_guard_robust.sh && echo "OK: run_llama_guard_robust.sh"

echo "[OK] Installation sanity check passed."

