#!/bin/bash
# Reproduce SyNG-R on PolBlogs (r=2, 200 samples) and evaluate paper metrics.
set -euo pipefail
R=2; DS=polblogs; OUT="runs/syngler/${DS}_r${R}"

# Redirect runs to cache if overlay is tight
if [ ! -e runs ] || [ "$(df --output=pcent / | tail -1 | tr -d ' %')" -gt 95 ]; then
    mkdir -p /autosota_cache/tmp/paper-3593/runs
    rm -f runs
    ln -sf /autosota_cache/tmp/paper-3593/runs runs
fi

echo "=== Step 1/2: Generate SyNG-R samples ==="
python experiments/real_data/run_syngler.py --dataset "${DS}" --r "${R}" --fitted_pkl "data/real/${DS}/run/r=${R}/seed=0.pkl" --output "${OUT}" --methods res --num_samples 200

echo "=== Step 2/2: Evaluate paper metrics ==="
python scripts/eval_polblogs.py --samples_dir "${OUT}/syngr/samples" --ref_adj "data/real/${DS}/generator/seed=0.npy" --output "${OUT}/eval_results.json" --device cpu

echo "=== Results ==="
python -c "import json; r=json.load(open('${OUT}/eval_results.json')); [print(f'{k}: {r[k]:.8f}') for k in ['Tri_RMSE','Clus_RMSE','DegC_KS','Eig_MMD']]; print(f'Time: {r[\"eval_time_s\"]:.1f}s')"
