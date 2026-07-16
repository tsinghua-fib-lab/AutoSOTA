#!/bin/bash
# Reproduce ATLAS HellaSwag metrics (SE=0.1)
# Usage: bash eval_hellaswag.sh
set -euo pipefail
cd /repo

# Ensure R can find installed packages
export R_LIBS_USER=/autosota_cache/r-libs

# Extract response matrix data if not already extracted
if [ ! -f data/gaussian_sampled_hellaswag_response_matrix_test.csv ]; then
    python3 -c "import zipfile; zipfile.ZipFile('data/data.zip').extractall('.')"
fi

# Step 3: Run ATLAS adaptive testing for HellaSwag (SE=0.1)
echo '=== Running ATLAS (adaptive testing) ==='
Rscript scripts/03_atlas_cat.r --benchmark_name=hellaswag --se_theta_stop=0.1 --n_cores=8

# Step 4: Compute p-IRT accuracy
echo '=== Computing p-IRT accuracy ==='
Rscript scripts/04_pirt_accuracy.r --benchmark=hellaswag --se_theta_stop=0.1

# Step 5: Compute actual accuracy
echo '=== Computing actual accuracy ==='
Rscript scripts/05_compute_actual_acc.r --benchmark=hellaswag

# Step 6: Compare p-IRT vs actual
echo '=== Comparing p-IRT vs actual ==='
Rscript scripts/analysis/compare_pirt_actual.r --benchmark=hellaswag --se_theta_stop=0.1

# Extract and display metrics
echo ''
echo '=== REPRODUCED METRICS ==='
python3 -c "
import csv, math

# Ability MAE
with open('hellaswag/atlas_hellaswag_random/irt_person_scores_ATLAS_0.1.csv') as f:
    rows = list(csv.DictReader(f))
    theta_diffs = [abs(float(r['Theta_ATLAS']) - float(r['Theta_WLE'])) for r in rows]
    ability_mae = sum(theta_diffs) / len(rows)
    items = [float(r['Num_Items']) for r in rows]
    avg_items = sum(items) / len(rows)

# Accuracy MAE
with open('hellaswag/pirt_vs_actual_se_0.1.csv') as f:
    rows2 = list(csv.DictReader(f))
    acc_diffs = [abs(float(r['pirt_accuracy']) - float(r['actual_accuracy'])) for r in rows2]
    accuracy_mae = sum(acc_diffs) / len(rows2)

print(f'Ability_MAE: {ability_mae:.4f}')
print(f'Accuracy_MAE: {accuracy_mae:.4f}')
print(f'Average_Items: {avg_items:.1f}')
print(f'Models: {len(rows)}')
"
