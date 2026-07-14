#!/bin/bash
set -e

# Step 1: Generate representations (skip if already done)
LATEST=$(ls -td /repo/logs/UrbanFusionV2_1/runs/*/ 2>/dev/null | head -1)
if [ -z "$LATEST" ] || [ ! -f "$LATEST/plots/representations_test_epoch_0_masked_modality_none_dl_0.pt" ]; then
    echo "Generating representations..."
    cd /repo
    CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
      experiment=placepulse2/UrbanFusionV2_1 \
      ckpt_path=/models/UrbanFusion/UrbanFusion.ckpt \
      paths.data_dir=/repo/ \
      paths.log_dir=/repo/logs/ \
      data.coordinate_predictions=null \
      seed=42 \
      trainer=gpu \
      "trainer.devices=[0]"
fi

# Step 2: Update config with latest run date
LATEST=$(ls -td /repo/logs/UrbanFusionV2_1/runs/*/ 2>/dev/null | head -1)
DATE_STR=$(basename "$LATEST")

python3 << PYEOF
import json
cfg = json.load(open("/repo/svi_data/place-pulse-2.0/downstreamtask_data/results/crime_usa/crime_usa_within_region_UrbanFusion_trainedV2_1_ridge.json"))
cfg["UrbanFusionV2_1"]["date"] = "${DATE_STR}"
cfg["UrbanFusionV2_1"]["epoch"] = 0
json.dump(cfg, open("/repo/svi_data/place-pulse-2.0/downstreamtask_data/results/crime_usa/crime_usa_within_region_UrbanFusion_trainedV2_1_ridge.json", "w"), indent=4)
print(f"Config updated to use run: ${DATE_STR}")
PYEOF

# Step 3: Run crime evaluation
cd /repo
CUDA_VISIBLE_DEVICES=0 python scripts/downstream_tasks/crime_usa.py \
  --settings crime_usa_within_region_UrbanFusion_trainedV2_1_ridge.json \
  --model ridge

# Step 4: Display best result
echo ""
echo "Results:"
python3 << PYEOF
import pandas as pd
df = pd.read_csv("/repo/svi_data/place-pulse-2.0/downstreamtask_data/results/crime_usa/crime_usa_within_region_UrbanFusion_trainedV2_1_ridge_ridge.csv")
best = df.loc[df["r2"].idxmax()]
print(f"Best R^2: {best['r2']*100:.2f}% ({best['modality_name']})")
PYEOF
