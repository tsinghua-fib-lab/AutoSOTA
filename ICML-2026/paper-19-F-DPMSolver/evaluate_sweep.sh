#!/bin/bash
set -e
cd /repo

RHO=${RHO:-7}
SIGMA_MIN=${SIGMA_MIN:-0.002}
SIGMA_MAX=${SIGMA_MAX:-80}
ORDER=${ORDER:-2}
NFE=${NFE:-4}
SUBDIR=${SUBDIR:-cifar10_uncond_eval}

echo "=== Config: rho=${RHO} sigma_min=${SIGMA_MIN} sigma_max=${SIGMA_MAX} order=${ORDER} NFE=${NFE} ==="

torchrun --standalone --nproc_per_node=2 main.py \
    --subdirs "${SUBDIR}" \
    --seeds=0-49999 \
    --NFE=${NFE} \
    --batch=64 \
    --algorithm_name="F-DPMSolver" \
    --model_name "CIFAR10-uncond" \
    --order=${ORDER} \
    --rho=${RHO} \
    --sigma_min=${SIGMA_MIN} \
    --sigma_max=${SIGMA_MAX}

echo "=== Computing FID ==="
python3 -c "
from cleanfid import fid
score = fid.compute_fid(\"sample/${SUBDIR}\", dataset_name=\"cifar10\", dataset_res=32, dataset_split=\"train\", mode=\"clean\", num_workers=0)
print(f\"FID: {score:.2f}\")
"
