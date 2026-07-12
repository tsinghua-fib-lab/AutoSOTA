#!/bin/bash
set -e

# Usage: bash scripts/benchmark_diffid/latent_gig.sh <vae_type> <fraction> <use_slerp>
# Example: bash scripts/benchmark_diffid/latent_gig.sh sd1 0.1 false
#          bash scripts/benchmark_diffid/latent_gig.sh sd2 0.05 true

if [ $# -ne 3 ]; then
    echo "Usage: $0 <vae_type> <fraction> <use_slerp>"
    echo "  vae_type: sd1, sd2, mar, kd"
    echo "  fraction: 0.05, 0.1, 0.2, ..., 0.9"
    echo "  use_slerp: true, false"
    exit 1
fi

VAE_TYPE=$1
FRACTION=$2
USE_SLERP=$3

if [ "$USE_SLERP" = "true" ]; then
    SAVE_NAME=latent_gig_slerp_${VAE_TYPE}_${FRACTION}
else
    SAVE_NAME=latent_gig_${VAE_TYPE}_${FRACTION}
fi

for dataset in imagenet oxfordpet oxfordflower; do
    for model_name in vgg16 resnet18 inception; do
    python scripts/diffid.py \
        --config-name=latent_gig \
        dataset=$dataset \
        model_name=$model_name \
        vae_type=$VAE_TYPE \
        fraction=$FRACTION \
        use_slerp=$USE_SLERP \
        max_eval_samples=500 \
        save_dir=results/benchmark_diffid/$dataset/$SAVE_NAME/$model_name
    done
done
