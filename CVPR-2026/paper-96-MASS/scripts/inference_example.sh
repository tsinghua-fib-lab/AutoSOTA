#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-./mass_base.pth}
GPU=${GPU:-0}

python inference.py \
  --checkpoint "${CHECKPOINT}" \
  --test-image /path/to/test_image.nii.gz \
  --reference-image /path/to/reference_image.nii.gz \
  --reference-mask /path/to/reference_mask.nii.gz \
  --output outputs/test_image_seg.nii.gz \
  --gpu "${GPU}" \
  --use-ema \
  --modality ct \
  --orientation RAS \
  --target-spacing 1.5 1.5 1.5 \
  --window-size 128 128 128 \
  --overlap 0.5
