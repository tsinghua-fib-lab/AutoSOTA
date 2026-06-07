#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for item in \
    "PhyGaP/bud_corridor"\
    "PhyGaP/ox_corridor"\
    "PhyGaP/pop_garden"

do
    python train.py -s "./data/${item}" --eval --indirect_type obj_env --lambda_stokes 4 --use_LP --double_view --envmap_max_roughness 1
done 

for item in \
    "PANDORA/owl_quat_white"\
    "PANDORA/vase_quat_white"\
    "MitsubaSynthetic/mitsuba_david_museum"\
    "MitsubaSynthetic/mitsuba_matpreview_museum"\
    "MitsubaSynthetic/mitsuba_teapot_museum"
do
    python train.py -s "./data/${item}" --eval --indirect_type obj_env --lambda_stokes 4
done

   

# "PhyGaP/bud_corridor" \
# "PhyGaP/ox_corridor" \
# "PhyGaP/pop_garden" \
# "MitsubaSynthetic/mitsuba_david_museum"\
# "MitsubaSynthetic/mitsuba_matpreview_museum"\
# "MitsubaSynthetic/mitsuba_teapot_museum"\
# "PANDORA/owl_quat_white"\
# "PANDORA/vase_quat_white"\
#    "SMVP3D/david"\
#     "SMVP3D/snail"\
