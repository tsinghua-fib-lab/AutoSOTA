#!/bin/env python
SETTINGS=(
    # "0.1667 0.8333 0.2"
    # "0.2 0.8 0.25"
    "0.25 0.75 0.33"
    # "0.33 0.66 0.5"
    # "0.5 0.5 1.0"
    # "0.66 0.33 2.0"
    # "0.75 0.25 3.0"
    # "0.8 0.2 4.0"
    # "0.8333 0.1667 5.0" 
)
DATA=$1
NTEST=150
device=-1
for config in "${SETTINGS[@]}";
    do
        ((device++))

        if [ "$device" -eq 4 ]; then
        device=0
        fi
        echo "device: ${device}"
        
        IFS=" " read -ra params <<< "$config"
        pz0="${params[0]}"
        pz1="${params[1]}"
        alpha_test="${params[2]}"
        
        CUDA_VISIBLE_DEVICES=${device} python scripts/naive_cf.py \
        -data ${DATA} \
        -model bert-base-uncased -out output/local/${DATA}/ -rep 5 \
        --n_test ${NTEST} \
        --config ${pz0} ${pz1} ${alpha_test}  &

    done
wait