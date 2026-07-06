# ./scripts/eval_pta.sh 0 ulip weights/ulip/pointbert_ulip1.pt modelnet_c obj_only 1024 vitg14 ulip1 so_obj_only_9

gpu=$1
lm3d=$2         # uni3d, openshape, ulip
ckpt_path=$3    # weights/uni3d/lvis/model.pt, weights/uni3d/modelnet40/model.pt, weights/uni3d/scanobjnn/model.pt
                # weights/openshape/openshape-pointbert-vitg14-rgb/model.pt, weights/ulip/pointbert_ulip2.pt
dataset=$4     # modelnet_c, sonn_c, snv2_c, omniobject3d
sonn_variant=$5 # obj_only, obj_bg, hardest
npoints=$6      # 1024/4096/16384 for `omniobject3d`
os_version=$7   # vitl14, vitg14 for `openshape`
ulip_version=$8 # ulip1, ulip2
s2r_type=${9}  # so_obj_only_9', 'so_obj_only_11'

export CUDA_VISIBLE_DEVICES=${gpu}

# Define the list of cor_types to run
cor_types=("add_global_2" "add_local_2" "dropout_global_2" "dropout_local_2" "jitter_2" "rotate_2" "scale_2")

# Loop through each cor_type
for cor_type in "${cor_types[@]}"; do
    echo "Running with cor_type: $cor_type"

    if [ "$lm3d" = "ulip" ]; then
        python ./run_pta.py \
        --config configs \
        --wandb-log \
        --lm3d ${lm3d} \
        --ckpt_path ${ckpt_path} \
        --dataset ${dataset} \
        --sonn_variant ${sonn_variant} \
        --cor_type ${cor_type} \
        --sim2real_type ${s2r_type} \
        --npoints ${npoints} \
        --ulip-version ${ulip_version}

    else
        echo "The model does not match any of the supported ones."
    fi
done
