#!/bin/bash
# read debug flag
DEBUG=0
while getopts "d" flag; do
        case "${flag}" in
                d) DEBUG=1;;
        esac
done
shift $((OPTIND - 1))

train_data_dirs=(
#     ./data/improved/skew_mixedp_ic16/train
#     ./data/improved/final_skew40/train
#     ./data/improved/final_skew40/train_z5_z10
#     ./data/improved/final_base40/train
#     ./data/improved/final_base40/train_z5_z10
      ./data/train
)

# --- START: Modified section (replaces jq) ---
# Goal: Convert the bash array to a JSON array string.
# Example: (path1 path2) -> ["path1","path2"]

train_data_dirs_json="[" # Start the JSON array string
num_dirs=${#train_data_dirs[@]} # Get the total number of directories
i=0
for dir in "${train_data_dirs[@]}"; do
    train_data_dirs_json+="\"$dir\"" # Add the directory path in quotes
    i=$((i + 1))
    if [ "$i" -lt "$num_dirs" ]; then
        train_data_dirs_json+="," # Add a comma if it's not the last element
    fi
done
train_data_dirs_json+="]" # Close the JSON array string
# --- END: Modified section ---

echo "train_data_dirs: $train_data_dirs_json"

ulimit -n 999999
if [ "$DEBUG" -eq 0 ]; then

        TOTAL_CORES=$(nproc)
        CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
        CORES_PER_JOB=$(( $CORES_PER_GROUP / 4 ))

        CUDA_DEVICES=6
        # CUDA_DEVICES=4,5,6,7
        NUM_DEVICES=$(echo "$CUDA_DEVICES" | tr -d ' ' | tr ',' '\n' | wc -l)

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$CORES_PER_JOB python3 -m torch.distributed.run \
                --nproc-per-node $NUM_DEVICES \
                --master-port 29536 \
                scripts/scaleformer/train.py \
                shuffle_buffer_length=100_000 \
                train_data_dirs=$train_data_dirs_json \
                scaleformer.mode=predict \
                scaleformer.use_dynamics_embedding=true \
                scaleformer.pretrained_encoder_path=null \
                scaleformer.context_length=512 \
                scaleformer.prediction_length=128 \
                scaleformer.patch_length=8 \
                scaleformer.patch_stride=8 \
                scaleformer.num_hidden_layers=8 \
                scaleformer.num_attention_heads=8 \
                scaleformer.d_model=48 \
                scaleformer.num_rff=24 \
                scaleformer.rff_scale=1.0 \
                scaleformer.rff_trainable=false \
                scaleformer.num_poly_feats=8 \
                scaleformer.poly_degrees=2 \
                scaleformer.channel_attention=true \
                scaleformer.max_wavelength=500 \
                scaleformer.rope_percent=0.75 \
                scaleformer.pooling_type=mean \
                scaleformer.loss=mse \
                scaleformer.distribution_output=null \
                train.per_device_train_batch_size=1024 \
                train.max_steps=100000 \
                train.save_steps=10000 \
                train.log_steps=100 \
                train.warmup_ratio=0.1 \
                train.torch_compile=true \
                train.weight_decay=0.0 \
                train.output_dir="./checkpoints/" \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/scaleformer/train.py \
                run_name=DEBUG \
                scaleformer.pretrained_encoder_path=null \
                shuffle_buffer_length=100 \
                scaleformer.mode=predict \
                train.ddp_backend=null \
                train.torch_compile=false \
                "$@"
fi

