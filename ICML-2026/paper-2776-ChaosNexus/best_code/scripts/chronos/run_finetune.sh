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

ulimit -n 99999
if [ "$DEBUG" -eq 0 ]; then

        CUDA_DEVICES=0,2,5,6
        #CUDA_DEVICES=4,5,6,7
        echo "CUDA Devices: $CUDA_DEVICES"

        NUM_DEVICES=$(echo "$CUDA_DEVICES" | tr -d ' ' | tr ',' '\n' | wc -l)
        TOTAL_PROCESSES=$(nproc)
        PER_GROUP=$(( $TOTAL_PROCESSES / 2 ))
        PER_RANK=$(( $PER_GROUP / $NUM_DEVICES ))

        export OMP_NUM_THREADS=$PER_RANK
        export MKL_NUM_THREADS=$PER_RANK
        export OPENBLAS_NUM_THREADS=$PER_RANK
        export NUMEXPR_NUM_THREADS=$PER_RANK

        if python -c "import torch; print(torch.version.hip)" 2>/dev/null | grep -vq "None"; then
                echo "ROCm detected - disabling P2P for distributed training"
                export NCCL_P2P_DISABLE=1
        fi
        export PYTHONWARNINGS="ignore"

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun \
                --nproc-per-node $NUM_DEVICES \
                --standalone \
                scripts/chronos/train.py \
                shuffle_buffer_length=100_000 \
                train_data_dirs=$train_data_dirs_json \
                chronos.model_id="./huggingface/amazon/chronos-t5-large" \
                chronos.model_type=seq2seq \
                chronos.random_init=false \
                chronos.tie_embeddings=true \
                chronos.context_length=512 \
                chronos.prediction_length=128 \
                chronos.num_samples=20 \
                chronos.n_tokens=4096 \
                chronos.n_special_tokens=2 \
                chronos.pad_token_id=0 \
                chronos.eos_token_id=1 \
                chronos.use_eos_token=true \
                chronos.tokenizer_class=MeanScaleUniformBins \
                chronos.tokenizer_kwargs.low_limit=-15.0 \
                chronos.tokenizer_kwargs.high_limit=15.0 \
                chronos.temperature=1.0 \
                chronos.top_k=50 \
                chronos.top_p=1.0 \
                train.max_steps=100_000 \
                train.save_steps=10_000 \
                train.log_steps=1000 \
                shuffle_buffer_length=100_000 \
                train.per_device_train_batch_size=16 \
                train.warmup_ratio=0.05 \
                train.torch_compile=true \
                train.weight_decay=1e-4 \
                train.output_dir=./checkpoints/t5-mini-sft \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/chronos/train.py \
                run_name=DEBUG \
                shuffle_buffer_length=100 \
                train.ddp_backend=null \
                train.torch_compile=false \
                "$@"
fi