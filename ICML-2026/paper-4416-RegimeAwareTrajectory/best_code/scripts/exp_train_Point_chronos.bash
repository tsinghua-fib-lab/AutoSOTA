max_epochs=$1
model_path=PointForecasting/$2
experiment_name=$2
load_name="SysBio-Traj_index.csv"
devices="[$3]"
device_map="cuda:$3"

batch_size=$4
check_epoch=$5
seeds=($6)
local_model_dir=$7
offline_mode=${8:-1}

if [[ -z "${local_model_dir}" ]]; then
    echo "[ERROR] Please provide local model directory (e.g., Materials/PretrainedModel/Chronos_bolt/20M)"
    exit 1
fi

if [[ ! -f "${local_model_dir}/config.json" || ! -f "${local_model_dir}/model.safetensors" ]]; then
    echo "[ERROR] Invalid local model directory: ${local_model_dir}"
    echo "[HINT] Directory must contain config.json and model.safetensors"
    exit 1
fi

extra_overrides="model.model.args.pretrained_model_path=${local_model_dir} model.model.args.local_files_only=true model.model.args.device_map=${device_map}"

echo "Starting PointForecasting Chronos runs for seeds: ${seeds[*]}"

for seed in "${seeds[@]}"; do
    echo "Launching seed ${seed} on devices ${devices}..."
    if [[ "${offline_mode}" == "1" ]]; then
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
        python exp/train_bioTFM.py -cp ./configs -cn config model=${model_path} \
            data.load_name=${load_name} data.data_type=PointTrajectory \
            experiment_name=${experiment_name}_seed${seed} training.use_ema=false \
            test_only=true test_without_ckpt=true \
            "hardware.devices=${devices}" data.batch_size=${batch_size} \
            seed=${seed} ${extra_overrides}
    else
        WANDB_MODE=offline \
        python exp/train_bioTFM.py -cp ./configs -cn config model=${model_path} \
            data.load_name=${load_name} data.data_type=PointTrajectory \
            experiment_name=${experiment_name}_seed${seed} training.use_ema=false \
            test_only=true test_without_ckpt=true \
            "hardware.devices=${devices}" data.batch_size=${batch_size} \
            seed=${seed} ${extra_overrides}
    fi
done

echo "All seeds have finished runs!"
