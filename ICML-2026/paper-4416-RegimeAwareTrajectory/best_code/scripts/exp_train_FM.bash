max_epochs=$1
model_path=ProbabilityForecasting/$2
experiment_name=$2
load_name="SysBio-Traj_index.csv"
devices="[$3]"

bs=$4
check_epoch=$5
seeds=($6)
# seeds=(53 25 81)
echo "Starting parallel training for seeds: ${seeds[*]}"

for seed in "${seeds[@]}"; do
    echo "Launching seed ${seed} on devices ${devices}..."
    WANDB_MODE=offline \
    python exp/train_bioTFM.py -cp ./configs -cn config model=${model_path} \
        data.load_name=${load_name} data.data_type=BioTFM \
        experiment_name=${experiment_name}_seed${seed} training.use_ema=true \
        training.max_epochs=${max_epochs} training.warmup_epochs=0 "hardware.devices=${devices}" data.batch_size=${bs} \
        seed=${seed} training.check_val_every_n_epoch=${check_epoch} data.num_workers=0 \
        training.limit_val_batches=1.0
done

echo "All seeds have finished training!"
