#!/bin/bash
set -e

cd /repo
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
unset HF_ENDPOINT

SAVE_ROOT="./llm_unlearn_results_v2"
OPTIM_CFG="dual_adam_plus"
BETA1=0.9
BETA2=0.95
BASE_BETA1=0.9
BASE_BETA2=0.95

RESULTS_FILE="$SAVE_ROOT/aggregated_results.txt"
mkdir -p "$SAVE_ROOT"
echo "Task UFE TFE MU OVR" > $RESULTS_FILE

for TASK_ID in 1 2 3 4 5; do
    echo ""
    echo "===== Task $TASK_ID ====="
    export TASK_LIST=$TASK_ID
    MASTER_PORT=$((29500 + TASK_ID))

    echo "Training (task $TASK_ID)..."
    torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
      forget.py \
      --config-name=phi1-5_tofu.yaml \
      task_id=$TASK_ID \
      use_LoRA=false \
      forget_coeff=1.0 \
      regularization_coeff=1.0 \
      lr=1e-5 \
      forget_lr=1e-5 \
      split=forget05 \
      forget_loss=IDK+GD \
      num_epochs=5 \
      mask=true \
      fix_ref_model=false \
      save_root=$SAVE_ROOT \
      save_checkpoint=false \
      alternate=true \
      optim_cfg=$OPTIM_CFG \
      retain_freq=5 \
      alpha=1.0 \
      beta1=$BETA1 \
      beta2=$BETA2 \
      base_beta1=$BASE_BETA1 \
      base_beta2=$BASE_BETA2 \
      max_steps=300 \
      save_steps=last

    echo "Evaluating (task $TASK_ID)..."
    torchrun --nproc_per_node=1 --master_port=$((MASTER_PORT+100)) \
      eval.py \
      --config-name=phi1-5_tofu.yaml \
      task_id=$TASK_ID \
      use_LoRA=false \
      forget_coeff=1.0 \
      regularization_coeff=1.0 \
      lr=1e-5 \
      forget_lr=1e-5 \
      split=forget05 \
      forget_loss=IDK+GD \
      num_epochs=5 \
      mask=true \
      fix_ref_model=false \
      save_root=$SAVE_ROOT \
      save_checkpoint=false \
      alternate=true \
      optim_cfg=$OPTIM_CFG \
      retain_freq=5 \
      alpha=1.0 \
      beta1=$BETA1 \
      beta2=$BETA2 \
      base_beta1=$BASE_BETA1 \
      base_beta2=$BASE_BETA2 \
      max_steps=300 \
      save_steps=last \
      eval_unlearn_step=last

    # Find the latest result file for this task
    UNLEARN_RESULT=$(find $SAVE_ROOT -path "*/unlearn_times_1/unlearning_results.txt" -newer $SAVE_ROOT -type f 2>/dev/null | tail -1)
    if [ -n "$UNLEARN_RESULT" ] && [ -f "$UNLEARN_RESULT" ]; then
        # Extract metrics using grep
        UFE=$(grep "Untargeted Forget Efficacy" "$UNLEARN_RESULT" | awk '{print $NF}')
        TFE=$(grep "Targeted Forget Efficacy" "$UNLEARN_RESULT" | awk '{print $NF}')
        MU=$(grep "^Model Utility" "$UNLEARN_RESULT" | awk '{print $NF}')
        if [ -n "$UFE" ] && [ -n "$TFE" ] && [ -n "$MU" ]; then
            OVR=$(python3 -c "print(round(0.25*($TFE + $UFE) + 0.5*$MU, 6))")
            echo "Task $TASK_ID: UFE=$UFE TFE=$TFE MU=$MU OVR=$OVR"
            echo "$TASK_ID $UFE $TFE $MU $OVR" >> $RESULTS_FILE
        else
            echo "Task $TASK_ID: Could not parse metrics from $UNLEARN_RESULT"
        fi
    fi
done

echo ""
echo "===== Aggregated Results ====="
cat $RESULTS_FILE
echo ""
python3 << 'PYEOF'
import sys
results_file = "./llm_unlearn_results_v2/aggregated_results.txt"
lines = open(results_file).read().strip().split('\n')[1:]
ufes, tfes, mus, ovrs = [], [], [], []
for line in lines:
    parts = line.split()
    if len(parts) == 5:
        ufes.append(float(parts[1]))
        tfes.append(float(parts[2]))
        mus.append(float(parts[3]))
        ovrs.append(float(parts[4]))
if ufes:
    avg_ufe = sum(ufes) / len(ufes)
    avg_tfe = sum(tfes) / len(tfes)
    avg_mu = sum(mus) / len(mus)
    avg_ovr = sum(ovrs) / len(ovrs)
    n = len(ufes)
    print(f"Number of runs: {n}")
    print(f"Average UFE: {avg_ufe:.6f} (scaled: {avg_ufe*100:.2f})")
    print(f"Average TFE: {avg_tfe:.6f} (scaled: {avg_tfe*100:.2f})")
    print(f"Average MU:  {avg_mu:.6f} (scaled: {avg_mu*100:.2f})")
    print(f"Average OVR: {avg_ovr:.6f} (scaled: {avg_ovr*100:.2f})")
    print(f"\nPaper reference values:")
    print(f"  UFE: 67.63 (CI: 67.14-72.55)")
    print(f"  TFE: 67.60 (CI: 64.34-67.93)")
    print(f"  MU:  51.52 (CI: 49.50-51.72)")
    print(f"  OVR: 59.57 (CI: 57.96-59.73)")
else:
    print("No valid results found")
PYEOF
