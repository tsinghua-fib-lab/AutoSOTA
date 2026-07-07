#!/bin/bash
# Full 5-task evaluation for best config verification
set -e
cd /repo
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
unset HF_ENDPOINT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FORGET_COEFF=${1:-0.8}
FORGET_LR=${2:-1.4e-5}
SAVE_ROOT=${3:-./llm_unlearn_results_full}

rm -rf "$SAVE_ROOT"

UFES=""; TFES=""; MUS=""
for TASK_ID in 1 2 3 4 5; do
    export TASK_LIST=$TASK_ID
    TRAIN_PORT=$((29500 + TASK_ID))
    EVAL_PORT=$((TRAIN_PORT + 100))

    echo "===== Task $TASK_ID / 5 ====="
    echo "[Train] task_id=$TASK_ID"
    torchrun --nproc_per_node=2 --master_port=$TRAIN_PORT \
      forget.py --config-name=phi1-5_tofu.yaml \
      task_id=$TASK_ID use_LoRA=false forget_coeff=$FORGET_COEFF regularization_coeff=1.0 \
      lr=1e-5 forget_lr=$FORGET_LR split=forget05 forget_loss=IDK+GD \
      num_epochs=5 mask=true fix_ref_model=false save_root=$SAVE_ROOT \
      save_checkpoint=true alternate=true optim_cfg=dual_adam_plus retain_freq=5 \
      alpha=1.0 beta1=0.9 beta2=0.95 base_beta1=0.9 base_beta2=0.95 \
      max_steps=300 save_steps=last 2>&1 | tail -1

    echo "[Eval] task_id=$TASK_ID"
    RESULT=$(CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$EVAL_PORT \
      eval.py --config-name=phi1-5_tofu.yaml \
      task_id=$TASK_ID use_LoRA=false forget_coeff=$FORGET_COEFF regularization_coeff=1.0 \
      lr=1e-5 forget_lr=$FORGET_LR split=forget05 forget_loss=IDK+GD \
      num_epochs=5 mask=true fix_ref_model=false save_root=$SAVE_ROOT \
      save_checkpoint=true alternate=true optim_cfg=dual_adam_plus retain_freq=5 \
      alpha=1.0 beta1=0.9 beta2=0.95 base_beta1=0.9 base_beta2=0.95 \
      max_steps=300 save_steps=last eval_unlearn_step=last 2>&1 | grep "After Unlearn Task")
    
    UFE=$(echo "$RESULT" | grep -oP "Untargeted Forget Efficacy \K[0-9.]+")
    TFE=$(echo "$RESULT" | grep -oP "Targeted Forget Efficacy \K[0-9.]+")
    MU=$(echo "$RESULT" | grep -oP "Model Utility \K[0-9.]+")
    
    UFES="$UFES $UFE"; TFES="$TFES $TFE"; MUS="$MUS $MU"
    echo "Task $TASK_ID: UFE=$UFE TFE=$TFE MU=$MU"
    
    find "$SAVE_ROOT" -type d -name "checkpoint-last" -exec rm -rf {} + 2>/dev/null || true
done

echo ""
echo "===== Final Results ====="
python3 -c "
ufes = [$(echo $UFES | tr ' ' ',')]
tfes = [$(echo $TFES | tr ' ' ',')]
mus = [$(echo $MUS | tr ' ' ',')]
avg_ufe = sum(ufes)/5; avg_tfe = sum(tfes)/5; avg_mu = sum(mus)/5
avg_ovr = 0.25*(avg_tfe+avg_ufe) + 0.5*avg_mu
print(f'UFE={avg_ufe*100:.2f} TFE={avg_tfe*100:.2f} MU={avg_mu*100:.2f} OVR={avg_ovr*100:.2f}')
for i in range(5):
    ovr = 0.25*(tfes[i]+ufes[i]) + 0.5*mus[i]
    print(f'Task {i+1}: UFE={ufes[i]*100:.2f} TFE={tfes[i]*100:.2f} MU={mus[i]*100:.2f} OVR={ovr*100:.2f}')
"
