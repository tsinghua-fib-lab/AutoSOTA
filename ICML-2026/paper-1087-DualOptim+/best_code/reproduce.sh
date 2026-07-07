#!/bin/bash
# Reproduction script for DualOptim+ paper (ID 1087)
# Reproduces DO+ (dual_adam_plus) on TOFU forget05, Phi-1.5, IDK+GD
# Averaged over 5 forget sets (tasks 1-5)

set -e
cd /repo
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
unset HF_ENDPOINT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_ROOT="./llm_unlearn_results"
OPTIM_CFG="dual_adam_plus"
BETA1=0.9
BETA2=0.95
BASE_BETA1=0.9
BASE_BETA2=0.95

echo "DualOptim+ Reproduction: forget05, Phi-1.5, IDK+GD, 5 runs"
echo "============================================================"

for TASK_ID in 1 2 3 4 5; do
    echo ""
    echo "===== Task $TASK_ID / 5 ====="
    export TASK_LIST=$TASK_ID
    TRAIN_PORT=$((29500 + TASK_ID))
    EVAL_PORT=$((TRAIN_PORT + 100))

    echo "[Train] forget.py task_id=$TASK_ID ..."
    torchrun --nproc_per_node=2 --master_port=$TRAIN_PORT \
      forget.py --config-name=phi1-5_tofu.yaml \
      task_id=$TASK_ID use_LoRA=false forget_coeff=1.0 regularization_coeff=1.0 \
      lr=1e-5 forget_lr=1e-5 split=forget05 forget_loss=IDK+GD \
      num_epochs=5 mask=true fix_ref_model=false save_root=$SAVE_ROOT \
      save_checkpoint=false alternate=true optim_cfg=$OPTIM_CFG retain_freq=5 \
      alpha=1.0 beta1=$BETA1 beta2=$BETA2 base_beta1=$BASE_BETA1 base_beta2=$BASE_BETA2 \
      max_steps=300 save_steps=last

    echo "[Eval] eval.py task_id=$TASK_ID ..."
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$EVAL_PORT \
      eval.py --config-name=phi1-5_tofu.yaml \
      task_id=$TASK_ID use_LoRA=false forget_coeff=1.0 regularization_coeff=1.0 \
      lr=1e-5 forget_lr=1e-5 split=forget05 forget_loss=IDK+GD \
      num_epochs=5 mask=true fix_ref_model=false save_root=$SAVE_ROOT \
      save_checkpoint=false alternate=true optim_cfg=$OPTIM_CFG retain_freq=5 \
      alpha=1.0 beta1=$BETA1 beta2=$BETA2 base_beta1=$BASE_BETA1 base_beta2=$BASE_BETA2 \
      max_steps=300 save_steps=last eval_unlearn_step=last
done

echo ""
echo "===== Aggregated Results ====="
python3 -c "
import os, re
ufes, tfes, mus = [], [], []
for root, dirs, files in os.walk('$SAVE_ROOT'):
    for f in files:
        if f == 'unlearning_results.txt':
            m = re.search(r'forget05_task_(\d+)', root)
            if not m: continue
            tid = int(m.group(1))
            with open(os.path.join(root, f)) as fh:
                content = fh.read()
            ufe = tfe = mu = None
            for line in content.split('\n'):
                if line.startswith('Untargeted Forget Efficacy:'):
                    ufe = float(line.split(':')[1].strip())
                elif line.startswith('Targeted Forget Efficacy:'):
                    tfe = float(line.split(':')[1].strip())
                elif line.strip().startswith('Model Utility:') and 'Retain' not in line:
                    mu = float(line.split(':')[1].strip())
            if ufe and tfe and mu:
                ufes.append(ufe); tfes.append(tfe); mus.append(mu)
                ovr = 0.25*(tfe+ufe) + 0.5*mu
                print(f'Task {tid}: UFE={ufe*100:.2f} TFE={tfe*100:.2f} MU={mu*100:.2f} OVR={ovr*100:.2f}')
n = len(ufes)
avg_ufe = sum(ufes)/n; avg_tfe = sum(tfes)/n; avg_mu = sum(mus)/n
avg_ovr = 0.25*(avg_tfe+avg_ufe) + 0.5*avg_mu
print(f'\nAverage ({n} runs): UFE={avg_ufe*100:.2f} TFE={avg_tfe*100:.2f} MU={avg_mu*100:.2f} OVR={avg_ovr*100:.2f}')
print(f'Paper:           UFE=67.63   TFE=67.60   MU=51.52   OVR=59.57')
"
