#!/usr/bin/env bash
# Wrapper to evaluate TOP1-PG across seeds, one seed per Python process.
# This avoids CUDA resource buildup that causes intermittent SIGSEGV.
set -eo pipefail

SEEDS=${1:-10}
START_SEED=${2:-0}
N_EPOCH=${3:-5000}
K=${4:-50}
DEVICE=${5:-cuda:0}
OUTPUT=${6:-/repo/results.json}
MAX_TRIES=${7:-3}
DIM_EMB=${8:-10}
N_MOE=${9:-1}
LR=${10:-0.01}

ALL_VALS=()
RESULTS_JSON="{"
SUCCESS_COUNT=0

for ((s=START_SEED; s<START_SEED+SEEDS; s++)); do
    SEED_OUT="/tmp/results_seed_${s}.json"

    for ((attempt=1; attempt<=MAX_TRIES; attempt++)); do
        echo "=== Seed $s (attempt $attempt/$MAX_TRIES) ==="

        cd /repo
        PYTHONPATH=/repo:${PYTHONPATH:-} python3 -c "
import sys, os, json
sys.path.insert(0, '/repo')
os.chdir('/repo/experiments/synthetic')
from experiments.synthetic.function_kuairec import setup_data_generation_process, initialize_trainable_policy, train_online_pg_policy
import torch, numpy as np

device = torch.device('${DEVICE}')
env = setup_data_generation_process(
    dataset_path='/repo/experiments/synthetic/data/kuairec_small_matrix.csv',
    n_output_action=1, device=device, random_seed=12345)
torch.manual_seed(${s})
if torch.cuda.is_available():
    torch.cuda.manual_seed(${s})
policy, _ = initialize_trainable_policy(
    env=env, dim_model_emb=${DIM_EMB}, n_moe_model=${N_MOE},
    device=device, random_seed=${s})
_, logs = train_online_pg_policy(
    env=env, early_stage_policy=policy, early_stage_lr=${LR},
    late_stage_optimality='optimal', credit_assignment_type='TOP1',
    is_vanilla_replacement=False, n_epoch=${N_EPOCH}, n_epochs_per_log=100,
    n_candidate_action_train=${K}, n_candidate_action_eval=${K},
    device=device, random_seed=${s}, use_wandb=False)
val = float(logs['policy_values'][-1].item())
history = [float(x) for x in logs['policy_values'].cpu().tolist()]
config = {'dim_emb': ${DIM_EMB}, 'n_moe': ${N_MOE}, 'lr': ${LR}, 'K': ${K}, 'n_epoch': ${N_EPOCH}}
result = {'seed': ${s}, 'policy_value': val, 'history': history, 'config': config}
with open('${SEED_OUT}', 'w') as f:
    json.dump(result, f)
print('SEED_RESULT: seed=${s} val=' + str(val), flush=True)
" 2>&1 | grep "SEED_RESULT" || true

        if [ -f "$SEED_OUT" ]; then
            SEED_VAL=$(python3 -c "import json; print(json.load(open('${SEED_OUT}'))['policy_value'])")
            echo "Seed $s: policy_value = $SEED_VAL (saved)"
            ALL_VALS+=("$SEED_VAL")
            if [ $SUCCESS_COUNT -gt 0 ]; then
                RESULTS_JSON+=", "
            fi
            HISTORY=$(python3 -c "import json; print(json.dumps(json.load(open('${SEED_OUT}'))['history']))")
            RESULTS_JSON+="\"seed_${s}\": {\"policy_value\": ${SEED_VAL}, \"history\": ${HISTORY}}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            break
        else
            echo "Seed $s attempt $attempt: no output file (likely crashed), retrying..."
            sleep 2
        fi
    done

    if [ ! -f "$SEED_OUT" ]; then
        echo "ERROR: Seed $s failed after $MAX_TRIES attempts"
        exit 1
    fi
done

RESULTS_JSON+="}"

# Compute mean and std
MEAN=$(python3 -c "
import numpy as np
vals = [${ALL_VALS[*]}]
print(np.mean(vals))
")
STD=$(python3 -c "
import numpy as np
vals = [${ALL_VALS[*]}]
print(np.std(vals))
")

python3 -c "
import json
per_seed = json.loads('''${RESULTS_JSON}''')
output = {
    'method': 'TOP1-PG (CA-PG-SwR)',
    'benchmark': 'KuaiRec',
    'K': ${K},
    'n_epoch': ${N_EPOCH},
    'gradient_steps': ${N_EPOCH} * 10,
    'n_seeds': ${SUCCESS_COUNT},
    'policy_value_mean': ${MEAN},
    'policy_value_std': ${STD},
    'config': {'dim_emb': ${DIM_EMB}, 'n_moe': ${N_MOE}, 'lr': ${LR}},
    'per_seed': per_seed,
}
with open('${OUTPUT}', 'w') as f:
    json.dump(output, f, indent=2)
print()
print('=== FINAL RESULT ===')
print('Policy Value: {:.4f} +/- {:.4f}'.format(${MEAN}, ${STD}))
print('Saved to ${OUTPUT}')
"
