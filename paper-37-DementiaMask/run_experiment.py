"""
Experiment script for DualFilter paper reproduction.
Runs 5 repetitions for alpha_train=0.2 (pz0=0.8333, pz1=0.1667, alpha_test=5.0)
Evaluates DF M_I at 10% ablation.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import random
import json
from pathlib import Path
import shutil
from transformers import set_seed

from src.model.weights_filter import DualFilter
from src.preprocessing.data_generator import cf_train_test_split

# Hyperparameters from paper
PZ0 = 0.8333
PZ1 = 0.1667
ALPHA_TEST = 5.0
N_TEST = 150
N_VALIDATION = 120
N_REPS = 5
NUM_EPOCHS = 30
MASK_RATIO = 0.10  # 10% ablation as stated in rubric
MASK_TYPE = 'I'    # Intersection

MODEL_NAME = '/models/'

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def compute_fpr(metrics_group):
    """FPR = 1 - recall_neg (recall for non-dementia label)"""
    return 1.0 - metrics_group['recall_neg']

all_auprc = []
all_delta_fpr = []

base_seed = 24  # data_seed starts from this (incremented per rep)

for rep in range(N_REPS):
    print(f"\n===== Repetition {rep+1}/{N_REPS} =====")

    random_state = base_seed + rep
    set_all_seeds(42)

    # Create train/eval/test split
    train_df, eval_df, test_df = cf_train_test_split(
        data_name="pitts",
        train_pos_z0=PZ0,
        train_pos_z1=PZ1,
        alpha_test=ALPHA_TEST,
        test_size=N_TEST,
        validation_size=N_VALIDATION,
        random_state=random_state,
        verbose=True
    )

    print(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}, Test size: {len(test_df)}")

    # Initialize model
    model = DualFilter(MODEL_NAME)

    # Train and apply mask
    model.train(
        train_df, eval_df, test_df,
        num_epochs=NUM_EPOCHS,
        mask_type=MASK_TYPE,
        mask_ratio=MASK_RATIO,
        ablat_rate=0.0,
        output_dir=Path(f'/tmp/train_logs_df_rep{rep}')
    )

    # Get predictions
    model.predict(test_df)

    # Cleanup checkpoint files to save disk space (weights already in memory)
    shutil.rmtree(Path(f'/tmp/train_logs_df_rep{rep}'), ignore_errors=True)

    mask_label = f'DF_{MASK_TYPE}_{MASK_RATIO}_0.0'
    print(f"\nMask label: {mask_label}")
    print(f"Full metrics: {json.dumps(model.metrics, indent=2, default=str)}")

    if mask_label in model.metrics:
        m = model.metrics[mask_label]
        auprc = m['full']['aps']

        # FPR per gender group
        # confounder_0 = male (gender=0), confounder_1 = female (gender=1)
        fpr_male = compute_fpr(m['confounder_0'])
        fpr_female = compute_fpr(m['confounder_1'])
        delta_fpr = abs(fpr_female - fpr_male)

        print(f"\nRep {rep+1}: AUPRC={auprc:.4f}, FPR_male={fpr_male:.4f}, FPR_female={fpr_female:.4f}, Delta_FPR={delta_fpr:.4f}")
        all_auprc.append(auprc)
        all_delta_fpr.append(delta_fpr)
    else:
        print(f"WARNING: mask_label {mask_label} not in metrics: {list(model.metrics.keys())}")

print("\n\n===== FINAL RESULTS =====")
print(f"AUPRC values: {all_auprc}")
print(f"Delta FPR values: {all_delta_fpr}")

if all_auprc:
    mean_auprc = np.mean(all_auprc)
    std_auprc = np.std(all_auprc)
    mean_delta_fpr = np.mean(all_delta_fpr)
    std_delta_fpr = np.std(all_delta_fpr)

    print(f"\nMean AUPRC: {mean_auprc:.4f} +/- {std_auprc:.4f}")
    print(f"Mean Delta FPR: {mean_delta_fpr:.4f} +/- {std_delta_fpr:.4f}")
    print(f"\n=== KEY RESULTS ===")
    print(f"AUPRC (DF M_I, alpha_train=0.2, 10% ablation): {mean_auprc:.4f}")
    print(f"Delta FPR (DF M_I, alpha_train=0.2, 10% ablation): {mean_delta_fpr:.4f}")
else:
    print("No results collected!")
