"""Re-run TAP with full-dataset TabDiff and fixed anchor mapping."""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tabcamel.data.dataset import TabularDataset
from config import get_default_config
from generators import TabDiffGenerator
from tap import TAPTrainer
from utils import set_seed, setup_logging
from environment import DataGenerationEnv
from kto_controller import KTOAgent

set_seed(42)

# Load data
dataset = TabularDataset(dataset_name='MiceProtein', task_type='classification')
full_df = dataset.data_df.copy()
target_col = dataset.target_col

# Sample 20 real rows
def sample_real_rows(df, tc, n_rows, seed):
    if n_rows >= len(df):
        return df.reset_index(drop=True)
    try:
        return df.groupby(tc, group_keys=False).sample(
            frac=n_rows/len(df), random_state=seed
        ).sample(n=n_rows, replace=False, random_state=seed).reset_index(drop=True)
    except ValueError:
        pass
    return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)

np.random.seed(42)

# Sample 20 rows, preserving original indices
def sample_real_rows_with_idx(df, tc, n_rows, seed):
    if n_rows >= len(df):
        return df.reset_index(drop=True), list(range(len(df)))
    try:
        sampled = (
            df.groupby(tc, group_keys=False)
            .sample(frac=n_rows/len(df), random_state=seed)
            .sample(n=n_rows, replace=False, random_state=seed)
        )
        return sampled.reset_index(drop=True), list(sampled.index)
    except ValueError:
        pass
    sampled = df.sample(n=n_rows, random_state=seed)
    return sampled.reset_index(drop=True), list(sampled.index)

real_df, real_full_indices = sample_real_rows_with_idx(full_df, target_col, 20, 42)
print("Real train:", len(real_df), "rows")
print("Full indices:", real_full_indices)
assert len(real_full_indices) == 20, f"Only found {len(real_full_indices)} matches"

# Load pre-trained full-dataset TabDiff
model_dir = 'runs_v3/MiceProtein_n20/model'
generator = TabDiffGenerator(model_dir, model_dir + '/data', 'cuda')
print("Loaded TabDiff. Encoded data shape:", generator._encoded_train_data.shape)

# Config
config = get_default_config()
config.data.task_type = 'classification'
config.data.target_column = target_col
config.generator.device = 'cuda'
config.train.seed = 42
config.train.checkpoint_dir = 'runs_v3/MiceProtein_n20/checkpoints'
config.train.log_dir = 'runs_v3/MiceProtein_n20/logs'

# Setup
checkpoint_dir = Path(config.train.checkpoint_dir)
log_dir = Path(config.train.log_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(str(log_dir))

env = DataGenerationEnv(config, generator, real_df)

# Patch anchor selection to use full-dataset indices
def patched_select(target_idx, anchor_rule):
    y = env.D_real[env.target_col].values
    if env.is_regression:
        bins = env._get_bin(y)
        mask = (bins == target_idx)
    else:
        y_int = y.astype(int)
        mask = (y_int == target_idx)
    candidate_indices = np.where(mask)[0]
    if len(candidate_indices) == 0:
        return np.array([], dtype=int)
    n_anchors = min(
        config.inpaint.samples_per_step // config.inpaint.samples_per_anchor,
        len(candidate_indices)
    )
    rule = env.anchor_rules[anchor_rule]
    X = env.D_real.iloc[candidate_indices][env.columns].values

    if rule == 'high_uncertainty':
        if env.is_regression:
            preds = env.tabpfn.predict(X)
            residuals = np.abs(y[candidate_indices].astype(float) - preds)
            selected_local = np.argsort(residuals)[-n_anchors:]
        else:
            probs = env.tabpfn.predict_proba(X)
            entropy = -np.sum(probs * np.log(probs.clip(1e-10, 1)), axis=1)
            selected_local = np.argsort(entropy)[-n_anchors:]
        result = candidate_indices[selected_local]
    elif rule == 'high_error':
        y_cand = y[candidate_indices]
        preds = env.tabpfn.predict(X)
        if env.is_regression:
            residuals = np.abs(y_cand.astype(float) - preds)
            selected_local = np.argsort(residuals)[-n_anchors:]
            result = candidate_indices[selected_local]
        else:
            wrong_local = np.where(preds != y_cand.astype(int))[0]
            if len(wrong_local) >= n_anchors:
                result = candidate_indices[wrong_local[:n_anchors]]
            else:
                result = candidate_indices[:min(n_anchors, len(candidate_indices))]
    elif rule == 'minority_class':
        result = candidate_indices[:min(n_anchors, len(candidate_indices))]
    else:
        result = np.random.choice(
            candidate_indices,
            size=min(n_anchors, len(candidate_indices)),
            replace=False
        )

    full_indices = np.array([real_full_indices[i] for i in result])
    return full_indices

env._select_anchor_indices = patched_select

agent = KTOAgent(
    state_dim=env.get_state_dim(),
    num_targets=env.num_targets,
    num_templates=env.num_templates,
    config=config.kto,
    device='cuda',
)

best_reward = float('-inf')
best_step = 0
logger.info('Training TAP for 200 steps (full-dataset TabDiff)')

for step in range(200):
    state = env.get_state()
    action = agent.select_action(state)
    splits = [(np.arange(len(env.D_real)), np.arange(len(env.D_real)))]
    reward, info = env.step(action, splits=splits)
    if info['n_passed'] > 0:
        env.proposals.append({'batch': info['passed_batch'], 'ig': reward})
    loss = agent.update(state, action, reward)
    if reward > best_reward:
        best_reward = reward
        best_step = step
    if (step + 1) % config.inpaint.commit_interval == 0:
        env.commit_top_proposals()
    if step % config.train.log_every == 0:
        logger.info(
            'step=%d reward=%+.4f generated=%d passed=%d loss=%.4f',
            step, reward, info['n_generated'], info['n_passed'], loss,
        )
    if len(env.synthetic_buffer) >= 500:
        break

if env.proposals:
    env.commit_top_proposals()

synthetic = env.get_synthetic_data()
if len(synthetic) > 500:
    synthetic = synthetic.sample(n=500, random_state=42)
synthetic = synthetic.reset_index(drop=True)

out_path = 'runs_v3/MiceProtein_n20/synthetic_data.csv'
synthetic.to_csv(out_path, index=False)
logger.info('Finished! Wrote %d rows to %s', len(synthetic), out_path)
print(f'Wrote {len(synthetic)} rows to {out_path}')
