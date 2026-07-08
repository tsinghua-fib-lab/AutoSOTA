"""Modified TAP runner: trains TabDiff on full dataset, runs TAP on 20-row subset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabcamel.data.dataset import TabularDataset

from config import get_default_config
from generators import TabDiffGenerator, train_tabdiff
from tap import TAPTrainer
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MiceProtein")
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--n_real", type=str, default="20")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--final_samples", type=int, default=500)
    parser.add_argument("--gen_steps", type=int, default=8000)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--full_data_train", action="store_true",
                        help="Train TabDiff on full dataset (not just n_real rows)")
    return parser.parse_args()


def sample_real_rows(df, target_col, n_rows, seed):
    if n_rows >= len(df):
        return df.reset_index(drop=True)
    try:
        return (
            df.groupby(target_col, group_keys=False)
            .sample(frac=n_rows / len(df), random_state=seed)
            .sample(n=n_rows, replace=False, random_state=seed)
            .reset_index(drop=True)
        )
    except ValueError:
        pass
    return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load full dataset
    dataset = TabularDataset(dataset_name=args.dataset, task_type=args.task_type)
    full_df = dataset.data_df.copy()
    target_col = dataset.target_col
    print(f"Full dataset: {full_df.shape}, target={target_col}")

    # Sample n_real rows
    n_real = int(float(args.n_real))
    if n_real <= 1.0:
        n_real = int(len(full_df) * n_real)
    n_real = max(1, min(n_real, len(full_df)))

    np.random.seed(args.seed)
    real_df = sample_real_rows(full_df, target_col, n_real, args.seed)
    print(f"Real train rows: {len(real_df)}")

    # Find indices of real rows in full_df (by matching values)
    real_full_indices = []
    for real_idx in real_df.index:
        real_row = real_df.loc[real_idx]
        for full_idx in full_df.index:
            if (real_row == full_df.loc[full_idx]).all():
                real_full_indices.append(full_idx)
                break
    print(f"Real row indices in full dataset: {real_full_indices}")

    run_dir = Path(args.output_dir) / f"{args.dataset}_n{n_real}"
    model_dir = run_dir / "model"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.full_data_train:
        # Train TabDiff on full dataset
        generator = train_tabdiff(
            train_data=full_df,
            target_col=target_col,
            save_path=str(model_dir),
            steps=args.gen_steps,
            device=args.device,
            seed=args.seed,
            task_type=args.task_type,
        )
        print(f"TabDiff trained on full dataset ({len(full_df)} rows)")
        # Save anchor mapping
        with open(model_dir / "anchor_indices.json", "w") as f:
            json.dump({"real_full_indices": real_full_indices}, f)
    else:
        # Train TabDiff only on real rows (original behavior)
        generator = train_tabdiff(
            train_data=real_df,
            target_col=target_col,
            save_path=str(model_dir),
            steps=args.gen_steps,
            device=args.device,
            seed=args.seed,
            task_type=args.task_type,
        )

    config = get_default_config()
    config.data.task_type = args.task_type
    config.data.target_column = target_col
    config.generator.device = args.device
    config.train.seed = args.seed
    config.train.checkpoint_dir = str(run_dir / "checkpoints")
    config.train.log_dir = str(run_dir / "logs")

    trainer = TAPTrainer(config=config, task_type=args.task_type)

    if args.full_data_train:
        # Patch the environment to use full-dataset anchor indices
        # We monkey-patch at runtime
        synthetic = _train_with_full_generator(
            trainer, real_df, generator, target_col,
            real_full_indices, args.num_steps, args.final_samples,
        )
    else:
        synthetic = trainer.train_policy(
            train_data=real_df,
            generator=generator,
            target_col=target_col,
            num_steps=args.num_steps,
            final_samples=args.final_samples,
        )

    out_path = run_dir / "synthetic_data.csv"
    synthetic.to_csv(out_path, index=False)
    print(f"Wrote {len(synthetic)} rows to {out_path}")


def _train_with_full_generator(trainer, real_df, generator, target_col,
                                real_full_indices, num_steps, final_samples):
    """Run TAP with a generator trained on the full dataset.

    Monkey-patches the environment to use full-dataset anchor indices.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from environment import DataGenerationEnv
    from kto_controller import KTOAgent
    from utils import save_checkpoint, setup_logging

    config = trainer.config
    train_df = real_df.copy()
    config.data.target_column = target_col
    if num_steps is not None:
        config.train.num_steps = num_steps

    checkpoint_dir = Path(config.train.checkpoint_dir)
    log_dir = Path(config.train.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_dir))

    env = DataGenerationEnv(config, generator, train_df)

    # Monkey-patch: replace anchor selection to use full-dataset indices
    orig_select = env._select_anchor_indices
    def patched_select(target_idx, anchor_rule):
        """Select anchors using full-dataset indices."""
        y = env.D_real[env.target_col].values

        if env.is_regression:
            bins = env._get_bin(y)
            mask = (bins == target_idx)
        else:
            mask_int = (y.astype(int) == target_idx)
            # Handle encoded labels
            if y.dtype == 'object':
                mask_int = (y == target_idx)
            mask = mask_int

        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:
            return np.array([], dtype=int)

        n_anchors = min(
            config.inpaint.samples_per_step // config.inpaint.samples_per_anchor,
            len(candidate_indices)
        )

        rule = env.anchor_rules[anchor_rule]
        X = env.D_real.iloc[candidate_indices][env.columns].values

        if rule == "high_uncertainty":
            if env.is_regression:
                preds = env.tabpfn.predict(X)
                y_cand = y[candidate_indices]
                residuals = np.abs(y_cand - preds)
                selected_local = np.argsort(residuals)[-n_anchors:]
            else:
                probs = env.tabpfn.predict_proba(X)
                entropy = -np.sum(probs * np.log(probs.clip(1e-10, 1)), axis=1)
                selected_local = np.argsort(entropy)[-n_anchors:]
            result = candidate_indices[selected_local]
        elif rule == "high_error":
            y_cand = y[candidate_indices]
            preds = env.tabpfn.predict(X)
            if env.is_regression:
                residuals = np.abs(y_cand - preds)
                selected_local = np.argsort(residuals)[-n_anchors:]
            else:
                wrong_local = np.where(preds != y_cand)[0]
                if len(wrong_local) >= n_anchors:
                    result = candidate_indices[wrong_local[:n_anchors]]
                else:
                    selected_local = np.arange(min(n_anchors, len(candidate_indices)))
                    result = candidate_indices[selected_local]
            result = candidate_indices[selected_local] if 'result' not in dir() else result
        else:
            result = np.random.choice(candidate_indices, size=min(n_anchors, len(candidate_indices)), replace=False)

        # Map from D_real indices to full dataset indices
        full_indices = np.array([real_full_indices[i] for i in result])
        return full_indices

    env._select_anchor_indices = patched_select

    agent = KTOAgent(
        state_dim=env.get_state_dim(),
        num_targets=env.num_targets,
        num_templates=env.num_templates,
        config=config.kto,
        device=config.generator.device,
    )

    best_reward = float("-inf")
    best_step = 0
    last_step = 0
    logger.info("Training TAP for %s steps (full-dataset TabDiff)", config.train.num_steps)

    for step in range(config.train.num_steps):
        last_step = step
        state = env.get_state()
        action = agent.select_action(state)
        reward, info = env.step(action, splits=trainer._reward_splits(len(env.D_real)))

        if info["n_passed"] > 0:
            env.proposals.append({"batch": info["passed_batch"], "ig": reward})

        loss = agent.update(state, action, reward)
        if reward > best_reward:
            best_reward = reward
            best_step = step

        if (step + 1) % config.inpaint.commit_interval == 0:
            env.commit_top_proposals()

        if step % config.train.log_every == 0:
            logger.info(
                "step=%d reward=%+.4f generated=%d passed=%d loss=%.4f",
                step, reward, info["n_generated"], info["n_passed"], loss,
            )

        if len(env.synthetic_buffer) >= final_samples:
            break

    if env.proposals:
        env.commit_top_proposals()

    synthetic = env.get_synthetic_data()
    if len(synthetic) > final_samples:
        synthetic = synthetic.sample(n=final_samples, random_state=config.train.seed)

    save_checkpoint(
        str(checkpoint_dir), last_step, agent,
        {"num_targets": env.num_targets, "num_templates": env.num_templates},
        best_reward,
    )
    logger.info(
        "Finished TAP: synthetic_rows=%d best_reward=%+.4f best_step=%d",
        len(synthetic), best_reward, best_step,
    )
    return synthetic.reset_index(drop=True)


if __name__ == "__main__":
    main()
