"""TAP policy training."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from config import Config, get_default_config
from environment import DataGenerationEnv
from generators.base import BaseGenerator
from kto_controller import KTOAgent
from utils import save_checkpoint, set_seed, setup_logging


class TAPTrainer:
    """Train a TAP policy and return committed synthetic rows."""

    def __init__(self, config: Optional[Config] = None, task_type: str = "classification"):
        self.config = config or get_default_config()
        self.config.data.task_type = task_type
        set_seed(self.config.train.seed)

    def train_policy(
        self,
        train_data: Union[str, pd.DataFrame],
        generator: BaseGenerator,
        target_col: str,
        num_steps: Optional[int] = None,
        final_samples: int = 500,
    ) -> pd.DataFrame:
        train_df = pd.read_csv(train_data) if isinstance(train_data, str) else train_data.copy()
        self.config.data.target_column = target_col
        if num_steps is not None:
            self.config.train.num_steps = num_steps

        checkpoint_dir = Path(self.config.train.checkpoint_dir)
        log_dir = Path(self.config.train.log_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(str(log_dir))

        env = DataGenerationEnv(self.config, generator, train_df)
        agent = KTOAgent(
            state_dim=env.get_state_dim(),
            num_targets=env.num_targets,
            num_templates=env.num_templates,
            config=self.config.kto,
            device=self.config.generator.device,
        )

        best_reward = float("-inf")
        best_step = 0
        last_step = 0
        logger.info("Training TAP for %s steps", self.config.train.num_steps)

        for step in range(self.config.train.num_steps):
            last_step = step
            state = env.get_state()
            action = agent.select_action(state)
            reward, info = env.step(action, splits=self._reward_splits(len(env.D_real)))

            if info["n_passed"] > 0:
                env.proposals.append({"batch": info["passed_batch"], "ig": reward})

            loss = agent.update(state, action, reward)
            if reward > best_reward:
                best_reward = reward
                best_step = step

            if (step + 1) % self.config.inpaint.commit_interval == 0:
                env.commit_top_proposals()

            if step % self.config.train.log_every == 0:
                logger.info(
                    "step=%d reward=%+.4f generated=%d passed=%d loss=%.4f",
                    step,
                    reward,
                    info["n_generated"],
                    info["n_passed"],
                    loss,
                )

            if len(env.synthetic_buffer) >= final_samples:
                break

        if env.proposals:
            env.commit_top_proposals()

        synthetic = env.get_synthetic_data()
        if len(synthetic) > final_samples:
            synthetic = synthetic.sample(n=final_samples, random_state=self.config.train.seed)

        save_checkpoint(
            str(checkpoint_dir),
            last_step,
            agent,
            {"num_targets": env.num_targets, "num_templates": env.num_templates},
            best_reward,
        )
        logger.info(
            "Finished TAP: synthetic_rows=%d best_reward=%+.4f best_step=%d",
            len(synthetic),
            best_reward,
            best_step,
        )
        return synthetic.reset_index(drop=True)

    @staticmethod
    def _reward_splits(n_rows: int, n_splits: int = 5, support_ratio: float = 0.8):
        if n_rows < 2:
            idx = np.arange(n_rows)
            return [(idx, idx) for _ in range(n_splits)]

        n_support = max(1, min(n_rows - 1, int(n_rows * support_ratio)))
        splits = []
        for _ in range(n_splits):
            idx = np.random.permutation(n_rows)
            splits.append((idx[:n_support], idx[n_support:]))
        return splits
