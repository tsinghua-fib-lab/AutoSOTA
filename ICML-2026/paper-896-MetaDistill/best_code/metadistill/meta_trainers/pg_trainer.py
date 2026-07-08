import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from optimizers.lde import LDE, LDETrainingInfo
from tasks import TaskProblem
from tasks.utils import genOffset
from torch_basic_settings import DEVICE, DTYPE


def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_training_info(info: Dict) -> LDETrainingInfo:
    if not isinstance(info, dict) or "training" not in info:
        raise ValueError("LDE forward() must return info dict with key 'training' (LDETrainingInfo).")
    training = info["training"]
    if not isinstance(training, LDETrainingInfo):
        raise ValueError("info['training'] must be an instance of LDETrainingInfo.")
    return training


@dataclass
class LDEPGStats:
    loss: float
    mean_return: float
    mean_final_best: float


class LDEPolicyGradientTrainer:
    """
    Policy Gradient trainer for LDE (IEEE TEVC 2021).

    Paper anchors:
    - Eq.(8): U_t statistics (histogram + moving average)
    - Eq.(9): Gaussian policy pi(A_t|S_t)
    - Eq.(10): reward as relative improvement of best fitness
    - Eq.(11)-(14) + Algorithm 1: REINFORCE update on W
    """

    def __init__(
        self,
        model: LDE,
        training_funcs: Sequence[dict],
        *,
        n_epochs: int,
        n_trajectories: int,
        traj_len: int,
        lr: float,
        seed: Optional[int] = None,
        ckpt_dir: Optional[str] = None,
        minimize: bool = True,
    ):
        self.model = model.to(device=DEVICE, dtype=DTYPE)
        self.training_funcs = list(training_funcs)

        self.n_epochs = int(n_epochs)
        self.n_trajectories = int(n_trajectories)
        self.traj_len = int(traj_len)
        self.lr = float(lr)

        self.seed = int(seed) if seed is not None else None
        self.ckpt_dir = ckpt_dir
        self.minimize = bool(minimize)

        if self.n_epochs <= 0 or self.n_trajectories <= 0 or self.traj_len <= 0:
            raise ValueError("n_epochs, n_trajectories and traj_len must be positive.")
        if len(self.training_funcs) == 0:
            raise ValueError("training_funcs cannot be empty.")

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _init_population(self, task: TaskProblem) -> torch.Tensor:
        pop = task.genRandomPop((1, self.model.popsize, self.model.problemdim))
        pop, _ = task.calfitness(pop)
        return pop

    def _trajectory(self, pop0: torch.Tensor, task: TaskProblem) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one trajectory of length T.
        Returns:
          sum_log_prob: scalar tensor
          sum_reward: scalar tensor
          final_best: scalar tensor
        """
        self.model.reset()
        pop = pop0
        sum_log_prob = torch.zeros((), device=DEVICE, dtype=DTYPE)
        sum_reward = torch.zeros((), device=DEVICE, dtype=DTYPE)

        for _ in range(self.traj_len):
            pop, info = self.model(pop, task)
            tr = _extract_training_info(info)
            sum_log_prob = sum_log_prob + tr.log_prob
            sum_reward = sum_reward + tr.reward

        best = pop[0, :, 0].min() if self.minimize else pop[0, :, 0].max()
        return sum_log_prob, sum_reward, best.detach()

    def train(self) -> List[LDEPGStats]:
        if self.ckpt_dir is not None:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        if self.seed is not None:
            _set_global_seed(self.seed)

        self.model.train()

        stats: List[LDEPGStats] = []
        best_loss = float("inf")

        epoch_bar = tqdm(range(self.n_epochs), ncols=100)
        for epoch in epoch_bar:
            loss = torch.zeros((), device=DEVICE, dtype=DTYPE)
            returns: List[float] = []
            finals: List[float] = []

            for func in self.training_funcs:
                # BBOB functions require offsets (xopt/fopt) to be set.
                if isinstance(func.get("fid", None), int):
                    genOffset(dim=self.model.problemdim, fun=func)

                task = TaskProblem(fun=func, repaire=True, dim=self.model.problemdim)
                pop0 = self._init_population(task).detach()

                for _ in range(self.n_trajectories):
                    sum_log_prob, sum_reward, final_best = self._trajectory(pop0, task)

                    # REINFORCE (Eq.(14)): maximize E[r(τ) * Σ log π]
                    loss = loss - (sum_reward.detach() * sum_log_prob)
                    returns.append(float(sum_reward.detach().item()))
                    finals.append(float(final_best.item()))

            denom = max(1, len(self.training_funcs) * self.n_trajectories)
            loss = loss / denom

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            mean_return = float(sum(returns) / max(1, len(returns)))
            mean_final_best = float(sum(finals) / max(1, len(finals)))
            loss_val = float(loss.detach().item())
            epoch_bar.set_description(f"epoch {epoch}/{self.n_epochs} loss={loss_val:.4e} R={mean_return:.4e}")

            cur = LDEPGStats(loss=loss_val, mean_return=mean_return, mean_final_best=mean_final_best)
            stats.append(cur)

            if self.ckpt_dir is not None:
                # Save best-on-loss checkpoint (paper does not specify checkpointing; keep it minimal).
                if loss_val < best_loss:
                    best_loss = loss_val
                    torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "lde_pg_best.pth"))

        return stats

