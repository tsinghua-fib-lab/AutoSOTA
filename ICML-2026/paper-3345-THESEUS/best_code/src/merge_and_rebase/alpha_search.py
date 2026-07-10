from __future__ import annotations

from dataclasses import dataclass, field


def average_scores(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass
class PerTaskAlphaTracker:
    task_names: list[str]
    initial_alpha: float = 0.0
    patience: int = 0
    best_alpha: list[float] = field(init=False)
    best_baseline_acc: list[float] = field(init=False)
    best_rebase_acc: list[float] = field(init=False)
    active: list[bool] = field(init=False)
    bad_steps: list[int] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.task_names)
        self.best_alpha = [float(self.initial_alpha)] * n
        self.best_baseline_acc = [0.0] * n
        self.best_rebase_acc = [float("-inf")] * n
        self.active = [True] * n
        self.bad_steps = [0] * n

    def active_indices(self) -> list[int]:
        return [i for i, is_active in enumerate(self.active) if is_active]

    def update(
        self,
        *,
        alpha: float,
        indices: list[int],
        baseline_accs: list[float],
        rebase_accs: list[float],
    ) -> list[int]:
        if len(indices) != len(baseline_accs) or len(indices) != len(rebase_accs):
            raise ValueError("indices, baseline_accs, and rebase_accs must have the same length.")

        stopped: list[int] = []
        eps = 1e-12
        for idx, baseline_acc, rebase_acc in zip(indices, baseline_accs, rebase_accs, strict=True):
            baseline_acc = float(baseline_acc)
            rebase_acc = float(rebase_acc)
            best_rebase_acc = float(self.best_rebase_acc[idx])
            if rebase_acc > best_rebase_acc + eps:
                self.best_alpha[idx] = float(alpha)
                self.best_baseline_acc[idx] = baseline_acc
                self.best_rebase_acc[idx] = rebase_acc
                self.bad_steps[idx] = 0
                continue
            if rebase_acc + eps >= best_rebase_acc:
                self.bad_steps[idx] = 0
                continue
            self.bad_steps[idx] += 1
            if self.bad_steps[idx] > self.patience:
                self.active[idx] = False
                stopped.append(idx)
        return stopped

    def best_avg(self) -> float:
        vals = [v for v in self.best_rebase_acc if v != float("-inf")]
        return average_scores(vals)
