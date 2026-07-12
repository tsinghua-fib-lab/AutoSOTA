import torch


class ConvergenceTracker:
    """Tracks per-sample steps spent below given convergence thresholds."""

    def __init__(
        self,
        max_optimization_steps: int,
        batch_size: int,
        thresholds: tuple[float, ...] = (0.95, 0.99, 1.0),
        convergence_threshold: float = 1.0,
    ):
        self._buffers = {
            threshold: torch.zeros((max_optimization_steps, batch_size), dtype=torch.long) for threshold in thresholds
        }
        self.convergence_threshold = convergence_threshold
        self.fully_converged: torch.Tensor | None = None  # [batch]

    def update(self, step_i: int, convergence_per_sample: torch.Tensor) -> bool:
        """Record one step. Returns True if every sample reached ``convergence_threshold``."""
        for threshold, buffer in self._buffers.items():
            buffer[step_i, :] = (convergence_per_sample < threshold).to(torch.long)
        self.fully_converged = convergence_per_sample >= self.convergence_threshold
        return self.fully_converged.all().item()

    def steps_below(self, threshold: float) -> torch.Tensor:
        """Per-sample number of steps where convergence was strictly below `threshold`."""
        return self._buffers[threshold].sum(dim=0)


class ConvergedSamplesGuard:
    """Zeroes gradients and restores parameter values for samples already at full convergence."""

    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters  # [batch, compression, hidden]
        self._snapshot: torch.Tensor | None = None

    def before_step(self, converged_mask: torch.Tensor | None) -> None:
        """Zero out grads for frozen indices and snapshot their current values."""
        if converged_mask is None or self.parameters.grad is None:
            self._snapshot = None
        else:
            self.parameters.grad[converged_mask] = 0
            self._snapshot = self.parameters.detach().clone()

    def after_step(self, converged_mask: torch.Tensor | None) -> None:
        """Restore the pre-step values for frozen indices (undoing weight decay / momentum drift)."""
        if converged_mask is None or self._snapshot is None:
            return
        with torch.no_grad():
            self.parameters[converged_mask] = self._snapshot[converged_mask]


class ProgressiveSampleStateMachine:
    """Per-sample state for progressive cramming: active / converged-in-stage / permanently skipped."""

    def __init__(self, batch_size: int, threshold: float):
        self.batch_size = batch_size
        self.threshold = threshold
        self.skipped: list[bool] = [False] * batch_size
        self.converged_in_stage: list[bool] = [False] * batch_size
        self.steps_taken: list[int] = [0] * batch_size

    def reset_stage(self) -> None:
        """Start of a new stage; permanently skipped samples remain frozen."""
        self.converged_in_stage = [self.skipped[j] for j in range(self.batch_size)]

    def is_active(self, j: int) -> bool:
        """Sample j is still being optimized in the current stage."""
        return not self.skipped[j] and not self.converged_in_stage[j]

    def update(self, convergence_per_sample: torch.Tensor) -> bool:
        """Mark active samples that crossed the threshold. Returns True iff all samples done."""
        for j in range(self.batch_size):
            if self.is_active(j) and convergence_per_sample[j].item() >= self.threshold:
                self.converged_in_stage[j] = True
        return all(self.converged_in_stage[j] or self.skipped[j] for j in range(self.batch_size))

    def increment_steps(self, j: int) -> None:
        """Bump the per-sample optimizer step counter for active sample j."""
        self.steps_taken[j] += 1

    def mark_skipped_if_not_converged(self, seq_len: int) -> None:
        """End of stage: permanently skip samples that never reached the threshold this stage."""
        for j in range(self.batch_size):
            if self.is_active(j):
                self.skipped[j] = True
                print(f"Sample {j} failed to converge at seq_len={seq_len}, marking as skipped.")

    def mark_exhausted(self, max_steps_per_sample: int) -> bool:
        """Permanently skip active samples whose cumulative ``steps_taken`` reached the cap.

        Returns True iff every sample is now done (skipped or converged-in-stage),
        so the caller can short-circuit the stage loop.
        """
        for j in range(self.batch_size):
            if self.is_active(j) and self.steps_taken[j] >= max_steps_per_sample:
                self.skipped[j] = True
                print(
                    f"Sample {j} exhausted step budget "
                    f"({self.steps_taken[j]} >= {max_steps_per_sample}), marking as skipped."
                )
        return all(self.converged_in_stage[j] or self.skipped[j] for j in range(self.batch_size))

    @property
    def all_skipped(self) -> bool:
        return all(self.skipped)
