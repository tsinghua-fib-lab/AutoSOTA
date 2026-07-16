from dataclasses import dataclass
from logging import Logger
from math import cos, exp, pi
from typing import Callable

from transformers import TrainerCallback, TrainerControl, TrainerState

import wandb


@dataclass
class LinearSchedule:
    v_init: float
    v_final: float

    def __call__(self, t: float) -> float:
        return self.v_init + (self.v_final - self.v_init) * t


@dataclass
class CosineSchedule:
    v_init: float
    v_final: float
    eps: float = 0.008
    """
    From Improved DDPM (IDDPM) paper: https://arxiv.org/pdf/2102.09672
    """

    def __call__(self, t: float) -> float:
        return (
            self.v_final
            + (self.v_init - self.v_final)
            * cos((t + self.eps) / (1 + self.eps) * pi / 2) ** 2
        )


@dataclass
class ExponentialSchedule:
    v_init: float
    v_final: float  # unused
    decay_rate: float = 1.0

    def __call__(self, t: float) -> float:
        return self.v_init * exp(-self.decay_rate * t)


@dataclass
class StepSchedule:
    v_init: float
    v_final: float
    num_steps: int = 4

    def __call__(self, t: float) -> float:
        current_step = int(t * self.num_steps)
        if current_step >= self.num_steps:
            return self.v_final

        step_value = self.v_init + (self.v_final - self.v_init) * (
            current_step / self.num_steps
        )
        return step_value


@dataclass
class Scheduler:
    """
    General Scheduler for the training process in the model forward pass.

    NOTE: this is not a learning rate scheduler

    Args:
        schedule_name: schedule type. Options are "linear", "exponential", "cosine", "step"
        init_value: initial value for the schedule
        final_value: final value for the schedule
        decay_rate: decay rate for the exponential decay schedule
        eps: epsilon for the cosine schedule (for numerical stability)
        num_steps: number of steps for the step schedule
        epoch_stop: epoch (as a fraction of total epochs) at which to stop the schedule
    """

    schedule_name: str
    init_value: float
    final_value: float
    decay_rate: float = 8.0
    eps: float = 0.008
    num_steps: int = 4
    epoch_stop: float = 1.0

    def __post_init__(self):
        if self.epoch_stop > 1.0 or self.epoch_stop < 0.0:
            raise ValueError("Epoch stop must be between 0.0 and 1.0")

        self.schedule_fn = {
            "linear": LinearSchedule(self.init_value, self.final_value),
            "cosine": CosineSchedule(self.init_value, self.final_value, self.eps),
            "exponential": ExponentialSchedule(self.init_value, self.decay_rate),
            "step": StepSchedule(self.init_value, self.final_value, self.num_steps),
        }[self.schedule_name]

    def __call__(self, epoch: float) -> float:
        t = epoch / self.epoch_stop
        schedule_param = self.schedule_fn(t)
        if t >= 1.0:  # stop if epoch is greater than or equal to the stop epoch
            schedule_param = self.final_value
        return schedule_param


@dataclass
class SchedulerLoggingCallback(TrainerCallback):
    scheduler: Callable
    logger: Logger
    log_interval: int = 100
    log_value_name: str = "schedule_param"

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        should_log = state.global_step % self.log_interval == 0 or state.epoch == 1.0
        if should_log and state.epoch < self.scheduler.epoch_stop:  # type: ignore
            epoch = float(state.epoch)  # type: ignore
            schedule_param = self.scheduler(epoch)

            if wandb.run is not None and args.report_to:
                for report in args.report_to:
                    if report == "wandb":
                        wandb.log(
                            {self.log_value_name: schedule_param},
                            step=state.global_step,
                        )
                    elif report == "tensorboard":
                        if control.should_log:
                            if not hasattr(self, "log_history"):
                                self.log_history = []
                            self.log_history.append(
                                {self.log_value_name: schedule_param}
                            )
