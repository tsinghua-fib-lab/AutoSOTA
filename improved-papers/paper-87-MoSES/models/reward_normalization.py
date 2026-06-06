import math
import torch

from tensordict import TensorDict
from rl4co.utils.ops import unbatchify


class BaseValues:
    def __init__(self, init_val=0) -> None:
        self.cvrp = init_val
        self.ovrp = init_val
        self.vrpb = init_val
        self.vrpl = init_val
        self.vrptw = init_val

        self.ovrptw = init_val
        self.ovrpb = init_val
        self.ovrpl = init_val
        self.vrpbl = init_val
        self.vrpbtw = init_val

        self.vrpltw = init_val
        self.ovrpbl = init_val
        self.ovrpbtw = init_val
        self.ovrpltw = init_val
        self.vrpbltw = init_val

        self.ovrpbltw = init_val
        self.vrpmb = init_val
        self.ovrpmb = init_val
        self.vrpmbl = init_val
        self.vrpmbtw = init_val

        self.ovrpmbl = init_val
        self.ovrpmbtw = init_val
        self.vrpmbltw = init_val
        self.ovrpmbltw = init_val

    def apply_to_all(self, other, func):
        for attr in vars(self):
            self.apply_to_variant(other=other, attr=attr, func=func)

    def apply_to_variant(self, other, attr, func):
        setattr(self, attr, func(getattr(self, attr), getattr(other, attr)))


class RewardNormalization:
    def __init__(self) -> None:
        self.norm_vals = BaseValues()

    def __call__(
        self, td: TensorDict, rewards: torch.Tensor, operation: str
    ) -> torch.Tensor:
        normalized_rewards = rewards.clone()
        norm_vals = torch.zeros_like(normalized_rewards)
        new_vals = BaseValues()
        for variant in vars(new_vals):
            mask = self.get_problem_mask(problem_variant=variant, td=td)
            if mask.any():
                setattr(new_vals, variant, rewards[mask].mean().item())  # new mean reward
                self.update(new_vals=new_vals, variant=variant)  # update mean reward
                normalized_rewards[mask] = self.norm_reward(
                    attr=variant, mask=mask, rewards=rewards, operation=operation
                )
                norm_vals[mask] = getattr(self.norm_vals, variant)
        return normalized_rewards, norm_vals

    def norm_reward(self, attr, mask, rewards, operation: str):
        assert operation in ["div", "sub"], "Normalizing operation must be div or sub"
        # norm_val = getattr(self.norm_vals, attr)
        norm_val = getattr(self.norm_vals, attr)  # I assume we'd use the abs. val?
        if operation == "div":
            return rewards[mask] / abs(norm_val)
        elif operation == "sub":
            return rewards[mask] - norm_val

    @staticmethod
    def get_problem_mask(problem_variant: str, td: TensorDict) -> torch.Tensor:
        backhaul_1 = (td["backhaul_class"] == 1).squeeze(dim=-1)
        backhaul_2 = (td["backhaul_class"] == 2).squeeze(dim=-1)
        time_windows = td["time_windows"][..., 0, 1] != float("inf")  # depot end time
        open_route = td["open_route"].squeeze(dim=-1)
        distance_limit = (td["distance_limit"] != float("inf")).squeeze(dim=-1)
        mask_mapping = {
            "cvrp": (
                ~backhaul_1 & ~backhaul_2 & ~time_windows & ~open_route & ~distance_limit
            ),
            "ovrp": (
                ~backhaul_1 & ~backhaul_2 & ~time_windows & open_route & ~distance_limit
            ),
            "vrpb": (
                backhaul_1 & ~backhaul_2 & ~time_windows & ~open_route & ~distance_limit
            ),
            "vrpl": (
                ~backhaul_1 & ~backhaul_2 & ~time_windows & ~open_route & distance_limit
            ),
            "vrptw": (
                ~backhaul_1 & ~backhaul_2 & time_windows & ~open_route & ~distance_limit
            ),
            "ovrptw": (
                ~backhaul_1 & ~backhaul_2 & time_windows & open_route & ~distance_limit
            ),
            "ovrpb": (
                backhaul_1 & ~backhaul_2 & ~time_windows & open_route & ~distance_limit
            ),
            "ovrpl": (
                ~backhaul_1 & ~backhaul_2 & ~time_windows & open_route & distance_limit
            ),
            "vrpbl": (
                backhaul_1 & ~backhaul_2 & ~time_windows & ~open_route & distance_limit
            ),
            "vrpbtw": (
                backhaul_1 & ~backhaul_2 & time_windows & ~open_route & ~distance_limit
            ),
            "vrpltw": (
                ~backhaul_1 & ~backhaul_2 & time_windows & ~open_route & distance_limit
            ),
            "ovrpbl": (
                backhaul_1 & ~backhaul_2 & ~time_windows & open_route & distance_limit
            ),
            "ovrpbtw": (
                backhaul_1 & ~backhaul_2 & time_windows & open_route & ~distance_limit
            ),
            "ovrpltw": (
                ~backhaul_1 & ~backhaul_2 & time_windows & open_route & distance_limit
            ),
            "vrpbltw": (
                backhaul_1 & ~backhaul_2 & time_windows & ~open_route & distance_limit
            ),
            "ovrpbltw": (
                backhaul_1 & ~backhaul_2 & time_windows & open_route & distance_limit
            ),
            "vrpmb": (
                ~backhaul_1 & backhaul_2 & ~time_windows & ~open_route & ~distance_limit
            ),
            "ovrpmb": (
                ~backhaul_1 & backhaul_2 & ~time_windows & open_route & ~distance_limit
            ),
            "vrpmbl": (
                ~backhaul_1 & backhaul_2 & ~time_windows & ~open_route & distance_limit
            ),
            "vrpmbtw": (
                ~backhaul_1 & backhaul_2 & time_windows & ~open_route & ~distance_limit
            ),
            "ovrpmbl": (
                ~backhaul_1 & backhaul_2 & ~time_windows & open_route & distance_limit
            ),
            "ovrpmbtw": (
                ~backhaul_1 & backhaul_2 & time_windows & open_route & ~distance_limit
            ),
            "vrpmbltw": (
                ~backhaul_1 & backhaul_2 & time_windows & ~open_route & distance_limit
            ),
            "ovrpmbltw": (
                ~backhaul_1 & backhaul_2 & time_windows & open_route & distance_limit
            ),
        }
        return mask_mapping.get(problem_variant, torch.tensor([]))

    def update(self, new_vals: torch.Tensor, variant: str) -> torch.Tensor:
        raise NotImplementedError("Implement reward normalization in child classes")


class CumulativeMean(RewardNormalization):
    def __init__(self) -> None:
        super(CumulativeMean, self).__init__()
        self.n = 0

    def __call__(
        self, td: TensorDict, rewards: torch.Tensor, operation: str
    ) -> torch.Tensor:
        result = super().__call__(td=td, rewards=rewards, operation=operation)
        self.n += 1
        return result

    def update(self, new_vals: torch.Tensor, variant: str) -> torch.Tensor:
        self.norm_vals.apply_to_variant(
            other=new_vals,
            attr=variant,
            func=lambda x, y: (self.n * x + y) / (self.n + 1),
        )


class ExponentialMean(RewardNormalization):
    def __init__(self, alpha: float = 0.1) -> None:
        super(ExponentialMean, self).__init__()
        self.t = 0
        self.alpha = alpha

    def __call__(
        self, td: TensorDict, rewards: torch.Tensor, operation: str
    ) -> torch.Tensor:
        result = super().__call__(td=td, rewards=rewards, operation=operation)
        self.t += 1
        return result

    def update(self, new_vals: torch.Tensor, variant: str) -> torch.Tensor:
        if self.t == 0:
            self.norm_vals = new_vals
        else:
            self.norm_vals.apply_to_variant(
                other=new_vals,
                func=lambda x, y: ((1 - self.alpha) * x) + (self.alpha * y),
                attr=variant,
            )


class NoNormalization(RewardNormalization):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, td: TensorDict, rewards: torch.Tensor, operation: str
    ) -> torch.Tensor:
        return rewards, torch.zeros_like(rewards)


class ZNormalization(RewardNormalization):
    """
    Inspired by 'Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>.'
    """

    def __init__(self, alpha: float, epsilon: float = 0.0) -> None:
        super(ZNormalization, self).__init__()
        self.t = 0
        self.alpha = alpha
        self.epsilon = epsilon  # default for torch batch normalization is 1e-5
        self.running_var = BaseValues(init_val=1)

    def __call__(
        self, td: TensorDict, rewards: torch.Tensor, operation: str
    ) -> torch.Tensor:
        normalized_rewards = rewards.clone()
        norm_vals = torch.zeros_like(normalized_rewards)
        new_means = BaseValues()
        new_vars = BaseValues()
        for variant in vars(new_means):
            mask = self.get_problem_mask(problem_variant=variant, td=td)
            if mask.any():
                # new mean reward
                setattr(new_means, variant, rewards[mask].mean().item())
                # new variance
                setattr(new_vars, variant, rewards[mask].var().item())
                # update running mean and variance
                self.update(new_means=new_means, new_vars=new_vars, variant=variant)
                normalized_rewards[mask] = self.norm_reward(
                    attr=variant, mask=mask, rewards=rewards, operation=operation
                )
                norm_vals[mask] = getattr(self.norm_vals, variant)
        return normalized_rewards, norm_vals

    def update(
        self, new_means: torch.Tensor, new_vars: torch.Tensor, variant: str
    ) -> torch.Tensor:
        if self.t == 0:
            self.norm_vals = new_means
            self.running_var = new_vars
        else:
            self.norm_vals.apply_to_variant(
                other=new_means,
                func=lambda x, y: ((1 - self.alpha) * x) + (self.alpha * y),
                attr=variant,
            )
            self.running_var.apply_to_variant(
                other=new_vars,
                func=lambda x, y: ((1 - self.alpha) * x) + (self.alpha * y),
                attr=variant,
            )

    def norm_reward(self, attr, mask, rewards, operation: str):
        return (rewards[mask] - getattr(self.norm_vals, attr)) / math.sqrt(
            getattr(self.running_var, attr) + self.epsilon
        )
