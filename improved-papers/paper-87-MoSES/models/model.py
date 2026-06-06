from typing import Any, Optional, Union
import os, sys
import torch
import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict
from models.policy import LoRAPolicy, MultiLoRAPolicy
from envs.mtvrp import MTVRPEnv


from models.reward_normalization import (
    CumulativeMean,
    ExponentialMean,
    NoNormalization,
    ZNormalization,
)

log = get_pylogger(__name__)


class RouteFinderBase(POMO):
    """
    Main RouteFinder RL model based on POMO.
    This automatically include the Mixed Batch Training (MBT) from the environment.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        use_aug: bool = False,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        alpha = kwargs.pop("alpha", 0.1)
        epsilon = kwargs.pop("epsilon", 1e-5)
        normalize_reward = kwargs.pop("normalize_reward", "none")
        self.norm_operation = kwargs.pop("norm_operation", "div")  # "div" or "sub"
        self.use_aug = use_aug

        # Initialize with the shared baseline
        super(RouteFinderBase, self).__init__(env, policy, **kwargs)

        allowed_normalizations = [
            "cumulative",
            "exponential",
            "none",
            "normal",
            "z",
            "z-score",
        ]
        assert (
            normalize_reward in allowed_normalizations
        ), f"normalize_reward must lie in {allowed_normalizations}."

        if normalize_reward == "cumulative":
            self.normalization = CumulativeMean()
        elif normalize_reward == "exponential":
            self.normalization = ExponentialMean(alpha=alpha)
        elif normalize_reward == "none":
            self.normalization = NoNormalization()
        elif normalize_reward in ["normal", "z", "z-score"]:
            self.normalization = ZNormalization(alpha=alpha, epsilon=epsilon)
        else:
            raise NotImplementedError("Normalization not implemented")

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        costs_bks = batch.get("costs_bks", None)

        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start

        # During training, we do not augment the data
        if self.use_aug:
            td = self.augment(td)
        else:
            if phase == "train":
                n_aug = 0
            elif n_aug > 1:
                td = self.augment(td)

        # Evaluate policy
        out = self.policy(
            td, self.env, phase=phase, num_starts=n_start, return_actions=True
        )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))

        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            normalized_reward, norm_vals = self.normalization(
                td=unbatchify(x=td, shape=n_aug),
                rewards=reward,
                operation=self.norm_operation,
            )
            out.update({"norm_vals": norm_vals, "norm_reward": normalized_reward})
            self.calculate_loss(td, batch, out, normalized_reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            max_norm_reward, _ = normalized_reward.max(dim=-1)
            out.update({"max_reward": max_reward, "max_norm_reward": max_norm_reward})
        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # note: torch.abs is here as a temporary fix, since we forgot to
                    # convert rewards to costs. Does not affect the results.
                    gap_to_bks = (
                        100
                        * (-max_aug_reward - torch.abs(costs_bks))
                        / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 100})  # Dummy value

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm.

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # REINFORCE baseline
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )

        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        advantage = self.advantage_scaler(advantage)
        reinforce_loss = -(advantage * log_likelihood).mean()
        loss = reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        return policy_out


class CadaModel(RouteFinderBase):
    def __init__(self,
                 env: RL4COEnvBase,
                 policy: nn.Module,
                 **kwargs,
                 ):
        super(CadaModel, self).__init__(env, policy, **kwargs)


class LoRABaseModel(RouteFinderBase):
    def __init__(
            self,
            env: RL4COEnvBase,
            policy: Union[LoRAPolicy, MultiLoRAPolicy],
            **kwargs,
    ):
        super(LoRABaseModel, self).__init__(env, policy, **kwargs)
        self.trainable_parameters = policy.lora_trainable_params()

    def configure_optimizers(self, parameters=None):
        return super(LoRABaseModel, self).configure_optimizers(parameters=self.trainable_parameters)



class MultiLoRAModel(RouteFinderBase):
    def __init__(
            self,
            env: RL4COEnvBase,
            policy: Union[LoRAPolicy, MultiLoRAPolicy],
            **kwargs,
    ):  
        super(MultiLoRAModel, self).__init__(env, policy, **kwargs)
        self.trainable_parameters = policy.lora_trainable_params()

    def configure_optimizers(self, parameters=None):
        return super(MultiLoRAModel, self).configure_optimizers(parameters=self.trainable_parameters)






