"""Wrappers for additional helper functions, logging, etc.
"""
import torch as th
from stable_baselines3 import PPO


__author__ = "Hannes Unger"
__version__ = "1.0"
__email__ = "hannes.unger@fh-salzburg.ac.at"


class PolynomialPPOLoggerWrapper(PPO):
    def train(self):
        self.policy.current_progress_remaining = self._current_progress_remaining
        super().train()
        self.logger.record("train/mean_policy_gradients", th.mean(self.policy.policy.policy_approximator.coeffs.grad).item())
        self.logger.record("train/mean_value_gradients", th.mean(self.policy.policy.value_approximator.coeffs.grad).item())
        if self.policy.use_fixed_std_schedule:
            self.logger.record("train/std", self.policy.get_std().item())
        else:
            self.logger.record("train/mean_sigma_gradients", th.mean(self.policy.policy.sigma_approximator.coeffs.grad).item())


class MLPPPOLoggerWrapper(PPO):
    def train(self):
        super().train()
        actor_grads_mean, value_grads_mean = self.get_grads_mean()
        self.logger.record("train/mean_policy_gradients", actor_grads_mean.item())
        self.logger.record("train/mean_value_gradients", value_grads_mean.item())
    
    def get_grads_mean(self):
        value_params = list(self.policy.value_net.parameters())
        actor_params = list(self.policy.action_net.parameters())
        return th.mean(actor_params[0]), th.mean(value_params[0])