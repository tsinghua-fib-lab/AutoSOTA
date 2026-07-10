import torch
import wandb
from lightning import Callback
from pc_e import PCE
from torch.linalg import vector_norm


class CompareWithStateOptimization(PCE):
    def __init__(self, other: PCE):
        # Just copy all the attributes from the original PCE
        self.__dict__.update(other.__dict__)

    def log(self, *args, **kwargs):
        # Disable all intermediary logging
        return None

    @staticmethod
    def relative_state_comparison(s1, s2):
        return (
            (
                vector_norm((s1 - s2).flatten(1), dim=1, ord=2)
                / vector_norm(s2.flatten(1), dim=1, ord=2)
            )
            .mean(dim=0)
            .log10()
        )

    @torch.no_grad()
    def compare_equilibrium_states(self, x: torch.Tensor, y: torch.Tensor, iters: int, s_lr: float):
        # Get equilibrium states from error optimization
        equilibrium_error_states = self.get_states_from_errors(x)

        # Get equilibrium states from state optimization
        with torch.enable_grad():
            equilibrium_states = self.minimize_state_energy(x, y, iters, s_lr)

        # Compare final values for each layer and each batch item
        comparison = torch.stack(
            [
                self.relative_state_comparison(s_i, s_e)
                for s_i, s_e in zip(equilibrium_states, equilibrium_error_states)
            ]
        )  # shape: (nr_states, )

        if self.logger:
            # We disabled self.log, but we can still log via self.logger!
            self.logger.experiment.log(
                {"Hist(log10_state_comparison)": wandb.Histogram(comparison.cpu())}
            )


class StateComparisonCallback(Callback):
    """
    A Callback to compare the error-based implementation with the state-based one.
    It checks to what extent both approaches reach the same equilibrium.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Stores the batch for use in on_after_backward (which doesn't get 'batch' as an input)"""
        # Only run for one batch (takes too long otherwise)
        self.batch = batch if batch_idx == 0 else None

    def on_after_backward(self, trainer, pl_module):
        """Runs after the backward pass, so that we cannot accidentally modify the gradients.
        Runs before optimizer.step(), so that the comparison uses the exact same parameters"""
        if self.batch is not None:
            # Unpack batch
            x, y = self.batch["img"], self.batch["y"]

            # Create a wrapper over PCE to compare the final optimal states of error optim vs state optim
            state_opt = CompareWithStateOptimization(pl_module)
            state_opt.compare_equilibrium_states(x, y, iters=5 ** len(state_opt.layers), s_lr=0.1)
