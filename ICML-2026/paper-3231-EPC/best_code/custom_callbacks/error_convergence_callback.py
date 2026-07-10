import torch
import wandb
from lightning.pytorch.callbacks import Callback
from torch.linalg import vector_norm


class ErrorConvergenceCallback(Callback):
    @torch.no_grad()
    def on_before_backward(self, trainer, pl_module, loss):
        """
        We track the relative residuals of the two FB-DEQs.
        We make sure to check before the optimizer backward pass,
        which may change the grads of self.errors.
        """
        errors = pl_module.errors

        if trainer.logger:
            # Calculate all grad ratios (in log10 space)
            error_grad_ratio = torch.stack(
                tuple(map(ErrorConvergenceCallback.log10_grad_norm_ratio, errors)),
            )  # shape = len(errors), batch_size

            # Track overall distribution of the grad ratio: what values does it take?
            trainer.logger.experiment.log(
                {"Hist(log10_grad_ratio)": wandb.Histogram(error_grad_ratio.cpu())}
            )

        # Track ||g||_\infty (i.e., the max grad component)
        error_max_grad = max(map(lambda e: e.grad.max(), errors))
        self.log("errors/max_grad", error_max_grad, prog_bar=True)

    @staticmethod
    def log10_grad_norm_ratio(error: torch.Tensor):
        """Returns a batched version of the gradient norm ratio ||g_e||/||e||.
        This is basically the same as the relative residual in DEQs."""
        return (
            vector_norm(error.grad.flatten(1), dim=1, ord=2)
            / vector_norm(error.flatten(1), dim=1, ord=2)
        ).log10()

    ### Test time code ###
    # At inference, convergence is perfect, since a feedforward pass is the exact energy minimum
