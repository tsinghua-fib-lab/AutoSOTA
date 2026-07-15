import copy
import time
import numpy as np
from .fl_base import BaseClient
from appfl.misc.deprecation import deprecated
from appfl.misc.utils import (
    save_partial_model_iteration,
    model_parameters_clip_factor,
    scale_update,
    get_torch_optimizer,
)


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.trainer instead."
)
class PersonalizedClientOptim(BaseClient):
    def __init__(
        self,
        id,
        weight,
        model,
        loss_fn,
        dataloader,
        cfg,
        outfile,
        test_dataloader,
        metric,
        **kwargs,
    ):
        super().__init__(
            id,
            weight,
            model,
            loss_fn,
            dataloader,
            cfg,
            outfile,
            test_dataloader,
            metric,
        )
        self.__dict__.update(kwargs)
        super().client_log_title()

    def update(self):
        self.model.to(self.cfg.device)
        optimizer = get_torch_optimizer(
            optimizer_name=self.optim,
            model_parameters=self.model.parameters(),
            **self.optim_args,
        )

        ## initial evaluation
        if self.cfg.validation and self.test_dataloader is not None:
            start_time = time.time()
            test_loss, test_accuracy = super().client_validation()
            per_iter_time = time.time() - start_time
            super().client_log_content(0, per_iter_time, 0, 0, test_loss, test_accuracy)

        ## local training
        for t in range(self.num_local_epochs):
            start_time = time.time()
            train_loss, target_true, target_pred = 0, [], []
            for data, target in self.dataloader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                pre_update_params = [
                    param.clone() for param in self.model.parameters()
                ]  # save pre-update params
                loss.backward()
                optimizer.step()
                # ----- Implementing clipping. Using idea that if per-update clip is C/n then total clip for n epochs
                # is at most C by triangle inequality.
                clip_factor = model_parameters_clip_factor(
                    self.model,
                    pre_update_params,
                    self.clip_value / self.num_local_epochs,
                    self.clip_norm,
                )
                if self.clip_grad or self.use_dp:
                    scale_update(self.model, pre_update_params, scale=clip_factor)
                # -----
                train_loss += loss.item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())

            train_loss /= len(self.dataloader)
            target_true, target_pred = (
                np.concatenate(target_true),
                np.concatenate(target_pred),
            )
            train_accuracy = float(self.metric(target_true, target_pred))

            ## Validation
            if self.cfg.validation and self.test_dataloader is not None:
                test_loss, test_accuracy = super().client_validation()
            else:
                test_loss, test_accuracy = 0, 0
            per_iter_time = time.time() - start_time
            super().client_log_content(
                t + 1,
                per_iter_time,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
            )

        self.round += 1

        ## Differential Privacy
        self.primal_state = copy.deepcopy(self.model.state_dict())
        if self.use_dp:
            sensitivity = 2.0 * self.clip_value * self.optim_args.lr
            scale_value = sensitivity / self.epsilon
            super().laplace_mechanism_output_perturb_personalized(scale_value)

        ## Save each client model periodically
        if (
            self.cfg.personalization
            and self.cfg.save_model_state_dict
            and (
                (self.round) % self.cfg.checkpoints_interval == 0
                or self.round == self.cfg.num_epochs
            )
        ):
            save_partial_model_iteration(
                self.round, self.model, self.cfg, client_id=self.id
            )

        ## Move the model parameter to CPU (if not) for communication
        if self.cfg.device == "cuda":
            for k in self.primal_state:
                self.primal_state[k] = self.primal_state[k].cpu()

        return self.primal_state
