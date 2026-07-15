import copy
import torch
import logging
from collections import OrderedDict
from .ppfl_base import PPFLServer, PPFLClient
from appfl.misc.deprecation import deprecated
from appfl.misc.utils import get_torch_optimizer

log = logging.getLogger(__name__)


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.aggregator instead."
)
class ICEADMMServer(PPFLServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, **kwargs):
        super().__init__(weights, model, loss_fn, num_clients, device)
        self.__dict__.update(kwargs)
        self.is_first_iter = 1

    def update(self, local_states: OrderedDict):
        """Inputs for the global model update"""
        self.global_state = copy.deepcopy(self.model.state_dict())
        super().primal_recover_from_local_states(local_states)
        super().dual_recover_from_local_states(local_states)
        super().penalty_recover_from_local_states(local_states)

        """ residual calculation """
        super().primal_residual_at_server()
        super().dual_residual_at_server()

        total_penalty = 0
        for i in range(self.num_clients):
            total_penalty += self.penalty[i]

        """ global_state calculation """
        for name, param in self.model.named_parameters():
            tmp = 0.0
            for i in range(self.num_clients):
                ## change device
                self.primal_states[i][name] = self.primal_states[i][name].to(
                    self.device
                )
                self.dual_states[i][name] = self.dual_states[i][name].to(self.device)
                ## computation
                tmp += (self.penalty[i] / total_penalty) * self.primal_states[i][
                    name
                ] + (1.0 / total_penalty) * self.dual_states[i][name]

            self.global_state[name] = tmp

        """ model update """
        self.model.load_state_dict(self.global_state)

    def logging_iteration(self, cfg, logger, t):
        if t == 0:
            title = super().log_title()
            title = title + "%12s %12s %12s %12s" % (
                "PrimRes",
                "DualRes",
                "RhoMin",
                "RhoMax",
            )
            logger.info(title)

        contents = super().log_contents(cfg, t)
        contents = contents + "{:12.4e} {:12.4e} {:12.4e} {:12.4e}".format(
            self.prim_res,
            self.dual_res,
            min(self.penalty.values()),
            max(self.penalty.values()),
        )
        logger.info(contents)

    def logging_summary(self, cfg, logger):
        super().log_summary(cfg, logger)


@deprecated(
    "Imports from appfl.algorithm is deprecated and will be removed in the future. Please use appfl.algorithm.trainer instead."
)
class ICEADMMClient(PPFLClient):
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

        """
        At initial, (1) primal_state = global_state, (2) dual_state = 0
        """
        self.model.to(self.cfg.device)
        for name, param in model.named_parameters():
            self.primal_state[name] = param.data
            self.dual_state[name] = torch.zeros_like(param.data)

        self.penalty = kwargs["init_penalty"]
        self.proximity = kwargs["init_proximity"]
        self.is_first_iter = 1

    def update(self):
        self.model.train()
        self.model.to(self.cfg.device)

        optimizer = get_torch_optimizer(
            optimizer_name=self.optim,
            model_parameters=self.model.parameters(),
            **self.optim_args,
        )

        """ Inputs for the local model update """
        global_state = copy.deepcopy(self.model.state_dict())

        """ Adaptive Penalty (Residual Balancing) """
        if self.residual_balancing.res_on:
            prim_res = super().primal_residual_at_client(global_state)
            dual_res = super().dual_residual_at_client()
            super().residual_balancing(prim_res, dual_res)

        """ Multiple local update """
        for i in range(self.num_local_epochs):
            for data, target in self.dataloader:
                for name, param in self.model.named_parameters():
                    param.data = self.primal_state[name].to(self.cfg.device)

                if (
                    self.residual_balancing.res_on
                    and self.residual_balancing.res_on_every_update
                ):
                    prim_res = super().primal_residual_at_client(global_state)
                    dual_res = super().dual_residual_at_client()
                    super().residual_balancing(prim_res, dual_res)

                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)

                if not self.accum_grad:
                    optimizer.zero_grad()

                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                if self.clip_grad or self.use_dp:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_value,
                        norm_type=self.clip_norm,
                    )

                ## STEP: Update primal and dual
                coefficient = 1
                if self.coeff_grad:
                    coefficient = (
                        self.weight * len(target) / len(self.dataloader.dataset)
                    )

                self.iceadmm_step(coefficient, global_state)

        """ Differential Privacy  """
        if self.use_dp:
            sensitivity = 2.0 * self.clip_value / self.penalty
            scale_value = sensitivity / self.epsilon
            super().laplace_mechanism_output_perturb(scale_value)

        ## store data in cpu before sending it to server
        if self.cfg.device == "cuda":
            for name, param in self.model.named_parameters():
                self.primal_state[name] = param.data.cpu()

        """ Update local_state """
        self.local_state = OrderedDict()
        self.local_state["primal"] = self.primal_state
        self.local_state["dual"] = self.dual_state
        self.local_state["penalty"] = OrderedDict()
        self.local_state["penalty"][self.id] = self.penalty

        ## Back to "cuda"
        if self.cfg.device == "cuda":
            for name, param in self.model.named_parameters():
                self.primal_state[name] = param.data.cuda()

        return self.local_state

    def iceadmm_step(self, coefficient, global_state):
        for name, param in self.model.named_parameters():
            self.primal_state[name] = self.primal_state[name].to(self.cfg.device)
            self.dual_state[name] = self.dual_state[name].to(self.cfg.device)
            global_state[name] = global_state[name].to(self.cfg.device)

            grad = param.grad * coefficient
            ## Update primal
            self.primal_state[name] = self.primal_state[name] - (
                self.penalty * (self.primal_state[name] - global_state[name])
                + grad
                + self.dual_state[name]
            ) / (self.weight * self.proximity + self.penalty)
            ## Update dual
            self.dual_state[name] = self.dual_state[name] + self.penalty * (
                self.primal_state[name] - global_state[name]
            )
