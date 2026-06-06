import torch
import torch.nn.functional as F

from utils import *


class LESAM_S(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LESAM_S, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        # self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
            # group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, g_update):
        # first order sum
        grad_norm = 0
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                else:
                    grad_norm += g_update[idx].norm(p=2)

        for group in self.param_groups:
            # if g_update !=None:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                # original SAM
                # e_w = p.grad * scale.to(p)
                # ASAM

                # e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                else:
                    e_w = -g_update[idx] * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self, g_update):

        inputs, labels, loss_func, model, delta_list, lamb = self.paras

        self.zero_grad()

        self.first_step(g_update)

        param_list = param_to_vector(model)
        predictions = model(inputs)
        loss = loss_func(predictions, labels, param_list, delta_list, lamb)
        self.zero_grad()
        loss.backward()

        self.second_step()