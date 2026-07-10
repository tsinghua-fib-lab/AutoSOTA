from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from src import utils


class Sampler:
    """Drives the sampling loop using injected components.

    Expects dependencies with the following minimal interfaces:
    - model(X, E, y, node_mask) -> PlaceHolder(X, E, y)
    - extra_builder(noisy_data) -> (extras: PlaceHolder, t_emb: Tensor)
    - kappa_scheduler.kappa_and_grad(t) -> (kappa_t, kappa_dot_t)
    - stepper.propose(t_norm, kappa_t, kappa_dot_t) -> dt
    - rate_computer.compute(kappa_t, kappa_dot_t, node_mask, G_t, G_1_pred, z_0)
    - projector.project(R_t_X, R_t_E, X_t, E_t, dt) -> (prob_X, prob_E)
    - postproc.maybe_apply(prob_X, prob_E, pred_X, pred_E, near_one) -> (prob_X, prob_E)
    - noise_model.sample_from_probs(prob_X, prob_E, node_mask) -> PlaceHolder
    """

    def __init__(
        self,
        model,
        extra_builder,
        kappa_scheduler,
        stepper,
        rate_computer,
        projector,
        postproc,
        noise_model,
        use_sid=False,
        temperature=1.0,
    ):
        self.model = model
        self.extra_builder = extra_builder
        self.kappa_scheduler = kappa_scheduler
        self.stepper = stepper
        self.rate_computer = rate_computer
        self.projector = projector
        self.postproc = postproc
        self.noise_model = noise_model
        self.use_sid = use_sid
        self.temperature = temperature
        # self.sample_G1 = sample_G1

    @torch.no_grad()
    def run(self, X, E, y, node_mask, z_0):
        bs = X.shape[0]
        current_t = torch.zeros((bs, 1), device=X.device, dtype=y.dtype)
        # loop_guard = 0
        # max_loops = int(5 * sample_steps)
        G_t = (X, E)
        # if self.conditional:
            #     y_to_save = y_t
        y = torch.zeros([y.shape[0], 0], device=X.device)

        while torch.any(current_t < 1.0):
            kappa_t, kappa_dot_t = self.kappa_scheduler.kappa_and_grad(current_t)
            h = self.stepper.propose(current_t, kappa_t, kappa_dot_t)
            next_t = torch.clamp(current_t + h, max=1.0)

            noisy_data = {"X_t": G_t[0], "E_t": G_t[1], "y_t": y, "t": kappa_t, "node_mask": node_mask}
            extras, t_emb = self.extra_builder(noisy_data)
            pred = self.model(
                torch.cat((noisy_data["X_t"], extras.X), dim=2).float(),
                torch.cat((noisy_data["E_t"], extras.E), dim=3).float(),
                torch.hstack((noisy_data["y_t"], extras.y)).float(),
                t_emb.float(),
                node_mask,
            )
            pred_X = F.softmax(pred.X / self.temperature, dim=-1)
            pred_E = F.softmax(pred.E / self.temperature, dim=-1)
            # if self.sample_G1:
            #     sampled_G1 = self.noise_model.sample_from_probs(pred_X, pred_E, node_mask)
            #     pred_X = F.one_hot(sampled_G1.X, num_classes=pred_X.shape[-1]).float()
            #     pred_E = F.one_hot(sampled_G1.E, num_classes=pred_E.shape[-1]).float()

            if self.use_sid:
                # SID-style re-noising: blend prediction with prior distribution
                # alpha = kappa_t: more prediction weight as t -> 1
                alpha_X = kappa_t.view(-1, 1, 1)
                alpha_E = kappa_t.view(-1, 1, 1, 1)
                prob_X = alpha_X * pred_X + (1 - alpha_X) * z_0.X.unsqueeze(0).unsqueeze(0)
                prob_E = alpha_E * pred_E + (1 - alpha_E) * z_0.E.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            else:
                R_t_X, R_t_E = self.rate_computer.compute(kappa_t, kappa_dot_t, node_mask, G_t, (pred_X, pred_E), z_0)
            # Optional guidance blending with unconditional prediction
            if not self.use_sid:
                prob_X, prob_E = self.projector.project(R_t_X, R_t_E, G_t[0], G_t[1], h)
            prob_X, prob_E = self.postproc.apply(current_t, kappa_t, h, prob_X, prob_E, pred_X, pred_E)
            sampled = self.noise_model.sample_from_probs(prob_X, prob_E, node_mask)
            X_next_t = F.one_hot(sampled.X, num_classes=G_t[0].shape[-1]).float()
            E_next_t = F.one_hot(sampled.E, num_classes=G_t[1].shape[-1]).float()
            assert (E_next_t == torch.transpose(E_next_t, 1, 2)).all()
            assert (G_t[0].shape == X_next_t.shape) and (G_t[1].shape == E_next_t.shape)
            out_one_hot = utils.PlaceHolder(X=X_next_t, E=E_next_t, y=y)
            out_one_hot = out_one_hot.mask(node_mask).type_as(y)
            G_t = (out_one_hot.X, out_one_hot.E)
            current_t = next_t

        # out_one_hot = utils.PlaceHolder(X=G_t[0], E=G_t[1], y=y)
        # out_one_hot = out_one_hot.mask(node_mask).type_as(y)
        out_discrete = utils.PlaceHolder(X=G_t[0], E=G_t[1], y=y)
        out_discrete = out_discrete.mask(node_mask, one_hot_to_index=True).type_as(y)
        return out_one_hot, out_discrete, # loop_guard
