from typing import Tuple

import torch
import torch.nn.functional as F

from src.flow_matching import flow_matching_utils
from src.flow_matching.utils import p_xt_g_x1
from src import utils


class NoiseModel:
    """Unifies noise sampling and application for training and sampling."""

    def __init__(self, limit_dist):
        self.limit_dist = limit_dist

    def sample_limit(self, node_mask: torch.Tensor) -> utils.PlaceHolder:
        return flow_matching_utils.sample_discrete_feature_noise(self.limit_dist, node_mask)

    def p_xt_given_x1(self, X1_label: torch.Tensor, E1_label: torch.Tensor, t: torch.Tensor):
        return p_xt_g_x1(X1_label, E1_label, t, self.limit_dist)

    def sample_from_probs(self, prob_X: torch.Tensor, prob_E: torch.Tensor, node_mask: torch.Tensor):
        return flow_matching_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

    def apply_noise(self, X: torch.Tensor, E: torch.Tensor, y: torch.Tensor, node_mask: torch.Tensor, t: torch.Tensor):
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        prob_X_t, prob_E_t = self.p_xt_given_x1(X_1_label, E_1_label, t)
        sampled_t = self.sample_from_probs(prob_X=prob_X_t, prob_E=prob_E_t, node_mask=node_mask)
        X_t = F.one_hot(sampled_t.X, num_classes=len(self.limit_dist.X))
        E_t = F.one_hot(sampled_t.E, num_classes=len(self.limit_dist.E))
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
        return {
            "t": t,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }


