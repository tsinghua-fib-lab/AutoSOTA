# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Policy class for the early stage retrieval and late stage ranking policies."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .action_set import BaseActionSet
from .base import (
    BaseEarlyStagePolicy,
    BaseJointPolicy,
    BaseLateStagePolicy,
    BaseSingleStagePolicy,
)


@dataclass
class BaselineEarlyStagePolicy(BaseEarlyStagePolicy):
    """Early stage retrieval policy.

    Input
    ------
    base_model: nn.Module
        The base model.

    action_set: BaseActionSet
        The action set.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: nn.Module
    action_set: BaseActionSet
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.base_model = self.base_model.to(self.device)

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def _n_candidate_per_model(
        self,
        n_candidate_action: int,
        n_model: int,
    ):
        """Retrieve the number of candidate actions per model.

        Input
        ------
        n_candidate_action: int
            The number of candidate actions.

        n_model: int
            The number of models.

        Output
        ------
        assignment: torch.Tensor, shape (n_model, )
            The number of candidate actions per model.

        """
        base = n_candidate_action // n_model
        remainder = n_candidate_action % n_model
        assignment = [base + 1 if i < remainder else base for i in range(n_model)]
        return torch.tensor(assignment, device=self.device)

    def _topk(
        self,
        logits: torch.Tensor,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_actions)
            The logits of the actions.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        n_model = logits.shape[0]

        if n_candidate_per_model is None:
            n_candidate_per_model = self._n_candidate_per_model(
                n_candidate_action=n_candidate_action, n_model=n_model
            )
        else:
            assert len(n_candidate_per_model) == n_model
            assert sum(n_candidate_per_model) == n_candidate_action

        candidate_actions = []
        model_idx = torch.arange(n_model, device=self.device).view(-1, 1, 1)
        sample_idx = torch.arange(len(logits[0]), device=self.device).view(1, -1, 1)
        for model_id_ in range(n_model):
            _, candidate_ = torch.topk(
                logits[model_id_], k=n_candidate_per_model[model_id_], dim=1
            )
            logits[model_idx, sample_idx, candidate_.unsqueeze(0)] = -torch.inf
            candidate_actions.append(candidate_)

        return torch.cat(candidate_actions, dim=1)

    def _topk_exluding_actions(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions excluding the given actions.

        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_actions)
            The logits of the actions.

        actions: torch.Tensor, shape (n_samples, n_output_action)
            The actions to exclude.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action, n_candidate_action)
            The topk actions.

        """
        n_model = logits.shape[0]
        n_output_action = actions.shape[1]

        if n_candidate_per_model is None:
            n_candidate_per_model = self._n_candidate_per_model(
                n_candidate_action=n_candidate_action, n_model=n_model
            )
        else:
            assert len(n_candidate_per_model) == n_model
            assert sum(n_candidate_per_model) == n_candidate_action

        with torch.no_grad():
            candidate_actions = []
            for i in range(n_output_action):
                candidate_actions_ = []
                logits_ = logits.clone().detach()
                model_idx = torch.arange(n_model, device=self.device).view(-1, 1, 1)
                sample_idx = torch.arange(len(logits[0]), device=self.device).view(
                    1, -1, 1
                )

                for model_id_ in range(n_model):
                    _, candidate_ = torch.topk(
                        logits_[model_id_],
                        k=n_candidate_per_model[model_id_] + 1,
                        dim=1,
                    )

                    # remove the selected action from the candidate
                    candidate_ = torch.where(
                        candidate_[:, : n_candidate_per_model[model_id_]]
                        == actions[:, i].unsqueeze(1),
                        candidate_[:, n_candidate_per_model[model_id_]].unsqueeze(1),
                        candidate_[:, : n_candidate_per_model[model_id_]],
                    )

                    logits_[model_idx, sample_idx, candidate_.unsqueeze(0)] = -torch.inf
                    candidate_actions_.append(candidate_)

                candidate_actions.append(
                    torch.cat(candidate_actions_, dim=1).unsqueeze(1)
                )

        return torch.cat(candidate_actions, dim=1)

    def _candidate_wise_log_prob_given_logits_and_actions(
        self,
        logits: torch.Tensor,
        candidate_actions: torch.Tensor,
        n_candidate_per_model: Optional[List[int]] = None,
        require_grad_model_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the plackett-luce probability of the given candidate_actions.

        Note
        ------
        For each position, the log probability is given by:

            log_prob = logit(kth selected) - A - log1p(-exp(B-A))

        where A = logsumexp( logit(all) ) and B = logsumexp( logit(selected by kth) )


        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_action)
            The logits of the actions.

        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        require_grad_model_id: int
            The model id to calculate the gradient.

        Output
        ------
        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the actions (differential).

        """
        n_model, n_sample, _ = logits.shape
        n_candidate_action = candidate_actions.shape[1]

        if n_candidate_per_model is None:
            n_candidate_per_model = self._n_candidate_per_model(
                n_candidate_action=n_candidate_action, n_model=n_model
            )
        else:
            assert len(n_candidate_per_model) == n_model
            assert sum(n_candidate_per_model) == n_candidate_action

        model_id = torch.tensor(
            [i for i in range(n_model) for j in range(n_candidate_per_model[i])],
            device=self.device,
        )

        if require_grad_model_id is None:
            # first calculate the logits(kth selected) (n_samples, n_candidate_action)
            sample_indices = torch.arange(n_sample, device=self.device).unsqueeze(1)
            a = logits[model_id.unsqueeze(0), sample_indices, candidate_actions]

            # next, we calculate A = logsumexp( logit(all) )
            logsumexp_all = torch.logsumexp(logits, dim=2)  # (n_model, n_samples)
            A = logsumexp_all[model_id].T  # (n_samples, n_candidate_action)

            # then, we calculate B = logsumexp( logit(selected by kth) )
            selected_mask = torch.tril(
                torch.ones((n_candidate_action, n_candidate_action), device=self.device)
            ) - torch.eye(n_candidate_action, device=self.device)

            logits_topk = torch.gather(
                logits[model_id],
                2,
                candidate_actions.unsqueeze(0).expand(n_candidate_action, -1, -1),
            )  # (n_candidate_action, n_samples, n_candidate_action)

            if n_candidate_action > 1:
                logsumexp_selected = []
                for k in range(1, n_candidate_action):
                    selected_logits = torch.masked_select(
                        logits_topk[k], selected_mask[k].unsqueeze(0).bool()
                    ).reshape(n_sample, -1)  # (n_samples, n_candidate_action - k)
                    logsumexp_selected.append(
                        torch.logsumexp(selected_logits, dim=1).unsqueeze(1)
                    )  # (n_samples, )

                B = torch.cat(
                    logsumexp_selected, dim=1
                )  # (n_samples, n_candidate_action - 1)

            # denominator  (n_sample, n_candidate_action)
            normalizer = A[:, 0].unsqueeze(-1)
            if n_candidate_action > 1:
                A = A[:, 1:]
                normalizer_ = A + torch.log1p(-torch.exp(B - A) + 1e-10)
                normalizer = torch.cat([normalizer, normalizer_], dim=-1)

            # kth prob  (n_sample, n_candidate_action)
            kth_log_prob = a - normalizer

            # overall prob  (n_sample, )
            log_prob = kth_log_prob.sum(dim=-1)

        else:
            raise NotImplementedError()

        return log_prob
    
    def _candidate_wise_replacement_log_prob_given_logits_and_actions(
        self,
        logits: torch.Tensor,
        candidate_actions: torch.Tensor,
        n_candidate_per_model: Optional[List[int]] = None,
        require_grad_model_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the plackett-luce probability of the given candidate_actions.

        Note
        ------
        For each position, the log probability is given by:

            log_prob = logit(kth selected) - A

        where A = logsumexp( logit(all) )


        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_action)
            The logits of the actions.

        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        require_grad_model_id: int
            The model id to calculate the gradient.

        Output
        ------
        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the actions (differential).

        """
        n_model, n_sample, _ = logits.shape
        n_candidate_action = candidate_actions.shape[1]

        if n_candidate_per_model is None:
            n_candidate_per_model = self._n_candidate_per_model(
                n_candidate_action=n_candidate_action, n_model=n_model
            )
        else:
            assert len(n_candidate_per_model) == n_model
            assert sum(n_candidate_per_model) == n_candidate_action

        model_id = torch.tensor(
            [i for i in range(n_model) for j in range(n_candidate_per_model[i])],
            device=self.device,
        )

        if require_grad_model_id is None:
            # first calculate the logits(kth selected) (n_samples, n_candidate_action)
            sample_indices = torch.arange(n_sample, device=self.device).unsqueeze(1)
            a = logits[model_id.unsqueeze(0), sample_indices, candidate_actions]

            # next, we calculate A = logsumexp( logit(all) )
            logsumexp_all = torch.logsumexp(logits, dim=2)  # (n_model, n_samples)
            A = logsumexp_all[model_id].T  # (n_samples, n_candidate_action)

            # overall logprob  (n_sample, )
            log_prob = (a - A).sum(dim=-1)

        else:
            raise NotImplementedError()

        return log_prob

    def _credit_assigned_log_prob_given_logits_and_actions(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
        require_grad_model_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the plackett-luce probability of the given actions.

        Note
        ------
        The gradient is given as follows:

            \nabla lop1p ( -exp ( \sum_{k=1}^K lop1p (- \pi_{ESR}(a is selected at k'th) ) ) )

        where

            \pi_{ESR}(a is selected at k'th) = \frac{ exp(a) }{ exp( A + log1p( - exp( B - A ) ) ) }

        where

            A = logsumexp( all logits )
            B = logsumexp( topk logits excluding a )


        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_action)
            The logits of the actions.

        actions: torch.Tensor, shape (n_samples, n_output_action)
            The actions to calculate the probability.

        n_candidate_action: int
            The number of candidate actions.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        require_grad_model_id: int
            The model id to calculate the gradient.

        Output
        ------
        log_prob: torch.Tensor, shape (n_samples, n_output_action)
            The log joint probability of the actions (differential).

        """
        n_model, n_sample, _ = logits.shape
        n_output_action = actions.shape[1]

        if n_candidate_per_model is None:
            n_candidate_per_model = self._n_candidate_per_model(
                n_candidate_action=n_candidate_action, n_model=n_model
            )
        else:
            assert len(n_candidate_per_model) == n_model
            assert sum(n_candidate_per_model) == n_candidate_action

        model_id = torch.tensor(
            [i for i in range(n_model) for j in range(n_candidate_per_model[i])],
            device=self.device,
        )

        topk_candidate_ex_a = self._topk_exluding_actions(
            logits=logits,
            actions=actions,
            n_candidate_action=n_candidate_action,
            n_candidate_per_model=n_candidate_per_model,
        )

        if require_grad_model_id is None:
            # calculate exp(a)  (n_sample, n_output_action, n_candidate_action)
            target_logits = torch.gather(
                logits, 2, actions.unsqueeze(0).expand(n_model, -1, -1)
            )
            a = target_logits[model_id].permute(1, 2, 0)

            # calculate A  (n_sample, n_candidate_action)
            logsumexp_all = torch.logsumexp(logits, dim=-1)
            A = logsumexp_all[model_id].T

            # calculate B  (n_sample, n_output_action, n_candidate_action - 1)
            if n_candidate_action > 1:
                logsumexp_selected = []
                for k in range(1, n_candidate_action):
                    prev_topk = topk_candidate_ex_a[:, :, :k]
                    logit_ = logits[model_id[k]]
                    selected_logits = torch.gather(
                        logit_.unsqueeze(1).expand(-1, n_output_action, -1),
                        2,
                        prev_topk,
                    )
                    logsumexp_selected.append(
                        torch.logsumexp(selected_logits, dim=2).unsqueeze(-1)
                    )

                B = torch.cat(logsumexp_selected, dim=-1)

            # denominator  (n_sample, n_output_action, n_candidate_action)
            A = A.unsqueeze(1)
            normalizer = A[:, :, 0].unsqueeze(-1).expand(-1, n_output_action, 1)

            if n_candidate_action > 1:
                A = A[:, :, 1:]

                normalizer_ = A + torch.log1p(-torch.exp(B - A) + 1e-10)
                normalizer = torch.cat([normalizer, normalizer_], dim=-1)

            # kth prob  (n_sample, n_output_action, n_candidate_action)
            kth_prob = torch.exp(a - normalizer)

            # overall prob  (n_sample, n_output_action)
            log_prob = torch.log1p(
                -torch.exp(torch.sum(torch.log1p(-kth_prob + 1e-10), dim=-1)) + 1e-10
            )

        if require_grad_model_id is not None:
            raise NotImplementedError()

        return log_prob
    
    def _top1_log_prob_given_logits_and_actions(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
        require_grad_model_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the plackett-luce probability of the given actions.

        Note
        ------
        The gradient is given as follows:

            \pi_{ESR}(a is selected at top-1) = \frac{ exp(a) }{ exp(A) }

        where

            A = logsumexp( all logits )
            B = logit a


        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_action)
            The logits of the actions.

        actions: torch.Tensor, shape (n_samples, n_output_action)
            The actions to calculate the probability.

        n_candidate_action: int
            The number of candidate actions.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        require_grad_model_id: int
            The model id to calculate the gradient.

        Output
        ------
        log_prob: torch.Tensor, shape (n_samples, n_output_action)
            The log probability of the actions (differential).

        """
        n_model = logits.shape[0]

        if n_candidate_per_model is None:
            n_candidate_per_model = self._n_candidate_per_model(
                n_candidate_action=n_candidate_action, n_model=n_model
            )
        else:
            assert len(n_candidate_per_model) == n_model
            assert sum(n_candidate_per_model) == n_candidate_action

        if require_grad_model_id is None and n_model > 1:
            actions_expanded = actions.unsqueeze(0).expand(n_model, -1, -1)
            a = torch.gather(logits, 1, actions_expanded)
            A = torch.logsumexp(logits, dim=-1)

            log_prob = a - A.unsqueeze(dim=-1)
            log_prob = (n_candidate_per_model.view(-1, 1, 1) * log_prob).sum(dim=0)

        elif require_grad_model_id is None and n_model == 1:
            logits = logits[0]
            a = torch.gather(logits, 1, actions)
            A = torch.logsumexp(logits, dim=-1)
            log_prob = a - A.unsqueeze(dim=-1)

        if require_grad_model_id is not None:
            raise NotImplementedError()

        return log_prob

    def _log_prob_given_logits_and_actions(
        self,
        logits: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        candidate_actions: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
        require_grad_model_id: Optional[int] = None,
        is_credit_assigned_gradient: bool = False,
        is_top1_gradient: bool = False,
        is_vanilla_replacement_gradient: bool = False,
    ) -> torch.Tensor:
        """Calculate the plackett-luce probability of the given actions.

        Note
        ------
        The credit-assigned gradient is given as follows:

            \nabla lop1p ( -exp ( \sum_{k=1}^K lop1p (- \pi_{ESR}(a is selected at k'th) ) ) )

        where

            \pi_{ESR}(a is selected at k'th) = \frac{ exp(a) }{ exp( A + log1p( - exp( B - A ) ) ) }

        where

            A = logsumexp( all logits )
            B = logsumexp( topk logits excluding a )


        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_action)
            The logits of the actions.

        actions: torch.Tensor, shape (n_samples, n_output_action)
            The actions to calculate the probability.

        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        n_candidate_action: int
            The number of candidate actions.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        require_grad_model_id: int
            The model id to calculate the gradient.

        is_credit_assigned_gradient: bool, default=False
            Whether to use the credit-assigned policy gradient.

        is_top1_gradient: bool, default=False
            Whether to use the top1 policy gradient.

        is_vanilla_replacement_gradient: bool, default=False
            Whether to use the vanilla policy gradient under sampling w/ replacement (SwR) approximation.

        Output
        ------
        log_prob: torch.Tensor, shape (n_samples, n_output_action) or (n_samples, )
            The log joint probability of the actions (differential).

        """
        n_model, n_sample, _ = logits.shape

        if is_top1_gradient:
            if n_candidate_per_model is None:
                n_candidate_per_model = self._n_candidate_per_model(
                    n_candidate_action=n_candidate_action, n_model=n_model
                )
            else:
                assert len(n_candidate_per_model) == n_model
                assert sum(n_candidate_per_model) == n_candidate_action

            if require_grad_model_id is None and n_model > 1:
                actions_expanded = actions.unsqueeze(0).expand(n_model, -1, -1)
                a = torch.gather(logits, 2, actions_expanded)
                A = torch.logsumexp(logits, dim=-1)

                log_prob = a - A.unsqueeze(dim=-1)
                log_prob = (n_candidate_per_model.view(-1, 1, 1) * log_prob).sum(dim=0)

            elif require_grad_model_id is None and n_model == 1:
                logits = logits[0]
                a = torch.gather(logits, 1, actions)
                A = torch.logsumexp(logits, dim=-1)
                log_prob = a - A.unsqueeze(dim=-1)

        elif is_vanilla_replacement_gradient:
            if n_candidate_per_model is None:
                n_candidate_per_model = self._n_candidate_per_model(
                    n_candidate_action=n_candidate_action, n_model=n_model
                )
            else:
                assert len(n_candidate_per_model) == n_model
                assert sum(n_candidate_per_model) == n_candidate_action

            model_id = torch.tensor(
                [i for i in range(n_model) for j in range(n_candidate_per_model[i])],
                device=self.device,
            )

            if require_grad_model_id is None:
                # first calculate the logits(kth selected) (n_samples, n_candidate_action)
                sample_indices = torch.arange(n_sample, device=self.device).unsqueeze(1)
                a = logits[model_id.unsqueeze(0), sample_indices, candidate_actions]

                # next, we calculate A = logsumexp( logit(all) )
                logsumexp_all = torch.logsumexp(logits, dim=2)  # (n_model, n_samples)
                A = logsumexp_all[model_id].T  # (n_samples, n_candidate_action)

                # overall logprob  (n_sample, )
                log_prob = (a - A).sum(dim=-1)

        else:
            if is_credit_assigned_gradient:
                n_output_action = actions.shape[1]
            else:
                n_candidate_action = candidate_actions.shape[1]

            if n_candidate_per_model is None:
                n_candidate_per_model = self._n_candidate_per_model(
                    n_candidate_action=n_candidate_action, n_model=n_model
                )
            else:
                assert len(n_candidate_per_model) == n_model
                assert sum(n_candidate_per_model) == n_candidate_action

            model_id = torch.tensor(
                [i for i in range(n_model) for j in range(n_candidate_per_model[i])],
                device=self.device,
            )

            topk_candidate_ex_a = self._topk_exluding_actions(
                logits=logits,
                actions=actions,
                n_candidate_action=n_candidate_action,
                n_candidate_per_model=n_candidate_per_model,
            )

            topk_candidate_inc_a = candidate_actions

            if require_grad_model_id is None:
                # calculate exp(a)  (n_sample, n_output_action, n_candidate_action)
                if is_credit_assigned_gradient:
                    target_logits = torch.gather(
                        logits,
                        2,
                        actions.unsqueeze(0).expand(n_model, -1, -1),
                    )
                    a = target_logits[model_id].permute(1, 2, 0)

                else:
                    target_logits = torch.gather(
                        logits,
                        2,
                        candidate_actions.unsqueeze(0).expand(n_model, -1, -1),
                    )
                    a = target_logits[model_id].permute(1, 2, 0)
                    a = a.diagonal(dim1=1, dim2=2)

                # calculate A  (n_sample, n_candidate_action)
                logsumexp_all = torch.logsumexp(logits, dim=-1)
                A = logsumexp_all[model_id].T

                # calculate B  (n_sample, n_output_action, n_candidate_action - 1)
                if n_candidate_action > 1:
                    logsumexp_selected = []
                    for k in range(1, n_candidate_action):
                        logit_ = logits[model_id[k]]

                        if is_credit_assigned_gradient:
                            prev_topk = topk_candidate_ex_a[:, :, :k]
                            selected_logits = torch.gather(
                                logit_.unsqueeze(1).expand(
                                    -1,
                                    n_output_action,
                                    -1,
                                ),
                                2,
                                prev_topk,
                            )

                        else:
                            prev_topk = topk_candidate_inc_a[:, :k]
                            selected_logits = torch.gather(
                                logit_,
                                1,
                                prev_topk,
                            )

                        logsumexp_selected.append(
                            torch.logsumexp(selected_logits, dim=-1).unsqueeze(-1)
                        )

                    B = torch.cat(logsumexp_selected, dim=-1)

                # denominator  (n_sample, n_output_action, n_candidate_action)
                if is_credit_assigned_gradient:
                    A = A.unsqueeze(1)
                    normalizer = (
                        A[:, :, 0]
                        .unsqueeze(-1)
                        .expand(-1, n_output_action, 1)
                    )

                    if n_candidate_action > 1:
                        A = A[:, :, 1:]

                        normalizer_ = A + torch.log1p(
                            -torch.exp(B - A) + 1e-10
                        )
                        normalizer = torch.cat([normalizer, normalizer_], dim=-1)

                    # kth prob  (n_sample, n_output_action, n_candidate_action)
                    kth_prob = torch.exp(a - normalizer)

                    # overall prob  (n_sample, n_output_action)
                    log_prob = torch.log1p(
                        -torch.exp(torch.sum(torch.log1p(-kth_prob + 1e-10), dim=-1))
                        + 1e-10
                    )

                else:
                    normalizer = A[:, 0].unsqueeze(-1)

                    if n_candidate_action > 1:
                        A = A[:, 1:]

                        normalizer_ = A + torch.log1p(
                            -torch.exp(B - A) + 1e-10
                        )
                        normalizer = torch.cat([normalizer, normalizer_], dim=-1)

                    # kth prob  (n_sample, n_candidate_action)
                    kth_prob = torch.exp(a - normalizer)

                    # overall prob  (n_sample, )
                    log_prob = torch.log1p(
                        -torch.exp(torch.log1p(-kth_prob + 1e-10)) + 1e-10
                    ).sum(dim=-1)

            if require_grad_model_id is not None:
                raise NotImplementedError()

        return log_prob

    def retrieve_topk(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        context: Optional[Tensor], shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Optional[Tensor], shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        with torch.no_grad():
            logits = self.base_model(context=context, context_id=context_id)

        topk = self._topk(
            logits=logits,
            n_candidate_action=n_candidate_action,
            n_candidate_per_model=n_candidate_per_model,
        )
        return topk

    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
        is_deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        context: Optional[Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to sample.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        with torch.no_grad():
            logits = self.base_model(context=context, context_id=context_id)

        if is_deterministic:
            gumbel_noise = torch.zeros_like(logits).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(logits.shape).to(
                self.device
            )

        actions = self._topk(
            logits=logits + gumbel_noise.squeeze(0),
            n_candidate_action=n_candidate_action,
            n_candidate_per_model=n_candidate_per_model,
        )
        return actions

    def calc_prob_given_actions(
        self,
        actions: Optional[torch.Tensor] = None,
        candidate_actions: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        n_candidate_per_model: Optional[List[int]] = None,
        is_deterministic: bool = False,
        is_credit_assigned: bool = False,
        is_top1: bool = False,
        is_vanilla_replacement: bool = False,
        require_grad_model_id: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the plackett-luce probability given actions.

        Input
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The actions to calculate the probability.

        context: Tensor, shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        n_candidate_actions: int
            The number of candidate actions to sample.

        n_candidate_per_model: List[int]
            The number of candidate actions per model.

        is_deterministic: bool
            Whether to use deterministic sampling.

        is_credit_assigned: bool
            Whether to use credit assignment.

        is_top1: bool
            Whether to use top-1 policy gradient.

        is_vanilla_replacement: bool, default=False
            Whether to use the vanilla policy gradient under sampling w/ replacement (SwR) approximation.

        require_grad_model_id: int
            The model id to calculate the gradient.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, n_output_action)
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        log_prob: torch.Tensor, shape (n_samples, n_output_action)
            The log joint probability of the sampled actions (used for caluculating the policy gradient, differential).

        """
        if is_credit_assigned and is_top1:
            raise ValueError("Gradient type is unselected. Please choose to use credit-assigned PG or top-1 PG.")
        
        if is_credit_assigned and is_vanilla_replacement:
            raise ValueError("Gradient type is unselected. Please choose to use credit-assigned PG or vanilla PG (sampling w/ replacement).")
        
        if is_top1 and is_vanilla_replacement:
            raise ValueError("Gradient type is unselected. Please choose to use top-1 PG or vanilla PG (sampling w/ replacement).")
        
        
        if is_credit_assigned and actions is None:
            raise ValueError("actions must be provided for credit-assigned PG.")
        elif not is_credit_assigned and candidate_actions is None:
            raise ValueError(
                "candidate_actions must not be provided for candidate-wise PG."
            )

        if is_deterministic:
            raise NotImplementedError()

        else:
            logits = self.base_model(context=context, context_id=context_id)
            n_model = logits.shape[0]

            log_prob = self._log_prob_given_logits_and_actions(
                logits=logits,
                actions=actions,
                candidate_actions=candidate_actions,
                n_candidate_action=n_candidate_action,
                n_candidate_per_model=n_candidate_per_model,
                require_grad_model_id=require_grad_model_id,
                is_credit_assigned_gradient=is_credit_assigned,
                is_top1_gradient=is_top1,
                is_vanilla_replacement_gradient=is_vanilla_replacement,
            )

            if is_top1 and n_model > 1:
                # this is not the exact probability, just a proxy (can be exact only in the single logit model case)
                prob = (log_prob.clone().detach() / n_candidate_action).exp()  # non-differential
            else:
                prob = log_prob.clone().detach().exp()  # non-differential

        return prob, log_prob


@dataclass
class BaselineLateStagePolicy(BaseLateStagePolicy):
    """Late stage ranking/recommendation policy.

    Input
    ------
    base_model: nn.Module
        The base model.

    action_set: BaseActionSet
        The action set.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: nn.Module
    action_set: BaseActionSet
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.base_model = self.base_model.to(self.device)

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def _topk(
        self,
        candidate_actions: torch.Tensor,
        logits: torch.Tensor,
        n_output_action: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The index of candidate actions to retrieve from.

        logits: torch.Tensor, shape (n_samples, n_candidate_action)
            The logits of the actions.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        within_candidate_idx: torch.Tensor, shape (n_samples, n_output_action)
            The index of the sampled actions within the candidate actions.

        """
        _, within_candidate_idx = torch.topk(logits, k=n_output_action, dim=1)
        actions = torch.gather(candidate_actions, 1, within_candidate_idx)
        return actions, within_candidate_idx

    def retrieve_topk(
        self,
        candidate_actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The index of candidate actions to retrieve from.

        context: Tensor, shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to retrieve actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to retrieve actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        with torch.no_grad():
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                action_ids=candidate_actions,
            )

        topk, _ = self._topk(
            candidate_actions=candidate_actions,
            logits=logits,
            n_output_action=n_output_action,
        )
        return topk

    def sample(
        self,
        candidate_actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        is_deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The index of candidate actions to sample from.

        context: Tensor, shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor
            The sampled actions.

        """
        with torch.no_grad():
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                action_ids=candidate_actions,
            )

        if is_deterministic:
            gumbel_noise = torch.zeros_like(logits).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(logits.shape).to(
                self.device
            )

        topk, _ = self._topk(
            candidate_actions=candidate_actions,
            logits=logits + gumbel_noise,
            n_output_action=n_output_action,
        )
        return topk


@dataclass
class BaselineJointPolicy(BaseJointPolicy):
    """Joint (early stage + late stage) recommendation/ranking policy.

    Input
    ------
    early_stage_policy: BaseEarlyStagePolicy
        The early stage policy.

    late_stage_policy: BaseLateStagePolicy
        The late stage policy.

    action_set: BaseActionSet
        The action set.

    is_deterministic_early_stage: bool
        Whether the early stage policy is deterministic.

    is_deterministic_late_stage: bool
        Whether the late stage policy is deterministic.

    is_model_free_early_stage: bool
        Whether the early stage policy is model free.

    is_model_free_late_stage: bool
        Whether the late stage policy is model free.

    n_candidate_action: int
        The number of candidate actions.

    n_output_action: int
        The number of output actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    early_stage_policy: BaseEarlyStagePolicy
    late_stage_policy: BaseLateStagePolicy
    action_set: BaseActionSet
    is_deterministic_early_stage: bool = False
    is_deterministic_late_stage: bool = False
    is_model_free_early_stage: bool = False
    is_model_free_late_stage: bool = False
    n_candidate_action: int = 1
    n_output_action: int = 1
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

            if not self.is_model_free_early_stage:
                self.early_stage_policy.base_model = (
                    self.early_stage_policy.base_model.to(self.device)
                )
            else:
                self.late_stage_policy.base_model = (
                    self.late_stage_policy.base_model.to(self.device)
                )

            self.action_prob_regressor = self.action_prob_regressor.to(self.device)

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def retrieve_topk(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to retrieve actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to retrieve actions from. Either latent or latent_id must be provided.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        candidate_actions = self.early_stage_policy.retrieve_topk(
            context=context,
            context_id=context_id,
            n_candidate_action=self.n_candidate_action,
            n_output_action=self.n_output_action,
            is_deterministic=self.is_deterministic_early_stage,
        )
        actions = self.late_stage_policy.retrieve_topk(
            candidate_actions=candidate_actions,
            context=context,
            context_id=context_id,
            latent=latent,
            latent_id=latent_id,
            n_output_action=self.n_output_action,
            is_deterministic=self.is_deterministic_late_stage,
        )
        return actions

    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        if self.is_deterministic_early_stage:
            candidate_actions = self.early_stage_policy.retrieve_topk(
                context=context,
                context_id=context_id,
                n_candidate_action=self.n_candidate_action,
                n_output_action=self.n_output_action,  # only when using greedy algorithm
                is_deterministic=self.is_deterministic_early_stage,
            )
        else:
            candidate_actions = self.early_stage_policy.sample(
                context=context,
                context_id=context_id,
                n_candidate_action=self.n_candidate_action,
                is_deterministic=self.is_deterministic_early_stage,
            )
        if self.is_deterministic_late_stage:
            actions = self.late_stage_policy.retrieve_topk(
                candidate_actions=candidate_actions,
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_output_action=self.n_output_action,
                is_deterministic=self.is_deterministic_late_stage,
            )
        else:
            actions = self.late_stage_policy.sample(
                candidate_actions=candidate_actions,
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_output_action=self.n_output_action,
                is_deterministic=self.is_deterministic_late_stage,
            )
        return actions


@dataclass
class BaselineSingleStagePolicy(BaseSingleStagePolicy):
    """Single stage ranking/recommendation policy.

    Input
    ------
    base_model: nn.Module
        The base model.

    action_set: BaseActionSet
        The action set.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: nn.Module
    action_set: BaseActionSet
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.base_model = self.base_model.to(self.device)

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def _topk(
        self,
        logits: torch.Tensor,
        n_output_action: int = 1,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        logits: torch.Tensor, shape (n_samples, n_action)
            The logits of the actions.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        within_candidate_idx: torch.Tensor, shape (n_samples, n_output_action)
            The index of the sampled actions within the candidate actions.

        """
        _, actions = torch.topk(logits, k=n_output_action, dim=1)
        return actions

    def retrieve_topk(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        context: Tensor, shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to retrieve actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to retrieve actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        with torch.no_grad():
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
            )

        topk = self._topk(
            logits=logits,
            n_output_action=n_output_action,
        )
        return topk

    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        is_deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        context: Tensor, shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Tensor, shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Tensor, shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        with torch.no_grad():
            logits = self.base_model(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
            )

        if is_deterministic:
            gumbel_noise = torch.zeros_like(logits).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(logits.shape).to(
                self.device
            )

        topk = self._topk(
            logits=logits + gumbel_noise,
            n_output_action=n_output_action,
        )
        return topk
