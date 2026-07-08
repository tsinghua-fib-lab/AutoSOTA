from abc import ABC
from typing import Any, Optional

import torch
from torchmetrics import Metric


class MetricMultiBatchMultiAlphaMixin:
    @staticmethod
    def _split_per_alpha(Y: Optional[torch.Tensor], q_low: torch.Tensor, q_high: torch.Tensor, alpha):
        if isinstance(alpha, float):
            alpha = torch.tensor([alpha], dtype=torch.float, device=q_low.device)
        if len(alpha.shape) == 1:  # One Batch  (different alphas after another)
            no_alphas = alpha.shape[0]
            batches = 1
            multi_batch = False
        else:  # Multi Batch   (first different batches then different alphas after another)
            no_alphas = alpha.shape[1]
            batches = alpha.shape[0]
            alpha = alpha[0]  # All batches have same alphas
            multi_batch = True

        assert q_low.shape[0] % no_alphas == 0
        eval_len = q_low.shape[0] // (no_alphas * batches)
        if Y is not None:
            if multi_batch:
                Y = Y[:, -eval_len:, :]
                Y = Y.reshape(-1, 1)
            else:
                Y = Y[-eval_len:]
            Y = Y.repeat(no_alphas, 1)

        return torch.split(Y, eval_len * batches) if Y is not None else None, torch.split(q_low, eval_len * batches), \
               torch.split(q_high, eval_len * batches), alpha


class MetricPerCoverageMixin:

    def __init__(self, coverage_values) -> None:
        self._coverage_values = list(coverage_values)

    def _get_idx(self, alpha):
        return self._coverage_values.index(alpha)

    def _get_compute_out(self, per_alpha_values, overall_value=None, base_name=None):
        base_name = base_name if base_name is not None else self.__class__.__name__
        per_coverage = {f"{base_name}_A_{self._get_alpha_str(alpha)}": per_alpha_values[idx] for idx, alpha in enumerate(self._coverage_values)}
        if overall_value is not None:
            per_coverage[f"{base_name}"] = overall_value
        return per_coverage

    @staticmethod
    def _get_alpha_str(alpha):
        return f'{alpha:.2f}'[-2:]


class MissCoverage(Metric, MetricPerCoverageMixin, MetricMultiBatchMultiAlphaMixin):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def __init__(self, coverage_values) -> None:
        Metric.__init__(self)
        MetricPerCoverageMixin.__init__(self, coverage_values)
        MetricMultiBatchMultiAlphaMixin.__init__(self)
        self.add_state("total_covered", default=torch.zeros((len(self._coverage_values), 1)), dist_reduce_fx="sum")
        self.add_state("total_steps", default=torch.zeros((len(self._coverage_values), 1)), dist_reduce_fx="sum")

    def update(self, Y: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor, alpha, **kwargs: Any) -> None:
        """
        :param Y:       [batch_size, 1]    or  [batch, batch_size, 1]
        :param q_low:   [batch_size, 1]    or  [batch, batch_size, 1]
        :param q_high:  [batch_size, 1]    or  [batch, batch_size, 1]
        :param alpha:   float or List () or Tensor
        """
        Y_c, q_low_c, q_high_c, alphas = self._split_per_alpha(Y, q_low, q_high, alpha)

        loss_mask = kwargs.get('loss_mask', None)
        for idx, alpha in enumerate(alphas):
            if alpha not in self._coverage_values:
                raise ValueError("Alpha not available for ")
            if loss_mask is None:
                self.total_covered[self._get_idx(alpha)] += torch.sum(
                    torch.logical_and(torch.ge(Y_c[idx], q_low_c[idx]), torch.le(Y_c[idx], q_high_c[idx])).to(torch.int32))
                self.total_steps[self._get_idx(alpha)] += Y_c[idx].shape[0]
            else:
                tmp = torch.logical_and(torch.ge(Y_c[idx], q_low_c[idx]), torch.le(Y_c[idx], q_high_c[idx])).masked_fill(loss_mask, False)
                self.total_covered[self._get_idx(alpha)] += torch.sum(tmp.to(torch.int32))
                self.total_steps[self._get_idx(alpha)] += Y_c[idx].shape[0] - loss_mask.sum()

    def compute(self):
        coverages = self.total_covered / self.total_steps
        miss_coverage = (coverages + torch.tensor(self._coverage_values, device=coverages.device).unsqueeze(1) - 1) * - 1.0
        miss_coverage = torch.relu(miss_coverage)
        return self._get_compute_out(miss_coverage, overall_value=miss_coverage.sum())


class CoverageDiff(Metric, MetricPerCoverageMixin, MetricMultiBatchMultiAlphaMixin):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def __init__(self, coverage_values) -> None:
        Metric.__init__(self)
        MetricPerCoverageMixin.__init__(self, coverage_values)
        MetricMultiBatchMultiAlphaMixin.__init__(self)
        self.add_state("total_covered", default=torch.zeros((len(self._coverage_values), 1)), dist_reduce_fx="sum")
        self.add_state("total_steps", default=torch.zeros((len(self._coverage_values), 1)), dist_reduce_fx="sum")

    def update(self, Y: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor, alpha, **kwargs: Any) -> None:
        """
        :param Y:       [batch_size, 1]    or  [batch, batch_size, 1]
        :param q_low:   [batch_size, 1]    or  [batch, batch_size, 1]
        :param q_high:  [batch_size, 1]    or  [batch, batch_size, 1]
        :param alpha:   float or List () or Tensor
        """
        Y_c, q_low_c, q_high_c, alphas = self._split_per_alpha(Y, q_low, q_high, alpha)

        loss_mask = kwargs.get('loss_mask', None)
        for idx, alpha in enumerate(alphas):
            if alpha not in self._coverage_values:
                raise ValueError("Alpha not available for ")
            if loss_mask is None:
                self.total_covered[self._get_idx(alpha)] += torch.sum(
                    torch.logical_and(torch.ge(Y_c[idx], q_low_c[idx]), torch.le(Y_c[idx], q_high_c[idx])).to(torch.int32))
                self.total_steps[self._get_idx(alpha)] += Y_c[idx].shape[0]
            else:
                tmp = torch.logical_and(torch.ge(Y_c[idx], q_low_c[idx]), torch.le(Y_c[idx], q_high_c[idx])).masked_fill(loss_mask, False)
                self.total_covered[self._get_idx(alpha)] += torch.sum(tmp.to(torch.int32))
                self.total_steps[self._get_idx(alpha)] += Y_c[idx].shape[0] - loss_mask.sum()

    def compute(self):
        coverages = self.total_covered / self.total_steps
        diff_coverage = (coverages + torch.tensor(self._coverage_values, device=coverages.device).unsqueeze(1) - 1)
        return self._get_compute_out(diff_coverage)


class PIWidth(Metric, MetricPerCoverageMixin, MetricMultiBatchMultiAlphaMixin):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def __init__(self, coverage_values) -> None:
        Metric.__init__(self)
        MetricPerCoverageMixin.__init__(self, coverage_values)
        MetricMultiBatchMultiAlphaMixin.__init__(self)
        self.add_state("total_width", default=torch.zeros((len(self._coverage_values), 1), dtype=torch.float),
                       dist_reduce_fx="sum")
        self.add_state("total_steps", default=torch.zeros((len(self._coverage_values), 1)),
                       dist_reduce_fx="sum")

    def update(self, q_low: torch.Tensor, q_high: torch.Tensor, alpha, **kwargs: Any) -> None:
        """
        :param Y:       [batch_size, 1]    or  [batch, batch_size, 1]
        :param q_low:   [batch_size, 1]    or  [batch, batch_size, 1]
        :param q_high:  [batch_size, 1]    or  [batch, batch_size, 1]
        :param alpha:   float or List () or Tensor
        """
        _, q_low_c, q_high_c, alphas = self._split_per_alpha(None, q_low, q_high, alpha)

        loss_mask = kwargs.get('loss_mask', None)
        for idx, a in enumerate(alphas):
            if loss_mask is None:
                self.total_width[self._get_idx(a)] += (q_high_c[idx] - q_low_c[idx]).abs().sum()  # abs() to avoid to make negative intervals look good
                self.total_steps[self._get_idx(a)] += q_low_c[idx].shape[0]
            else:
                self.total_width[self._get_idx(a)] += (q_high_c[idx] - q_low_c[idx]).abs().masked_fill(loss_mask, 0.0).sum()
                self.total_steps[self._get_idx(a)] += q_low_c[idx].shape[0] - loss_mask.sum()


    def compute(self) -> Any:
        avg_width = self.total_width / self.total_steps
        avg_width_total = self.total_width.sum() / self.total_steps.sum()
        return self._get_compute_out(avg_width, overall_value=avg_width_total)


class WinklerScore(Metric, MetricPerCoverageMixin, MetricMultiBatchMultiAlphaMixin):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def __init__(self, coverage_values) -> None:
        Metric.__init__(self)
        MetricPerCoverageMixin.__init__(self, coverage_values)
        MetricMultiBatchMultiAlphaMixin.__init__(self)
        self.add_state("total_score", default=torch.zeros((len(self._coverage_values), 1), dtype=torch.float),
                       dist_reduce_fx="sum")
        self.add_state("total_steps", default=torch.zeros((len(self._coverage_values), 1)),
                       dist_reduce_fx="sum")

    def update(self, Y: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor, alpha, **kwargs: Any) -> None:
        """
        :param Y:       [batch_size, 1]    or  [batch, batch_size, 1]
        :param q_low:   [batch_size, 1]    or  [batch, batch_size, 1]
        :param q_high:  [batch_size, 1]    or  [batch, batch_size, 1]
        :param alpha:   float or List () or Tensor
        """
        # https://otexts.com/fpp3/distaccuracy.html

        Y_c, q_low_c, q_high_c, alphas = self._split_per_alpha(Y, q_low, q_high, alpha)

        loss_mask = kwargs.get('loss_mask', None)
        for idx, a in enumerate(alphas):
            width = (q_high_c[idx] - q_low_c[idx]).abs()  # abs() to avoid to make negative intervals look good
            undershoot = torch.lt(Y_c[idx], q_low_c[idx]).long()
            overshoot = torch.gt(Y_c[idx], q_high_c[idx]).long()
            if loss_mask is not None:
                width = width.masked_fill(loss_mask, 0)
                undershoot = undershoot.masked_fill(loss_mask, 0)
                overshoot = overshoot.masked_fill(loss_mask, 0)
                self.total_steps[self._get_idx(a)] += Y_c[idx].shape[0] - loss_mask.sum()
            else:
                self.total_steps[self._get_idx(a)] += Y_c[idx].shape[0]
            self.total_score[self._get_idx(a)] += \
                (width + (undershoot * ((q_low_c[idx] - Y_c[idx]) * 2 / a)) + (overshoot * ((Y_c[idx] - q_high_c[idx]) * 2 / a))).sum()

    def compute(self):
        avg_scores = self.total_score / self.total_steps
        avg_scores_total = self.total_score.sum() / self.total_steps.sum()
        return self._get_compute_out(avg_scores, overall_value=avg_scores_total)


class DummyMetric(Metric):

    def __init__(self, **kwargs: Any) -> None:
        Metric.__init__(self)
        self.add_state("dummy_score", default=torch.zeros((1, 1), dtype=torch.float))

    def update(self, *_: Any, **__: Any):
        pass

    def compute(self) -> Any:
        return {'dummy_score': self.dummy_score}
