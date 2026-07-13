from typing import Any

from torch.nn import Identity
from torchmetrics import Metric

from tsl.metrics.torch import MaskedMetric


class MaskedMetricWrapper(Metric):
    def __init__(self,
                 metric: MaskedMetric,
                 input_preprocessing=None,
                 target_preprocessing=None,
                 mask_preprocessing=None,
                 **kwargs):
        self.__dict__['is_differentiable'] = metric.is_differentiable
        self.__dict__['higher_is_better'] = metric.higher_is_better
        super(MaskedMetricWrapper, self).__init__(**kwargs)
        # super(MaskedMetricWrapper, self).__init__(lambda *args, **kwargs: None, **kwargs)
        self.metric = metric

        if input_preprocessing is None:
            input_preprocessing = Identity()

        if target_preprocessing is None:
            target_preprocessing = Identity()

        if mask_preprocessing is None:
            mask_preprocessing = Identity()

        self.input_preprocessing = input_preprocessing
        self.target_preprocessing = target_preprocessing
        self.mask_preprocessing = mask_preprocessing

    def update(self, y_hat, y, mask=None):
        y_hat = self.input_preprocessing(y_hat)
        y = self.target_preprocessing(y)
        if mask is not None:
            mask = self.mask_preprocessing(mask)
        self.metric.update(y_hat, y, mask)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        super(MaskedMetricWrapper, self).reset()
        self.metric.reset()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.metric(*args, **kwargs)
