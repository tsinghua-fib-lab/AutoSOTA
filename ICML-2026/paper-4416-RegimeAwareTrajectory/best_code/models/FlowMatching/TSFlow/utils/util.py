import math
import re
from argparse import ArgumentParser, ArgumentTypeError
from functools import partial
from typing import Dict, Optional

import numpy as np
import ot as pot
import pandas as pd
import torch
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.dataset.util import period_index
from gluonts.model.forecast import SampleForecast
from gluonts.torch.scaler import Scaler
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    ValidationSplitSampler,
)


class ConcatDataset:
    def __init__(self, test_pairs, axis=-1) -> None:
        self.test_pairs = test_pairs
        self.axis = axis

    def _concat(self, test_pairs):
        for t1, t2 in test_pairs:
            data = {
                "target": np.concatenate([t1["target"], t2["target"]], axis=self.axis),
                "start": t1["start"],
            }
            if "item_id" in t1.keys():
                data["item_id"] = t1["item_id"]
            if "feat_static_cat" in t1.keys():
                data["feat_static_cat"] = t1["feat_static_cat"]
            yield data

    def __iter__(self):
        yield from self._concat(self.test_pairs)


def create_splitter(past_length: int, future_length: int, mode: str = "train"):
    if mode == "train":
        instance_sampler = ExpectedNumInstanceSampler(
            num_instances=1,
            min_past=past_length,
            min_future=future_length,
        )
    elif mode == "val":
        instance_sampler = ValidationSplitSampler(min_future=future_length)
    elif mode == "test":
        instance_sampler = TestSplitSampler()

    splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=past_length,
        future_length=future_length,
        time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
    )
    return splitter


def add_config_to_argparser(config: Dict, parser: ArgumentParser):
    for k, v in config.items():
        sanitized_key = re.sub(r"[^\w\-]", "", k).replace("-", "_")
        val_type = type(v)
        if val_type not in {int, float, str, bool}:
            print(f"WARNING: Skipping key {k}!")
            continue
        if val_type == bool:  # noqa: E721
            parser.add_argument(f"--{sanitized_key}", type=str2bool, default=v)
        else:
            parser.add_argument(f"--{sanitized_key}", type=val_type, default=v)
    return parser


def filter_metrics(metrics, select={"ND", "NRMSE", "CRPS"}):
    return {m: metrics[m].item() for m in select}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


class LongScaler(Scaler):
    def __call__(
        self, data: torch.Tensor, scale: torch.Tensor, loc: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if loc is None:
            loc = torch.zeros_like(scale)
        scaled_data = (data - loc) / scale
        return scaled_data, loc, scale


def quantile_loss(y_prediction, y_target, q):
    assert y_target.shape == y_prediction.shape
    e = y_target - y_prediction
    # var = (1 - 2 * q + 2 * q**2) / (q**2 * (1 - q) ** 2)
    # norm = torch.sqrt(torch.sqrt(1 / var))
    norm = 1
    loss = (norm * torch.max(q * e, (q - 1) * e)).mean()  # .sum()
    return loss


def descale(data, scale, scaling_type):
    if scaling_type == "mean":
        return data * scale
    elif scaling_type == "min-max" or scaling_type == "zscore":
        mean, loc, scale = scale
        return (data * scale + loc) * mean
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}")


def predict_and_descale(predictor, dataset, num_samples, scaling_type):
    """Generates forecasts using the predictor on the test
    dataset and then scales them back to the original space
    using the scale feature from `ScaleAndAddMeanFeature`
    or `ScaleAndAddMinMaxFeature` transformation.

    Parameters
    ----------
    predictor
        GluonTS predictor
    dataset
        Test dataset
    num_samples
        Number of forecast samples
    scaling_type
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Yields
    ------
        SampleForecast objects

    Raises
    ------
    ValueError
        If the predictor generates Forecast objects other than SampleForecast
    """
    forecasts = predictor.predict(dataset, num_samples=num_samples)
    for input_ts, fcst in zip(dataset, forecasts):
        scale = input_ts["scale"]
        if isinstance(fcst, SampleForecast):
            fcst.samples = descale(fcst.samples, scale, scaling_type=scaling_type)
        else:
            raise ValueError("Only SampleForecast objects supported!")
        yield fcst


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


def to_dataframe_and_descale(input_label, scaling_type) -> pd.DataFrame:
    """Glues together "input" and "label" time series and scales
    the back using the scale feature from transformation.

    Parameters
    ----------
    input_label
        Input-Label pair generated from the test template
    scaling_type
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Returns
    -------
        A DataFrame containing the time series
    """
    start = input_label[0][FieldName.START]
    scale = input_label[0]["scale"]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    full_target = descale(full_target, scale, scaling_type=scaling_type)
    index = period_index({FieldName.START: start, FieldName.TARGET: full_target})
    return pd.DataFrame(full_target.transpose(), index=index)


def make_evaluation_predictions_with_scaling(dataset, predictor, num_samples: int = 100, scaling_type="mean"):
    """A customized version of `make_evaluation_predictions` utility
    that first scales the test time series, generates the forecast and
    the scales it back to the original space.

    Parameters
    ----------
    dataset
        Test dataset
    predictor
        GluonTS predictor
    num_samples, optional
        Number of test samples, by default 100
    scaling_type, optional
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Returns
    -------
        A tuple of forecast and time series iterators
    """
    window_length = predictor.prediction_length + predictor.lead_time
    _, test_template = split(dataset, offset=-window_length)
    test_data = test_template.generate_instances(window_length)
    input_test_data = list(test_data.input)

    return (
        predict_and_descale(
            predictor,
            input_test_data,
            num_samples=num_samples,
            scaling_type=scaling_type,
        ),
        map(
            partial(to_dataframe_and_descale, scaling_type=scaling_type),
            test_data,
        ),
    )


class GluonTSNumpyDataset:
    """GluonTS dataset from a numpy array.

    Parameters
    ----------
    data
        Numpy array of samples with shape [N, T].
    start_date, optional
        Dummy start date field, by default pd.Period("2023", "H")
    """

    def __init__(self, data: np.ndarray, start_date: pd.Period = pd.Period("2023", "H")):
        self.data = data
        self.start_date = start_date

    def __iter__(self):
        for ts in self.data:
            item = {"target": ts, "start": self.start_date}
            yield item

    def __len__(self):
        return len(self.data)
