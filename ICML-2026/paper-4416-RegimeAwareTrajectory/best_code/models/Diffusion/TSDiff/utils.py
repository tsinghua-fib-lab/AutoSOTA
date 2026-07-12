# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from typing import Type, Dict
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
from functools import partial
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import Dataset
from pandas.tseries.frequencies import to_offset

from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.dataset.util import period_index
from gluonts.transform import (
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    MapTransformation,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    ValidationSplitSampler,
)
from gluonts.model.forecast import SampleForecast

sns.set(
    style="white",
    font_scale=1.1,
    rc={"figure.dpi": 125, "lines.linewidth": 2.5, "axes.linewidth": 1.5},
)


def filter_metrics(metrics, select={"ND", "NRMSE", "mean_wQuantileLoss"}):
    return {m: metrics[m].item() for m in select}


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = (
        torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.1
    return torch.linspace(beta_start, beta_end, timesteps)


def plot_train_stats(df: pd.DataFrame, y_keys=None, skip_first_epoch=True):
    if skip_first_epoch:
        df = df.iloc[1:, :]
    if y_keys is None:
        y_keys = ["train_loss", "valid_loss"]

    fix, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    for y_key in y_keys:
        sns.lineplot(
            ax=ax,
            data=df,
            x="epochs",
            y=y_key,
            label=y_key.replace("_", " ").capitalize(),
        )
    ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    plt.show()


def get_lags_for_freq(freq_str: str):
    offset = to_offset(freq_str)
    if offset.n > 1:
        raise NotImplementedError(
            "Lags for freq multiple > 1 are not implemented yet."
        )
    if offset.name == "H":
        lags_seq = [24 * i for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]]
    elif offset.name == "D" or offset.name == "B":
        # TODO: Fix lags for B
        lags_seq = [30 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    else:
        raise NotImplementedError(
            f"Lags for {freq_str} are not implemented yet."
        )
    return lags_seq


def create_transforms(
    num_feat_dynamic_real,
    num_feat_static_cat,
    num_feat_static_real,
    time_features,
    prediction_length,
):
    remove_field_names = []
    if num_feat_static_real == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if num_feat_dynamic_real == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
            if not num_feat_static_cat > 0
            else []
        )
        + (
            [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
            if not num_feat_static_real > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=int,
            ),
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_REAL,
                expected_ndim=1,
            ),
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,
                pred_length=prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            AddMeanAndStdFeature(
                target_field=FieldName.TARGET,
                output_field="stats",
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if num_feat_dynamic_real > 0
                    else []
                ),
            ),
        ]
    )


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


def get_next_file_num(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
):
    """Gets the next available file number in a directory.
    e.g., if `base_fname="results"` and `base_dir` has
    files ["results-0.yaml", "results-1.yaml"],
    this function returns 2.

    Parameters
    ----------
    base_fname
        Base name of the file.
    base_dir
        Base directory where files are located.

    Returns
    -------
        Next available file number
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and x.name.startswith(base_fname),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: x.name.startswith(base_fname),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    return max(run_nums) + 1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def add_config_to_argparser(config: Dict, parser: ArgumentParser):
    for k, v in config.items():
        sanitized_key = re.sub(r"[^\w\-]", "", k).replace("-", "_")
        val_type = type(v)
        if val_type not in {int, float, str, bool}:
            print(f"WARNING: Skipping key {k}!")
            continue
        if val_type == bool:
            parser.add_argument(f"--{sanitized_key}", type=str2bool, default=v)
        else:
            parser.add_argument(f"--{sanitized_key}", type=val_type, default=v)
    return parser


class AddMeanAndStdFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        dtype: Type = np.float32,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data[self.feature_name] = np.array(
            [data[self.target_field].mean(), data[self.target_field].std()]
        )

        return data


class ScaleAndAddMeanFeature(MapTransformation):
    def __init__(
        self, target_field: str, output_field: str, prediction_length: int
    ) -> None:
        """Scale the time series using mean scaler and
        add the scale to `output_field`.

        Parameters
        ----------
        target_field
            Key for target time series
        output_field
            Key for the mean feature
        prediction_length
            prediction length, only the time series before the
            last `prediction_length` timesteps is used for
            scale computation
        """
        self.target_field = target_field
        self.feature_name = output_field
        self.prediction_length = prediction_length

    def map_transform(self, data, is_train: bool):
        scale = np.mean(
            np.abs(data[self.target_field][..., : -self.prediction_length]),
            axis=-1,
            keepdims=True,
        )
        scale = np.maximum(scale, 1e-7)
        scaled_target = data[self.target_field] / scale
        data[self.target_field] = scaled_target
        data[self.feature_name] = scale

        return data


class ScaleAndAddMinMaxFeature(MapTransformation):
    def __init__(
        self, target_field: str, output_field: str, prediction_length: int
    ) -> None:
        """Scale the time series using min-max scaler and
        add the scale to `output_field`.

        Parameters
        ----------
        target_field
            Key for target time series
        output_field
            Key for the min-max feature
        prediction_length
            prediction length, only the time series before the
            last `prediction_length` timesteps is used for
            scale computation
        """
        self.target_field = target_field
        self.feature_name = output_field
        self.prediction_length = prediction_length

    def map_transform(self, data, is_train: bool):
        full_seq = data[self.target_field][..., : -self.prediction_length]
        min_val = np.min(full_seq, axis=-1, keepdims=True)
        max_val = np.max(full_seq, axis=-1, keepdims=True)
        loc = min_val
        scale = np.maximum(max_val - min_val, 1e-7)
        scaled_target = (full_seq - loc) / scale
        data[self.target_field] = scaled_target
        data[self.feature_name] = (loc, scale)

        return data


def descale(data, scale, scaling_type):
    if scaling_type == "mean":
        return data * scale
    elif scaling_type == "min-max":
        loc, scale = scale
        return data * scale + loc
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
            fcst.samples = descale(
                fcst.samples, scale, scaling_type=scaling_type
            )
        else:
            raise ValueError("Only SampleForecast objects supported!")
        yield fcst


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
    index = period_index(
        {FieldName.START: start, FieldName.TARGET: full_target}
    )
    return pd.DataFrame(full_target.transpose(), index=index)


def make_evaluation_predictions_with_scaling(
    dataset, predictor, num_samples: int = 100, scaling_type="mean"
):
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


class PairDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class GluonTSNumpyDataset:
    """GluonTS dataset from a numpy array.

    Parameters
    ----------
    data
        Numpy array of samples with shape [N, T].
    start_date, optional
        Dummy start date field, by default pd.Period("2023", "H")
    """

    def __init__(
        self, data: np.ndarray, start_date: pd.Period = pd.Period("2023", "H")
    ):
        self.data = data
        self.start_date = start_date

    def __iter__(self):
        for ts in self.data:
            item = {"target": ts, "start": self.start_date}
            yield item

    def __len__(self):
        return len(self.data)


class MaskInput(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        observed_field: str,
        context_length: int,
        missing_scenario: str,
        missing_values: int,
        dtype: Type = np.float32,
    ) -> None:
        # FIXME: Remove hardcoding of fields
        self.target_field = target_field
        self.observed_field = observed_field
        self.context_length = context_length
        self.missing_scenario = missing_scenario
        self.missing_values = missing_values
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data = deepcopy(data)
        data["orig_past_target"] = data["past_target"].copy()
        if self.missing_scenario == "BM-E" and self.missing_values > 0:
            data["past_target"][-self.missing_values :] = 0
            data["past_observed_values"][-self.missing_values :] = 0
        elif self.missing_scenario == "BM-B" and self.missing_values > 0:
            data["past_target"][
                -self.context_length : -self.context_length
                + self.missing_values
            ] = 0
            data["past_observed_values"][
                -self.context_length : -self.context_length
                + self.missing_values
            ] = 0
        elif self.missing_scenario == "RM" and self.missing_values > 0:
            weights = torch.ones(self.context_length)
            missing_idxs = -self.context_length + torch.multinomial(
                weights, self.missing_values, replacement=False
            )
            data["past_target"][missing_idxs] = 0
            data["past_observed_values"][missing_idxs] = 0
        return data


class ConcatDataset:
    def __init__(self, test_pairs, axis=-1) -> None:
        self.test_pairs = test_pairs
        self.axis = axis

    def _concat(self, test_pairs):
        for t1, t2 in test_pairs:
            yield {
                "target": np.concatenate(
                    [t1["target"], t2["target"]], axis=self.axis
                ),
                "start": t1["start"],
            }

    def __iter__(self):
        yield from self._concat(self.test_pairs)
