import math
from typing import Type

import numpy as np
from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpandDimArray,
    MapTransformation,
    TargetDimIndicator,
)
import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.TSFlow.utils.variables import Setting


def create_transforms(
    time_features,
    prediction_length,
    freq,
    train_length,
):
    return Chain(
        [
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
            AddMeanFeature(
                target_field=FieldName.TARGET,
                output_field="mean",
                train_length=train_length,
            ),
            AddMeanAndStdFeature(
                target_field=FieldName.TARGET,
                output_field="stats",
            ),
        ]
    )


def create_multivariate_transforms(
    time_features,
    prediction_length,
    freq,
    train_length,
    target_dim,
):
    return Chain(
        [
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=2,
            ),
            # maps the target to (1, T)
            # if the target data is uni dimensional
            ExpandDimArray(
                field=FieldName.TARGET,
                axis=None,
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
            AddIDFeature(
                output_field="id",
                train_length=train_length,
                setting="multivariate",
            ),
            AddMeanFeature(
                target_field=FieldName.TARGET,
                output_field="mean",
                train_length=train_length,
                setting="multivariate",
            ),
            AddMeanAndStdFeature(
                target_field=FieldName.TARGET,
                output_field="stats",
            ),
            TargetDimIndicator(
                field_name="target_dimension_indicator",
                target_field=FieldName.TARGET,
            ),
        ]
    )


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
        data[self.feature_name] = np.array([data[self.target_field].mean(), data[self.target_field].std()])

        return data


class AddIDFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        output_field: str,
        train_length: int,
        setting: Setting = Setting.UNIVARIATE,
    ) -> None:
        self.feature_name = output_field
        self.train_length = train_length
        self.setting = setting

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        id = data["feat_static_cat"][0]
        if not is_train:
            id = id % self.train_length
        data[self.feature_name] = id
        return data


class AddMeanFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        train_length: int,
        dtype: Type = np.float32,
        minimum_scale: float = 0.01,
        setting: Setting = Setting.UNIVARIATE,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.dtype = dtype
        self.minimum_scale = minimum_scale
        self.train_means = {}
        self.train_length = train_length
        self.setting = setting

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        if "item_id" in data:
            id = data["item_id"]
        else:
            id = data["feat_static_cat"][0]
        if is_train:
            if id in self.train_means.keys():
                scale = self.train_means[id]
            else:
                scale = np.array([data[self.target_field].mean(-1)])
                scale = np.clip(scale, a_min=self.minimum_scale, a_max=np.inf)
                self.train_means[id] = scale
        else:
            if id in self.train_means.keys():
                scale = self.train_means[id]
            else:
                scale = self.train_means[id % self.train_length]

        data[self.feature_name] = scale
        return data


class AddStdFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        train_length: int,
        dtype: Type = np.float32,
        minimum_scale: float = 0.1,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.dtype = dtype
        self.minimum_scale = minimum_scale
        self.train_means = {}
        self.train_length = train_length

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        if "item_id" in data:
            id = data["item_id"]
        else:
            id = data["feat_static_cat"][0]
        if is_train:
            if id in self.train_means.keys():
                scale = self.train_means[id]
            else:
                scale = np.array([data[self.target_field].std(-1)])
                scale = np.clip(scale, a_min=self.minimum_scale, a_max=np.inf)
                self.train_means[id] = scale
        else:
            if id in self.train_means.keys():
                scale = self.train_means[id]
            else:
                scale = self.train_means[id % self.train_length]

        data[self.feature_name] = scale
        return data


class AddStdFreqFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        freq: int,
        dtype: Type = np.float32,
        minimum_scale: float = 1e-10,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.freq = freq
        self.dtype = dtype
        self.minimum_scale = minimum_scale

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = data[self.target_field].shape[0]
        stds = data[self.target_field][: ((length // self.freq) * self.freq)].reshape(-1, self.freq).std(0)
        stds = np.tile(stds, math.ceil(length / self.freq))[:length]
        data[self.feature_name] = stds
        return data


class ScaleAndAddMeanFeature(MapTransformation):
    def __init__(self, target_field: str, output_field: str, prediction_length: int) -> None:
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


class ScaleAndAddLongMeanFeature(MapTransformation):
    def __init__(self, target_field: str, output_field: str, prediction_length: int) -> None:
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
        scaled_target = data[self.target_field] / data["mean"]
        data[self.target_field] = scaled_target
        data[self.feature_name] = data["mean"]
        return data


class ScaleAndAddMinMaxFeature(MapTransformation):
    def __init__(self, target_field: str, output_field: str, prediction_length: int) -> None:
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
