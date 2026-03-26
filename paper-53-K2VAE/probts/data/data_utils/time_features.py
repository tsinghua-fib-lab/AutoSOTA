# ---------------------------------------------------------------------------------
# Portions of this file are derived from GluonTS
# - Source: https://github.com/awslabs/gluonts
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------



from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.transform import MapTransformation
from typing import List, Type

class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


class FourierDateFeatures(TimeFeature):
    def __init__(self, freq: str) -> None:
        super().__init__()
        # reocurring freq
        freqs = [
            "month",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        return np.vstack([np.cos(steps), np.sin(steps)])


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]

    # Pandas has start and end frequencies, e.g `AS` and `A` for yearly start
    # and yearly end frequencies. We don't make that difference and instead
    # rely only on the end frequencies which don't have the `S` prefix.
    # Note: Secondly ("S") frequency exists, where we don't want to remove the
    # "S"!
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]

    return base_freq


def fourier_time_features_from_frequency(freq_str: str) -> List[TimeFeature]:
    offset = to_offset(freq_str)
    granularity = norm_freq_str(offset.name)
    granularity = granularity.upper()
    features = {
        "M": ["weekofyear"],
        "W": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: List[TimeFeature] = [
        FourierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes


def get_lags(freq_str:str):
    """
    Calculate appropriate lag values for time series forecasting based on data frequency.

    Parameters
    ----------
    freq_str : str
        The frequency of the time series data. Supported values include:

    Returns
    -------
    lags : list[int]
        A list of lag values, representing the offsets of past observations to include in the model.
        The lags are tailored to capture autocorrelation and seasonality patterns for the specified frequency.

    Examples
    --------
    >>> get_lags("H")
    [1, 24, 168]  # Captures hourly, daily, and weekly seasonality

    >>> get_lags("D")
    [1, 7, 14]  # Captures daily, weekly, and bi-weekly seasonality
    """
    freq_str = freq_str.upper()
    if freq_str == "M":
        lags = [1, 12]
    elif freq_str == "D":
        lags = [1, 7, 14]
    elif freq_str == "B":
        lags = [1, 2]
    elif freq_str == "H":
        lags = [1, 24, 168]
    elif freq_str in ("T", "min"):
        lags = [1, 4, 12, 24, 48]
    else:
        lags = [1]

    return lags


def target_transformation_length(
    target: np.ndarray, pred_length: int, is_train: bool
) -> int:
    return target.shape[-1] + (0 if is_train else pred_length)


class AddCustomizedTimeFeatures(MapTransformation):
    """
    Adds a set of time features.

    If `is_train=True` the feature matrix has the same length as the `target`
    field. If `is_train=False` the feature matrix has length
    `len(target) + pred_length`

    Parameters
    ----------
    start_field
        Field with the start time stamp of the time series
    target_field
        Field with the array containing the time series values
    output_field
        Field name for result.
    time_features
        list of time features to use.
    pred_length
        Prediction length
    """

    @validated()
    def __init__(
        self,
        start_field: str,
        target_field: str,
        output_field: str,
        time_features,
        pred_length: int,
        dtype: Type = np.float32,
    ) -> None:
        self.date_features = time_features
        self.pred_length = pred_length
        self.start_field = start_field
        self.target_field = target_field
        self.output_field = output_field
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )

        if len(self.date_features.shape) == 2:
            data[self.output_field] = self.date_features[:length].astype(self.dtype)
        else:
            data[self.output_field] = self.date_features[:length].astype(np.float64)
        data[self.output_field] = self.date_features[:length].astype(np.float64)
        data[self.output_field] = np.transpose(data[self.output_field])
        
        return data
