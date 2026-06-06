from copy import deepcopy
import math
import pandas as pd
import numpy as np
from datetime import datetime
from distutils.util import strtobool
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName


def split_train_val(train_set, num_test_windows, context_length, prediction_length, freq):
    """
    Splits a training dataset into a truncated training set and a validation set.

    Parameters:
    - train_set: The input training dataset.
    - num_test_windows: Number of rolling windows for validation.
    - context_length: Context length for the model.
    - prediction_length: Prediction horizon for the model.
    - freq: Data frequency (e.g., 'H' for hourly).

    Returns:
    - trunc_train_set: Truncated training dataset (ListDataset).
    - val_set: Validation dataset (ListDataset).
    """
    trunc_train_list = []
    val_set_list = []
    univariate = False

    for train_seq in iter(train_set):
        # truncate train set
        offset = num_test_windows * prediction_length
        trunc_train_seq = deepcopy(train_seq)

        if len(train_seq[FieldName.TARGET].shape) == 1:
            trunc_train_len = train_seq[FieldName.TARGET].shape[0] - offset
            trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][:trunc_train_len]
            univariate = True
        elif len(train_seq[FieldName.TARGET].shape) == 2:
            trunc_train_len = train_seq[FieldName.TARGET].shape[1] - offset
            trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][:, :trunc_train_len]
        else:
            raise ValueError(f"Invalid Data Shape: {str(len(train_seq[FieldName.TARGET].shape))}")

        trunc_train_list.append(trunc_train_seq)

        # construct val set by rolling
        for i in range(num_test_windows):
            val_seq = deepcopy(train_seq)
            rolling_len = trunc_train_len + prediction_length * (i+1)
            if univariate:
                val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][trunc_train_len + prediction_length * (i-1) - context_length : rolling_len]
            else:
                val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][:, :rolling_len]
            
            val_set_list.append(val_seq)

    trunc_train_set = ListDataset(
        trunc_train_list, freq=freq, one_dim_target=univariate
    )

    val_set = ListDataset(
        val_set_list, freq=freq, one_dim_target=univariate
    )
    
    return trunc_train_set, val_set


def truncate_test(test_set, context_length, prediction_length, freq):
    """
    Truncates the test dataset to ensure only the last context and prediction lengths are retained.

    Parameters:
    - test_set: The input test dataset.
    - context_length: Context length for the model.
    - prediction_length: Prediction horizon for the model.
    - freq: Data frequency.

    Returns:
    - trunc_test_set: Truncated test dataset (ListDataset).
    """
    trunc_test_list = []
    for test_seq in iter(test_set):
        # truncate train set
        trunc_test_seq = deepcopy(test_seq)

        trunc_test_seq[FieldName.TARGET] = trunc_test_seq[FieldName.TARGET][- (prediction_length * 2 + context_length):]

        trunc_test_list.append(trunc_test_seq)

    trunc_test_set = ListDataset(
        trunc_test_list, freq=freq, one_dim_target=True
    )

    return trunc_test_set


def get_rolling_test(stage, test_set, border_begin_idx, border_end_idx, rolling_length, pred_len, freq=None):
    """
    Using rolling windows to build the test dataset.

    Parameters:
    - stage: Stage name (e.g., 'test', 'val').
    - test_set: The test dataset.
    - border_begin_idx: Start index for rolling windows.
    - border_end_idx: End index for rolling windows.
    - rolling_length: Gap length of each rolling window.
    - pred_len: Prediction length.
    - freq: Data frequency.

    Returns:
    - rolling_test_set: Rolling test dataset (ListDataset).
    """
    num_test_windows = math.ceil(((border_end_idx - border_begin_idx - pred_len) / rolling_length))
    print(f"{stage}  pred_len: {pred_len} : num_test_windows: {num_test_windows}")

    test_set = next(iter(test_set))
    rolling_test_seq_list = list()
    for i in range(num_test_windows):
        rolling_test_seq = deepcopy(test_set)
        rolling_end = border_begin_idx + pred_len + i * rolling_length
        rolling_test_seq[FieldName.TARGET] = rolling_test_seq[FieldName.TARGET][:, :rolling_end]
        rolling_test_seq_list.append(rolling_test_seq)

    rolling_test_set = ListDataset(
        rolling_test_seq_list, freq=freq, one_dim_target=False
    )
    return rolling_test_set


def get_rolling_test_of_gift_eval(dataset, prediction_length, windows):
    """
    Using rolling windows to build the test dataset for GiftEval.
    https://github.com/SalesforceAIResearch/gift-eval/blob/61ec5e563188bc4b2d7e86f6a7fcc78270607ae7/src/gift_eval/data.py#L213
    Get the windows from the back of the dataset, for example if the dataset has N time points:
    - The first window will be from the first time point to the N - prediction_length * windows time point.
    - The second window will be from the first time point to the N - prediction_length * (windows - 1) time point.
    - The last window will be from the first time point to the N time point.

    Parameters:
    - dataset: The input dataset.
    - prediction_length: Prediction length.
    - windows: Number of rolling windows.

    Returns:
    - rolling_test_set: Rolling test dataset (ListDataset).
    """
    rolling_test_seq_list = list()
    dataset = next(iter(dataset))
    if "freq" not in dataset.keys():
        raise ValueError("The dataset must contain the 'freq' key.")
    freq = dataset["freq"]
    is_univariate = len(dataset[FieldName.TARGET].shape) == 1

    for i in range(windows):
        rolling_test_seq = deepcopy(dataset)
        rolling_end = dataset[FieldName.TARGET].shape[-1] - prediction_length * (windows - i)
        if is_univariate:
            rolling_test_seq[FieldName.TARGET] = dataset[FieldName.TARGET][:rolling_end]
        elif len(dataset[FieldName.TARGET].shape) == 2:
            rolling_test_seq[FieldName.TARGET] = dataset[FieldName.TARGET][:, :rolling_end]
        else:
            raise ValueError(f"Invalid Data Shape: expected 1 or 2 dimensions, got {len(dataset[FieldName.TARGET].shape)}")
        rolling_test_seq_list.append(rolling_test_seq)

    rolling_test_set = ListDataset(
        rolling_test_seq_list, freq=freq, one_dim_target=is_univariate
    )
    return rolling_test_set



def df_to_mvds(df, freq='H'):
    """
    Converts a pandas DataFrame to a multivariate ListDataset for GluonTS.

    Parameters:
    - df: Input DataFrame where columns represent time series variables.
    - freq: Data frequency (e.g., 'H' for hourly).

    Returns:
    - dataset: Multivariate ListDataset.
    """
    datasets = []
    for variable in df.keys():
        ds = {"item_id" : variable, "target" : df[variable], "start": str(df.index[0])}
        datasets.append(ds)
    dataset = ListDataset(datasets,freq=freq)
    return dataset


def convert_monash_data_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

def monash_format_convert(loaded_data, frequency, multivariate):
    series_names = loaded_data['series_name'].values

    if str(frequency) == '10_minutes':
        freq = '10min'
    elif str(frequency) == 'daily':
        freq = 'D'
    else:
        freq = frequency

    if multivariate:
        timestamps = pd.date_range(start=loaded_data['start_timestamp'][0], periods=len(loaded_data['series_value'][0]), freq=freq)
        new_df = pd.DataFrame({ 'date': timestamps })

        series_df = pd.DataFrame({ series: loaded_data['series_value'][i] for i, series in enumerate(series_names) })
        result_df = pd.concat([new_df, series_df], axis=1)
    else:
        result = []
        for idx, row in loaded_data.iterrows():
            result.append({
                'target': np.array(row['series_value'], dtype=np.float32),
                'start': pd.Period(row['start_timestamp'], freq=freq),
                'feat_static_cat': np.array([idx], dtype=np.int32),
                'item_id': idx,
            })
        result_df = pd.DataFrame(result)
    return result_df