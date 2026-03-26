#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from zipfile import ZipFile
import os

import gluonts
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets, CategoricalFeatureInfo
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.repository._tsf_datasets import Dataset as MonashDataset
from gluonts.dataset.repository._tsf_datasets import TSFReader, convert_data
from gluonts.dataset.repository._tsf_reader import frequency_converter
from gluonts.dataset.repository._util import metadata
from gluonts.dataset.split import split
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.repository.datasets import get_dataset
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

LSF_PATH = "../long_term_tsf/dataset"

def default_prediction_length_from_frequency(freq: str) -> int:
    prediction_length_map = {
        "T": 60 * 24 * 7,
        "H": 24 * 7,
        "D": 30,
        "W-SUN": 8,
        "M": 12,
        "Y": 4,
        "S": 60 * 60 * 24 * 7,
    }
    try:
        freq = to_offset(freq).name
        return prediction_length_map[freq]
    except KeyError as err:
        raise ValueError(
            f"Cannot obtain default prediction length from frequency `{freq}`."
        ) from err


gluonts.dataset.repository._tsf_datasets.default_prediction_length_from_frequency = (
    default_prediction_length_from_frequency
)


def generate_forecasting_dataset(
    dataset_path: Path,
    dataset_name: str,
    dataset_writer: DatasetWriter,
    prediction_length: Optional[int] = None,
):
    dataset = gluonts.dataset.repository._tsf_datasets.datasets[dataset_name]
    dataset_path.mkdir(exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        with ZipFile(dataset.download(temp_path)) as archive:
            archive.extractall(path=temp_path)

        # only one file is exptected
        reader = TSFReader(temp_path / archive.namelist()[0])
        meta, data = reader.read()

    if dataset_name.startswith("cif_2016") and len(dataset_name) > len("cif_2016"):
        horizon = int(dataset_name[len("cif_2016_") :])
        data = list(filter(lambda x: x if x["horizon"] == horizon else False, data))
        meta.forecast_horizon = horizon

    if dataset_name.startswith("monash_m3_other"):
        meta.frequency = "quarterly"

    freq = frequency_converter(meta.frequency)
    if prediction_length is None:
        if hasattr(meta, "forecast_horizon"):
            prediction_length = int(meta.forecast_horizon)
        else:
            prediction_length = default_prediction_length_from_frequency(freq)

    # Impute missing start dates with unix epoch and remove time series whose
    # length is less than or equal to the prediction length
    data = [
        {**d, "start_timestamp": d.get("start_timestamp", "1970-01-01")}
        for d in data
        if len(d[FieldName.TARGET]) > prediction_length
    ]
    train_data, test_data = convert_data(data, prediction_length)

    meta = MetaData(
        **metadata(
            cardinality=len(data),
            freq=freq,
            prediction_length=prediction_length,
        )
    )

    dataset = TrainDatasets(metadata=meta, train=train_data, test=test_data)
    dataset.save(path_str=str(dataset_path), writer=dataset_writer, overwrite=True)


gluonts.dataset.repository._tsf_datasets.generate_forecasting_dataset = (
    generate_forecasting_dataset
)


additional_datasets = {
    "bitcoin": MonashDataset(
        file_name="bitcoin_dataset_without_missing_values.zip",
        record="5122101",
        ROOT="https://zenodo.org/record",
    ),
    "wind_power": MonashDataset(
        file_name="wind_4_seconds_dataset.zip",
        record="4656032",
        ROOT="https://zenodo.org/record",
    ),
    "us_births": MonashDataset(
        file_name="us_births_dataset.zip",
        record="4656049",
        ROOT="https://zenodo.org/record",
    ),
    "traffic_hourly": MonashDataset(
        file_name="traffic_hourly_dataset.zip",
        record="4656132",
        ROOT="https://zenodo.org/record",
    ),
    "traffic_weekly": MonashDataset(
        file_name="traffic_weekly_dataset.zip",
        record="4656135",
        ROOT="https://zenodo.org/record",
    ),
    "solar_power": MonashDataset(
        file_name="solar_4_seconds_dataset.zip",
        record="4656027",
        ROOT="https://zenodo.org/record",
    ),
    "oikolab_weather": MonashDataset(
        file_name="oikolab_weather_dataset.zip",
        record="5184708",
        ROOT="https://zenodo.org/record",
    ),
    "elecdemand": MonashDataset(
        file_name="elecdemand_dataset.zip",
        record="4656069",
        ROOT="https://zenodo.org/record",
    ),
    "covid_mobility": MonashDataset(
        file_name="covid_mobility_dataset_with_missing_values.zip",
        record="4663762",
        ROOT="https://zenodo.org/record",
    ),
    "extended_web_traffic_with_missing": MonashDataset(
        file_name="web_traffic_extended_dataset_with_missing_values.zip",
        record="7370977",
        ROOT="https://zenodo.org/record",
    ),
    "monash_m3_monthly": MonashDataset(
        file_name="m3_monthly_dataset.zip",
        record="4656298",
        ROOT="https://zenodo.org/record",
    ),
    "monash_m3_quarterly": MonashDataset(
        file_name="m3_quarterly_dataset.zip",
        record="4656262",
        ROOT="https://zenodo.org/record",
    ),
    "monash_m3_yearly": MonashDataset(
        file_name="m3_yearly_dataset.zip",
        record="4656222",
        ROOT="https://zenodo.org/record",
    ),
    "monash_m3_other": MonashDataset(
        file_name="m3_other_dataset.zip",
        record="4656335",
        ROOT="https://zenodo.org/record",
    ),
    "cif_2016_12": MonashDataset(
        file_name="cif_2016_dataset.zip",
        record="4656042",
        ROOT="https://zenodo.org/record",
    ),
    "cif_2016_6": MonashDataset(
        file_name="cif_2016_dataset.zip",
        record="4656042",
        ROOT="https://zenodo.org/record",
    ),
    "sunspot_with_missing": MonashDataset(
        file_name="sunspot_dataset_with_missing_values.zip",
        record="4654773",
        ROOT="https://zenodo.org/record",
    ),
    "temperature_rain_with_missing": MonashDataset(
        file_name="temperature_rain_dataset_with_missing_values.zip",
        record="5129073",
        ROOT="https://zenodo.org/record",
    ),
    "rideshare_with_missing": MonashDataset(
        file_name="rideshare_dataset_with_missing_values.zip",
        record="5122114",
        ROOT="https://zenodo.org/record",
    ),
    "car_parts_with_missing": MonashDataset(
        file_name="car_parts_dataset_with_missing_values.zip",
        record="4656022",
        ROOT="https://zenodo.org/record",
    ),
    "kdd_cup_2018_with_missing": MonashDataset(
        file_name="kdd_cup_2018_dataset_with_missing_values.zip",
        record="4656719",
        ROOT="https://zenodo.org/record",
    ),
    "vehicle_trips_with_missing": MonashDataset(
        file_name="vehicle_trips_dataset_with_missing_values.zip",
        record="5122535",
        ROOT="https://zenodo.org/record",
    ),
    "bitcoin_with_missing": MonashDataset(
        file_name="bitcoin_dataset_with_missing_values.zip",
        record="5121965",
        ROOT="https://zenodo.org/record",
    ),
    "london_smart_meters_with_missing": MonashDataset(
        file_name="london_smart_meters_dataset_with_missing_values.zip",
        record="4656072",
        ROOT="https://zenodo.org/record",
    ),
    "wind_farms_with_missing": MonashDataset(
        file_name="wind_farms_minutely_dataset_with_missing_values.zip",
        record="4654909",
        ROOT="https://zenodo.org/record",
    ),
    "nn5_daily_with_missing": MonashDataset(
        file_name="nn5_daily_dataset_with_missing_values.zip",
        record="4656110",
        ROOT="https://zenodo.org/record",
    ),
}

gluonts.dataset.repository._tsf_datasets.datasets.update(additional_datasets)
gluonts.dataset.repository.datasets.dataset_recipes.update({
    k: partial(
        generate_forecasting_dataset,
        dataset_name=k,
    )
    for k in additional_datasets.keys()
})


def _load_walmart(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(
        os.path.join(
            LSF_PATH, "walmart-recruiting-store-sales-forecasting/train.csv"
        )
    )

    data = []
    for id, row in df[["Store", "Dept"]].drop_duplicates().iterrows():
        row_df = df.query(f"Store == {row.Store} and Dept == {row.Dept}")
        if len(row_df) != 143:
            continue
        data.append(row_df.Weekly_Sales.to_numpy())
    data = np.stack(data, 1)

    start = pd.to_datetime("2010-02-05")
    freq = "W"
    prediction_length = prediction_length or 8
    rolling_evaluations = 4
    return data, start, freq, prediction_length, rolling_evaluations


def _load_jena_weather(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(os.path.join(LSF_PATH, "weather/weather.csv"))
    cols = list(df.columns)
    cols.remove("OT")
    cols.remove("date")
    df = df[["date"] + cols + ["OT"]]
    data = df[df.columns[1:]].to_numpy()

    start = pd.to_datetime(df[["date"]].iloc[0].item())
    freq = "10T"
    prediction_length = prediction_length or 6 * 24
    rolling_evaluations = 7
    return data, start, freq, prediction_length, rolling_evaluations


def _load_istanbul_traffic(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(
        os.path.join(LSF_PATH, "istanbul-traffic-index/istanbul_traffic.csv")
    )
    df.datetime = pd.to_datetime(df.datetime)
    df = df.set_index("datetime")
    df = df.resample("h").mean()

    data = df.values
    start = df.index[0]
    freq = "h"
    prediction_length = prediction_length or 24
    rolling_evaluations = 7
    return data, start, freq, prediction_length, rolling_evaluations


def _load_turkey_power(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(
        os.path.join(
            LSF_PATH,
            "electrical-power-demand-in-turkey/power Generation and consumption.csv",
        )
    )
    df.Date_Time = pd.to_datetime(df.Date_Time, format="%d.%m.%Y %H:%M")
    df = df.set_index("Date_Time")

    data = df.values
    start = df.index[0]
    freq = "h"
    prediction_length = prediction_length or 24
    rolling_evaluations = 7
    return data, start, freq, prediction_length, rolling_evaluations


pf_load_func_map = {
    "walmart": _load_walmart,
    "jena_weather": _load_jena_weather,
    "istanbul_traffic": _load_istanbul_traffic,
    "turkey_power": _load_turkey_power,
}


def generate_pf_dataset(
    dataset_path: Path,
    dataset_name: str,
    dataset_writer: DatasetWriter,
    prediction_length: Optional[int] = None,
):
    load_func = pf_load_func_map[dataset_name]
    data, start, freq, prediction_length, rolling_evaluations = load_func(
        dataset_name, prediction_length
    )

    train_ts = []
    for cat in range(data.shape[-1]):
        sliced_ts = data[: -prediction_length * rolling_evaluations, cat]
        train_ts.append(
            {
                "target": sliced_ts,
                "start": start,
                "feat_static_cat": [cat],
                "item_id": cat,
            }
        )

    test_ts = []
    for window in range(rolling_evaluations - 1, -1, -1):
        for cat in range(data.shape[-1]):
            sliced_ts = data[: len(data) - prediction_length * window, cat]
            test_ts.append(
                {
                    "target": sliced_ts,
                    "start": start,
                    "feat_static_cat": [cat],
                    "item_id": cat,
                }
            )

    meta = MetaData(
        freq=freq,
        feat_static_cat=[
            CategoricalFeatureInfo(name="feat_static_cat_0", cardinality=data.shape[-1])
        ],
        prediction_length=prediction_length,
    )
    dataset = TrainDatasets(metadata=meta, train=train_ts, test=test_ts)
    dataset.save(path_str=str(dataset_path), writer=dataset_writer, overwrite=True)


gluonts.dataset.repository.dataset_recipes.update({
    k: partial(generate_pf_dataset, dataset_name=k) for k in pf_load_func_map.keys()
})


def get_gluonts_test_dataset(
    dataset_name: str,
    prediction_length: int = None,
    regenerate: bool = False,
):
    default_prediction_lengths = {
        "australian_electricity_demand": 336,
        "pedestrian_counts": 24,
    }
    if prediction_length is None and dataset_name in default_prediction_lengths:
        prediction_length = default_prediction_lengths[dataset_name]

    dataset = get_dataset(
        dataset_name, prediction_length=prediction_length, regenerate=regenerate
    )

    prediction_length = prediction_length or dataset.metadata.prediction_length
    _, test_template = split(dataset.test, offset=-prediction_length)
    test_data = test_template.generate_instances(prediction_length)
    metadata = MetaData(
        freq=dataset.metadata.freq,
        target_dim=1,
        prediction_length=prediction_length,
        split="test",
    )
    return test_data, metadata


