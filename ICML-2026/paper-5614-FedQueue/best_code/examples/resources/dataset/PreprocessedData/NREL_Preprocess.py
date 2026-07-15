"""
this script downloads power consumption data from NREL datasets, along with other features.
the website is https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2023%2F
dset_tags denotes the states whose data is to be downloaded
b_id denotes the building IDs of buildings whose data should be downloaded for each state
NOTE: the first three features will be the power consumption itself, time index and day index. they are default. the next 2 set of features are as follows
WEATHER FEATURES: these feature names should be passed in the argument COLUMNS_FROM_WEATHER
STATIC (NON TIME-VARYING FEATURES): these feature names should be passed in the argument STATIC_COLUMNS
"""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np


class NRELDataDownloader:
    def __init__(
        self,
        dset_tags=["CA", "IL", "NY"],
        b_id=[
            [
                15227,
                15233,
                15241,
                15222,
                15225,
                15228,
                15404,
                15429,
                15460,
                15445,
                15895,
                16281,
                16013,
                16126,
                16145,
                47395,
                15329,
                15504,
                15256,
                15292,
                15294,
                15240,
                15302,
                15352,
                15224,
                15231,
                15243,
                17175,
                17215,
                18596,
                15322,
                15403,
                15457,
                15284,
                15301,
                15319,
                15221,
                15226,
                15229,
                15234,
                15237,
                15239,
            ],
            [
                108872,
                109647,
                110111,
                108818,
                108819,
                108821,
                108836,
                108850,
                108924,
                108879,
                108930,
                108948,
                116259,
                108928,
                109130,
                113752,
                115702,
                118613,
                108816,
                108840,
                108865,
                108888,
                108913,
                108942,
                108825,
                108832,
                108837,
                109548,
                114596,
                115517,
                109174,
                109502,
                109827,
                108846,
                108881,
                108919,
                108820,
                108823,
                108828,
                108822,
                108864,
                108871,
            ],
            [
                205362,
                205863,
                205982,
                204847,
                204853,
                204854,
                204865,
                204870,
                204878,
                205068,
                205104,
                205124,
                205436,
                213733,
                213978,
                210920,
                204915,
                205045,
                204944,
                205129,
                205177,
                204910,
                205024,
                205091,
                204849,
                204860,
                204861,
                208090,
                210116,
                211569,
                204928,
                204945,
                205271,
                204863,
                204873,
                204884,
                204842,
                204843,
                204844,
                204867,
                204875,
                204880,
            ],
        ],
        COLUMNS_FROM_WEATHER=["Dry Bulb Temperature [°C]", "Wind Speed [m/s]"],
        STATIC_COLUMNS=[
            "in.sqft",
            "out.params.ext_wall_area..m2",
            "out.params.ext_window_area..m2",
        ],
    ):
        # record the input arguments
        self.dset_tags = dset_tags
        self.b_id = b_id
        self.COLUMNS_FROM_WEATHER = COLUMNS_FROM_WEATHER
        self.STATIC_COLUMNS = STATIC_COLUMNS

        # verify that there are as many lists of b_id's as dset_tags
        assert len(dset_tags) == len(b_id), (
            "Data manager didn't get the same number of dataset tags and building ID keys."
        )

        # populate list of power consumption files
        dset_urls = [
            "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2023/comstock_amy2018_release_1/timeseries_individual_buildings/by_state/upgrade=10/state=%s/%s-10.parquet"
            % (dt, "%d")
            for dt in dset_tags
        ]
        self.download_list = [
            [dl % b for b in b_id[didx]] for didx, dl in enumerate(dset_urls)
        ]

        # populate list of metadata files
        self.metadata_files = [
            "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2023/comstock_amy2018_release_1/metadata_and_annual_results/by_state/state=%s/parquet/%s_upgrade10_metadata_and_annual_results.parquet"
            % (dt, dt)
            for dt in dset_tags
        ]

        # indicator for whether data has downloaded
        self.is_data_available = False

    def download_data(self):
        # download metadata
        print("Downloading metadata", flush=True)
        self.metadata = [
            pd.read_parquet(mdfile) for mdfile in tqdm(self.metadata_files)
        ]

        # download electrical consumption data
        print("Downloading energy consumption files.", flush=True)
        self.pfiles = [
            [
                pd.read_parquet(self.download_list[didx][bidx])
                for bidx, _ in enumerate(self.b_id[didx])
            ]
            for didx, _ in enumerate(tqdm(self.dset_tags))
        ]

        # download weather data
        print("Downloading weather files.", flush=True)
        weather_url_template = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2023/comstock_amy2018_release_1/weather/amy2018/%s_2018.csv"
        wfiles_tmp = [
            [
                pd.read_csv(weather_url_template % id)
                for id in self.metadata[didx]["in.nhgis_county_gisjoin"].loc[
                    self.b_id[didx]
                ]
            ]
            for didx, _ in enumerate(tqdm(self.dset_tags))
        ]
        self.wfiles = [
            [data.loc[data.index.repeat(4)].reset_index(drop=True) for data in u]
            for u in wfiles_tmp
        ]  # the weather data has 1 sample per 4 samples of energy consumption, so repeat weather 4 times

        # indicate data is downloaded
        self.is_data_available = True

    def save_data(self, fname=os.getcwd() + "/NREL%sdataset.npz"):
        ##NOTE TO USER: pass your desired filename. If you want the name of the dataset in the filename, send a %s

        self.all_fnames = [
            fname % tuple([dt] * fname.count("%s")) for dt in self.dset_tags
        ]

        # check if files are downloaded
        if not self.is_data_available:
            print("Call the download_data() function before saving.")

        # save the data
        for didx, _ in enumerate(tqdm(self.dset_tags)):
            data_collector = []
            for bidx, b in enumerate(self.b_id[didx]):
                data = self.pfiles[didx][bidx]
                data["timestamp"] = pd.to_datetime(data["timestamp"])
                data["time_idx"] = data["timestamp"].apply(
                    lambda x: int(4 * x.hour + x.minute / 15)
                )
                data["day_idx"] = data["timestamp"].apply(lambda x: x.weekday())
                for cols in self.COLUMNS_FROM_WEATHER:
                    wcol = self.wfiles[didx][bidx][cols].to_numpy()
                    data[cols] = wcol
                for cols in self.STATIC_COLUMNS:
                    data[cols] = [float(self.metadata[didx].loc[b][cols])] * data.shape[
                        0
                    ]
                data = data.drop(columns="timestamp")
                data = data[
                    ["out.electricity.total.energy_consumption", "time_idx", "day_idx"]
                    + self.COLUMNS_FROM_WEATHER
                    + self.STATIC_COLUMNS
                ]
                data_collector.append(data.to_numpy())
            np.savez(self.all_fnames[didx], data=np.array(data_collector))
