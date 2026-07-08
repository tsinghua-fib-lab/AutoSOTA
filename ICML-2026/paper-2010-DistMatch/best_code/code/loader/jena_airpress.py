import itertools
import numpy as np
import pandas as pd
import torch

JENA_TYPE_PREFIX = "jena-air"

ROOF_DATA = list(zip(np.repeat(np.arange(2004, stop=2022), 2), itertools.cycle(("a", "b"))))
SAALE_DATA = list(zip(np.repeat(np.arange(2003, stop=2022), 2), itertools.cycle(("a", "b"))))
SOIL_DATA = list(zip(np.repeat(np.arange(2007, stop=2022), 2), itertools.cycle(("a", "b"))))
DATA_SETS = {
    'mpi_roof': ROOF_DATA,
    'mpi_saale': SAALE_DATA,
#    'MPI_Soil': SOIL_DATA  # Has no pressure since its "soil data"
}


# This Features are the overlap between the Saale and Roof Data
# Foldesi and Valdenegro-Toro (2022) does not use rain but in addtion maximum wind velocity (which is only in roof data)
USED_FEATURES = ['T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'wd (deg)', 'rain (mm)']

START_YEAR = 2004  # Available for all


FIND_GAPS = True

def get_jena_data(dataset_type, path):
    assert path.is_dir()
    if dataset_type.startswith(f'{JENA_TYPE_PREFIX}-1y'):  # 1 year
        take = 2
    elif dataset_type.startswith(f'{JENA_TYPE_PREFIX}-3y'):  # 3 year
        take = 6
    elif dataset_type.startswith(f'{JENA_TYPE_PREFIX}-5y'):  # 5 year
        take = 10
    elif dataset_type.startswith(f'{JENA_TYPE_PREFIX}-10y'):  # 10 year
        take = 20
    elif dataset_type.startswith(f'{JENA_TYPE_PREFIX}-15y'):  # 15 year
        take = 30
    elif dataset_type.startswith(f'{JENA_TYPE_PREFIX}-18y'):  # 18 year
        take = 36
    else:
        raise ValueError("Dataset not supported!")
    if dataset_type.endswith("60m"):
        down_sample = 6
    elif dataset_type.endswith("30m"):
        down_sample = 3
    elif dataset_type.endswith("10m"):
        down_sample = 1
    else:
        raise ValueError("Rate not supported")
    datasets = []
    for key, data in DATA_SETS.items():
        data_used = list(map(lambda x: f"{x[0]}{x[1]}", filter(lambda x: x[0] >= START_YEAR, data)))[-take:]
        _id = f"{key}_{data_used[0]}-{data_used[-1]}"
        tables = []
        for table in data_used:
            p_str = f"{path.resolve()}/{key}_{table}.csv"
            ts_df = pd.read_csv(p_str, encoding='iso8859_15', dayfirst=True, parse_dates=['Date Time'], skip_blank_lines=True)
            if FIND_GAPS:
                ts_df['gap'] = ts_df['Date Time'].sort_values().diff() > pd.to_timedelta('10 min')
                print(ts_df[ts_df.gap])
            tables.append(ts_df)
        ts = pd.concat(tables, axis=0)
        ts = ts.iloc[::down_sample, :]
        Y_full = ts['p (mbar)']
        X_full = ts.loc[:, ts.columns != 'p (mbar)']
        X_full.drop(columns=["Date Time"], inplace=True)
        X_full = X_full[USED_FEATURES]
        datasets.append((
            torch.from_numpy(X_full.to_numpy()).float(),
            torch.from_numpy(Y_full.to_numpy()).float(),
            _id
        ))
    return datasets
