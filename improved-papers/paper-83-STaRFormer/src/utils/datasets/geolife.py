import logging
import os
import os.path as osp
import numpy as np
import torch 
import random
import patoolib
import requests
import glob
import pandas as pd
import pyproj
import warnings

from typing import List, Dict, Tuple, Union
from tqdm.auto import tqdm

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from ..parameters import TrainingMethodOptions
from .utils import BaseData


LABEL2IDX = {
    'bike': 0,
    'bus': 1,
    'car': 2,
    'walk': 3,
    None: None 
}

IDX2LABEL = {
    0: 'bike',
    1: 'bus',
    2: 'car',
    3: 'walk',
    None: None 
}

LABELS_MAP = {
    'airplane': None,
    'bike': 'bike',
    'boat': None,
    'bus': 'bus',
    'car': 'car',
    'motorcycle': None,
    'run': 'walk', #'train
    'subway': None,
    'taxi': 'car',
    'train': None, #'train
    'walk': 'walk',
}

DEFAULT_LABELS = list(LABELS_MAP.keys())

V_MAX_LIMITS = {
    'bike': 60.0, # km/h
    'bus': 100.0, # km/h
    'car': 120.0, # km/h
    'run': 12.0, # 4 min/km pace / 4:30 --> 13.33 / 5:00 --> 12 km/h
    'taxi': 120.0, # km/h
    'walk': 12.0, #6.0, # km/h make same as run
}

########
# misc #
########

class GeoLifeData(BaseData):
    def __init__(
        self,
        traj_id: str=None, 
        data: Tensor=None, 
        label: Tensor=None, 
        **kwargs
        ) -> None:
        self.traj_id = traj_id
        self.data = data
        self.label = label
        self.seq_len = torch.tensor(data.size(0))
        super().__init__(**kwargs)


class GeoLifeBatchData(BaseData):
    def __init__(
        self,
        traj_id: List[str]=None, 
        data: Union[Tensor, PackedSequence]=None, 
        label: Tensor=None, 
        seq_len: Union[List, Tensor]=None,
        **kwargs
        ) -> None:
        self.traj_id = traj_id
        self.data = data
        self.label = label
        self.batch_size=len(traj_id) if traj_id is not None else None
        self.seq_len = seq_len.reshape(-1, 1)
        self.ptr = [torch.tensor([0])]
        if traj_id is not None:
            for idx, sl in enumerate(self.seq_len):
                self.ptr.extend([self.ptr[idx] + sl])
        self.ptr = torch.concat(self.ptr)
        super().__init__(**kwargs)


############
# download #
############

def _get_progress_log(part, total, progress_bar_length: int=50):
    """ If the total is unknown, just return the part """
    if total == -1:
        return f"Downloaded: {part / 1024 ** 2:.2f} MB"

    passed = "=" * int(progress_bar_length * part / total)
    rest = " " * (progress_bar_length - len(passed))
    p_bar = f"[{passed}{rest}] {part * 100/total:.2f}%"
    if part == total:
        p_bar += "\n"
    return p_bar

def start_download(url, dataset_name: str='Geolife'):
    """ """
    response = requests.get(url, allow_redirects=True, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download dataset {dataset_name}")

    return response

from pathlib import Path


def download_until_finish(url: str, response: requests.Response, dataset_path: Path, chunk_size: int=4096) -> Path:
    """ """
    data_length = int(response.headers.get("content-length", -1))
    size_mb_msg = (
        f"    Size: {data_length / 1024 ** 2:.2f} MB" if data_length != -1 else ""
    )
    dataset_file_path = dataset_path
    with open(dataset_file_path, "wb") as ds_file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                ds_file.write(chunk)
                downloaded += len(chunk)
                print(
                    _get_progress_log(downloaded, data_length) + size_mb_msg,
                    end="\r",
                    flush=True,
                )
    return dataset_file_path

def _download(url: str, dataset_name: str, dataset_file_path: Path) -> Path:
    """ Check if the dataset is already downloaded
    """
    if os.path.exists(dataset_file_path):
        return dataset_file_path
    
    # Make the download request
    response = start_download(url, dataset_name)

    # Download the dataset to a zip file
    return download_until_finish(url, response, dataset_file_path)

def download(
    url: str, 
    dataset_name: str, 
    dataset_file_path: Path | str,
    dataset_path: Path,
    uncompress: bool=True
    ):
    _download(url, dataset_name, dataset_file_path)

    if uncompress:
        patoolib.extract_archive(
            str(dataset_file_path),
            outdir=str(dataset_path),
            verbosity=1,
            interactive=False,
        )
    
    return dataset_file_path
        

##################
# preoprocessing #
##################

def read_plt_file(plt_file):
    """read the plt file
    Adjusted from https://heremaps.github.io/pptk/tutorials/viewer/geolife.html"""
    df = pd.read_csv(plt_file, skiprows=6, header=None)
    # convert the date and time to a timestamp
    df['timestamp'] = pd.to_datetime(df[5].astype(str) + ' ' + df[6].astype(str)) 
    # for clarity rename columns
    df.rename(inplace=True, columns={0: 'lat', 1: 'lon', 3: 'alt'})
    # remove unused columns
    df.drop(inplace=True, columns=[2, 4, 5, 6])
    return df


def read_labels(labels_file):
    """
    the attributes are whitespace-delimited, hence the keyword argument sep='\s+'
    Adjusted from https://heremaps.github.io/pptk/tutorials/viewer/geolife.html
    """
    # reda the labels file
    df = pd.read_csv(labels_file, skiprows=1, header=None, sep='\s+')
    df.rename(inplace=True, columns={4: 'label'})
    # convert the date and time to a timestamp
    df['start_time'] = pd.to_datetime(df[0].astype(str) + ' ' + df[1].astype(str))
    df['end_time'] = pd.to_datetime(df[2].astype(str) + ' ' + df[3].astype(str))
    # removed unsused columns
    df.drop(inplace=True, columns=[0,1,2,3])
    return df

def apply_labels(df_points, df_labels, user):
    """find indices where a lable is available
    Adjusted from https://heremaps.github.io/pptk/tutorials/viewer/geolife.html"""
    indices = df_labels['start_time'].searchsorted(df_points['timestamp'], side='right') - 1
    no_label = (indices < 0) | (df_points['timestamp'].values >= df_labels['end_time'].iloc[indices].values)
    # attach the available labels, if no label is found --> 'unk'
    df_points['label'] = df_labels['label'].iloc[indices].values
    df_points.loc[no_label, 'label'] = 'unk'

    # Generate unique IDs for rows with matching timestamps
    unique_ids = 0
    for index, row in df_labels.iterrows():
        mask = (df_points['timestamp'] >= row['start_time']) & (df_points['timestamp'] <= row['end_time'])
        df_points.loc[mask, 'id'] = 'id_'+ str(user) + '_' + str(unique_ids)
        unique_ids += 1

def preprocess(data_folder: Path | str, raw_data_file_path: Path | str):
    """Adjusted from https://heremaps.github.io/pptk/tutorials/viewer/geolife.html"""
    subfolders = [f for f in os.listdir(data_folder) if f != ".DS_Store"]
    dfs = []
    pbar = tqdm(sorted(subfolders))
    for i, sf in enumerate(pbar):
        pbar.set_description('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        user_folder = osp.join(data_folder, sf)

        plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
        df = pd.concat([read_plt_file(f) for f in plt_files])
        df['user'] = int(sf)
        df['id'] = 'none'

        labels_file = os.path.join(user_folder, 'labels.txt')
        if os.path.exists(labels_file):
            df_labels = read_labels(labels_file)
            apply_labels(df_points=df, df_labels=df_labels, user=sf)
            #add_traj_id(df=df, df_labels=df_labels)
        else:
            df['label'] = 'unk'
        
        if isinstance(df.timestamp.iloc[0], str):
            df.timestamp = df.timestamp.apply(lambda x: pd.to_datetime(x))
        assert isinstance(df.timestamp.iloc[0], pd._libs.tslibs.timestamps.Timestamp), f'{df.timestamp.iloc[0]}, {type(df.timestamp.iloc[0])}'
        dfs.append(df)

    df_geolife = pd.concat(dfs, ignore_index=True)
    df_geolife.to_pickle(raw_data_file_path, compression="gzip")
    return df_geolife


##############
# processing #
##############


def filter_1(df: pd.DataFrame) -> pd.DataFrame:
    """
    - filters indices without a trajectory id
    - filters indices without a label
    - filters indices where trajectory is less than 5
    """
    df = df[df.id != 'none'] # filter index with no traj id
    df = df[df.label != 'unk'] # filter index with no traj label
    value_counts = df.id.value_counts()
    mask = value_counts > 5
    return df[df.id.isin(value_counts[mask].index)]

def filter_utm_50(df: pd.DataFrame) -> pd.DataFrame:
    """ Adjusted from https://heremaps.github.io/pptk/tutorials/viewer/geolife.html """
    mask_50 = (df['lon'] > 114.0) & (df['lon'] < 120.0) & (df['lat'] > 32.0) & (df['lat'] < 48.0)
    return df[mask_50]

def convert_lon_lat_alt_to_meters():
    """ Adjusted from https://heremaps.github.io/pptk/tutorials/viewer/geolife.html """
    return pyproj.Proj(proj='utm', zone=50, ellps='WGS84')

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    proj = convert_lon_lat_alt_to_meters()
    grouped = df.groupby('id')
    dfs = [
        _create_features_per_trajectory(grouped.get_group(x), proj=proj)
        for idx, x in enumerate(tqdm(grouped.groups, desc='Creating Features'))
    ]
    return pd.concat(dfs)

def _calc_velocity(x, y, dt):
    return ((x[1] - x[0])**2 + (y[1] - y[0])**2)**0.5  / dt

def _calc_acc(v, dt):
    return (v[1] - v[0]) / dt


def _create_features_per_trajectory(df_grouped: pd.DataFrame, proj: pyproj.Proj) -> pd.DataFrame:
    x_data, y_data = proj(df_grouped['lon'].values, df_grouped['lat'].values)
    alt_data = (df_grouped['alt']*0.3048).values
    if isinstance(df_grouped['timestamp'].iloc[0], str):
        df_grouped['timestamp'] = df_grouped['timestamp'].apply(lambda x: pd.Timedelta(x))
    t_data = (df_grouped['timestamp'] - df_grouped['timestamp'].iloc[0])
    dt_data = (df_grouped['timestamp'] - df_grouped['timestamp'].shift(+1))
    # convert pd.timedelta to datetime.timedelta 
    t_data = t_data.apply(lambda x: pd.Timedelta(0).to_pytimedelta() if isinstance(x, pd._libs.tslibs.nattype.NaTType) else x.to_pytimedelta())
    dt_data = dt_data.apply(lambda x: pd.Timedelta(0).to_pytimedelta() if isinstance(x, pd._libs.tslibs.nattype.NaTType) else x.to_pytimedelta())
    
    v = np.zeros(len(x_data))
    v_x = np.zeros(len(x_data))
    v_y = np.zeros(len(x_data))
    a = np.zeros(len(x_data))
    a_x = np.zeros(len(x_data))
    a_y = np.zeros(len(x_data))

    for i in range(len(x_data)-1):
        if dt_data.iloc[i+1].total_seconds() == 0.0:
            v[i+1] = 0.
            v_x[i+1] = 0.
            v_y[i+1] = 0.
            a[i+1] = 0.
            a_x[i+1] = 0.
            a_y[i+1] = 0.
        else:
            x = [x_data[i], x_data[i+1]]
            y = [y_data[i], y_data[i+1]]
            dt = dt_data.iloc[i+1].total_seconds()
            v[i+1] = _calc_velocity(x, y, dt)
            #v[i+1] = ((x_data[i+1] - x_data[i])**2 + (y_data[i+1] - y_data[i])**2)**0.5  / ()
            v_x[i+1] = (x_data[i+1] - x_data[i])  / dt
            v_y[i+1] = (y_data[i+1] - y_data[i])  / dt
            a[i+1] = _calc_acc([v[i], v[i+1]], dt)
            a_x[i+1] = _calc_acc([v_x[i], v_x[i+1]], dt)
            a_y[i+1] = _calc_acc([v_y[i], v_y[i+1]], dt)
            #a[i+1] = (v[i+1] - v[i]) / (dt_data.iloc[i+1].total_seconds())
            #a_x[i+1] = (v_x[i+1] - v_x[i]) / dt
            #a_y[i+1] = (v_y[i+1] - v_y[i]) / dt
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
        df_grouped['x (m)'] = x_data
        df_grouped['y (m)'] = y_data
        df_grouped['alt (m)'] = alt_data
        df_grouped['v (m/s)'] = v
        df_grouped['v (km/h)'] = v*3.6
        df_grouped['vx (m/s)'] = v_x
        df_grouped['vy (m/s)'] = v_y
        df_grouped['a (m/s^2)'] = a
        df_grouped['ax (m/s^2)'] = a_x
        df_grouped['ay (m/s^2)'] = a_y
        df_grouped['time'] = t_data
        df_grouped['dt'] = dt_data
    return df_grouped


def filter_trajectories_velocity(df: pd.DataFrame, bound: float=0.2) -> pd.DataFrame:
    df_grouped = df.groupby('id')
    return pd.concat([
        df_grouped.get_group(x)
        for x in tqdm(df_grouped.groups, desc='Filtering by velocity')
        if df_grouped.get_group(x).label.iloc[0] in list(V_MAX_LIMITS.keys()) and \
            (df_grouped.get_group(x)['v (km/h)'].mean() <= V_MAX_LIMITS[df_grouped.get_group(x).label.iloc[0]] + bound)
    ])

def create_new_labels_for_training(df: pd.DataFrame) -> pd.DataFrame:
    df.label = df.label.apply(lambda x: LABELS_MAP[x])
    df.label = df.label.apply(lambda x: LABEL2IDX[x])
    
    df_grouped = df.groupby('id')
    keys= [k for k in LABEL2IDX.values() if k != None]
    return pd.concat([ 
        df_grouped.get_group(x)
        for x in tqdm(df_grouped.groups, desc="Creating new labels for training")
        if df_grouped.get_group(x).label.unique()[0] in keys 
    ])

def prepare_df(
    df: pd.DataFrame,
    feature_columns: List[str]=['x (m)', 'y (m)', 'v (m/s)', 'v (km/h)', 'vx (m/s)', 'vy (m/s)', \
'a (m/s^2)', 'ax (m/s^2)', 'ay (m/s^2)', 'time', 'dt']
    ) -> List[GeoLifeData]:

    df_grouped = df.groupby('id')
    df_to_list = []
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
        for x in tqdm(df_grouped.groups, desc='Processing data'):
            df_group = df_grouped.get_group(x)
            if 'time' in feature_columns:
                df_group.time = df_group.time.apply(lambda t: t.total_seconds())
            if 'dt' in feature_columns:
                df_group.dt = df_group.dt.apply(lambda t: t.total_seconds())

            df_to_list.append(
                GeoLifeData(
                    traj_id=df_grouped.get_group(x).id.iloc[0], 
                    data=torch.from_numpy(df_group[feature_columns].values), 
                    label=torch.from_numpy(np.array((df_grouped.get_group(x).label.values[0])))
                )
            )

    return df_to_list


def shorten_trajectory(trajectory_data: torch.Tensor, max_trajectory_length: int = 1024):
    """
    Shortens a trajectory sequence to a specified length N, keeping the first and last elements
    and selecting the rest in a special manner. shorten lengths proprtionally from 1024 onwards.
    
    Args:
        trajectory_data (torch.Tensor): The input trajectory sequence.
        max_trajectory_length (int): The desired maximum length of the shortened trajectory.
    
    Returns:
        torch.Tensor: The shortened trajectory sequence.
    """
    # Get the length of the input trajectory
    trajectory_length = trajectory_data.size(0)
    
    # Check if the trajectory is already shorter than the desired length
    if trajectory_length <= max_trajectory_length:
        return trajectory_data
    
    # Calculate the proportional length
    proportion = max_trajectory_length / 9052
    new_length = int(np.ceil(proportion * trajectory_length))
    
    # Ensure the new length does not exceed the maximum allowed length
    new_length = min(new_length, max_trajectory_length)
    
    # Create the shortened trajectory array
    shortened_trajectory_data = torch.zeros(new_length, trajectory_data.size(1))

    # Set the first and last elements
    shortened_trajectory_data[0, :] = trajectory_data[0, :]
    shortened_trajectory_data[-1, :] = trajectory_data[-1, :]
    
    len_remain = shortened_trajectory_data.size(0) - 2
    
    # Interpolate
    x_int = np.interp(
        np.linspace(1, trajectory_length, len_remain),
        np.arange(1, trajectory_length - 1),
        trajectory_data[1:-1, 0].numpy()
    )
    y_int = np.interp(
        np.linspace(1, trajectory_length, len_remain),
        np.arange(1, trajectory_length - 1),
        trajectory_data[1:-1, 1].numpy()
    )
    
    # Get xy interpolated positions
    xy = np.concatenate([x_int[:, np.newaxis], y_int[:, np.newaxis]], axis=1)
    pairwise_dist = np.sqrt(((xy[:, np.newaxis, :] - trajectory_data[:, :2].numpy())**2).sum(axis=2))
    
    # Find nearest features from pairwise distances
    nearest_indices = pairwise_dist.argmin(axis=1)
    selected_features = trajectory_data[nearest_indices, 2:]
    
    shortened_trajectory_data[1:-1, :2] = torch.from_numpy(xy)
    shortened_trajectory_data[1:-1, 2:] = selected_features
    shortened_trajectory_data[0, :] = trajectory_data[0, :]
    shortened_trajectory_data[-1, :] = trajectory_data[-1, :] 

    return shortened_trajectory_data


def create_train_test_split(
    data: list, 
    targets: list, 
    train_test_split: dict=None, 
    identical_training_class_label_distribution: bool=True,
    ):
    assert train_test_split.get('train', None) is not None
    assert train_test_split.get('test', None) is not None
    assert sum([v for v in train_test_split.values() if v is not None]) == 1.0 # assert sum is 1.0

    if identical_training_class_label_distribution:
        
        random.shuffle(data)
        target_keys = set([k for k in LABEL2IDX.values() if k != None])

        data_sorted_by_label = {k: [] for k in target_keys}
        for d in data:    
            #_, _, t = d
            data_sorted_by_label[d.label.item()].append(d)

        v, c = np.unique(targets, return_counts=True)
       #print(v, c)
        N_train_per_target = int(np.min(c)*(train_test_split.get('train')))

        if train_test_split.get('val', None) is not None:
            N_val_per_target = int(np.min(c)*(train_test_split.get('val'))) 
            #print(N_train_per_target, N_val_per_target)
            train_data = []
            val_data = []
            test_data = []
            for k, v in data_sorted_by_label.items():
                train_data.extend(v[:N_train_per_target])
                val_data.extend(v[N_train_per_target:(N_train_per_target+N_val_per_target)])
                test_data.extend(v[(N_train_per_target+N_val_per_target):])
            
            random.shuffle(train_data)
            random.shuffle(val_data)
            random.shuffle(test_data)
        else:
            train_data = []
            val_data = None
            test_data = []
            for k, v in data_sorted_by_label.items():
                train_data.extend(v[:N_train_per_target])
                test_data.extend(v[N_train_per_target:])
        
            random.shuffle(train_data)
            random.shuffle(test_data)
    else:
        random.shuffle(data)
        tot = len(data)
        N_train = int(tot*train_test_split.get('train', None))
        N_val = int(tot*train_test_split.get('val', None)) if train_test_split.get('val', None) is not None else None
        N_test = int(tot*train_test_split.get('test', None))

        #if tot != (N_train + N_val, N_test):
        #    diff = tot - (N_train + N_val, N_test)

        if N_val is not None:
            train_data = data[:N_train]
            val_data = data[N_train:(N_train+N_val)]
            test_data = data[(N_train+N_val):]

            random.shuffle(train_data)
            random.shuffle(val_data)
            random.shuffle(test_data)
        else:
            train_data = data[:N_train]
            val_data = None
            test_data = data[N_train:]

            random.shuffle(train_data)
            random.shuffle(test_data)

    return {
        'train_data': train_data, 
        'val_data': val_data,
        'test_data': test_data
    }

###########
# logging #
###########

def log_stats(
    method: str,
    cli_logger: logging.Logger,
    data: List[Tuple]=None,
    ds_config: Dict=None
    ):
    if method == TrainingMethodOptions.centralized:
        log_stats_centralized(cli_logger=cli_logger, data=data, ds_config=ds_config)
    #elif method == TrainingMethodOptions.federated:
        #log_stats_federated()
    else:
        RuntimeError

def log_stats_centralized(
    cli_logger: logging.Logger, 
    data: List, 
    ds_config: Dict
    ):
    cli_logger.info("-" * 50)
    cli_logger.info("Statistics")
    cli_logger.info("-" * 50)
    if isinstance(ds_config['indices']['train'][0], str):
        train_data = [sample.traj_id for sample in data if sample.traj_id in ds_config['indices']['train']]
        if ds_config['indices'].get('val', None) is not None:
            val_data = [sample.traj_id for sample in data if sample.traj_id in ds_config['indices']['val']]
        test_data = [sample.traj_id for sample in data if sample.traj_id in ds_config['indices']['test']]
    else:
        raise RuntimeError

    cli_logger.info(f"Size of Train Data:\t{len(train_data)}")
    if ds_config['indices'].get('val', None) is not None: cli_logger.info(f"Size of Val Data:\t {len(val_data)}")
    cli_logger.info(f"Size of Test Data:\t{len(test_data)}")
    cli_logger.info("-" * 50)
    cli_logger.info(f"Total Data Size:\t{len(data)}")
    cli_logger.info("-" * 50)


#########
# print #
#########

def print_label_distribution(labels: list):
    idx2label = IDX2LABEL

    v, c = np.unique(labels, return_counts=True)

    for vv,cc in zip(v, c):
        #print(type(vv), isinstance(vv, int))
        if isinstance(vv, (int, np.int32, np.int64)):
            print(f'{idx2label[vv]}: {cc} {(cc/len(labels)*100):.2f}%')
        elif isinstance(vv, Tensor):
            print(f'{idx2label[int(vv.item())]}: {cc} {(cc/len(labels)*100):.2f}%')
        else:
            print(f'{vv}: {cc} {(cc/len(labels)*100):.2f}%')

    

