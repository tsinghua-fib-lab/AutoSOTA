import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import urllib.request
import tarfile
from sklearn.model_selection import train_test_split


PHYSIONET_PARAMS = [
    'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
    'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
]
PHYSIONET_PARAMS_DICT = {k: i for i, k in enumerate(PHYSIONET_PARAMS)}


class PhysioNetDataset(Dataset):
    def __init__(self, root_dir='./data/physionet', split='train', download=True, quantization=0.1):
        self.root_dir = root_dir
        self.split = split
        self.quantization = quantization
        self.params = PHYSIONET_PARAMS
        self.params_dict = PHYSIONET_PARAMS_DICT

        processed_file = os.path.join(root_dir, f'norm_{split}.pt')
        if download and not os.path.exists(processed_file):
            self._download_and_process()

        self.data = torch.load(processed_file, map_location='cpu')

    def _download_and_process(self):
        os.makedirs(self.root_dir, exist_ok=True)
        raw_dir = os.path.join(self.root_dir, 'raw')
        processed_dir = os.path.join(self.root_dir, 'processed')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        url = 'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download'
        tar_path = os.path.join(raw_dir, 'set-a.tar.gz')

        if not os.path.exists(tar_path):
            print('Downloading PhysioNet dataset...')
            urllib.request.urlretrieve(url, tar_path)

        dirname = os.path.join(raw_dir, 'set-a')
        if not os.path.exists(dirname):
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(raw_dir)

        print('Processing PhysioNet data...')
        patients = []
        for txtfile in sorted(os.listdir(dirname)):
            if not txtfile.endswith('.txt'):
                continue
            record_id = txtfile.split('.')[0]
            with open(os.path.join(dirname, txtfile)) as f:
                lines = f.readlines()
                prev_time = 0
                tt = [0.]
                vals = [torch.zeros(len(self.params))]
                mask = [torch.zeros(len(self.params))]
                nobs = [torch.zeros(len(self.params))]
                for l in lines[1:]:
                    parts = l.strip().split(',')
                    if len(parts) != 3:
                        continue
                    time_str, param, val = parts
                    try:
                        time = float(time_str.split(':')[0]) + float(time_str.split(':')[1]) / 60.
                    except Exception:
                        continue
                    time = round(time / self.quantization) * self.quantization

                    if time != prev_time:
                        tt.append(time)
                        vals.append(torch.zeros(len(self.params)))
                        mask.append(torch.zeros(len(self.params)))
                        nobs.append(torch.zeros(len(self.params)))
                        prev_time = time

                    if param in self.params_dict:
                        try:
                            val_float = float(val)
                            n_observations = nobs[-1][self.params_dict[param]]
                            if n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + val_float) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = val_float
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        except Exception:
                            continue

                tt = torch.tensor(tt)
                vals = torch.stack(vals)
                mask = torch.stack(mask)
                patients.append((record_id, tt, vals, mask, None))

        torch.save(patients, os.path.join(processed_dir, f'set-a_{self.quantization}.pt'))

        train, test = train_test_split(patients, test_size=0.2, random_state=0)
        train, val = train_test_split(train, test_size=0.25, random_state=0)

        torch.save(train, os.path.join(processed_dir, 'train.pt'))
        torch.save(val, os.path.join(processed_dir, 'val.pt'))
        torch.save(test, os.path.join(processed_dir, 'test.pt'))

        for name, data in [('train', train), ('val', val), ('test', test)]:
            data_tv = self._remove_time_invariant(data)
            data_norm = self._normalize_data(data_tv)
            torch.save(data_norm, os.path.join(self.root_dir, f'norm_{name}.pt'))

        print(f'Saved: {len(train)} train, {len(val)} val, {len(test)} test samples')

    def _remove_time_invariant(self, data):
        result = []
        for sample in data:
            obs = sample[2][:, 4:]
            mask = sample[3][:, 4:]
            result.append((sample[0], sample[1], obs, mask, sample[4]))
        return result

    def _normalize_data(self, data):
        return normalize_samples(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record_id, tt, vals, mask, label = self.data[idx]
        return {
            'times': tt.float(),
            'values': vals.float(),
            'mask': mask.float()
        }


class USHCNDataset(Dataset):
    def __init__(self, root_dir='./data/ushcn', split='train', download=True, sample_rate=0.5):
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate

        processed_file = os.path.join(root_dir, f'norm_{split}.pt')
        if download and not os.path.exists(processed_file):
            self._download_and_process()

        self.data = torch.load(processed_file, map_location='cpu')

    def _download_and_process(self):
        os.makedirs(self.root_dir, exist_ok=True)

        url = 'https://raw.githubusercontent.com/usail-hkust/t-PatchGNN/main/data/ushcn/raw/small_chunked_sporadic.csv'
        csv_path = os.path.join(self.root_dir, 'small_chunked_sporadic.csv')

        if not os.path.exists(csv_path):
            print('Downloading USHCN dataset...')
            urllib.request.urlretrieve(url, csv_path)

        print('Processing USHCN data...')
        df = pd.read_csv(csv_path)

        value_cols = [c for c in df.columns if c.startswith('Value_')]
        mask_cols = [c for c in df.columns if c.startswith('Mask_')]

        grouped = df.groupby('ID')
        all_data = []

        for station_id, group in grouped:
            group = group.sort_values('Time')
            times = torch.tensor(group['Time'].values, dtype=torch.float32)
            values = torch.tensor(group[value_cols].values, dtype=torch.float32)
            mask = torch.tensor(group[mask_cols].values, dtype=torch.float32)
            values = torch.nan_to_num(values, nan=0.0)
            all_data.append((str(station_id), times, values, mask))

        train, test = train_test_split(all_data, test_size=0.2, random_state=0)
        train, val = train_test_split(train, test_size=0.25, random_state=0)

        for name, data in [('train', train), ('val', val), ('test', test)]:
            data_norm = self._normalize_data(data)
            torch.save(data_norm, os.path.join(self.root_dir, f'norm_{name}.pt'))

        print(f'Saved: {len(train)} train, {len(val)} val, {len(test)} test samples')

    def _normalize_data(self, data):
        return normalize_samples(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        station_id, tt, vals, mask = self.data[idx]
        return {
            'times': tt.float(),
            'values': vals.float(),
            'mask': mask.float()
        }


def compute_normalization_stats(data):
    if not data:
        raise RuntimeError('Cannot compute normalization stats on empty data')

    num_variables = data[0][2].shape[1]
    min_vals = []
    max_vals = []

    for i in range(num_variables):
        valid_chunks = []
        for sample in data:
            obs = sample[2]
            mask = sample[3]
            valid_vals = obs[:, i][mask[:, i] == 1]
            if len(valid_vals) > 0:
                valid_chunks.append(valid_vals)

        if valid_chunks:
            all_valid = torch.cat(valid_chunks)
            min_vals.append(all_valid.min())
            max_vals.append(all_valid.max())
        else:
            min_vals.append(torch.tensor(0.0))
            max_vals.append(torch.tensor(1.0))

    return torch.stack(min_vals), torch.stack(max_vals)


def normalize_samples_with_stats(data, min_vals, max_vals):
    max_vals = max_vals.clone()
    max_vals[max_vals == min_vals] = min_vals[max_vals == min_vals] + 1

    result = []
    for sample in data:
        prefix = sample[:2]
        obs = sample[2]
        mask = sample[3]
        suffix = sample[4:]
        obs_norm = (obs - min_vals) / (max_vals - min_vals)
        obs_norm[mask == 0] = 0
        result.append((*prefix, obs_norm, mask, *suffix))
    return result


def normalize_samples(data):
    min_vals, max_vals = compute_normalization_stats(data)
    return normalize_samples_with_stats(data, min_vals, max_vals)


def split_samples(data, random_state=0):
    train, test = train_test_split(data, test_size=0.2, random_state=random_state)
    train, val = train_test_split(train, test_size=0.25, random_state=random_state)
    return train, val, test


def process_generic_csv_directory(root_dir, time_col='date_time', id_col='record_id', time_unit='hours'):
    processed_dir = os.path.join(root_dir, 'processed')
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f'Processed directory not found: {processed_dir}')

    sample_dirs = sorted(
        entry.path for entry in os.scandir(processed_dir)
        if entry.is_dir() and os.path.exists(os.path.join(entry.path, 'time_series.csv'))
    )

    all_data = []
    for sample_dir in sample_dirs:
        csv_path = os.path.join(sample_dir, 'time_series.csv')
        sample = load_csv_sample(csv_path, time_col=time_col, id_col=id_col, time_unit=time_unit)
        if sample is not None:
            all_data.append(sample)

    if not all_data:
        raise RuntimeError(f'No valid samples found in {processed_dir}')

    return all_data


def load_csv_sample(csv_path, time_col='date_time', id_col='record_id', time_unit='hours'):
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f'Missing time column {time_col} in {csv_path}')

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    sample_id = os.path.basename(os.path.dirname(csv_path))
    if id_col in df.columns and df[id_col].notna().any():
        sample_id = str(df[id_col].dropna().iloc[0])

    value_cols = [c for c in df.columns if c not in {time_col, id_col}]
    if not value_cols:
        return None

    value_df = df[value_cols].apply(pd.to_numeric, errors='coerce')
    mask = ~value_df.isna()

    if not mask.to_numpy().any():
        return None

    base_time = df[time_col].iloc[0]
    time_delta = (df[time_col] - base_time).dt.total_seconds()
    if time_unit == 'hours':
        times = time_delta / 3600.0
    elif time_unit == 'days':
        times = time_delta / 86400.0
    elif time_unit == 'seconds':
        times = time_delta
    else:
        raise ValueError(f'Unsupported time_unit: {time_unit}')

    values = torch.tensor(value_df.fillna(0.0).to_numpy(dtype=np.float32))
    mask_tensor = torch.tensor(mask.to_numpy(dtype=np.float32))
    times_tensor = torch.tensor(times.to_numpy(dtype=np.float32))
    return sample_id, times_tensor, values, mask_tensor


class GenericProcessedCSVDataset(Dataset):
    def __init__(self, root_dir, split='train', time_unit='hours', cache_prefix='norm'):
        self.root_dir = root_dir
        self.split = split
        self.time_unit = time_unit
        self.cache_prefix = cache_prefix

        processed_file = os.path.join(root_dir, f'{cache_prefix}_{split}.pt')
        if not os.path.exists(processed_file):
            self._process()

        self.data = torch.load(processed_file, map_location='cpu')

    def _process(self):
        all_data = process_generic_csv_directory(self.root_dir, time_unit=self.time_unit)
        train, val, test = split_samples(all_data)

        for name, data in [('train', train), ('val', val), ('test', test)]:
            data_norm = normalize_samples(data)
            torch.save(data_norm, os.path.join(self.root_dir, f'{self.cache_prefix}_{name}.pt'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id, tt, vals, mask = self.data[idx]
        return {
            'times': tt.float(),
            'values': vals.float(),
            'mask': mask.float()
        }


class ClusterTraceDataset(Dataset):
    def __init__(
        self,
        root_dir='./data/ClusterTrace',
        split='train',
        window_hours=168,
        min_points=48,
        min_observations=96,
        min_variables=3,
        step_hours=None,
        force_reprocess=False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.window_hours = window_hours
        self.min_points = min_points
        self.min_observations = min_observations
        self.min_variables = min_variables
        self.step_hours = step_hours if step_hours is not None else window_hours // 2
        self.force_reprocess = force_reprocess
        self.cache_prefix = (
            f'norm_w{self.window_hours}_s{self.step_hours}_p{self.min_points}_'
            f'o{self.min_observations}_v{self.min_variables}'
        )

        processed_file = os.path.join(root_dir, f'{self.cache_prefix}_{split}.pt')
        if force_reprocess or not os.path.exists(processed_file):
            self._process()

        self.data = torch.load(processed_file, map_location='cpu')

    def _process(self):
        all_data = process_generic_csv_directory(self.root_dir, time_unit='hours')
        windowed_data = []

        for sample_id, times, values, mask in all_data:
            windowed_data.extend(
                self._create_windows(sample_id, times.float(), values.float(), mask.float())
            )

        if not windowed_data:
            raise RuntimeError('No valid ClusterTrace sliding windows were created')

        feature_dim = max(sample[2].shape[1] for sample in windowed_data)
        padded_data = [self._pad_sample(sample, feature_dim) for sample in windowed_data]

        train, val, test = split_samples(padded_data)
        train_min, train_max = compute_normalization_stats(train)

        split_map = {
            'train': normalize_samples_with_stats(train, train_min, train_max),
            'val': normalize_samples_with_stats(val, train_min, train_max),
            'test': normalize_samples_with_stats(test, train_min, train_max),
        }

        for name, data in split_map.items():
            torch.save(data, os.path.join(self.root_dir, f'{self.cache_prefix}_{name}.pt'))

    def _create_windows(self, sample_id, times, values, mask):
        windows = []
        if len(times) == 0:
            return windows

        start_time = float(times.min().item())
        end_time = float(times.max().item())
        window_start = start_time
        window_idx = 0

        while window_start < end_time:
            window_end = window_start + self.window_hours
            in_window = (times >= window_start) & (times < window_end)

            if in_window.any():
                window_times = times[in_window]
                window_values = values[in_window]
                window_mask = mask[in_window]

                keep_cols = window_mask.sum(dim=0) > 0
                if keep_cols.sum().item() >= self.min_variables:
                    window_values = window_values[:, keep_cols]
                    window_mask = window_mask[:, keep_cols]

                    rows_with_obs = window_mask.sum(dim=1) > 0
                    if rows_with_obs.any():
                        window_times = window_times[rows_with_obs]
                        window_values = window_values[rows_with_obs]
                        window_mask = window_mask[rows_with_obs]

                        if (
                            len(window_times) >= self.min_points
                            and window_mask.sum().item() >= self.min_observations
                        ):
                            rel_times = window_times - window_times.min()
                            window_id = f'{sample_id}_w{window_idx:04d}'
                            windows.append((window_id, rel_times, window_values, window_mask))

            window_start += self.step_hours
            window_idx += 1

        return windows

    def _pad_sample(self, sample, feature_dim):
        sample_id, times, values, mask = sample
        current_dim = values.shape[1]
        if current_dim == feature_dim:
            return sample

        pad_width = feature_dim - current_dim
        value_pad = torch.zeros(values.shape[0], pad_width, dtype=values.dtype)
        mask_pad = torch.zeros(mask.shape[0], pad_width, dtype=mask.dtype)
        values = torch.cat([values, value_pad], dim=1)
        mask = torch.cat([mask, mask_pad], dim=1)
        return sample_id, times, values, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id, tt, vals, mask = self.data[idx]
        return {
            'times': tt.float(),
            'values': vals.float(),
            'mask': mask.float()
        }


class EPAAirDataset(Dataset):
    def __init__(
        self,
        root_dir='./data/EPA-Air',
        split='train',
        window_hours=168,
        min_points=48,
        min_observations=96,
        min_variables=3,
        step_hours=None,
        force_reprocess=False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.window_hours = window_hours
        self.min_points = min_points
        self.min_observations = min_observations
        self.min_variables = min_variables
        self.step_hours = step_hours if step_hours is not None else window_hours // 2
        self.force_reprocess = force_reprocess
        self.cache_prefix = (
            f'norm_w{self.window_hours}_s{self.step_hours}_p{self.min_points}_'
            f'o{self.min_observations}_v{self.min_variables}'
        )

        processed_file = os.path.join(root_dir, f'{self.cache_prefix}_{split}.pt')
        if force_reprocess or not os.path.exists(processed_file):
            self._process()

        self.data = torch.load(processed_file, map_location='cpu')

    def _process(self):
        all_data = process_generic_csv_directory(self.root_dir, time_unit='hours')
        windowed_data = []

        for sample_id, times, values, mask in all_data:
            windowed_data.extend(
                self._create_windows(sample_id, times.float(), values.float(), mask.float())
            )

        if not windowed_data:
            raise RuntimeError('No valid EPA-Air sliding windows were created')

        feature_dim = max(sample[2].shape[1] for sample in windowed_data)
        padded_data = [self._pad_sample(sample, feature_dim) for sample in windowed_data]

        train, val, test = split_samples(padded_data)
        train_min, train_max = compute_normalization_stats(train)

        split_map = {
            'train': normalize_samples_with_stats(train, train_min, train_max),
            'val': normalize_samples_with_stats(val, train_min, train_max),
            'test': normalize_samples_with_stats(test, train_min, train_max),
        }

        for name, data in split_map.items():
            torch.save(data, os.path.join(self.root_dir, f'{self.cache_prefix}_{name}.pt'))

    def _create_windows(self, sample_id, times, values, mask):
        windows = []
        if len(times) == 0:
            return windows

        start_time = float(times.min().item())
        end_time = float(times.max().item())
        window_start = start_time
        window_idx = 0

        while window_start < end_time:
            window_end = window_start + self.window_hours
            in_window = (times >= window_start) & (times < window_end)

            if in_window.any():
                window_times = times[in_window]
                window_values = values[in_window]
                window_mask = mask[in_window]

                keep_cols = window_mask.sum(dim=0) > 0
                if keep_cols.sum().item() >= self.min_variables:
                    window_values = window_values[:, keep_cols]
                    window_mask = window_mask[:, keep_cols]

                    rows_with_obs = window_mask.sum(dim=1) > 0
                    if rows_with_obs.any():
                        window_times = window_times[rows_with_obs]
                        window_values = window_values[rows_with_obs]
                        window_mask = window_mask[rows_with_obs]

                        if (
                            len(window_times) >= self.min_points
                            and window_mask.sum().item() >= self.min_observations
                        ):
                            rel_times = window_times - window_times.min()
                            window_id = f'{sample_id}_w{window_idx:04d}'
                            windows.append((window_id, rel_times, window_values, window_mask))

            window_start += self.step_hours
            window_idx += 1

        return windows

    def _pad_sample(self, sample, feature_dim):
        sample_id, times, values, mask = sample
        current_dim = values.shape[1]
        if current_dim == feature_dim:
            return sample

        pad_width = feature_dim - current_dim
        value_pad = torch.zeros(values.shape[0], pad_width, dtype=values.dtype)
        mask_pad = torch.zeros(mask.shape[0], pad_width, dtype=mask.dtype)
        values = torch.cat([values, value_pad], dim=1)
        mask = torch.cat([mask, mask_pad], dim=1)
        return sample_id, times, values, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id, tt, vals, mask = self.data[idx]
        return {
            'times': tt.float(),
            'values': vals.float(),
            'mask': mask.float()
        }


class FNSPIDDataset(Dataset):
    def __init__(
        self,
        root_dir='./data/FNSPID',
        split='train',
        window_days=252,
        min_points=120,
        min_observations=500,
        min_variables=6,
        step_days=None,
        force_reprocess=False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.window_days = window_days
        self.min_points = min_points
        self.min_observations = min_observations
        self.min_variables = min_variables
        self.step_days = step_days if step_days is not None else window_days // 2
        self.force_reprocess = force_reprocess
        self.cache_prefix = (
            f'norm_w{self.window_days}_s{self.step_days}_p{self.min_points}_'
            f'o{self.min_observations}_v{self.min_variables}'
        )

        processed_file = os.path.join(root_dir, f'{self.cache_prefix}_{split}.pt')
        if force_reprocess or not os.path.exists(processed_file):
            self._process()

        self.data = torch.load(processed_file, map_location='cpu')

    def _process(self):
        all_data = process_generic_csv_directory(self.root_dir, time_unit='days')
        windowed_data = []

        for sample_id, times, values, mask in all_data:
            windowed_data.extend(
                self._create_windows(sample_id, times.float(), values.float(), mask.float())
            )

        if not windowed_data:
            raise RuntimeError('No valid FNSPID sliding windows were created')

        feature_dim = max(sample[2].shape[1] for sample in windowed_data)
        padded_data = [self._pad_sample(sample, feature_dim) for sample in windowed_data]

        train, val, test = split_samples(padded_data)
        train_min, train_max = compute_normalization_stats(train)

        split_map = {
            'train': normalize_samples_with_stats(train, train_min, train_max),
            'val': normalize_samples_with_stats(val, train_min, train_max),
            'test': normalize_samples_with_stats(test, train_min, train_max),
        }

        for name, data in split_map.items():
            torch.save(data, os.path.join(self.root_dir, f'{self.cache_prefix}_{name}.pt'))

    def _create_windows(self, sample_id, times, values, mask):
        windows = []
        if len(times) == 0:
            return windows

        start_time = float(times.min().item())
        end_time = float(times.max().item())
        window_start = start_time
        window_idx = 0

        while window_start < end_time:
            window_end = window_start + self.window_days
            in_window = (times >= window_start) & (times < window_end)

            if in_window.any():
                window_times = times[in_window]
                window_values = values[in_window]
                window_mask = mask[in_window]

                keep_cols = window_mask.sum(dim=0) > 0
                if keep_cols.sum().item() >= self.min_variables:
                    window_values = window_values[:, keep_cols]
                    window_mask = window_mask[:, keep_cols]

                    rows_with_obs = window_mask.sum(dim=1) > 0
                    if rows_with_obs.any():
                        window_times = window_times[rows_with_obs]
                        window_values = window_values[rows_with_obs]
                        window_mask = window_mask[rows_with_obs]

                        if (
                            len(window_times) >= self.min_points
                            and window_mask.sum().item() >= self.min_observations
                        ):
                            rel_times = window_times - window_times.min()
                            window_id = f'{sample_id}_w{window_idx:04d}'
                            windows.append((window_id, rel_times, window_values, window_mask))

            window_start += self.step_days
            window_idx += 1

        return windows

    def _pad_sample(self, sample, feature_dim):
        sample_id, times, values, mask = sample
        current_dim = values.shape[1]
        if current_dim == feature_dim:
            return sample

        pad_width = feature_dim - current_dim
        value_pad = torch.zeros(values.shape[0], pad_width, dtype=values.dtype)
        mask_pad = torch.zeros(mask.shape[0], pad_width, dtype=mask.dtype)
        values = torch.cat([values, value_pad], dim=1)
        mask = torch.cat([mask, mask_pad], dim=1)
        return sample_id, times, values, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id, tt, vals, mask = self.data[idx]
        return {
            'times': tt.float(),
            'values': vals.float(),
            'mask': mask.float()
        }


class PersonActivityDataset(Dataset):
    def __init__(self, root_dir='./data/PersonActivity', split='train'):
        self.root_dir = root_dir
        self.split = split
        processed_file = os.path.join(root_dir, f'norm_{split}.pt')
        if not os.path.exists(processed_file):
            self._process()
        self.data = torch.load(processed_file, map_location='cpu')

    def _locate_source_file(self):
        candidates = [
            os.path.join(self.root_dir, 'processed', 'data.pt'),
            os.path.join(self.root_dir, 'PersonActivity', 'processed', 'data.pt')
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError('PersonActivity processed/data.pt not found')

    def _process(self):
        source_file = self._locate_source_file()
        raw_data = torch.load(source_file, map_location='cpu')

        all_data = []
        for sample in raw_data:
            if len(sample) < 4:
                continue
            sample_id = str(sample[0])
            times = sample[1].float()
            values = sample[2].float()
            mask = sample[3].float()
            all_data.append((sample_id, times, values, mask))

        if not all_data:
            raise RuntimeError('No valid PersonActivity samples found')

        train, val, test = split_samples(all_data)

        for name, data in [('train', train), ('val', val), ('test', test)]:
            data_norm = normalize_samples(data)
            torch.save(data_norm, os.path.join(self.root_dir, f'norm_{name}.pt'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id, tt, vals, mask = self.data[idx]
        return {
            'times': tt.float(),
            'values': vals.float(),
            'mask': mask.float()
        }


def create_interpolation_task(times, values, mask, sample_rate=0.5):
    observed_mask = mask.clone().bool()
    n_total = observed_mask.sum().item()
    n_observed = max(1, int(n_total * sample_rate))

    observed_indices = torch.where(observed_mask)
    if len(observed_indices[0]) == 0:
        return observed_mask, torch.zeros_like(observed_mask, dtype=torch.bool)

    perm = torch.randperm(len(observed_indices[0]))[:n_observed]

    new_mask = torch.zeros_like(observed_mask, dtype=torch.bool)
    new_mask[observed_indices[0][perm], observed_indices[1][perm]] = True

    target_mask = observed_mask & (~new_mask)

    return new_mask, target_mask
