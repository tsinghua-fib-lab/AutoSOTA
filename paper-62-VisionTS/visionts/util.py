import inspect
import os
import requests

import pandas as pd
from torchvision.transforms import Resize
from tqdm import tqdm

def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(local_filename, 'wb') as file:
        with tqdm(
            desc=f"Download: {local_filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))


def safe_resize(size, interpolation):
    signature = inspect.signature(Resize)
    params = signature.parameters
    if 'antialias' in params:
        return Resize(size, interpolation, antialias=False)
    else:
        return Resize(size, interpolation)


POSSIBLE_SEASONALITIES = {
    "S": [3600],  # 1 hour
    "T": [1440, 10080],  # 1 day or 1 week
    "H": [24, 168],  # 1 day or 1 week
    "D": [7, 30, 365],  # 1 week, 1 month or 1 year
    "W": [52, 4], # 1 year or 1 month
    "M": [12, 6, 3], # 3 months, 6 months or 1 year
    "B": [5],
    "Q": [4, 2], # 6 months or 1 year
}


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]
    return base_freq


def freq_to_seasonality_list(freq: str, mapping_dict=None) -> int:
    if mapping_dict is None:
        mapping_dict = POSSIBLE_SEASONALITIES
    offset = pd.tseries.frequencies.to_offset(freq)
    base_seasonality_list = mapping_dict.get(norm_freq_str(offset.name), [])
    seasonality_list = []
    for base_seasonality in base_seasonality_list:
        seasonality, remainder = divmod(base_seasonality, offset.n)
        if not remainder:
            seasonality_list.append(seasonality)
    seasonality_list.append(1) # we append P=1 for those without significant periodicity
    return seasonality_list

