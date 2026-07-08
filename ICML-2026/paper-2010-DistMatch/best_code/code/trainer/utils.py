import operator
from functools import reduce
from typing import List


def map_identity():
    return lambda x: x


def merge_dicts():
    return lambda *x: reduce(operator.ior, x, {})


def map_merge_dicts(map_dict):
    return lambda *x: {map_dict[key]: val for key, val in reduce(operator.ior, x, {}).items() if key in map_dict}


def batch_to_device_all():
    return lambda batch_data, device: {key: val.to(device) for key, val in batch_data.items()}


def batch_to_device_all_but(exclude: List[str]):
    return lambda batch_data, device: {key: val.to(device) for key, val in batch_data.items() if key not in exclude}
