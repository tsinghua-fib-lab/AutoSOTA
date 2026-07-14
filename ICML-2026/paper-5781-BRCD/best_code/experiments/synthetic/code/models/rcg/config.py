import json
import yaml
from typing import List
from dataclasses import dataclass


@dataclass
class GraphGenConf:
    seed: int
    nodes: int
    states: int
    samples: int

    verbose: bool


@dataclass
class DataGenParams:
    n: int # number of DAGs for every node
    nodes: List[int]
    threading: bool
    workers: int
    verbose: bool


class DataGenConf:
    def __init__(self, **conf):
        self.graph_gen_conf: GraphGenConf= GraphGenConf(**conf['graph_gen_conf'])
        self.params: DataGenParams = DataGenParams(**conf['params'])


@dataclass
class LearnPriorConf:
    k: List[int]
    oracle: bool
    threading: bool
    workers: int
    verbose: bool


@dataclass
class ExperimentConf:
    anomalous_samples: int
    oracle: bool
    L: List[int]
    int_samples: List[int]
    workers: int
    threading: bool

    verbose: bool
    # Only for internal use
    l_value: int = 1
    interventional_samples: int = 100


def load_config(name: str, type: type):
    with open(f'config/{name}', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return type(**cfg)


def dump_config(conf, src_dir: str):
    with open(f'{src_dir}/readme.txt', 'w') as readme:
        readme.write(json.dumps(_obj_to_dict(conf)))
    readme.close()


def _obj_to_dict(obj):
    if type(obj) is dict:
        res = {}
        for k, v in obj.items():
            res[k] = _obj_to_dict(v)
        return res
    elif not hasattr(obj, '__dict__'):
        return obj
    else:
        return _obj_to_dict(vars(obj))
