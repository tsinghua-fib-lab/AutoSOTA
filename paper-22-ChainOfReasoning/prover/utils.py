# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/deepseek-ai/DeepSeek-Prover-V1.5

import os
import json
import pytz
from pathlib import Path
from datetime import datetime
from collections import UserDict
from importlib.machinery import SourceFileLoader
from easydict import EasyDict as AttrDict


LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

def get_datetime(readable=False):
    if readable:
        return datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y/%m/%d %H:%M:%S")
    return datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")

def load_config(fname):
    name = Path(fname).stem
    mod = SourceFileLoader(name, fname).load_module()

    config = {}
    for n in dir(mod):
        if not n.startswith("__"):
            config[n] = getattr(mod, n)
    config = AttrDict(config)

    return config

def load_jsonl_objects(input_path):
    objects = []
    with open(input_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            objects.append(json.loads(line))
    return objects


class ConcurrentJob(object):
    def __init__(self, stage_list):
        assert len(stage_list) > 1
        self.stage_list = stage_list
        self.reset()
    
    def is_idle(self):
        return self._stage_idx is None
    
    def reset(self):
        self._stage_idx = None
        self._stage_cache = None
    
    def start(self, **kwargs):
        self._stage_idx = 1
        self._stage_cache = self.stage_list[0](**kwargs)
    
    def get_status(self):
        assert not self.is_idle()
        while True:
            status = self.stage_list[self._stage_idx](**self._stage_cache)
            if status is None:
                return None
            self._stage_idx += 1
            if self._stage_idx == len(self.stage_list):
                self.reset()
                return status
            self._stage_cache = status
