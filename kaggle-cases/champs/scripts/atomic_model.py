"""Load model definitions from the runner-up's original notebook.

Only model definition cells are executed. FastAI's three small layer helpers
are supplied locally, so its obsolete training dependencies are unnecessary.
Architecture and state-dict names are retained for strict checkpoint loading.
"""
import json
import math
from enum import Enum
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

ROOT = Path(__file__).resolve().parents[1]
TYPES = ["1JHC", "2JHH", "1JHN", "2JHN", "2JHC", "3JHH", "3JHC", "3JHN"]


class Module(nn.Module):
    """FastAI v1 initializes nn.Module before subclass __init__."""
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        nn.Module.__init__(obj)
        return obj


class SequentialEx(Module):
    def __init__(self, *layers):
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        result = x
        for layer in self.layers:
            result.orig = x
            new_result = layer(result)
            result.orig = None
            result = new_result
        return result


class MergeLayer(Module):
    def __init__(self, dense=False):
        self.dense = dense

    def forward(self, x):
        return torch.cat([x, x.orig], dim=1) if self.dense else x + x.orig


def ifnone(value, fallback):
    return fallback if value is None else value


def definitions():
    notebook = ROOT / "repos/champs_reference/solutions/2/transformer-experiments.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    namespace = dict(globals(), types=TYPES, atoms="CFHNO", n_types=8, max_atoms=29)
    for index in (33, 34, 35):
        exec(compile("".join(cells[index]["source"]), str(notebook) + "#cell%d" % index, "exec"), namespace)
    # The original hard-coded .cuda() assumes the default GPU.
    def swish_forward(self, x):
        return self.slope.to(x.device) * x * torch.sigmoid(x)
    namespace["LearnedSwish"].forward = swish_forward
    return namespace


def build_model(state=None, d_model=1024, n_layers=6, n_heads=32, d_inner=4096):
    if state is not None:
        d_model = state["atom_embedding.weight"].shape[1] * 2
        n_layers = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("transformer.layers."))
        d_inner = state["transformer.layers.0.ff.layers.0.weight"].shape[0]
    namespace = definitions()
    model = namespace["AtomTransformer"](
        n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_inner=d_inner,
        act=namespace["Activation"].LearnedSwish, attn_cls=namespace["MultiHeadAttention3"])
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model
