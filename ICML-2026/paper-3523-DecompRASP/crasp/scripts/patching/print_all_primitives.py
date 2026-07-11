from dataclasses import dataclass
import tyro
import torch
import glob
from pathlib import Path
import json
import yaml
from transformers import GPT2LMHeadModel
from collections import defaultdict
from torch.utils.data import DataLoader
import multiprocessing as mp
from typing import Optional
import os
import re
from itertools import product

from patching_utils import get_logging_function, set_seed, restore_int_keys
from find_primitives import try_find_primitives
from patching_data import get_tokenizer_and_dataset_for_task
from train_new_models import customCollator, customBCECollator
from pruning_model import PruningModelWithHooksForQK, MaskSamplerForQK
from convert_mlp import convert_mlp
from primitives_helpers import plot_and_save_primitives_matrices, get_num_dims, is_token_dim_activation, is_cartesian_act
from primitives_for_coefficients import LOGITS_CONST_ALL_PRIMITIVES, LOGITS_ALL_PRIMITIVES
from primitives_for_coefficients import ATTENTION_CONST_ALL_PRIMITIVES, ATTENTION_ALL_PRIMITIVES

save_path = "SAVE_PATH"
task = "ababstar"

def run():
    save_path.mkdir(exist_ok=True, parents=True)
    tokenizer, iterable_dataset = get_tokenizer_and_dataset_for_task(task, (0, 150), 150, {"period_for_data":3})
    ticks_x = [t for t in tokenizer.vocab]
    ticks_y = [t for t in tokenizer.vocab]
    dims_left, dims_right = len(tokenizer.vocab), len(tokenizer.vocab)
    for primitive in LOGITS_CONST_ALL_PRIMITIVES:
        primitive_matrix = primitive.contruct_matrix(dims_right, tokenizer)
        plot_and_save_primitives_matrices(primitive_matrix, save_path / "const_logits" / f"{primitive.name}.png",
                                          ticks_x=ticks_x, ticks_y=None)
    for primitive in LOGITS_ALL_PRIMITIVES:
        primitive_matrix = primitive.contruct_matrix(dims_left, dims_right, tokenizer)
        plot_and_save_primitives_matrices(primitive_matrix, save_path / "logits" / f"{primitive.name}.png",
                                          ticks_x=ticks_x, ticks_y=ticks_y)
    for primitive in ATTENTION_CONST_ALL_PRIMITIVES:
        primitive_matrix = primitive.contruct_matrix(dims_right, tokenizer)
        plot_and_save_primitives_matrices(primitive_matrix, save_path / "const_attention" / f"{primitive.name}.png",
                                          ticks_x=ticks_x, ticks_y=None)
    for primitive in ATTENTION_ALL_PRIMITIVES:
        primitive_matrix = primitive.contruct_matrix(dims_left, dims_right, tokenizer)
        plot_and_save_primitives_matrices(primitive_matrix, save_path / "attention" / f"{primitive.name}.png",
                                          ticks_x=ticks_x, ticks_y=ticks_y)


if __name__ == "__main__":
    run()