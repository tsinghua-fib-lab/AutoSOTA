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

from patching_utils import get_logging_function, set_seed, restore_int_keys
from find_primitives import try_find_primitives
from patching_data import get_tokenizer_and_dataset_for_task
from train_new_models import customCollator, customBCECollator
from pruning_model import PruningModelWithHooksForQK, MaskSamplerForQK
from convert_mlp import convert_mlp

from primitives_for_coefficients import ATTENTION_CONST_ALL_PRIMITIVES, ATTENTION_ALL_PRIMITIVES
from primitives_classes import AbstractPrimitive
from primitives_classes import AttentionInteraction, LogitsInteraction
from primitives_for_coefficients import LOGITS_CONST_ALL_PRIMITIVES, LOGITS_ALL_PRIMITIVES
from round_primitive import MatrixRounder

@dataclass
class Args:
    series_path: str = None
    device: str = "cuda:0"

BASE_PATH = "PATH_TO_EXPERIMENTS_FOLDER"
PATH_TO_GOOD_MODELS_FILE = "good_models.json"
PATH_TO_SAVED_MODELS = "PATH_TO_SAVED_MODELS"

import torch
import json
from functools import partial
import copy
import re
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Any
import torch.nn.functional as F

from logits_primitives_hook import lm_head_hook
from attention_primitives_hook import primitives_attention_forward


INSTANCE_PER_TASK_NAME = {
    "012_0_2": {
        "positive": {
            "instance": ['<bos>', '2', '1', '1', '0', '2', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', '0', '2', '2', '0', '0', '1', '0', '1', '<sep>', '0'],
        }
    },
    "bce_012_0_2": {
        "positive": {
            "instance": ['<bos>', '2', '1', '1', '0', '2', '<eos>'],
        },
    },
    "aaaastar": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'a', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', 'a', 'a', 'a', '<sep>', '0'],
        },
    },
    "bce_aaaastar": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'a', '<eos>'],
        },
    },
    "aastar": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'a', '<sep>', '1'],
            "label": ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '1'],
        },
        "negative": {
            "instance": ['<bos>', 'a', 'a', 'a', '<sep>', '0'],
            "label": ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '0'],
        },
    },
    "bce_aastar": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'a', '<eos>'],
        },
    },
    "ab_d_bc": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'b', 'a', 'd', 'b', 'c', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', 'b', 'b', 'c', 'd', 'a', 'c', 'a', '<sep>', '0'],
        },
    },
    "bce_ab_d_bc": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'b', 'a', 'd', 'b', 'c', '<eos>'],
        },
    },
    "ababstar": {
        "positive": {
            "instance":  ['<bos>', 'a', 'b', 'a', 'b', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', 'a', 'a', 'b', 'a', 'b', 'a', '<sep>', '0'],
        },
    },
    "bce_ababstar": {
        "positive": {
            "instance":  ['<bos>', 'a', 'b', 'a', 'b', '<eos>'],
        },
    },
    "abcde": {
        "positive": {
            "instance": ['<bos>', 'a', 'b', 'c', 'd', 'd', 'd', 'd', 'e', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', 'a', 'b', 'c', 'd', 'b', 'c', '<sep>', '0'],
        },
    },
    "bce_abcde": {
        "positive": {
            "instance": ['<bos>', 'a', 'b', 'c', 'd', 'd', 'd', 'd', 'e', '<eos>'],
        },
    },
    "bin_majority": {
        "positive": {
            "instance": ['<bos>', '0', '0', '0', '1', '0', '<sep>', '0'],
        },
    },
    "bin_majority_interleave": {
        "positive": {
            "instance": ['<bos>', '1', '0', '1', '1', '0', '1', '<sep>', '1', '0', '1', '<eos>'],
        },
    },
    "count": {
        "positive": {
            "instance": ['<bos>', '72', '75', '<sep>', '72', '73', '74', '75', '<eos>'],
        },
    },
    "D2": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'b', 'a', 'b', 'b', 'a', 'b', 'a', 'b', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', 'a', 'a', 'b', 'a', 'a', 'b', 'a', 'b', '<sep>', '0'],
        },
    },
    "bce_D2": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'b', 'a', 'b', 'b', 'a', 'b', 'a', 'b', '<eos>'],
        },
    },
    "D3": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'b', 'b', 'b', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', 'a', 'a', 'a', 'b', '<sep>', '0'],
        },
    },
    "bce_D3": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'b', 'b', 'b', '<eos>'],
        },
    },
    "D4": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', 'a', 'b', 'a', 'a', '<sep>', '0'],
        },
    },
    "bce_D4": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', '<eos>'],
        },
    },
    "D12": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', 'a', 'a', 'a', 'a', 'b', 'b', '<sep>', '0'],
        },
    },
    "bce_D12": {
        "positive": {
            "instance": ['<bos>', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', '<eos>'],
        },
    },
    "majority": {
        "positive": {
            "instance": ['<bos>', 'o', 'b', 'r', 'o', '<sep>', 'o'],
        },
    },
    "sort": {
        "positive": {
            "instance": ['<bos>', '66', '115', '81', '55', '136', '118', '<sep>', '55', '66', '81', '115', '118', '136', '<eos>'],
        },
    },
    "tomita1": {
        "positive": {
            "instance": ['<bos>', '1', '1', '1', '1', '1', '1', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', '0', '1', '0', '1', '<sep>', '0'],
        },
    },
    "bce_tomita1": {
        "positive": {
            "instance": ['<bos>', '1', '1', '1', '1', '1', '1', '<eos>'],
        },
    },
    "tomita2": {
        "positive": {
            "instance": ['<bos>', '1', '0', '1', '0', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', '0', '1', '1', '1', '<sep>', '0'],
        },
    },
    "bce_tomita2": {
        "positive": {
            "instance": ['<bos>', '1', '0', '1', '0', '<eos>'],
        },
    },
    "tomita4": {
        "positive": {
            "instance": ['<bos>', '1', '1', '0', '1', '1', '0', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', '1', '1', '0', '0', '0', '1', '1', '<sep>', '0'],
        },
    },
    "bce_tomita4": {
        "positive": {
            "instance": ['<bos>', '1', '1', '0', '1', '1', '0', '<eos>'],
        },
    },
    "tomita7": {
        "positive": {
            "instance": ['<bos>', '0', '0', '1', '0', '0', '0', '1', '1', '<sep>', '1'],
        },
        "negative": {
            "instance": ['<bos>', '1', '0', '1', '1', '0', '1', '1', '<sep>', '0'],
        },
    },
    "bce_tomita7": {
        "positive": {
            "instance": ['<bos>', '0', '0', '1', '0', '0', '0', '1', '1', '<eos>'],
        },
    },
    "unique_bigram_copy": {
        "positive": {
            "instance": ['<bos>', '6', '0', '9', '12', '7', '<sep>', '6', '0', '9', '12', '7', '<eos>'],
        },
    },
    "unique_copy": {
        "positive": {
            "instance": ['<bos>', '145', '45', '101', '76', '<sep>', '145', '45', '101', '76', '<eos>'],
        },
    },
    "unique_reverse": {
        "positive": {
            "instance": ['<bos>', '75', '73', '105', '135', '<sep>', '135', '105', '73', '75', '<eos>'],
        },
    },
}

def get_primitives_in_classes(primitives):
    primitives_to_try = {}
        
    for layer in primitives:
        if layer == "lm_head":
            continue
        primitives_to_try[int(layer)] = {}
        for head in primitives[layer]:
            primitives_to_try[int(layer)][int(head)] = {}
            for interaction in primitives[layer][head]["qk_interactions"] + primitives[layer][head]["k_interactions"]:
                interaction_class = AttentionInteraction(
                    activation_name_to_keep_q=interaction["activation_name_to_keep_q"],
                    activation_name_to_keep_k=interaction["activation_name_to_keep_k"]
                )
                if interaction["primitive"] is not None:
                    if "predefined_primitive" in interaction:
                        attention_primitives_set = ATTENTION_ALL_PRIMITIVES
                        if interaction_class.activation_name_to_keep_q is None:
                            attention_primitives_set = ATTENTION_CONST_ALL_PRIMITIVES
                        predefined_primitive = [p for p in attention_primitives_set if p.name == interaction["predefined_primitive"]]
                        assert len(predefined_primitive) == 1
                        predefined_primitive = predefined_primitive[0]
                        if "predefined_special_primitive" in interaction:
                            predefined_special_primitive = [p for p in attention_primitives_set if p.name == interaction["predefined_special_primitive"]]
                            assert len(predefined_special_primitive) == 1
                            predefined_special_primitive = predefined_special_primitive[0]
                        else:
                            predefined_special_primitive = None

                        scaling_factor = interaction["scaling_factor_primitive"]

                        primitive_class = AbstractPrimitive(
                            name="predefined_primitive",
                            primitive=predefined_primitive,
                            special_primitive=predefined_special_primitive,
                            scaling_factor_primitive=scaling_factor,
                        )
                    if "replacement_matrix" in interaction:
                        rounder = MatrixRounder(torch.tensor(interaction["replacement_matrix"]))
                        primitive_class = AbstractPrimitive(
                            name="replacement_matrix",
                            replacement_matrix=rounder,
                        )
                    primitives_to_try[int(layer)][int(head)][interaction_class] = primitive_class
    primitives_to_try["lm_head"] = {}
    for interaction in primitives["lm_head"]:
        interaction_class = LogitsInteraction(
            activation_name_to_keep=interaction["activation_name_to_keep"],
        )
        if interaction["primitive"] is not None:
            if "predefined_primitive" in interaction:
                logits_primitives_set = LOGITS_ALL_PRIMITIVES
                if interaction_class.activation_name_to_keep == "vocab_bias":
                    logits_primitives_set = LOGITS_CONST_ALL_PRIMITIVES
                predefined_primitive = [p for p in logits_primitives_set if p.name == interaction["predefined_primitive"]]
                assert len(predefined_primitive) == 1
                predefined_primitive = predefined_primitive[0]
                if "predefined_special_primitive" in interaction:
                    predefined_special_primitive = [p for p in logits_primitives_set if p.name == interaction["predefined_special_primitive"]]
                    assert len(predefined_special_primitive) == 1
                    predefined_special_primitive = predefined_special_primitive[0]
                else:
                    predefined_special_primitive = None

                scaling_factor = interaction["scaling_factor_primitive"]

                primitive_class = AbstractPrimitive(
                    name="predefined_primitive",
                    primitive=predefined_primitive,
                    special_primitive=predefined_special_primitive,
                    scaling_factor_primitive=scaling_factor,
                )
            if "replacement_matrix" in interaction:
                rounder = MatrixRounder(torch.tensor(interaction["replacement_matrix"]))
                primitive_class = AbstractPrimitive(
                    name="replacement_matrix",
                    replacement_matrix=rounder,
                )
            primitives_to_try["lm_head"][interaction_class] = primitive_class
    return primitives_to_try

def save_wte_inputs(module, input, output, model):
    model.wte_inputs = input[0].detach()
def save_wpe_inputs(module, input, output, model):
    model.wpe_inputs = input[0].detach()

def run_on_one_example(hooked_model, primitives, mask_sampler, oa_vecs, tokenizer, converted_mlp,
                       inp_ids, pos_ids):
    hooks = []
    hooks.append(hooked_model.model.transformer.wte.register_forward_hook(partial(save_wte_inputs, model=hooked_model)))
    hooks.append(hooked_model.model.transformer.wpe.register_forward_hook(partial(save_wpe_inputs, model=hooked_model)))
    prev_attn_forward = hooked_model.attention_forward
    hooked_model.attention_forward = lambda module, layer: primitives_attention_forward(hooked_model, module, layer, primitives=primitives, tokenizer=tokenizer, converted_mlp=converted_mlp, oa_vecs=oa_vecs)
    hooks_to_remove = [h for h in hooked_model.hooks if h.id in h.__getstate__()[0] and h.__getstate__()[0][h.id] == hooked_model.lm_head_hook]
    assert len(hooks_to_remove) == 1, hooks_to_remove
    hooks_to_remove[0].remove()
    hooks.append(hooked_model.model.lm_head.register_forward_hook(partial(
        lm_head_hook,
        hooked_model=hooked_model, primitives=primitives, tokenizer=tokenizer, converted_mlp=converted_mlp,
        oa_vecs=oa_vecs
    )))
    with torch.no_grad():
        batch = {
                "input_ids": torch.tensor(inp_ids).unsqueeze(0),
                "position_ids": torch.tensor(pos_ids).unsqueeze(0),
        }
        batch = {k: v.to(hooked_model.device) for k, v in batch.items()}

        masks = mask_sampler.sample_binary_masks(1)
        hooked_model.input_ids = batch["input_ids"]
        hooked_model.position_ids = batch["position_ids"]
        
        result = hooked_model(masks=masks, oa_vecs=oa_vecs, **batch)
    
    hooked_model.attention_forward = prev_attn_forward
    for h in hooks:
        h.remove()
    hooked_model.hooks.append(hooked_model.model.lm_head.register_forward_hook(
        hooked_model.lm_head_hook
    ))

def run(args):
    device = torch.device(args.device)

    set_seed(0)
    torch.set_printoptions(sci_mode=False)
    torch.set_default_device(device)

    assert args.series_path is not None

    paths_to_run = []
    with open(PATH_TO_GOOD_MODELS_FILE, "r") as file:
        good_models = json.load(file)["chosen_exps"]
    series_path = BASE_PATH / Path(args.series_path)
    output_path = series_path / good_models[args.series_path] / "output.json"
    if output_path.exists():
        with open(output_path) as f:
            output_dict = json.load(f)
        if "result_patching_config_global_iteration_2" in output_dict:
            paths_to_run.append(output_path.parent)
    print("all path")
    print(paths_to_run)

    for path in paths_to_run:
        print("current path", path)

        with open(path / "config.yaml") as f:
            model_name = yaml.safe_load(f)["model_name"]
        task_name = model_name.split("-")[0]

        logger = get_logging_function(path, "running_with_example.txt")

        orig_model = GPT2LMHeadModel.from_pretrained(f"{PATH_TO_SAVED_MODELS}/{model_name}").to(device)
        orig_model.eval()

        model = GPT2LMHeadModel.from_pretrained(f"{PATH_TO_SAVED_MODELS}/{model_name}").to(device)
        model.eval()

        tokenizer, iterable_dataset = get_tokenizer_and_dataset_for_task(task_name, (0, 150), 150, {"period_for_data":3})

        with open(path / "output.json", "r") as file:
            logs = json.load(file)
        assert "result_patching_config_global_iteration_2" in logs
        is_config_empty = True
        loaded_config = logs["result_patching_config_global_iteration_2"]
        current_config = {}
        for layer in range(len(model.transformer.h)):
            current_config[layer] = {}
            for tp in ["v", "k", "qk"]:
                current_config[layer][tp] = {}
                for head in range(model.transformer.h[layer].attn.num_heads):
                    current_config[layer][tp][head] = loaded_config[str(layer)][tp][str(head)]
                    if len(current_config[layer][tp][head]) > 0:
                        is_config_empty = False
                    if tp == "qk":
                        current_config[layer][tp][head] = list(map(tuple, current_config[layer][tp][head]))
                        if len(current_config[layer][tp][head]) > 0:
                            is_config_empty = False
            current_config[layer]["mlp"] = loaded_config[str(layer)]["mlp"]
            if len(current_config[layer]["mlp"]) > 0:
                is_config_empty = False
        current_config["lm_head"] = loaded_config["lm_head"]
        if len(current_config["lm_head"]) > 0:
            is_config_empty = False
        if is_config_empty:
            logger(f"{path}: config is empty, skipping")
            continue


        logger("running a model with config: ", current_config)
        oa_vecs = torch.load(path / "oa_vecs.pt", weights_only=False, map_location=device)
        oa_vecs.requires_grad_(False)
        assert (path  / "converted_mlp.pt").exists()
        converted_mlp = torch.load(path  / "converted_mlp.pt", weights_only=False, map_location=device)
        with open(path / "attention_primitives" / "configs" / f"{task_name}.json", "r") as file:
            primitives = get_primitives_in_classes(json.load(file)["primitives"])
        
        with open(path / "output.json", "r") as file:
            output_dict = json.load(file)

        hooked_model = PruningModelWithHooksForQK(model, current_config, defaultdict(lambda : 0), hasattr(oa_vecs, "MLPs"), logger)
            # hooked_model.set_mask_sampler(mask_sampler)
        mask_sampler = MaskSamplerForQK(current_config).to(hooked_model.device)
        for posneg in ["positive", "negative"]:
            if posneg in INSTANCE_PER_TASK_NAME[task_name]:
                inp_ids = torch.tensor([tokenizer.vocab[i] for i in INSTANCE_PER_TASK_NAME[task_name][posneg]["instance"]])
                pos_ids = torch.arange(len(INSTANCE_PER_TASK_NAME[task_name][posneg]["instance"]))
                hooked_model.model_name = task_name
                hooked_model.save_matrices = True
                hooked_model.saved_matrices = set()
                hooked_model.save_matrices_path = path / "attention_primitives" / "heatmaps" / f"pretty_examples_{posneg}"
                run_on_one_example(hooked_model, primitives, mask_sampler, oa_vecs, tokenizer, converted_mlp,
                            inp_ids, pos_ids)

if __name__ == "__main__":
    args = tyro.cli(Args)
    run(args)

