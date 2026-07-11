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

@dataclass
class Args:
    exp_path: str = None    # path excluding or after patching_runs
    series_path: str = None  # path excluding or after patching_runs
    run_preconfiged: Optional[str] = None
    device: str = "cuda:0"

def load_preconfig(yaml_path: str = "preconfig.yaml"):
    """Load preconfig from YAML file and convert to Args objects."""
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    preconfig_list = []
    for config in config_data['preconfig']:
        args = Args(
            series_path=config['series_path'],
            device=config['device']
        )
        preconfig_list.append(args)
    
    return preconfig_list

PRIMITIVES_SEARCH_TYPE = "greedy_search_then_round"
HYPERPARAMETERS_PRIMITIVES = None # means we take the default ones

BASE_PATH = "PATH_TO_EXPERIMENTS"

PATH_TO_GOOD_MODELS_FILE = "good_models.json"
PATH_TO_SAVED_MODELS = "PATH_TO_SAVED_MODELS"

def run(args):
    device = torch.device(args.device)

    set_seed(0)
    torch.set_printoptions(sci_mode=False)
    torch.set_default_device(device)

    assert args.exp_path is not None or args.series_path is not None

    paths_to_run = []
    if args.exp_path is not None:
        output_path = BASE_PATH / Path(args.exp_path) / "output.json"
        assert output_path.exists()
        with open(output_path) as f:
            output_dict = json.load(f)
        assert "result_patching_config_global_iteration_2" in output_dict
        paths_to_run = [output_path.parent]
    else:
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

        num_test_step = 200
        batch_size = 120

        logger = get_logging_function(path, "primitives_logs.txt")

        orig_model = GPT2LMHeadModel.from_pretrained(f"{PATH_TO_SAVED_MODELS}/{model_name}").to(device)
        orig_model.eval()

        model = GPT2LMHeadModel.from_pretrained(f"{PATH_TO_SAVED_MODELS}/{model_name}").to(device)
        model.eval()

        tokenizer, iterable_dataset = get_tokenizer_and_dataset_for_task(task_name, (0, 150), 150, {"period_for_data":3})
        if not hasattr(iterable_dataset, "BCE") or not iterable_dataset.BCE:
            collator = customCollator(tokenizer.pad_token_id)
        else:
            collator = customBCECollator(tokenizer.pad_token_id)

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
        if (path  / "converted_mlp.pt").exists():
            converted_mlp = torch.load(path  / "converted_mlp.pt", weights_only=False, map_location=device)
        else:
            hooked_model.logger("Converting MLPs...")
            hooked_model = PruningModelWithHooksForQK(model, current_config, defaultdict(lambda : 0), hasattr(oa_vecs, "MLPs"), logger)
            dataloader = DataLoader(iterable_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
            converted_mlp = convert_mlp(hooked_model, oa_vecs, orig_model, dataloader, logger)
            hooked_model.logger("Saving converted MLPs...")
            torch.save(converted_mlp, path  / "converted_mlp.pt")
        with open(path / "output.json", "r") as file:
            output_dict = json.load(file)

        hooked_model = PruningModelWithHooksForQK(model, current_config, defaultdict(lambda : 0), hasattr(oa_vecs, "MLPs"), logger)
            # hooked_model.set_mask_sampler(mask_sampler)
        mask_sampler = MaskSamplerForQK(current_config).to(hooked_model.device)
        try_find_primitives(hooked_model, orig_model, iterable_dataset, batch_size,
                num_test_step, collator, mask_sampler, oa_vecs,
                tokenizer, converted_mlp, task_name, path / "attention_primitives",
                search_method=PRIMITIVES_SEARCH_TYPE,
                hyperparameters=HYPERPARAMETERS_PRIMITIVES)

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.run_preconfiged is not None:
        # Load preconfig from YAML file
        preconfig = load_preconfig(args.run_preconfiged)
        
        # Run each config in a separate process
        processes = []
        for config in preconfig:
            process = mp.Process(target=run, args=(config,))
            processes.append(process)
            process.start()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
    else:
        run(args)