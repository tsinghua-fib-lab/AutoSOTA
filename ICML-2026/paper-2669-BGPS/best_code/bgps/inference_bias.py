from typing import Dict, List, Tuple
import os
import time
import pprint
import csv
import cv2
import numpy as np
import torch
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.save_utils import save_predict
from utils.config_utils import prepare_config
from utils.wandb_utils import set_wandb
from utils.random_utils import set_seed
from utils.print_utils import time_log
from utils.param_utils import count_params
from wrapper_bias import BGPS
import re

torch.backends.cudnn.benchmark = False

def setup_gpu(gpu_id):
    """Properly set up GPU isolation and device handling"""
    if gpu_id is not None:
        # Set CUDA_VISIBLE_DEVICES to isolate this process to one GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Now device 0 will be the GPU we want
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            # Clear any existing CUDA cache
            torch.cuda.empty_cache()
        else:
            print(f"Warning: CUDA not available, using CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    # if device.type == 'cuda':
    #     print(f"GPU: {torch.cuda.get_device_name(device)}")
    #     print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    #     print(f"Memory cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    return device

def safe_name(s: str, maxlen: int = 80) -> str:
        # filesystem-friendly folder name from prompt text
        s = re.sub(r'[^A-Za-z0-9._-]+', '_', s.strip())
        return s[:maxlen] or "prompt"

def inference_epoch(
        model: BGPS,
        cfg: Dict,
        sampling_generator: torch.Generator = None,
        num_run: int = 0
) -> List:
    
    model.eval()
    torch.set_grad_enabled(False)
    result = []
    model.reset()

    # --------------------------- run classifier-guided inference --------------------------- #
    results = model.inference(
        num_images_per_prompt=cfg["model"]["sampling"]["n_samples"],
        num_inference_steps=cfg["model"]["sampling"]["steps"],
        guidance_scale=cfg["model"]["sampling"]["scale"],
        skip_inference=cfg["skip_inference"],
        sampling_generator=sampling_generator
    )

    run_seed=cfg["seed"]* cfg["model"].get("num_validation_runs", 1) + num_run
    print(f"Run seed: {run_seed}")
    

    s = f"... samples: {int(cfg['batch_size'])} (valid done: {100:.2f} %)"
    print(s)

    # --------------------------- saving --------------------------- #
    if not os.path.exists(cfg["save_dir"]):
        os.makedirs(cfg["save_dir"], exist_ok=True)

    if cfg["use_occupation_template"]:
        prompt_dir_name = safe_name(cfg["model"]["model_occupation_template"])
    else:
        prompt_dir_name = safe_name(cfg["model"]["model_prompt_primer"])

    if cfg["skip_inference"] or (cfg["llm_only"] and not (cfg["biased_gender"] or cfg["biased_race"])):
        bias_attr = "no_bias_attribute"
        bias_target = ""
        attribute_dir_name = "no_bias_attribute"
    else:
        if cfg["bias_attribute2"] is None:
            bias_attr = cfg["bias_attribute"]
            bias_target = str(cfg["attributes"][cfg["bias_attribute"]]["target"])
            attribute_dir_name = cfg["bias_attribute"]+ "_" + bias_target
        else:
            bias_attr = cfg["bias_attribute"] + "_" + cfg["bias_attribute2"]
            bias_target = str(cfg["attributes"][cfg["bias_attribute"]]["target"]) + "_" + str(cfg["attributes"][cfg["bias_attribute2"]]["target"])
            attribute_dir_name = cfg["bias_attribute"]+"_"+str(cfg["attributes"][cfg["bias_attribute"]]["target"])+ \
                                    "_"+cfg["bias_attribute2"]+"_"+str(cfg["attributes"][cfg["bias_attribute2"]]["target"])

    if not os.path.exists(os.path.join(cfg["save_dir"],prompt_dir_name)):
        os.makedirs(os.path.join(cfg["save_dir"],prompt_dir_name), exist_ok=True)
    
    csv_path = os.path.join(cfg["save_dir"], cfg["save_file"])
    file_exists = os.path.isfile(csv_path)

    for output in results:

        # CSV (always write prompt, seed, bias score)
        with open(csv_path, "a" if file_exists else "w", newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            if not file_exists:
                # writer.writerow(['generated_prompt', 'attribute', 'target' , 'seed', 'bias_score'])
                writer.writerow(['generated_prompt'])
                file_exists = True
            writer.writerow([output["prompt"]])
            # writer.writerow([output["prompt"], bias_attr, bias_target, output["seed"], output["similarity"]])

        # Console
        print(f"...bias_score: {output['similarity']}\n")

        # directory for this run (use prompt as folder name)
        org_filename = safe_name(output["prompt"])
        out_dir = os.path.join(cfg["save_dir"],prompt_dir_name,attribute_dir_name,"seed_"+str(run_seed))
        os.makedirs(out_dir, exist_ok=True)

        # save images (if generated)
        if not cfg["model"]["gen_prompt_only"] and output["image"] is not None:
            for idx, image in enumerate(output["image"]):
                filename = org_filename+f'_seed_{run_seed}_{idx}.png'
                cv2.imwrite(os.path.join(out_dir, filename),
                            cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        # save JSON sidecar
        payload = {
            "generated_prompt": output["prompt"],
            "llm_input": output["input_prompt"],
            "alt_prompts": output["alt_prompts"],
            "bias_score": output["similarity"],
            "clf_alpha": cfg["model"]["clf_alpha"],
            "llm_alpha": cfg["model"]["llm_alpha"],
            "sd_batch_size": cfg["model"]["sd_batch_size"],
            "clf2_alpha": cfg["model"]["clf2_alpha"],
            "bias_attribute": bias_attr,
            "bias_target": bias_target,
            "seed": output["seed"]
        }
        # include initial_condition if present
        if "initial_condition" in output and output["initial_condition"] is not None:
            payload["initial_condition"] = output["initial_condition"]

        save_predict(payload, os.path.join(out_dir, f"seed_{run_seed}.json"))
        result.append(payload)
        #save cfg file to dir
        with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f)
        # save beam history (if present)
        # if beam_history is not None:
        #     #save as is
        #     with open(os.path.join(out_dir, f"seed_{output['seed']}_beam_history.json"), 'w') as f:
        #         #formatted
        #         json.dump(beam_history, f, indent=4)

        csv_path_seed = os.path.join(out_dir, cfg["save_file"])
        file_exists = os.path.isfile(csv_path_seed)
        # CSV (always write prompt, seed, bias score)
        with open(csv_path_seed, "a" if file_exists else "w", newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            if not file_exists:
                writer.writerow(['generated_prompt', 'attribute', 'target' , 'seed', 'bias_score'])
                file_exists = True
            writer.writerow([output["prompt"], bias_attr, bias_target, run_seed, output["similarity"]])

        attr_csv_path = os.path.join(cfg["save_dir"], f"{bias_attr}_{bias_target}.csv")
        attr_file_exists = os.path.isfile(attr_csv_path)

        with open(attr_csv_path, "a" if attr_file_exists else "w", newline='') as attr_file:
            attr_writer = csv.writer(attr_file, quoting=csv.QUOTE_NONNUMERIC)
            if not attr_file_exists:
                attr_writer.writerow(['generated_prompt'])
            attr_writer.writerow([output["prompt"]])

        if cfg["model"]["create_eval_set"]:

            eval_dir = os.path.join(cfg["save_dir"],prompt_dir_name,attribute_dir_name,"seed_"+str(run_seed),"eval_set")
            model.create_eval_set(output["prompt"],cfg["model"]["eval_set_size"], eval_dir, num_run=num_run)

    return results

def run(cfg: Dict, debug: bool = False) -> None:
    # ======================================================================================== #
    # Initialize
    # ======================================================================================== #
    # device, local_rank = set_dist(device_type="cuda")
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # gpu_id = cfg.get('gpu', None)
    # device = setup_gpu(gpu_id)
    pprint.pprint(cfg)  # print config to check if all arguments are correctly given.

    _ = set_wandb(cfg, force_mode="disabled" if debug else None)
    set_seed(seed=cfg["seed"])

    # ======================================================================================== #
    # Model
    # ======================================================================================== #
    model = BGPS(cfg)
    model = model.to(device)
   
    # print(model)
    p1, p2 = count_params(model.parameters())
    print(f"Model parameters: {p1} tensors, {p2} elements.")

    # ======================================================================================== #
    # Evaluation
    # ======================================================================================== #
    sampling_generator = torch.Generator(device=device)
    sampling_generator.manual_seed(cfg["seed"])
    num_runs = cfg["model"].get("num_validation_runs", 1)
    for num_run in range(num_runs):
        s = time_log()
        s += f"Start validation"
        print(s)
        inference_start_time = time.time()  # second
        result = inference_epoch(model, cfg, sampling_generator=sampling_generator, num_run=num_run)
        inference_time = time.time() - inference_start_time

        s = time_log()
        s += f"End validation, time: {inference_time:.3f} s\n"

    

    print(s)


if __name__ == '__main__':
    args, config = prepare_config()
    run(config, args.debug)
