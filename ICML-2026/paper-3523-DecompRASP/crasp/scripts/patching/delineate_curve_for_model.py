from streamlit_app.my_paths import SUBMIT_HOST, CONDA_PATH, CONDA_ENV, REMOTE_SCRIPT_PATH, output_dir, HISTORY_FILE, RUNNING_EXPERIMENTS_FILE, path_to_saved_model, CACHE_DIR, USER_NAME, SUBMIT_REQUIREMENTS, UNIVERSE
import json
import os
import yaml
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time
import re
import random
import string
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys
import math
import click
from transformers import GPT2LMHeadModel
from patching_data import get_tokenizer_and_dataset_for_task
from pruning_model import PruningModelWithHooksForQK
from train_new_models import customCollator, customBCECollator
from collections import defaultdict
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from patching_utils import set_seed
import glob

base_config = {'batch_size_for_pruning': 120, 'device': 'cuda','find_graph_method': 'pruning', 'length_range': [0, 150], 'num_iterations': 100, 'num_repeat_for_pruning': 12, 'path_to_saved_model': path_to_saved_model, 'period_for_data': 3, 'prune_inputs_to_mlps_and_lm_head': True, 'seed': 0, 
               'training_steps0_for_pruning': 1000, 'training_steps1_for_pruning': 500, 'training_steps2_for_pruning': 500, 'split_mlps': True, 'train_new_attn_heads': False, 'find_attention_primitives': False, 'find_logits_primitives': False, 'find_primitives': False,
               'lr_LN_var_for_pruning': 0.1, 'lr_MLP_for_pruning': 0.001, 'lr_oa_for_pruning': 0.002, 'lr_sampler_for_pruning': 0.1, 'max_test_length': 150,}

cached_results = {'unique_reverse-%4l1h256d4lr01drop': (0.031948638029396535, 4.6875e-05), 'sort-@1l1h256d3lr01drop': (0.007941538429819048, 0.005921875), 'count-@1l4h256d4lr01drop': (0.0008619880203041248, 0.007203125), 'unique_copy-@2l1h64d3lr01drop': (0.03247358152829111, 1.5625e-05), 'bin_majority_interleave-@2l2h16d3lr00drop': (0.16622770835459233, 0.0), 'unique_bigram_copy-@2l4h256d3lr01drop': (0.1388480603992939, 0.0), 'unique_reverse-@2l1h64d3lr01drop': (0.032451047256588934, 3.125e-05), 'unique_copy-#4l4h256d4lr00drop': (0.010744351204717532, 0.0), 'unique_bigram_copy-#4l2h64d3lr00drop': (0.11686349717900157, 0.0004375), 'bin_majority_interleave-#2l2h64d4lr00drop': (0.15490948574244975, 0.072140625), 'bin_majority_interleave-%4l1h64d3lr01drop': (0.1656946315690875, 3.125e-05), 'count-#1l4h256d3lr01drop': (0.002105997174512595, 0.00025), 'count-%2l4h256d4lr01drop': (0.0008367605020757764, 0.00790625), 'unique_reverse-#2l4h256d4lr01drop': (0.030901854797266422, 6.25e-05), 'repeat_copy-@4l4h256d3lr00drop': (0.11505422690138221, 0.0), 'unique_copy-^2l1h64d3lr01drop-1100': (0.0038154607221949845, 3.125e-05), 'unique_copy-^2l1h64d3lr01drop-1200': (0.021356539892032744, 6.25e-05), 'unique_copy-^2l1h64d3lr01drop-1300': (0.029093256490305065, 0.0), 'unique_copy-^2l1h64d3lr01drop-1400': (0.03073668005876243, 1.5625e-05), 'unique_bigram_copy-^2l4h256d3lr01drop-500': (0.09800113148614764, 0.0), 'unique_bigram_copy-^2l4h256d3lr01drop-700': (0.11754926513507963, 3.125e-05), 'unique_bigram_copy-^2l4h256d3lr01drop-1400': (0.13134995344281197, 3.125e-05), 'unique_bigram_copy-^2l4h256d3lr01drop-1900': (0.13457509446889163, 0.0), 'unique_bigram_copy-^2l4h256d3lr01drop-2100': (0.13698141080886125, 0.0), 'unique_bigram_copy-^2l4h256d3lr01drop-2200': (0.13764136973023414, 0.0), 'unique_bigram_copy-^2l4h256d3lr01drop-3300': (0.13870336763560773, 0.0), 'addition-@4l2h64d3lr00drop': (0.12292155916243791, 1.5625e-05),}
cached_results.update({'D4-@4l1h64d3lr00drop': (0.08694661417230963, 0.5059375), 'aastar-@1l1h256d4lr01drop': (0.09944417619705201, 0.497328125), 'ababstar-@1l2h64d3lr01drop': (0.08689756329357624, 0.50103125), 'ab_d_bc-@2l1h16d3lr00drop': (0.06983417865261435, 0.4976875), '012_0_2-@2l4h64d3lr00drop': (0.0993219968341291, 0.494828125), 'tomita3-@2l2h256d4lr00drop': (0.09786376241594553, 0.5516875), 'tomita1-@1l1h16d3lr00drop': (0.11586838148161768, 0.498265625), 'tomita2-@1l2h256d4lr01drop': (0.11600471433997155, 0.499734375), 'tomita7-@4l1h16d3lr01drop': (0.11221353425458074, 0.50109375), 'D2-@2l2h64d3lr00drop': (0.08726279286667704, 0.4973125), 'tomita6-@4l4h256d4lr00drop': (0.08190702863968909, 0.426078125), 'tomita5-@2l4h256d4lr00drop': (0.04615162647422403, 0.5418125), 'aaaastar-@1l1h256d4lr01drop': (0.09944417619705201, 0.497328125), 'D3-@2l1h64d4lr00drop': (0.08710023204982281, 0.50525), 'abcde-@2l2h16d3lr00drop': (0.06143081364221871, 0.499578125), 'D12-@2l1h16d3lr00drop': (0.0834556778036058, 0.50740625), 'tomita4-@4l2h256d4lr01drop': (0.11504682051017881, 0.498125)})
cached_results.update({'D2-^2l2h64d3lr00drop-100': (0.00015466447974904441, 1.0), 'D2-^2l2h64d3lr00drop-7600': (0.058574771478772165, 0.599390625), 'aaaastar-%4l4h256d3lr01drop': (0.09953224632889032, 0.497328125), 'tomita2-#1l4h256d4lr01drop': (0.1097471906915307, 0.646046875), 'D2-^2l2h64d3lr00drop-4600': (0.018209616938373075, 0.260609375), 'ab_d_bc-%4l4h16d3lr01drop': (0.06937344709038734, 0.49709375), 'aaaastar-#1l1h256d3lr01drop': (0.03764293195167556, 0.7844375), 'tomita2-%2l1h64d3lr00drop': (0.11585032936558128, 0.50146875)})
cached_results.update({'parity-@4l4h256d4lr00drop': (0.05804645420890302, 0.305625), 'majority-@4l4h256d3lr00drop': (0.10801247358694673, 0.038484375), 'bin_majority-@1l1h16d3lr01drop': (0.10789108648151159, 0.499453125), })
cached_results.update({'bce_tomita2-@1l1h16d3lr00drop': (0.007244067153660581, 1.0), 'bce_tomita7-@2l1h16d3lr01drop': (0.06287697586975992, 0.335828125), 'bce_tomita5-@2l1h64d3lr00drop': (0.10912917731329799, 0.0), 'bce_tomita6-@2l2h256d4lr00drop': (0.1150262157022953, 0.0), 'bce_D12-@4l2h256d3lr00drop': (0.055119195733219384, 0.0), 'bce_tomita1-@1l1h16d3lr00drop': (0.008153964128810913, 1.0), 'bce_ababstar-@1l1h64d3lr00drop': (0.06268642069585621, 0.0), 'bce_D3-@1l1h16d4lr00drop': (0.2066849657520652, 0.0), 'bce_D2-@1l1h16d3lr00drop': (0.17839485596865415, 0.026875), 'bce_abcde-@1l1h16d3lr00drop': (0.010599749504588545, 1.0), 'bce_012_0_2-@1l1h64d4lr00drop': (0.0424677674267441, 0.143015625), 'bce_aastar-@1l2h16d3lr01drop': (0.14345939234644176, 0.0), 'bce_tomita4-@2l2h16d3lr01drop': (0.05230267773009837, 0.034), 'bce_D4-@1l2h256d4lr00drop': (0.13472282557934523, 0.0), 'bce_ab_d_bc-@1l1h16d3lr01drop': (0.17566769814118743, 0.10990625), 'bce_tomita3-@4l4h64d3lr00drop': (0.06696410210616886, 0.455703125)})
cached_results.update({'bce_aaaastar-@2l1h64d4lr01drop': (0.1191895402111113, 0.008015625), 'majority-@1l4h256d3lr01drop': (0.10591594930365682, 0.040515625)})
def run_ssh_command(command):
    """Run a command on the submit machine via SSH with conda environment"""
    # Escape any single quotes in the command
    escaped_command = command.replace("'", "'\\''")
    ssh_command = f"ssh {SUBMIT_HOST} '{escaped_command}'"
    return subprocess.run(ssh_command, shell=True, capture_output=True, text=True)

def run_command(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True)

def create_condor_submit(config_path, log_dir):
        submit_content = f"""{UNIVERSE}
executable = /bin/bash
arguments = "-c 'source {CONDA_PATH} {CONDA_ENV} && python {REMOTE_SCRIPT_PATH} --config_path {config_path}'"
output = {log_dir}/$(Cluster).out
error = {log_dir}/$(Cluster).err
log = {log_dir}/$(Cluster).log
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
request_cpus = 4
request_memory = 16GB
request_gpus = 1
requirements = {SUBMIT_REQUIREMENTS}
queue 1""" 
        return submit_content

class BaselineModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.unigram = nn.Parameter(torch.randn(vocab_size, vocab_size))
    
    def forward(self, input_ids):
        return self.unigram[input_ids]

def get_baseline_loss_acc(config):
    if config["model_name"] in cached_results:
        print("use cached values")
        return cached_results[config["model_name"]]
    
    device = torch.device("cuda") if config["device"] == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    set_seed(config["seed"])

    task_name = config["model_name"].split("-")[0]
    tokenizer, dataset = get_tokenizer_and_dataset_for_task(task_name, config["length_range"], config["max_test_length"], {"period_for_data":config["period_for_data"]})
    use_BCE = getattr(dataset, "BCE", False)

    if not use_BCE:
        collator = customCollator(tokenizer.pad_token_id)
    else:
        collator = customBCECollator(tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)

    baseline = BaselineModel(len(tokenizer)).to(device)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=3e-3)

    orig_model = GPT2LMHeadModel.from_pretrained(Path(config["path_to_saved_model"]) / config["model_name"]).to(device)
    orig_model.eval()

    kl_div = 0
    for current_step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        with torch.no_grad():
            target_logits = orig_model(**batch).logits

        logits = baseline(batch["input_ids"])

        if not use_BCE:
            target_logits = target_logits[:, :-1][labels[:, 1:]!=-100]
            logits = logits[:, :-1][labels[:, 1:]!=-100]
            loss = F.kl_div(F.log_softmax(logits, dim=-1), F.log_softmax(target_logits, dim=-1), log_target=True)
        else:
            mask = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)
            loss = F.binary_cross_entropy_with_logits(logits, F.sigmoid(target_logits), reduction="none")
            loss = loss[mask].mean()

        kl_div += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (current_step+1) % 200 == 0:
            print("training kl_div", kl_div/200)
            kl_div = 0
        if current_step+1 == 5000:
            break
    
    with torch.no_grad():
        num_match = 0
        kl_div = 0
        for current_step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            target_logits = orig_model(**batch).logits

            logits = baseline(batch["input_ids"])

            if not use_BCE:
                shift_target_logits = target_logits[:, :-1]
                shift_logits = logits[:, :-1]
                shift_labels = labels[:, 1:]
                target_predictions = shift_target_logits.argmax(dim=-1)
                predictions = shift_logits.argmax(dim=-1)

                match = ((predictions == target_predictions) | (shift_labels == -100)).all(dim=1)
                num_match += match.sum().item()

                kl_div += F.kl_div(F.log_softmax(shift_logits[shift_labels!=-100], dim=-1), F.log_softmax(shift_target_logits[shift_labels!=-100], dim=-1), log_target=True).item()
            
            else:
                mask = (batch["input_ids"] == tokenizer.pad_token_id) | (batch["input_ids"] == tokenizer.eos_token_id)
                target_predictions = (target_logits > 0).long()
                predictions = (logits > 0).long()

                match = ((predictions == target_predictions).all(dim=-1) | mask).all(dim=1)
                num_match += match.sum().item()
            
                kl_div += F.binary_cross_entropy_with_logits(logits, F.sigmoid(target_logits), reduction="none")[~mask].mean().item()

            if current_step+1 == 2000:
                break
  
        acc_match = num_match / (2000 * 32)
        kl_div /= 2000
    print("baseline loss", kl_div, "  acc", acc_match)
    return kl_div, acc_match

def insert_into_dict_(acc_to_coef, acc, coef, num_edges, exp_name):
    closest = None
    min_diff = 1.0
    for a in acc_to_coef:
        diff = abs(a - acc)
        if diff < min_diff:
            min_diff = diff
            closest = a
    acc_to_coef[closest].append((acc, coef, num_edges, exp_name))
    acc_to_coef[closest].sort(key=lambda x: x[1])

def determine_coef(acc_to_coef, running_coefs):
    all_acc = list(acc_to_coef.keys())
    random.shuffle(all_acc)
    for acc in all_acc:
        if len(acc_to_coef[acc]) == 0:
            target_acc = acc
            break
    else:
        return "complete"
    
    # look for closest non-empty neighbor
    all_acc = sorted(all_acc)
    idx = all_acc.index(target_acc)
    prev_coefs = []
    for acc in all_acc[:idx]:
        prev_coefs.extend([item[1] for item in acc_to_coef[acc]])
    if prev_coefs:
        coef_high = min(prev_coefs)
    else:
        coef_high = None
    
    prev_coefs = []
    for acc in all_acc[idx+1:]:
        prev_coefs.extend([item[1] for item in acc_to_coef[acc]])
    if prev_coefs:
        coef_low = max(prev_coefs)
    else:
        coef_low = None
    
    if coef_low is None and coef_high is None:
        return None
    elif coef_low is None:
        coef_candidate = coef_high / 5 if coef_high > 0 else coef_high * 5
    elif coef_high is None:
        coef_candidate = coef_low * 5 if coef_low > 0 else coef_low / 5
    else:
        if coef_low > 0 and coef_high > 0:
            coef_candidate = math.sqrt(coef_low * coef_high)
        elif coef_low < 0 and coef_high < 0:
            coef_candidate = -math.sqrt(coef_low * coef_high)
        else:
            coef_candidate = (coef_low + coef_high) / 2

    # avoid collision
    factor = 1.2
    if coef_candidate >= 1e-2:
        scale = 1/factor
    elif coef_candidate < 0:
        scale = factor
    else:
        if random.random() < 0.5:
            scale = 1/factor
        else:
            scale = factor

    while True:
        for coef in running_coefs:
            if (coef / coef_candidate < factor) and (coef / coef_candidate > 1/factor):
                coef_candidate *= scale
                break
        else:
            break
    return coef_candidate

def determine_current_coef(lis, threshold_acc):
    if len(lis) == 0:
        return None, False
    
    above_thr = []
    below_thr = []
    for acc, coef, _, _ in lis:
        if acc >= threshold_acc:
            above_thr.append(coef)
        else:
            below_thr.append(coef)
    
    if len(above_thr) == 0:
        coef = min(below_thr) / 5 if min(below_thr) > 0 else min(below_thr) * 5
        both_exist = False
    elif len(below_thr) == 0:
        coef = max(above_thr) * 5 if max(above_thr) > 0 else max(above_thr) / 5
        both_exist = False
    else:
        if max(above_thr) > 0 and min(below_thr) > 0:
            coef = math.sqrt(max(above_thr) * min(below_thr))
        elif max(above_thr) < 0 and min(below_thr) < 0:
            coef = -math.sqrt(max(above_thr) * min(below_thr))
        else:
            coef = (max(above_thr) + min(below_thr)) / 2

        both_exist = True

    return coef, both_exist

def pareto_frontier(points):
    """
    Extract the Pareto frontier where accuracy is maximized and num_edges minimized.
    points: list of (accuracy, coef, num_edges, exp_name)
    """
    points = sorted(points, key=lambda x: (x[2], -x[0]))
    
    frontier = []
    best_acc = -float("inf")
    for item in points:
        if item[0] > best_acc:  # strictly better accuracy
            frontier.append(item)
            best_acc = item[0]
    return frontier

def sample_frontier_existing(frontier, k):
    """
    Select k existing points from the frontier, spread as evenly as possible
    along curve length.
    """
    n = len(frontier)   # frontier already sorted
    if k > n:
        return frontier
    
    values = np.array([(acc, num_edges) for acc, _, num_edges, _ in frontier])
    
    # Compute cumulative arc length
    diffs = np.diff(values, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_len = np.insert(np.cumsum(seg_lengths), 0, 0)
    total_len = cum_len[-1]

    # Target positions
    target_lens = np.linspace(0, total_len, k)

    # For each target, assign the nearest *unused* frontier index in order
    chosen = []
    last_idx = -1
    for t in target_lens:
        idx = int(np.argmin(np.abs(cum_len - t)))
        # Ensure monotonic (don’t go backward)
        if idx <= last_idx:
            idx = last_idx + 1
        if idx >= n:  # clamp to end
            continue
        chosen.append(frontier[idx])
        last_idx = idx

    return chosen

@click.command
@click.option("--model_name")
@click.option("--num_thread", default=8)
@click.option("--series_label", default="")
@click.option("--granularity", default=10)
@click.option("--max_num_per_round", default=20)
@click.option("--ssh_command", default=True)
@click.option("--split_mlp", default=True)
def run(model_name, num_thread, series_label, granularity, max_num_per_round, ssh_command, split_mlp):
    base_config["model_name"] = model_name
    print(f"=============== start {model_name} ===============")
    start_time = time.time()
    timestamp = datetime.now().strftime("%m%d_%H%M")
    series_name = f"newprune-{model_name}-{timestamp}"  # a series means a series of exp for the same model
    if series_label:
        series_name += f"-{series_label}"   
    series_dir = Path(output_dir) / series_name
    assert not series_dir.exists()
    series_dir.mkdir(parents=True)

    baseline_loss, baseline_acc = get_baseline_loss_acc(base_config)
    base_config["baseline_loss"] = baseline_loss

    counter_base = 0
    print("******** determine zero stage coefs ********")
    # the stage where real LN is enabled, to see how many edges are needed to achieve 0.9

    base_config["init_sample_param"] = 10
    running_exps = set()
    init_coefs = [-1e-6, 1e-4, 1e-3, 1e-5, 1e-2, 1e-6, 1e-7, 3e-4, 3e-5]
    counter = 0
    
    results = []
    while True:
        time.sleep(1)
        result = run_ssh_command("condor_q") if ssh_command else run_command("condor_q")
        if result.returncode == 0:
            match = re.search(rf'Total for {USER_NAME}.*(\d+) idle, (\d+) running', result.stdout)
            if match and len(match.groups()) == 2:
                n1, n2 = match.groups()
                num_jobs = int(n1) + int(n2)
                if num_jobs < num_thread:
                    
                    for running_exp, coef in list(running_exps):
                        if err_path := glob.glob("*.err", root_dir=series_dir / running_exp / "logs"):
                            time.sleep(3)
                            output_path = series_dir / running_exp / "output.json"
                            if output_path.exists():
                                with open(output_path) as f:
                                    output_dict = json.load(f)
                                result_dict = output_dict["result_patching_performance_global_iteration_0"]
                                results.append((result_dict["acc_match"], result_dict["coef"], result_dict["num_edges"], running_exp))
                                print(f"{running_exp} finished, acc: {result_dict['acc_match']:.3f}, coef: {result_dict['coef']:.7f}, num_edges: {result_dict['num_edges']}")
                            else:
                                err_path = str(series_dir / running_exp / "logs" / err_path[0])
                                print(f"{running_exp} failed, err file path: {err_path}")
                            running_exps.remove((running_exp, coef))
                            print("remaining exps", running_exps)

                    coef, _ = determine_current_coef(results, 0.9)
                    if coef is None or counter < 4:
                        coef = init_coefs[counter]
                    elif len(results) >= max_num_per_round: # and both_exist:
                        if running_exps:
                            continue
                        else:
                            break
                    

                    config = deepcopy(base_config)
                    config["linear_LN"] = False
                    config["sparsity_coef0_for_pruning"] = coef
                    config["sparsity_coef1_for_pruning"] = 0
                    config["sparsity_coef2_for_pruning"] = 0
                    config["start_stage"] = 1
                    config["end_stage"] = 1
                    exp_name = f"exp{counter+counter_base:03d}"
                    config["output_dir"] = str(series_dir)
                    config["exp_name"] = exp_name
                    exp_dir = series_dir / exp_name
                    exp_dir.mkdir()

                    with open(exp_dir / "config.yaml", "w") as f:
                        yaml.dump(config, f)

                    log_dir = exp_dir / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create submit file
                    submit_content = create_condor_submit(str(exp_dir / "config.yaml"), str(log_dir))
                    with open(exp_dir / "submit.sub", 'w') as f:
                        f.write(submit_content)
                    
                    result = run_ssh_command(f'cd {exp_dir} && condor_submit submit.sub') if ssh_command else run_command(f'cd {exp_dir} && condor_submit submit.sub')
                    if result.returncode == 0:
                        match = re.search(r'submitted to cluster (\d+)\.', result.stdout)
                        if match:
                            print(f"new job ({exp_name}, coef={coef:.7f}) submitted to", match.group(1), "\n")
                            counter += 1
                            running_exps.add((exp_name, coef))
                    else:
                        print(result)
        else:
            print(result)
    counter_base += counter
    above_thr = [num_edges for acc, _, num_edges, _ in results if acc > 0.9]
    if len(above_thr) == 0:
        print(f"******** 0 stage acc too low, very weird ********")
    else:
        print(f" ------ minimum edges {min(above_thr)}")

    acc_centers = list(map(lambda x: round(x, 3), torch.linspace(baseline_acc, 1.0, granularity).tolist()))
    for init_sample_param in [8]:
        print("******** determine first stage coefs ********")
        print("init_sample_param", init_sample_param)
        base_config["init_sample_param"] = init_sample_param
        acc_to_coef = {acc: [] for acc in acc_centers}
        running_exps = set()
        init_coefs = [-1e-6, 1e-4, 1e-3, 1e-5, 1e-2, 1e-6, 1e-7, 3e-4, 3e-5]
        tried_coefs = []
        counter = 0
        
        while True:
            time.sleep(1)
            result = run_ssh_command("condor_q") if ssh_command else run_command("condor_q")
            if result.returncode == 0:
                match = re.search(rf'Total for {USER_NAME}.*(\d+) idle, (\d+) running', result.stdout)
                if match and len(match.groups()) == 2:
                    n1, n2 = match.groups()
                    num_jobs = int(n1) + int(n2)
                    if num_jobs < num_thread:
                        
                        for running_exp, coef in list(running_exps):
                            if err_path := glob.glob("*.err", root_dir=series_dir / running_exp / "logs"):
                                time.sleep(3)
                                output_path = series_dir / running_exp / "output.json"
                                if output_path.exists():
                                    with open(output_path) as f:
                                        output_dict = json.load(f)
                                    result_dict = output_dict["result_patching_performance_global_iteration_0"]
                                    insert_into_dict_(acc_to_coef, result_dict["acc_match"], result_dict["coef"], result_dict["num_edges"], running_exp)
                                    print(f"{running_exp} finished, acc: {result_dict['acc_match']:.3f}, coef: {result_dict['coef']:.7f}, num_edges: {result_dict['num_edges']}")
                                else:
                                    err_path = str(series_dir / running_exp / "logs" / err_path[0])
                                    print(f"{running_exp} failed, err file path: {err_path}")
                                running_exps.remove((running_exp, coef))
                                print("remaining exps", running_exps)

                        coef = determine_coef(acc_to_coef, tried_coefs)
                        if coef == "complete" or counter >= max_num_per_round:
                            # print("all acc interval contains at least one exp, finish running...")
                            if running_exps:
                                continue
                            else:
                                break
                        elif coef is None or counter < 4:
                            coef = init_coefs[counter]

                        config = deepcopy(base_config)
                        config["sparsity_coef0_for_pruning"] = coef
                        config["sparsity_coef1_for_pruning"] = 0
                        config["sparsity_coef2_for_pruning"] = 0
                        config["start_stage"] = 1
                        config["end_stage"] = 1
                        exp_name = f"exp{counter+counter_base:03d}"
                        config["output_dir"] = str(series_dir)
                        config["exp_name"] = exp_name
                        exp_dir = series_dir / exp_name
                        exp_dir.mkdir()

                        with open(exp_dir / "config.yaml", "w") as f:
                            yaml.dump(config, f)

                        log_dir = exp_dir / "logs"
                        log_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Create submit file
                        submit_content = create_condor_submit(str(exp_dir / "config.yaml"), str(log_dir))
                        with open(exp_dir / "submit.sub", 'w') as f:
                            f.write(submit_content)
                        
                        result = run_ssh_command(f'cd {exp_dir} && condor_submit submit.sub') if ssh_command else run_command(f'cd {exp_dir} && condor_submit submit.sub')
                        if result.returncode == 0:
                            match = re.search(r'submitted to cluster (\d+)\.', result.stdout)
                            if match:
                                print(f"new job ({exp_name}, coef={coef:.7f}) submitted to", match.group(1), "\n")
                                counter += 1
                                running_exps.add((exp_name, coef))
                                tried_coefs.append(coef)
                        else:
                            print(result)
            else:
                print(result)
        
        counter_base += counter
        if max([item[0] for lis in acc_to_coef.values() for item in lis]) > 0.95:
            break
    counter = counter_base

    # max_acc = max([item[0] for lis in acc_to_coef.values() for item in lis])
    above_thr = [item[2] for lis in acc_to_coef.values() for item in lis if item[0] >= 0.7]
    if len(above_thr) == 0 or min(above_thr) > 50:
        if len(above_thr) == 0:
            print(f"******** acc too low, searching terminated for {model_name} ********")
        elif min(above_thr) > 50:
            print(f"******** too many edges needed ({min(above_thr)}), searching terminated for {model_name} ********")
        print("Total number of exps:", counter, "\n")
        end_time = time.time()
        elapsed_hours = (end_time - start_time) / 3600
        print(f"Running time: {elapsed_hours:.4f} hours")
        exit()

    print("collected points", acc_to_coef)
    data_points = [item for lis in acc_to_coef.values() for item in lis]
    data_points = pareto_frontier(data_points)
    data_points = sample_frontier_existing(data_points, granularity)
    print("selected points\n", data_points)

    # for split_mlp in [True, False]:
    base_config["split_mlps"] = split_mlp
    print("******** determine second stage coefs ********")
    print("    split_mlp:", split_mlp)

    running_exps = set()

    acc_to_prev_coef = {}
    acc_to_coef = {}
    for acc, coef, _, pre_exp_name in data_points:
        acc_to_prev_coef[acc] = (coef, pre_exp_name)
        acc_to_coef[acc] = []   # (acc, 2nd coef, num_edges, exp) 
    max_num_per_bin = -(-max_num_per_round // len(acc_to_coef))

    queue = deque(list(acc_to_prev_coef.keys()))

    while True:
        time.sleep(1)
        result = run_ssh_command("condor_q") if ssh_command else run_command("condor_q")
        if result.returncode == 0:
            match = re.search(rf'Total for {USER_NAME}.*(\d+) idle, (\d+) running', result.stdout)
            if match and len(match.groups()) == 2:
                n1, n2 = match.groups()
                num_jobs = int(n1) + int(n2)
                if num_jobs < num_thread:
                    
                    for running_exp, prev_acc in list(running_exps):
                        if err_path := glob.glob("*.err", root_dir=series_dir / running_exp / "logs"):
                            time.sleep(3)
                            output_path = series_dir / running_exp / "output.json"
                            if output_path.exists():
                                with open(output_path) as f:
                                    output_dict = json.load(f)
                                result_dict = output_dict.get("result_patching_performance_global_iteration_1", None)
                                if result_dict is not None:
                                    acc_to_coef[prev_acc].append((result_dict["acc_match"], result_dict["coef"], result_dict["num_edges"], running_exp))
                                    print(f"{running_exp} finished, acc: {result_dict['acc_match']:.3f}, coef: {result_dict['coef']:.7f}, num_edges: {result_dict['num_edges']}")
                                    queue.append(prev_acc)
                                else:
                                    print(f"{running_exp} finished, stage skipped")
                            else:
                                err_path = str(series_dir / running_exp / "logs" / err_path[0])
                                print(f"{running_exp} failed, err file path: {err_path}")
                            running_exps.remove((running_exp, prev_acc))
                            print("remaining exps", running_exps)

                    if len(queue) == 0:
                        if len(running_exps) > 0:
                            continue
                        else:
                            break
                    prev_acc = queue.popleft()
                    first_coef, first_exp_name = acc_to_prev_coef[prev_acc]
                    threshold_acc = (prev_acc - baseline_acc) * 0.9 + baseline_acc
                    coef, both_exist = determine_current_coef(acc_to_coef[prev_acc], threshold_acc)

                    if coef is None:
                        coef = first_coef
                    elif len(acc_to_coef[prev_acc]) >= max_num_per_bin: # and both_exist:
                        continue

                    config = deepcopy(base_config)
                    config["sparsity_coef0_for_pruning"] = first_coef
                    config["sparsity_coef1_for_pruning"] = coef
                    config["sparsity_coef2_for_pruning"] = 0
                    config["input_exp_path"] = str(series_dir / first_exp_name)
                    config["start_stage"] = 2
                    config["end_stage"] = 2
                    exp_name = f"exp{counter:03d}"
                    config["output_dir"] = str(series_dir)
                    config["exp_name"] = exp_name
                    exp_dir = series_dir / exp_name
                    exp_dir.mkdir()

                    with open(exp_dir / "config.yaml", "w") as f:
                        yaml.dump(config, f)

                    log_dir = exp_dir / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create submit file
                    submit_content = create_condor_submit(str(exp_dir / "config.yaml"), str(log_dir))
                    with open(exp_dir / "submit.sub", 'w') as f:
                        f.write(submit_content)
                    
                    result = run_ssh_command(f'cd {exp_dir} && condor_submit submit.sub') if ssh_command else run_command(f'cd {exp_dir} && condor_submit submit.sub')
                    if result.returncode == 0:
                        match = re.search(r'submitted to cluster (\d+)\.', result.stdout)
                        if match:
                            print(f"new job ({exp_name}, 1st coef={first_coef:.7f}, 2nd coef={coef:.7f}) submitted to", match.group(1), "\n")
                            counter += 1
                            running_exps.add((exp_name, prev_acc))
                            
                    else:
                        print(result)
        else:
            print(result)

    print("******** determine third stage coefs ********")
    running_exps = set()
    print("collected points", acc_to_coef)
    data_points = []
    for prev_acc in acc_to_coef:
        first_coef, _ = acc_to_prev_coef[prev_acc]
        for acc, second_coef, num_edges, exp in acc_to_coef[prev_acc]:
            data_points.append((acc, (first_coef, second_coef), num_edges, exp))
    data_points = pareto_frontier(data_points)
    data_points = sample_frontier_existing(data_points, granularity)
    print("selected points\n", data_points)

    acc_to_prev_coef = {}
    acc_to_coef = {}
    for acc, coef, _, pre_exp_name in data_points:
        acc_to_prev_coef[acc] = (coef, pre_exp_name)
        acc_to_coef[acc] = []   # (acc, 3rd coef, num_edges, exp) 
    max_num_per_bin = -(-max_num_per_round // len(acc_to_coef))
    
    queue = deque(list(acc_to_prev_coef.keys()))

    while True:
        time.sleep(1)
        result = run_ssh_command("condor_q") if ssh_command else run_command("condor_q")
        if result.returncode == 0:
            match = re.search(rf'Total for {USER_NAME}.*(\d+) idle, (\d+) running', result.stdout)
            if match and len(match.groups()) == 2:
                n1, n2 = match.groups()
                num_jobs = int(n1) + int(n2)
                if num_jobs < num_thread:
                    
                    for running_exp, prev_acc in list(running_exps):
                        if err_path := glob.glob("*.err", root_dir=series_dir / running_exp / "logs"):
                            time.sleep(3)
                            output_path = series_dir / running_exp / "output.json"
                            if output_path.exists():
                                with open(output_path) as f:
                                    output_dict = json.load(f)
                                result_dict = output_dict.get("result_patching_performance_global_iteration_2", None)
                                if result_dict is not None:
                                    acc_to_coef[prev_acc].append((result_dict["acc_match"], result_dict["coef"], result_dict["num_edges"], running_exp))
                                    print(f"{running_exp} finished, acc: {result_dict['acc_match']:.3f}, coef: {result_dict['coef']:.7f}, num_edges: {result_dict['num_edges']}")
                                    queue.append(prev_acc)
                                else:
                                    print(f"{running_exp} finished, stage skipped")
                            else:
                                err_path = str(series_dir / running_exp / "logs" / err_path[0])
                                print(f"{running_exp} failed, err file path: {err_path}")
                            running_exps.remove((running_exp, prev_acc))
                            print("remaining exps", running_exps)

                    if len(queue) == 0:
                        if len(running_exps) > 0:
                            continue
                        else:
                            break
                    prev_acc = queue.popleft()
                    (first_coef, second_coef), second_exp_name = acc_to_prev_coef[prev_acc]
                    threshold_acc = (prev_acc - baseline_acc) * 0.9 + baseline_acc
                    coef, both_exist = determine_current_coef(acc_to_coef[prev_acc], threshold_acc)

                    if coef is None:
                        coef = second_coef
                    elif len(acc_to_coef[prev_acc]) >= max_num_per_bin: # and both_exist:
                        continue

                    config = deepcopy(base_config)
                    config["sparsity_coef0_for_pruning"] = first_coef
                    config["sparsity_coef1_for_pruning"] = second_coef
                    config["sparsity_coef2_for_pruning"] = coef
                    config["input_exp_path"] = str(series_dir / second_exp_name)
                    config["start_stage"] = 3
                    config["end_stage"] = 3
                    exp_name = f"exp{counter:03d}"
                    config["output_dir"] = str(series_dir)
                    config["exp_name"] = exp_name
                    exp_dir = series_dir / exp_name
                    exp_dir.mkdir()

                    with open(exp_dir / "config.yaml", "w") as f:
                        yaml.dump(config, f)

                    log_dir = exp_dir / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create submit file
                    submit_content = create_condor_submit(str(exp_dir / "config.yaml"), str(log_dir))
                    with open(exp_dir / "submit.sub", 'w') as f:
                        f.write(submit_content)
                    
                    result = run_ssh_command(f'cd {exp_dir} && condor_submit submit.sub') if ssh_command else run_command(f'cd {exp_dir} && condor_submit submit.sub')
                    if result.returncode == 0:
                        match = re.search(r'submitted to cluster (\d+)\.', result.stdout)
                        if match:
                            print(f"new job ({exp_name}, 1st coef={first_coef:.7f}, 2nd coef={second_coef:.7f}, 3rd coef={coef:.7f}) submitted to", match.group(1), "\n")
                            counter += 1
                            running_exps.add((exp_name, prev_acc))
                            
                    else:
                        print(result)
        else:
            print(result)

    print(f"******** searching finished for {model_name} ********")
    print("Total number of exps:", counter, "\n")
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    print(f"Running time: {elapsed_hours:.4f} hours")


if __name__ == "__main__":
    run()
