from transformers import GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer, TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import random
from copy import deepcopy
import string
import argparse
import itertools
import os
from collections import Counter
from typing import Optional
from patching_utils import set_seed
from patching_data import *
from pathlib import Path
from train_new_models import customCollator, compute_metrics, GPT2BCELMHeadModel, customBCECollator, compute_bce_metrics
from streamlit_app.my_paths import REMOTE_SCRIPT_PATH
import shutil


def parse_available_results(summary_path):
    data = {}
    with open(summary_path) as f:
        for line in f:
            if line.startswith("top 5"):
                break
            parts = line.strip().split()
            if not parts:
                continue
            
            if re.search(r"reach max step", line):
                steps = 1.0
            elif match := re.search(r"early stop ([\d.]+)", line):
                steps = float(match.group(1))
            else:
                raise RuntimeError(line)
            
            data[parts[0]] = steps
        else:
            raise RuntimeError("best arch not found")
        
        best_arch = next(f).strip().split()[0]
        
    return best_arch, data[best_arch]


class myCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        assert metrics["epoch"] >= getattr(self, "current_epoch", 0)
        if metrics["epoch"] > getattr(self, "current_epoch", 0):
            self.latest_acc = {}
            self.current_epoch = metrics["epoch"]
        for key in metrics.keys():
            if key.endswith("acc"):
                self.latest_acc[key] = metrics[key]
        if len(self.latest_acc) == len(test_length_ranges):
            if self.current_epoch >= stop_ratio:
                control.should_training_stop = True
            accs = tuple([self.latest_acc[f"eval_len{test_length_ranges[i][0]}-{test_length_ranges[i][1]}_acc"] for i in range(3)])
            global best_accs

            if any(acc - best_acc > 0.25 for acc, best_acc in zip(accs, best_accs)):
            # if acc - best_acc > 0.15:
                control.should_save = True
                print(f"{best_arch}\t\tepoch:{self.current_epoch}\t\tstep:{state.global_step}\t\t", "\t\t".join([f"{k}: {v}" for k, v in self.latest_acc.items()]), file=summary_f)
                summary_f.flush()

                best_accs = accs
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    task = args.task
    set_seed(0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_length_range = (0, 50)
    test_length_ranges = [train_length_range] + [(51, 100), (101, 150)]
    max_test_length = test_length_ranges[-1][1]
    batch_size = 64
    per_device_bz = batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size 
    test_num = 2_000

    tokenizer = get_tokenizer_for_task(task, max_test_length)
    train_dataset = get_dataset_for_task(task, tokenizer, train_length_range, max_test_length, {"period_for_data": 3})
    test_dataset = {
            f"len{test_range[0]}-{test_range[1]}": EvalDataset(get_dataset_for_task(task, tokenizer, test_range, -1, {"period_for_data": 3}), test_num)
                for test_range in test_length_ranges
        }
    use_BCE = getattr(train_dataset, "BCE", False)
    
    task_to_n_positions = {
        "bin_majority_interleave": max_test_length + 6,
        "unique_copy": max_test_length * 2 + 3,
        "sort": max_test_length * 2 + 3,
        "unique_reverse": max_test_length * 2 + 3,
        "unique_bigram_copy": max_test_length * 2 + 3,
        "count": max_test_length + 5,
        "repeat_copy": max_test_length*2 + 3,
        "addition": max_test_length*2 + 2,
    }
    if task in task_to_n_positions:
        n_positions = task_to_n_positions[task]
    elif use_BCE:
        n_positions = max_test_length + 2
    else:   # formal languages and others
        n_positions = max_test_length + 4
        
    for i in range(3):
        print("\ninput example:")
        print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][i][0])))
        if not use_BCE:
            print("label example:")
            print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][i][2])))

    task_path = Path(REMOTE_SCRIPT_PATH).parent.parent.parent / "saved_models" / f"lm-out-new-{task}"
    assert task_path.exists()
    
    best_arch, stop_ratio = parse_available_results(task_path / "summary.txt")

    summary_f = open(task_path / f"extra_checkpoints_summary.txt", "w")
    match = re.search(r"(\d+)l(\d+)h(\d+)d([34])lr(0[01])drop", best_arch)
    n_layer, n_head, d_model, lr, dropout = match.groups()
    n_layer, n_head, d_model = int(n_layer), int(n_head), int(d_model)
    lr = 1e-3 if lr == "3" else 1e-4
    dropout = 0.0 if dropout == "00" else 0.1

    set_seed(0)

    max_steps = 30_000
    warmup_steps = 0
    best_accs = (0, 0, 0)

    output_dir = task_path / "checkpoints"

    cfg = GPT2Config(vocab_size=len(tokenizer), 
                n_positions=n_positions,
                n_embd=d_model,
                n_layer=n_layer,
                n_head=n_head,
                bos_token_id=tokenizer.bos_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                attn_pdrop=dropout,
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                )

    if use_BCE:
        model = GPT2BCELMHeadModel(cfg)
    else:
        model = GPT2LMHeadModel(cfg)


    training_args = TrainingArguments(
        output_dir=output_dir,    
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_bz,
        per_device_eval_batch_size=per_device_bz,
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        save_only_model=True,
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=lr,
        weight_decay=0.01,
        optim='adamw_torch',
        lr_scheduler_type='linear',
        warmup_steps=warmup_steps,
        report_to="none",
        seed=0,
        data_seed=0,
        eval_do_concat_batches = not use_BCE,
    )

    if use_BCE:
        data_collator = customBCECollator(tokenizer.pad_token_id)
    else:
        data_collator = customCollator(tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_bce_metrics if use_BCE else compute_metrics,
        callbacks=[myCallback],
    )

    trainer.train()

    for checkpoint_dir in os.listdir(task_path / "checkpoints"):
        if match := re.search(r"checkpoint-(\d+)", checkpoint_dir):
            step = int(match.group(1))
            ratio = step / max_steps / stop_ratio
            if ratio < 0.99:
                # ratio = round(ratio, 4)
                shutil.copytree(task_path / "checkpoints" / checkpoint_dir, task_path.parent / f"{task}-^{best_arch}-{step}")
                print("copy finished")
        
    summary_f.close()