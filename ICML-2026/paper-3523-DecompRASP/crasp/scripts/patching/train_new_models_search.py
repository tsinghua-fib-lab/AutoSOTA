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
from train_new_models import customCollator, customBCECollator, GPT2BCELMHeadModel, compute_metrics, compute_bce_metrics
from streamlit_app.my_paths import REMOTE_SCRIPT_PATH
import shutil


class myCallback(TrainerCallback):
    def on_evaluate(self, state, args, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        assert metrics["epoch"] >= getattr(self, "current_epoch", 0)
        if metrics["epoch"] > getattr(self, "current_epoch", 0):
            self.latest_acc = {}
            self.current_epoch = metrics["epoch"]
        for key in metrics.keys():
            if key.endswith("acc"):
                self.latest_acc[key] = metrics[key]
        if len(self.latest_acc) == len(test_length_ranges):
            if (self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0) or (self.current_epoch == 1.0):  
                if self.latest_acc[f"eval_len{train_length_range[0]}-{train_length_range[1]}_acc"] == 1.0: 
                    control.should_training_stop = True
                    msg = f"early stop {self.current_epoch}\t\t"
                else:
                    msg = "reach max step\t\t"
                print(f"{arch_name}\t\t", msg, "\t\t".join([f"{k}: {v}" for k, v in self.latest_acc.items()]), file=summary_f)
                summary_f.flush()
                all_results[arch_name] = self.latest_acc
                if self.latest_acc[f"eval_len{test_length_ranges[2][0]}-{test_length_ranges[2][1]}_acc"] == 1.0:
                    global should_stop
                    should_stop = True
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--debug", action="store_true")
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

    configs = [(l, h, d, lr, dropout) for l in [1, 2, 4] for h in [1, 2, 4] for d in [16, 64, 256] for lr in [1e-3, 1e-4] for dropout in [0.0, 0.1]]

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
        
    task_path = Path(REMOTE_SCRIPT_PATH).parent.parent.parent / "saved_models" / (f"lm-out-new-{task}" if not args.debug else "temp")
    if not task_path.exists():
        task_path.mkdir(parents=True)
    summary_f = open(task_path / f"summary.txt", "w")

    for i in range(3):
        print("\ninput example:")
        print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][i][0])))
        if not use_BCE:
            print("label example:")
            print(" ".join(tokenizer.convert_ids_to_tokens(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][i][2])))
        else:
            print("label example:")
            print(test_dataset[f"len{test_length_ranges[0][0]}-{test_length_ranges[0][1]}"][i][2])

    should_stop = False
    all_results = {}
    for n_layer, n_head, d_model, lr, dropout in configs: 
        set_seed(0)

        max_steps = 30_000
        warmup_steps = 0

        arch_name = f"{n_layer}l{n_head}h{d_model}d{'4' if lr == 1e-4 else '3'}lr{'00' if dropout == 0 else '01'}drop"
        print("\n ********** start training for config:", arch_name, " ********** ")
        output_dir = task_path / arch_name

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
            eval_steps=3_000,
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=3_000,
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

        trainer.save_model(output_dir)

        if should_stop:
            break

    # select best
    temp = [(arch_name, results[f"eval_len{test_length_ranges[2][0]}-{test_length_ranges[2][1]}_acc"]) for arch_name, results in all_results.items()]
    temp = sorted(temp, key=lambda x: (-x[1], x[0]))

    print("top 5", file=summary_f)
    for arch_name, bin2_acc in temp[:5]:
        print(f"{arch_name}\t\t", "\t\t".join([f"{k}: {v}" for k, v in all_results[arch_name].items()]), file=summary_f)

    best_arch_name = temp[0][0]
    if not args.debug:
        shutil.copytree(task_path / best_arch_name, task_path.parent / f"{task}-@{best_arch_name}")
        print("copy finished")
    
    summary_f.close()