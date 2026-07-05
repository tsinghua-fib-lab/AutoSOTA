import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import math
from collections import OrderedDict

# Hugging Face & PEFT
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

# Local Utils
from ..utils import prepare_settings
from ..utils.data import get_dataloaders

SUPPORTED_LLM = {
    "opt-125m": "facebook/opt-125m",
    "gemma-3-270m": "google/gemma-3-270m",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1b",
}


class CentralizedMemoryProfilingRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        print("Initializing Centralized Training Memory Profiling Runner...")

        # 1. Data Setup
        self.train_loaders, _ = get_dataloaders(
            args.data_setting,
            num_train_split=1,
            seed=args.seed,
            hf_model_name=args.model_setting.get_hf_model_name(),
        )
        self.train_iter = self._inf_loader(self.train_loaders[0])

        # 2. Model Setup (Full Model + LoRA)
        self.model = self._load_centralized_model(args.model_setting)
        self.model = self.model.to(self.device)

        # 3. Optimizer Setup
        self.optimizer = prepare_settings.get_optimizer(
            self.model, args.data_setting.dataset, args.optimizer_setting
        )

        # 4. Metrics
        self.metric_packs = prepare_settings.get_metrics(
            args.data_setting.dataset, args.model_setting
        )

        print(
            f"Trainable Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

    def _inf_loader(self, dl):
        while True:
            for v in dl:
                yield v

    def _load_centralized_model(self, model_setting):
        model_key = model_setting.large_model.value
        if model_key not in SUPPORTED_LLM:
            hf_path = model_key
            print(
                f"Warning: {model_key} not in SUPPORTED_LLM, trying to load directly."
            )
        else:
            hf_path = SUPPORTED_LLM[model_key]

        print(f"Loading HF Model from: {hf_path}")

        torch_dtype = model_setting.get_torch_dtype()

        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype=torch_dtype,
        )

        if model_setting.lora:
            print(
                f"Applying LoRA: r={model_setting.lora_r}, alpha={model_setting.lora_alpha}"
            )
            # this step initialize lora parameters, which should be under control of seed
            lora_config = LoraConfig(
                r=model_setting.lora_r,
                lora_alpha=model_setting.lora_alpha,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config)

            model = model.to(torch_dtype)
            model.print_trainable_parameters()

        return model

    def run(self):
        total_steps = getattr(self.args.sfl_setting, "total_steps", 100)

        print(f"Start Centralized Profiling: {total_steps} Steps...")

        # GPU Warmup
        torch.cuda.synchronize()

        self.model.train()

        for step in tqdm(range(total_steps)):

            # 1. Get Data
            inputs, labels = next(self.train_iter)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=False,
            )

            loss = self.metric_packs.train_loss(outputs, labels)

            # 4. Backward
            loss.backward()

            # 5. Optimizer Step
            self.optimizer.step()

            # 6. Cleanup
            del inputs, labels, outputs, loss

            # torch.cuda.empty_cache()

        print("Centralized Profiling Completed.")
