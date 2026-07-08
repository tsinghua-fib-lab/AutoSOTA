from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import Trainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled, logging
from peft.tuners.lora.layer import Linear as LoraLinear

# include_keywords = ["block.0", "block.4"]
include_keywords = ["encoder.block.2","encoder.block.3","encoder.block.4"]  # for T5
# include_keywords = ["layers.27", "layers.6"]  # for Llama
do_log = False


def get_forward_hook(name):
    def hook(module, input, output):
        wandb.log(
            {
                f"{name}/input_mean": input[0].mean().item(),
                f"{name}/input_std": input[0].std().item(),
                f"{name}/output_mean": output.mean().item(),
                f"{name}/output_std": output.std().item(),
            },
            commit=False,
        )
    return hook

class LogTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.is_peft = "PeftModel" in type(model).__name__
        if self.is_peft:
            for name, module in model.named_modules():
                if isinstance(module, LoraLinear):
                    self.scaling = module.scaling["default"]
                    break
        self.orig_A = None
        self.orig_B = None
        self.orig_W = None
        self.gradient_accumulation_counter = 0

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        if not do_log:
            return super().training_step(model, inputs)
        if self.is_peft:
            if self.orig_A is None:
                self.orig_A = {}
                self.orig_B = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        if "lora_A" in name:
                            self.orig_A[name.split("lora_A.")[0]] = (
                                param.detach().clone()
                            )
                        elif "lora_B" in name:
                            self.orig_B[name.split("lora_B.")[0]] = (
                                param.detach().clone()
                            )
                for name, module in model.named_modules():
                    if any([kw in name for kw in include_keywords]) and isinstance(module, LoraLinear):
                        breakpoint()
                        hook = get_forward_hook(name)
                        module.register_forward_hook(hook)
        else:
            if self.orig_W is None:
                self.orig_W = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        self.orig_W[name] = param.detach().clone()

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        with torch.no_grad():
            if (
                self.gradient_accumulation_counter
                % self.args.gradient_accumulation_steps
                == self.args.gradient_accumulation_steps - 1
            ):
                if self.is_peft:
                    A_dict = {}
                    B_dict = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad and any(
                            [kw in name for kw in include_keywords]
                        ):
                            if "lora_A" in name:
                                A_dict[name.split("lora_A.")[0]] = param
                            elif "lora_B" in name:
                                B_dict[name.split("lora_B.")[0]] = param
                    assert (
                        len(A_dict)
                        == len(self.orig_A)
                        == len(B_dict)
                        == len(self.orig_B)
                    ), (
                        len(A_dict),
                        len(self.orig_A),
                        len(B_dict),
                        len(self.orig_B),
                    )
                    for key in A_dict.keys():
                        A = A_dict[key]
                        B = B_dict[key]
                        lora_r = A.shape[0]
                        A_grad = A_dict[key].grad
                        B_grad = B_dict[key].grad
                        A_0 = self.orig_A[key]
                        B_0 = self.orig_B[key]
                        A_diff = A - A_0
                        B_diff = B - B_0
                        BA = torch.matmul(B, A)
                        BA_0 = torch.matmul(B_0, A_0)
                        BA_diff = BA - BA_0
                        BA_diff_norm = torch.norm(BA_diff).item()
                        A_diff_norm = torch.norm(A_diff).item()
                        B_diff_norm = torch.norm(B_diff).item()
                        A_norm = torch.norm(A).item()
                        B_norm = torch.norm(B).item()
                        A_grad_norm = torch.norm(A_grad).item()
                        B_grad_norm = torch.norm(B_grad).item()
                        # BA_singular_values = torch.svd(BA_diff.float(), compute_uv=False).S[:lora_r]
                        BA_singular_values = torch.svd_lowrank(
                            BA_diff.float(), q=2 * lora_r
                        )[1][:lora_r]
                        top_1_ratio = (
                            BA_singular_values[0] / BA_singular_values.sum()
                        ).item()
                        top_4_ratio = (
                            BA_singular_values[:4].sum() / BA_singular_values.sum()
                        ).item()
                        wandb.log(
                            {
                                f"A_norm/{key}": A_norm,
                                f"B_norm/{key}": B_norm,
                                f"A_grad_norm/{key}": A_grad_norm,
                                f"B_grad_norm/{key}": B_grad_norm,
                                f"A_diff_norm/{key}": A_diff_norm,
                                f"B_diff_norm/{key}": B_diff_norm,
                                f"BA_diff_norm/{key}": BA_diff_norm,
                                f"scaled_BA_diff_norm/{key}": self.scaling
                                * BA_diff_norm,
                                f"BA_top_1_ratio/{key}": top_1_ratio,
                                f"BA_top_4_ratio/{key}": top_4_ratio,
                                "train/global_step": self.state.global_step,
                            }
                        )
                else:
                    W_dict = {}
                    for name, param in model.named_parameters():
                        if (
                            param.requires_grad
                            and any([kw in name for kw in include_keywords])
                            and len(param.shape) == 2
                        ):
                            W_dict[name] = param
                    for key in W_dict.keys():
                        W = W_dict[key]
                        W_grad = W.grad
                        W_0 = self.orig_W[key]
                        W_diff = W - W_0
                        W_diff_norm = torch.norm(W_diff).item()
                        W_norm = torch.norm(W).item()
                        W_grad_norm = torch.norm(W_grad).item()
                        U, S, V = torch.svd(W_diff.float())
                        top_1_ratio = S[0] / S.sum()
                        top_4_ratio = S[:4].sum() / S.sum()
                        wandb.log(
                            {
                                f"W_norm/{key}": W_norm,
                                f"W_grad_norm/{key}": W_grad_norm,
                                f"W_diff_norm/{key}": W_diff_norm,
                                "train/global_step": self.state.global_step,
                                f"W_top_1_ratio/{key}": top_1_ratio.item(),
                                f"W_top_4_ratio/{key}": top_4_ratio.item(),
                            }
                        )
        self.gradient_accumulation_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps
