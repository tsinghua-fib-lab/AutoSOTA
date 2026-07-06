from typing import Callable, TypeAlias
from functools import partial
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


from .language_utils import (
    LM_TEMPLATE_MAP,
    SUPPORTED_LLM,
    get_lm_loss,
    get_hf_tokenizer,
)
from .metrics import accuracy

from .cli_parser import (
    ModelSetting,
    OptimizerSetting,
)
from .data import (
    ImageClassificationTask,
    LmClassificationTask,
    LmGenerationTask,
)
from dataclasses import dataclass


SupportedDataset: TypeAlias = (
    ImageClassificationTask | LmClassificationTask | LmGenerationTask
)


def get_model(
    dataset: SupportedDataset,
    model_setting: ModelSetting,
    seed: int | None = None,
):
    torch_dtype = model_setting.get_torch_dtype()
    if seed:
        torch.manual_seed(seed)

    if isinstance(dataset, (LmClassificationTask, LmGenerationTask)):
        assert model_setting.large_model.value in SUPPORTED_LLM
        hf_model_name = model_setting.get_hf_model_name()
        if model_setting.split:
            from ..model_split_helper import ModelSplitHelper

            model_split_helper = ModelSplitHelper()
            client, server = model_split_helper.split(
                hf_model_name,
                split_point=model_setting.split_point,
                torch_dtype=torch_dtype,
                device_map="cpu",  # we will move it to device later
            )
            print(f"Split point: {model_setting.split_point}")
            if model_setting.lora:
                lora_config = LoraConfig(
                    r=model_setting.lora_r,
                    lora_alpha=model_setting.lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                )
                client = get_peft_model(client, lora_config).to(torch_dtype)
                server = get_peft_model(server, lora_config).to(torch_dtype)
                print(f"Client model: {client}")
                client.print_trainable_parameters()
                print(f"Server model: {server}")
                server.print_trainable_parameters()
            return (client, server)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_name, torch_dtype=torch_dtype
            )
            model.model_name = model_setting.large_model.value
            if model_setting.lora:
                # this step initialize lora parameters, which should be under control of seed
                lora_config = LoraConfig(
                    r=model_setting.lora_r,
                    lora_alpha=model_setting.lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                )
                model = get_peft_model(model, lora_config).to(torch_dtype)
                model.print_trainable_parameters()
            return model
    else:
        raise Exception(f"Dataset {dataset} is not supported")


def get_trainable_model_parameters(
    model: torch.nn.Module,
):
    for param in model.parameters():
        if param.requires_grad:
            yield param


def get_optimizer(
    model, dataset: SupportedDataset, optimizer_setting: OptimizerSetting
) -> torch.optim.SGD | torch.optim.Adam:
    trainable_model_parameters = get_trainable_model_parameters(model)
    if optimizer_setting.optimizer == "sgd":
        if dataset == ImageClassificationTask.mnist:
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                weight_decay=1e-5,
                momentum=optimizer_setting.momentum,
            )
        elif dataset == ImageClassificationTask.cifar10:
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                weight_decay=5e-4,
                momentum=optimizer_setting.momentum,
            )
        elif dataset == ImageClassificationTask.fashion:
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                weight_decay=1e-5,
                momentum=optimizer_setting.momentum,
            )
        elif isinstance(dataset, LmClassificationTask):
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                momentum=optimizer_setting.momentum,
                weight_decay=5e-4,
            )
        elif isinstance(dataset, LmGenerationTask):
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                momentum=optimizer_setting.momentum,
                weight_decay=0,
            )
        else:
            raise Exception(f"dataset {dataset.value} not supported")
    elif optimizer_setting.optimizer == "adamw":
        return torch.optim.AdamW(
            trainable_model_parameters,
            lr=optimizer_setting.lr,
            betas=(optimizer_setting.beta1, optimizer_setting.beta2),
            weight_decay=5e-4,
        )
    else:
        raise Exception(f"optimizer {optimizer_setting.optimizer} not supported")


@dataclass
class ModelInferences:
    train_inference: Callable
    test_inference: Callable


@dataclass
class MetricPacks:
    train_loss: Callable
    train_acc: Callable
    test_loss: Callable
    test_acc: Callable


def get_metrics(dataset: SupportedDataset, model_setting: ModelSetting) -> MetricPacks:
    if not isinstance(dataset, (LmClassificationTask, LmGenerationTask)):
        return MetricPacks(
            train_loss=nn.CrossEntropyLoss(),
            train_acc=accuracy,
            test_loss=nn.CrossEntropyLoss(),
            test_acc=accuracy,
        )
    else:
        hf_model_name = model_setting.get_hf_model_name()
        tokenizer = get_hf_tokenizer(hf_model_name)
        if isinstance(dataset, LmGenerationTask):
            # write in separate lines to differentiate from above cases, here acc=criterion
            train_criterion = get_lm_loss("full_sentence", verbalizer_id_map={})
            test_accuracy_func = get_lm_loss("f1", tokenizer=tokenizer)
            return MetricPacks(
                train_loss=train_criterion,
                train_acc=lambda pred, true: torch.tensor(
                    0.0
                ),  # noop training acc step here
                test_loss=lambda pred, true: torch.tensor(
                    0.0
                ),  # noop test loss step here
                test_acc=test_accuracy_func,
            )
        elif isinstance(dataset, LmClassificationTask):
            template = LM_TEMPLATE_MAP[dataset.value]()
            verbalizer_id_map = template.get_verbalizer_id(tokenizer)
            train_criterion = test_criterion = get_lm_loss(
                "last_token", verbalizer_id_map=verbalizer_id_map
            )
            train_accuracy_func = test_accuracy_func = get_lm_loss(
                "accuracy", verbalizer_id_map=verbalizer_id_map
            )
            return MetricPacks(
                train_loss=train_criterion,
                train_acc=train_accuracy_func,
                test_loss=test_criterion,
                test_acc=test_accuracy_func,
            )
        else:
            raise Exception(f"Dataset {dataset} not supported")
