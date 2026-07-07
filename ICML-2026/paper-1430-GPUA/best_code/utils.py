import math
import os
import random
import re
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
MAX_ITER_MAP = {
    "OxfordFlowers": 1,
    "OxfordPets": 3,
    "Caltech101": 5,
    "FGVCAircraft": 1,
    "EuroSAT": 5,
    "ImageUCF101": 3,
    "DescribableTextures": 5,
    "Food101": 5,
    "StanfordCars": 1,
    "SUN397": 5,
}

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True


def topk_accuracies(preds: torch.Tensor, labels: torch.Tensor, topk_vals: Tuple[int, ...]):
    assert preds.size(0) == labels.size(0)

    # Get class index for the top k probabilities
    top_max_k_inds = torch.topk(
        preds, max(topk_vals), dim=1, largest=True, sorted=True
    )[1]

    top_max_k_inds = top_max_k_inds.t()
    # duplicate the prediction top k time
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)

    # count the number of correct predictions
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    num_topks_correct = [top_max_k_correct[:k, :].float().sum()
                         for k in topk_vals]

    accuracies = [(x / preds.size(0)) * 100.0 for x in num_topks_correct]
    return accuracies


def process_class_names(cls_name):
    # converts str1-srt2- to str1 (srt2)
    cls_name = re.sub(r"(-)([^-]+)(-)", r" (\2)", cls_name)
    # converts str1_s_srt2 to str1's srt2
    cls_name = re.sub(r"_s_", r"'s ", cls_name)
    cls_name = cls_name.replace("_", " ")
    cls_name = cls_name if cls_name.endswith(".") else cls_name + "."
    cls_name = cls_name.lower()
    return cls_name


class ScalarMeter(object):
    def __init__(self, window_size: int):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_current_value(self):
        return self.deque[-1]

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


def get_grad_norm(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == "inf":
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device)
                 for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def convert_weights_to_fp16(layer):
    # From CLIP
    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        layer.weight.data = layer.weight.data.half()
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.half()

    if isinstance(layer, nn.MultiheadAttention):
        for attr in [
            *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
            "in_proj_bias",
            "bias_k",
            "bias_v",
        ]:
            tensor = getattr(layer, attr)
            if tensor is not None:
                tensor.data = tensor.data.half()

    for name in ["text_projection", "proj"]:
        if hasattr(layer, name):
            attr = getattr(layer, name)
            if attr is not None:
                attr.data = attr.data.half()


def gpu_mem_usage():
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def aggregate_predictions(preds: List[torch.Tensor], agg_type="max") -> torch.Tensor:
    # pred is n_clips x B x num_classes
    assert agg_type in ["majority", "mean", "max"]
    assert preds[0].ndim == 2

    if agg_type == "mean":
        return preds.mean(dim=0)

    if agg_type == "max":
        return preds.max(dim=0)[0]

    # take majority class
    majority_classes = preds.argmax(dim=-1).mode(dim=0)[0]
    agg_pred = torch.zeros_like(preds[0])
    return agg_pred.scatter_(1, majority_classes.view(-1, 1), 1)


class LRScheduler(object):
    def __init__(
        self,
        opt,
        lr_policy_type,
        lr_step_milestones,
        warmup_epochs,
        consine_end_lr,
        linear_end_lr,
        num_iters_per_epoch,
        num_epochs,
    ):
        assert lr_policy_type in ["cosine", "step", "linear", "constant"]

        self.opt = opt
        self.lr_policy_type = lr_policy_type

        self.base_lrs = [param_group["lr"] for param_group in opt.param_groups]
        self.consine_end_lr = consine_end_lr
        self.linear_end_lr = linear_end_lr
        self.lr_step_milestones = [
            i * num_iters_per_epoch for i in lr_step_milestones]

        self.warmup_steps = warmup_epochs * num_iters_per_epoch
        self.max_iters = num_epochs * num_iters_per_epoch

        self.curr_step = 0
        logger.info(f"Using {lr_policy_type} as a learning rate policy")

    def _warmup_lrs(self):
        multiplier = min(1.0, self.curr_step / self.warmup_steps)
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def _cosine_lrs(self):
        multiplier = (
            math.cos(
                math.pi
                * (self.curr_step - self.warmup_steps)
                / (self.max_iters - self.warmup_steps)
            )
            + 1.0
        ) * 0.5

        return [
            self.consine_end_lr + (base_lr - self.consine_end_lr) * multiplier
            for base_lr in self.base_lrs
        ]

    def _linear_lrs(self):
        multiplier = 1.0 - (self.curr_step - self.warmup_steps) / (
            self.max_iters - self.warmup_steps
        )

        return [
            self.linear_end_lr + (base_lr - self.linear_end_lr) * multiplier
            for base_lr in self.base_lrs
        ]

    def _step_lrs(self):
        multiplier = 0.1 ** [
            self.curr_step >= i for i in self.lr_step_milestones
        ].count(True)
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def step(self):
        self.curr_step += 1

        if self.curr_step <= self.warmup_steps:
            new_lrs = self._warmup_lrs()
        elif self.lr_policy_type == "cosine":
            new_lrs = self._cosine_lrs()
        elif self.lr_policy_type == "linear":
            new_lrs = self._linear_lrs()
        elif self.lr_policy_type == "step":
            new_lrs = self._step_lrs()
        else:
            new_lrs = [1.0] * len(self.base_lrs)

        for idx in range(len(self.opt.param_groups)):
            self.opt.param_groups[idx]["lr"] = new_lrs[idx]


def permute_embeddings(embeddings, labels):
    # Get the unique set of labels and their corresponding indices in the original tensor
    unique_labels, label_indices = torch.unique(labels, return_inverse=True)
    unique_labels = unique_labels.tolist()

    # Create a dictionary that maps each unique label to its corresponding indices in the original tensor
    label_index_dict = {label: torch.where(label_indices == i)[
        0] for i, label in enumerate(unique_labels)}

    shuffled_label_index_dict = {label: torch.where(label_indices == i)[
        0] for i, label in enumerate(unique_labels)}

    # Shuffle the indices for each unique label
    for label in unique_labels:
        perm = torch.randperm(
            shuffled_label_index_dict[label].size(0)).to(embeddings.device)
        shuffled_label_index_dict[label] = shuffled_label_index_dict[label][perm]

    # Concatenate the shuffled indices into a single tensor
    shuffled_indices = torch.cat([shuffled_label_index_dict[label]
                                 for label in unique_labels])

    indices = torch.cat([label_index_dict[label]
                         for label in unique_labels])

    # Use the shuffled indices to permute the embeddings and labels tensors
    shuffled_embeddings = embeddings.clone()
    shuffled_embeddings[indices] = embeddings[shuffled_indices]

    return shuffled_embeddings


def feature_augmentation(features, labels):

    def interpolation_weights(features, alpha=2.0):
        alpha = np.array([alpha] * features.size(0))
        interpolation_w = np.float32(np.random.beta(alpha, alpha))
        interpolation_w = torch.from_numpy(interpolation_w).to(features.device)
        interpolation_w = interpolation_w.view(-1, 1)
        return interpolation_w

    weights = interpolation_weights(features)
    aug_feats = permute_embeddings(features, labels)
    aug_feats = weights * aug_feats + (1 - weights) * features

    return (
        torch.cat([features, aug_feats], dim=0),
        torch.cat([labels, labels], dim=0)
    )
