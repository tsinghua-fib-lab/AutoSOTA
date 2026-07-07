from typing import Dict, Union, Any
from collections import OrderedDict

import torch
import copy
import importlib.metadata
import deepspeed
import datasets
from torch import nn
from transformers import Trainer
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .losses import get_loss

from trainer.custom_optimizer import AdamWDecouple8bit, AdamWDecoupleNormal
from trainer.custom_optimizer_gp import AdamWGPDecoupleNormal
from trainer.custom_optimizer_gp_block_vec import AdamWGPBlockVecDecoupleNormal
from trainer.custom_optimizer_plus import AdamWDecouplePlus, AdamWDecouplePlus8bit
from trainer.custom_sampler import AlternatingSampler, DistributedAlternatingSampler

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop("loss_type")
        self.ref_model = kwargs.pop("ref_model")

        # the coefficient of each part in the loss function. This is used in ablation study.
        self.forget_coeff = kwargs.pop("forget_coeff")
        self.regularization_coeff = kwargs.pop("regularization_coeff")
        # beta for NPO/DPO/RS
        self.beta = kwargs.pop("beta")

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        # Prepare the reference model with DeepSpeed
        if self.ref_model is not None and self.args.deepspeed is not None:
            self.ref_model = self.e_prepare_deepspeed(self.ref_model)
        if self.ref_model is not None and self.args.fsdp != "":
            self.ref_model = FSDP(self.ref_model,
                                  sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD if "full_shard" in self.args.fsdp_config
                                  else torch.distributed.fsdp.ShardingStrategy.NO_SHARD,
                                  backward_prefetch=self.args.fsdp_config["backward_prefetch"])
            self.ref_model.eval()
            if "full_shard" not in self.args.fsdp_config:
                self._move_model_to_device(self.ref_model, self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        forget_loss, regularization_loss = get_loss(
            model, self.ref_model, inputs, self.loss_type, self.beta
        )
        loss = (
            self.forget_coeff * forget_loss
            + self.regularization_coeff * regularization_loss
        )

        # print("Forget loss: ", forget_loss.item(), "Regularization loss: ", regularization_loss)
        # print(type(self.optimizer.optimizer))
        # print(self.accelerator.state.deepspeed_plugin.deepspeed_config)

        return (loss, None) if return_outputs else loss

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        # set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False

        return model



class CustomTrainerSafeAlign(CustomTrainerForgetting):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, None) if return_outputs else loss