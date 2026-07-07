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
from trainer.custom_optimizer_mix import AdamWDecoupleMix, AdamWDecoupleMix8bit
from trainer.custom_sampler import AlternatingSampler, DistributedAlternatingSampler


_smdistributed_available = importlib.util.find_spec("smdistributed") is not None


class CustomTrainerForgettingAlternate(Trainer):
    def __init__(
        self, alternate=True, optim_cfg="dual_adam", forget_lr=1e-5, retain_lr=1e-5, forget_freq=1, retain_freq=1,
        alpha=1., beta1=0.9, beta2=0.95, base_beta1=0.95, base_beta2=0.999, max_steps=300,
        *args, **kwargs
    ):
        self.loss_type = kwargs.pop("loss_type")
        self.ref_model = kwargs.pop("ref_model")
        self.forget_coeff = kwargs.pop("forget_coeff")
        self.regularization_coeff = kwargs.pop("regularization_coeff")
        self.beta = kwargs.pop("beta")

        self.optim_cfg = optim_cfg
        self.forget_lr = forget_lr
        self.retain_lr = retain_lr
        self.alternate = alternate

        self.forget_lr_ratio = self.forget_lr / self.retain_lr

        self.forget_freq = forget_freq
        self.retain_freq = retain_freq
        self.step_count = 0

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.base_beta1 = base_beta1
        self.base_beta2 = base_beta2

        self.forget_loss = 0.
        self.retain_loss = 0.

        self.max_steps = max_steps

        super(CustomTrainerForgettingAlternate, self).__init__(*args, **kwargs)

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

    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        """
        Override the default sampler to use AlternatingSampler
        """
        if not isinstance(self.train_dataset, ConcatDataset):
            raise ValueError("This CustomTrainer requires a ConcatDataset for train_dataset.")

        # train_dataset is a combination of dataset_a and dataset_b
        dataset_a, dataset_b = self.train_dataset.datasets

        return AlternatingSampler(
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                batch_size=self.args.train_batch_size,
                m=self.forget_freq * self.args.gradient_accumulation_steps,
                n=self.retain_freq * self.args.gradient_accumulation_steps,
            )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.,
            },
        ]

        if self.optim_cfg == "dual_adam":
            print("32bit DualAdam > Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)

            self.optimizer = AdamWDecoupleNormal(
                optimizer_grouped_parameters,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq,
            )
        elif self.optim_cfg == "dual_adam_8bit":
            print("8bit DualAdam > Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)

            self.optimizer = AdamWDecouple8bit(
                optimizer_grouped_parameters,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq
            )
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    # logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    # logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            print(f"skipped: {skipped / 2 ** 20}M params")
        elif self.optim_cfg == "dual_adam_plus":
            print("32bit DualAdamPlus > Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)

            self.optimizer = AdamWDecouplePlus(
                optimizer_grouped_parameters,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq,
                alpha=self.alpha,
                betas=(self.beta1, self.beta2, self.base_beta1, self.base_beta2)
            )
        elif self.optim_cfg.startswith("dual_adam_plus_8bit"):
            print("8bit DualAdamPlus> Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)

            quantize_delta = self.optim_cfg in ["dual_adam_plus_8bit", "dual_adam_plus_8bit_quantize_delta"]
            quantize_base = self.optim_cfg in ["dual_adam_plus_8bit", "dual_adam_plus_8bit_quantize_base"]

            self.optimizer = AdamWDecouplePlus8bit(
                optimizer_grouped_parameters,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq,
                alpha=self.alpha,
                betas=(self.beta1, self.beta2, self.base_beta1, self.base_beta2),
                quantize_delta=quantize_delta,
                quantize_base=quantize_base,
            )
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    # logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    # logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            print(f"skipped: {skipped / 2 ** 20}M params")

        elif self.optim_cfg == "dual_adam_mix":
            print("32bit DualAdamMix > Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)

            self.optimizer = AdamWDecoupleMix(
                optimizer_grouped_parameters,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq,
                betas=(self.beta1, self.beta2, self.base_beta1, self.base_beta2),
                max_steps=self.max_steps,
            )
        elif self.optim_cfg.startswith("dual_adam_mix_8bit"):
            print("8bit DualAdamMix> Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)
            self.optimizer = AdamWDecoupleMix8bit(
                optimizer_grouped_parameters,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq,
                betas=(self.beta1, self.beta2, self.base_beta1, self.base_beta2),
                max_steps=self.max_steps,
            )
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    # logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    # logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            print(f"skipped: {skipped / 2 ** 20}M params")

        elif self.optim_cfg == "dual_adam_gp":
            print("32bit Dual Adam GP > Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)

            original_param_shape_dict = self.get_original_parameter_shape()
            decay_parameters = [''.join(name.split("_fsdp_wrapped_module.")) for name in decay_parameters]
            grouped_original_params_shape = [
                [s for n, s in original_param_shape_dict.items() if n in decay_parameters],
                [s for n, s in original_param_shape_dict.items() if n not in decay_parameters]
            ]

            self.optimizer = AdamWGPDecoupleNormal(
                optimizer_grouped_parameters,
                grouped_original_params_shape,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq,
                svd_rank=self.svd_rank,
                proj_update_freq=self.proj_update_freq,
                project_1=self.gp_forget,
                project_2=self.gp_retain,
            )
        elif self.optim_cfg == "dual_adam_gp_block_vec":
            print("32bit Dual Adam GP > Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)

            self.optimizer = AdamWGPBlockVecDecoupleNormal(
                optimizer_grouped_parameters,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq,
                svd_rank=self.svd_rank,
                proj_update_freq=self.proj_update_freq,
                project_1=self.gp_forget,
                project_2=self.gp_retain,
            )
        else:
            self.create_optimizer()

        optimizer = self.optimizer
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=optimizer
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.alternate:  # alternating update
            if inputs[0] is not None or inputs[2] is not None:
                type = "forget"
            else:
                type = "retain"

            if type == "forget":
                forget_type = self.loss_type.split("+")[0]
                # print(forget_type)
                forget_loss, regularization_loss = get_loss(
                    model, self.ref_model, inputs, forget_type, self.beta
                )
                loss = self.forget_coeff * forget_loss
                self.forget_loss = forget_loss.item()

            else:
                retain_type = self.loss_type.split("+")[1]
                # print(retain_type)
                forget_loss, regularization_loss = get_loss(
                    model, self.ref_model, inputs, retain_type, self.beta
                )
                loss = self.regularization_coeff * regularization_loss
                self.retain_loss = regularization_loss.item()
            # print("Forget loss: ", self.forget_loss, "Regularization loss: ", self.retain_loss)

        else:  # joint update
            forget_loss, regularization_loss = get_loss(
                model, self.ref_model, inputs, self.loss_type, self.beta
            )

            loss = (
                self.forget_coeff * forget_loss
                + self.regularization_coeff * regularization_loss
            )

        return (loss, None) if return_outputs else loss

    def get_original_parameter_shape(self):
        param_shape_dict = OrderedDict()
        if self.args.deepspeed is not None:
            raise ValueError("DeepSpeed is not supported")
        if self.args.fsdp in [""]:
            for name, param in self.model.named_parameters():
                param_shape_dict[name] = param.shape
        else:
            with FSDP.summon_full_params(self.model, writeback=False):
                for name, param in self.model.named_parameters():
                    param_shape_dict[name] = param.shape

        return param_shape_dict

    def e_prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)
        # print(config_kwargs)

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
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0

        # Disable optimizer in DeepSpeed since we are using custom optimizers
        config_kwargs["optimizer"] = {"type": None}

        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        return model


class CustomTrainerSafeAlignAlternate(CustomTrainerForgettingAlternate):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, None) if return_outputs else loss