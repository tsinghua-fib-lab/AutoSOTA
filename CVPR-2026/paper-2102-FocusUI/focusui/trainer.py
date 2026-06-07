from datetime import timedelta
from functools import wraps, partial
from typing import Optional

import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin, InitProcessGroupKwargs
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    has_length,
    is_accelerate_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import LengthGroupedSampler as HFLengthGroupedSampler
from transformers.trainer_utils import seed_worker
from transformers.utils import logging

if is_datasets_available():
    import datasets


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


class FocusUITrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        original_save = self._save
        original_save_model = self.save_model

        def modify_eos_token(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                tokenizer = self.processing_class.tokenizer
                old_config_id = self.model.config.eos_token_id
                old_eos_token = tokenizer.eos_token
                old_generation_config_eos_token_id = (
                    self.model.generation_config.eos_token_id if hasattr(self.model, "generation_config") else None
                )

                try:
                    new_eos_token_id = tokenizer.convert_tokens_to_ids("<|diff_marker|>")
                    self.model.config.eos_token_id = [new_eos_token_id]
                    tokenizer.eos_token = "<|diff_marker|>"
                    if hasattr(self.model, "generation_config"):
                        self.model.generation_config.eos_token_id = [new_eos_token_id]

                    print("Set eos token id to", new_eos_token_id)
                    print("Set eos token to", "<|diff_marker|>")
                    print("Set generation config eos token id to", [new_eos_token_id])

                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.model.config.eos_token_id = old_config_id
                    tokenizer.eos_token = old_eos_token
                    if hasattr(self.model, "generation_config") and old_generation_config_eos_token_id is not None:
                        self.model.generation_config.eos_token_id = old_generation_config_eos_token_id

                    print("Set eos token id back to", old_config_id)
                    print("Set eos token back to", old_eos_token)
                    if old_generation_config_eos_token_id is not None:
                        print("Set generation config eos token id back to", old_generation_config_eos_token_id)

            return wrapper

        self._save = modify_eos_token(original_save)
        self.save_model = modify_eos_token(original_save_model)

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))

        # create accelerator object
        dispatch_batches = getattr(self.args, "dispatch_batches", None)
        split_batches = getattr(self.args, "split_batches", None)
        self.dataloader_config = DataLoaderConfiguration(
            dispatch_batches=dispatch_batches,
            split_batches=split_batches,
        )
        self.accelerator = Accelerator(
            dataloader_config=self.dataloader_config,
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            kwargs_handlers=[accelerator_kwargs],
        )
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                        "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                        "when using FSDP."
                    )

        self.is_tp_enabled = False


        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return HFLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return HFLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
            )
        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            # dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = (
                self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None
            )

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        return dataloader

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
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
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def create_optimizer_with_different_learning_rates(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            raise NotImplementedError("Sagemaker MP is not supported for separate learning rate yet")
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            new_parameters = []
            for name, param in opt_model.named_parameters():
                if ("pointer_head" in name) or ("embed_tokens" in name):
                    new_parameters.append(name)
            rank0_print(f"new_parameters: {len(new_parameters)}")
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if ((n in decay_parameters) and (n not in new_parameters) and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if ((n not in decay_parameters) and (n not in new_parameters) and p.requires_grad)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if ((n in decay_parameters) and (n in new_parameters) and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate_new_params,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if ((n not in decay_parameters) and (n in new_parameters) and p.requires_grad)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate_new_params,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args) # {'lr': 0.0001, 'betas': (0.9, 0.999), 'eps': 1e-08}
            optimizer_kwargs.pop("lr")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if hasattr(self.args, "past_index"):
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        
        # Log extra components when available
        lm_loss = getattr(outputs, "lm_loss", None)
        pointer_loss = getattr(outputs, "pointer_loss", None)
        ps_loss = getattr(outputs, "ps_loss", None)
        actual_token_percentage = getattr(outputs, "actual_token_percentage", None)
        visual_token_percentage = getattr(outputs, "visual_token_percentage", None)

        extra_loss_log_freq = 10
        
        if self.state.global_step % extra_loss_log_freq == 0:
            logs = {}
            if lm_loss is not None:
                logs["lm_loss"] = lm_loss.detach().float().item()
            if pointer_loss is not None:
                logs["pointer_loss"] = pointer_loss.detach().float().item()
            if ps_loss is not None:
                logs["ps_loss"] = ps_loss.detach().float().item()
            if actual_token_percentage is not None:
                logs["actual_token_percentage"] = actual_token_percentage
            if visual_token_percentage is not None:
                logs["visual_token_percentage"] = visual_token_percentage
            
            for k, v in logs.items():
                if isinstance(v, float) and k != "lm_loss":
                    logs[k] = round(v, 4)
                else:
                    logs[k] = v

            if logs and dist.get_rank() == 0:
                    self.log(logs)

        return (loss, outputs) if return_outputs else loss