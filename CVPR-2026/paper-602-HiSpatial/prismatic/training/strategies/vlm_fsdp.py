"""
fsdp.py

Core class definition for a strategy implementing Torch native Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).
"""

from datetime import datetime
import json
import math
from collections import OrderedDict
from functools import partial
import os
from pathlib import Path
import threading
from typing import Callable, Optional, Union, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup


from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.vlm_base_strategy import TrainingStrategy

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import OrderedDict

from collections import defaultdict
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLMMetrics,VLMMetricsWeb
import copy
import torch   
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

  
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def _get_constant_then_cosine_lr_lambda(
    current_step: int, *, num_constant_steps: int, num_training_steps: int, num_cycles: float
):
    """
    Piecewise LR multiplier:
      - [0, num_constant_steps): 1.0  (保持 base lr 不变)
      - [num_constant_steps, num_training_steps]: 余弦衰减到 0
    """
    if current_step < num_constant_steps:
        return 1.0

    denom = max(1, num_training_steps - num_constant_steps)
    progress = float(current_step - num_constant_steps) / float(denom)

    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_constant_then_cosine_schedule(
    optimizer: Optimizer,
    num_constant_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule where the learning rate is kept constant at the optimizer's initial lr
    for `num_constant_steps` steps, then decreases to 0 following a cosine function.

    Args:
        optimizer (`torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_constant_steps (`int`):
            The number of steps for the initial constant phase (keeps base lr).
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            Number of cosine waves in the decay phase (0.5 = half-cosine, i.e., from 1 down to 0).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_lambda = partial(
        _get_constant_then_cosine_lr_lambda,
        num_constant_steps=num_constant_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class VLMFSDPStrategy(TrainingStrategy):
    def __init__(
        self,
        vlm,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        sharding_strategy: str = "shard-grad-op",
        state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
        module_lr: Optional[Dict[str, float]] = None,
        module_tag=None,
    ) -> None:
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            module_lr=module_lr,
        )

        # FSDP-Specific Parameters
        if sharding_strategy == "shard-grad-op":
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif sharding_strategy == "full-shard":
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise ValueError(f"FSDP Sharding Strategy {sharding_strategy} is not supported!")

        assert state_dict_type == StateDictType.FULL_STATE_DICT, "Sharded state saving is not yet implemented!"
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        self.fsdp_save_optimizer_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

        self.module_tag = module_tag


                    
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, FSDP), "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"
        
        checkpoint_name = f"epoch={epoch}-step={global_step}.ckpt"
        checkpoint_dir = run_dir / "checkpoints"/ checkpoint_name
        if overwatch.is_rank_zero():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        def save_with_time(state_dict, path):
            overwatch.info(f"Saving state dict to {path} start at {datetime.now()}")
            torch.save(state_dict, path)
            overwatch.info(f"Saving state dict to {path} end at {datetime.now()}")            
        
        # Summon Full State Dictionary =>> Reconstitute from Shards
        with FSDP.state_dict_type(self.vlm, self.fsdp_state_dict_type, self.fsdp_save_policy, self.fsdp_save_optimizer_policy):
            model_state = self.vlm.state_dict()
            optim_state = FSDP.optim_state_dict(self.vlm, self.optimizer)
            meta_state = {
                "epoch": epoch,
                "global_step": global_step
            }
            if overwatch.is_rank_zero():
                with open(checkpoint_dir / "meta.json", "w") as f:
                    json.dump(meta_state, f)

            dist.barrier()
            if overwatch.is_rank_zero():
                threading.Thread(target=save_with_time, args=(model_state, checkpoint_dir / 'weights.pt')).start()
                threading.Thread(target=save_with_time, args=(optim_state, checkpoint_dir / 'optimizer.pt')).start()
                threading.Thread(target=save_with_time, args=(self.lr_scheduler.state_dict(), checkpoint_dir / 'scheduler.pt')).start()
                
            dist.barrier()
                # TODO (siddk) :: This breaks w/ Sagemaker default permissions (root vs. <user>)... skip?
                # shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    def load_optimizer_and_scheduler(self, checkpoint_folder: str) -> None:
        """Load a checkpoint from the specified `checkpoint_path`."""
        assert isinstance(self.vlm, FSDP), "FSDPStrategy.load_optimizer_and_scheduler assumes VLM is already wrapped in FSDP!"      
        checkpoint_folder = Path(checkpoint_folder)
        optimizer_path = checkpoint_folder / "optimizer.pt"
        scheduler_path = checkpoint_folder / "scheduler.pt"
        meta_path = checkpoint_folder / "meta.json"
        if not optimizer_path.exists():
            overwatch.warning(f"Optimizer checkpoint not found at {optimizer_path}!")
            return
        # Load Checkpoint =>> Note that FSDP will automatically handle device placement!
        optim_state_dict = torch.load(optimizer_path, map_location="cpu")
        with FSDP.state_dict_type(self.vlm, self.fsdp_state_dict_type, FullStateDictConfig(offload_to_cpu=True, rank0_only=False), FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            optim_state_dict = FSDP.optim_state_dict_to_load(self.vlm, self.optimizer, optim_state_dict)
            self.optimizer.load_state_dict(optim_state_dict)
        overwatch.info(f"Loaded optimizer state dict from {optimizer_path}")

        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            global_step = meta.get("global_step", 0)
        else:
            raise FileNotFoundError(f"[RESUME ERROR] Meta file not found at {meta_path}.")
        
        optim_step = global_step // self.grad_accumulation_steps

        num_warmup_steps = int(self.max_steps * self.warmup_ratio)
        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.max_steps,last_epoch=optim_step - 1,
            )
        elif self.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        elif self.lr_scheduler_type == "constant+cosine-decay":
            num_constant_steps = int(self.max_steps * self.warmup_ratio)
            self.lr_scheduler = get_constant_then_cosine_schedule(
                self.optimizer,
                num_constant_steps=num_constant_steps,
                num_training_steps=self.max_steps,
                num_cycles=0.5, 
                last_epoch=optim_step - 1,
            )

        else:
            raise ValueError(...)

        
        
    def run_setup(self, run_dir: Path, 
                  n_train_examples: int, 
                  auto_wrap_policy_modules,
                  checkpointing_policy_modules,
                  ) -> None:
       pass

    def clip_grad_norm(self) -> None:
        # Note =>> FSDP uses a custom `clip_grad_norm_` function; requires *uniform grad dtype*
        self.vlm.clip_grad_norm_(max_norm=self.max_grad_norm)

    def run_training(
        self,
        dataloader,
        metrics: VLMMetrics,
        save_interval: int = 2500,
        start_epoch: int = 0,
        start_global_step: int = 0,
        save_full_model: bool = True,
        tokenizer=None,
    ) -> None:
        pass

    # ------------------------------------------------
    #  iterable dataset FSDP init
    # ------------------------------------------------
    def run_setup_iterable(
        self,
        run_dir: Path,
        *,
        # At least one of max_steps or steps_per_epoch must be specified
        max_steps: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        # Others remain unchanged
        auto_wrap_policy_modules=None,
        checkpointing_policy_modules=None,
    ) -> None:
        """
        Setup for iterable-style dataset.

        - If `max_steps` is provided, total training steps are fixed; `steps_per_epoch` is optional.
        - If `max_steps` is not provided, `steps_per_epoch` must be given and total steps = steps_per_epoch × epochs.
        """
        # ------------------------------------------------
        # 0. Calculate total training / warm-up steps
        # ------------------------------------------------
        if max_steps is not None:
            num_training_steps = max_steps
            self.max_steps = max_steps
            # To control epoch progress bar, use steps_per_epoch if provided; otherwise, use a very large value
            if steps_per_epoch is not None:
                self.epochs = math.ceil(max_steps / steps_per_epoch)
            else:
                self.epochs = 10 ** 9
        elif steps_per_epoch is not None:
            num_training_steps = steps_per_epoch * self.epochs
        else:
            raise ValueError("WebDataset requires specifying at least one of `max_steps` or `steps_per_epoch`.")

        # ------------------------------------------------
        # 1. FSDP wrapping policy
        # ------------------------------------------------
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy_modules)

        # ------------------------------------------------
        # 2. Mixed precision policy
        # ------------------------------------------------
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_buffer_dtype,
                buffer_dtype=reduce_buffer_dtype,
            )
        else:
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

        # ------------------------------------------------
        # 3. Build FSDP model
        # ------------------------------------------------
        self.vlm = FSDP(
            self.vlm,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )

        # ------------------------------------------------
        # 4. Gradient Checkpointing
        # ------------------------------------------------
        if self.enable_gradient_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def check_fn(m):
                if isinstance(checkpointing_policy_modules, (list, set, tuple)):
                    return any(isinstance(m, t) for t in checkpointing_policy_modules)
                return isinstance(m, checkpointing_policy_modules)

            apply_activation_checkpointing(self.vlm, non_reentrant_wrapper, check_fn)

        dist.barrier()

        # ------------------------------------------------
        # 5. Optimizer & LR Scheduler — full three-branch logic
        # ------------------------------------------------
        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # ---- 1) Linear warm-up + cosine decay ----
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups = [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps, num_training_steps
            )
            # Cosine scheduler usually starts with lr = 0
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.0
        
        elif self.lr_scheduler_type == "constant+cosine-decay":
            # ---- 1.5) Constant + cosine decay ----
            num_warmup_steps = 0
            num_constant_steps = int(num_training_steps * self.warmup_ratio)
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups = [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_then_cosine_schedule(
                self.optimizer,
                num_constant_steps=num_constant_steps,
                num_training_steps=num_training_steps,
            )

        elif self.lr_scheduler_type == "constant" and self.module_tag is None:
            num_warmup_steps = 0
            # ---- 2) constant, no module_tag ----
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups = [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        elif self.lr_scheduler_type == "constant" and self.module_tag is not None:
            num_warmup_steps = 0
            # ---- 3) constant + module-wise learning rate ----
            from collections import defaultdict

            decay, no_decay = defaultdict(list), defaultdict(list)
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue
                tag = self.module_tag(name)
                bucket = decay if (param.ndim > 1 and not name.endswith(".bias")) else no_decay
                bucket[tag].append(param)

            param_groups = []
            for tag in set(decay) | set(no_decay):
                lr = self.module_lr.get(tag, self.learning_rate)
                if decay[tag]:
                    param_groups.append(
                        {"params": decay[tag], "weight_decay": self.weight_decay, "lr": lr}
                    )
                if no_decay[tag]:
                    param_groups.append(
                        {"params": no_decay[tag], "weight_decay": 0.0, "lr": lr}
                    )

            self.optimizer = AdamW(param_groups)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        else:
            raise ValueError(f"Learning Rate Schedule `{self.lr_scheduler_type}` is not supported")

        # ------------------------------------------------
        # 6. Logging
        # ------------------------------------------------
        overwatch.info(
            "\n===== FSDP  WebDataset  Setup Complete =====\n"
            f"| Global Batch Size          : {self.global_batch_size}\n"
            f"| Per-Device Batch Size      : {self.per_device_batch_size}\n"
            f"| World Size                 : {overwatch.world_size()}\n"
            f"| Grad Acc Steps             : {self.grad_accumulation_steps}\n"
            f"| Mixed Precision            : {self.enable_mixed_precision_training}\n"
            f"| Warm-up Steps ({self.warmup_ratio}) : {num_warmup_steps}\n"
            f"| Total Training Steps       : {num_training_steps}\n"
            f"| Logical Epochs             : {self.epochs}\n"
            "========================================\n"
        )


    def run_training_iterable(
        self,
        dataloader,  # iterable-style: WebDataset WebLoader
        metrics: VLMMetricsWeb,
        save_interval: int = 2500,
        start_global_step: int = 0,
        save_full_model: bool = True,
        tokenizer=None,
    ) -> None:
        """Training loop adapted for iterable-style WebDataset loaders."""

        self.vlm.train()
        metrics.global_step = start_global_step

        epoch = 0
        dataloader_iter = iter(dataloader)
        self.vlm.train()
        with tqdm(
            total=self.max_steps,
            desc=metrics.get_status(),
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            
            while True:
                if self.max_steps is not None and metrics.global_step >= self.max_steps * self.grad_accumulation_steps:
                    overwatch.info("Max steps reached, ending training.")
                    return

                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    epoch += 1
                    dataloader_iter = iter(dataloader) # new epoch, reset iterator
                    continue     
                except Exception as e:
                    overwatch.warning(f"Error loading batch: {e}")
                    continue

                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    prediction = self.vlm.forward(**batch, use_cache=False)
                    # print(f"decoded prompt: {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)}")
                    loss = prediction["loss"]

                metrics.commit(loss=loss)

                normalized_loss = loss / self.grad_accumulation_steps
                normalized_loss.backward()

                # ----------- Log gradient stats -----------
                total_grad_norm = 0.0
                grad_std_list = []

                for name, param in self.vlm.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.detach()
                        grad_norm = grad.norm(2).item()
                        total_grad_norm += grad_norm ** 2  # Accumulate squared norms
                        grad_std = grad.std().item()
                        grad_std_list.append(grad_std)

                total_grad_norm = total_grad_norm ** 0.5
                avg_grad_std = np.mean(grad_std_list) if grad_std_list else 0.0
                log_grad_std = np.log10(avg_grad_std + 1e-8)

                metrics.commit(grad_norm=total_grad_norm, grad_log_std=log_grad_std)

                

                if (metrics.global_step + 1) % self.grad_accumulation_steps == 0:
                    self.clip_grad_norm()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    metrics.commit(
                        update_step_time=True,
                        global_step=metrics.global_step + 1,
                        lr=self.lr_scheduler.get_last_lr()[0],
                        epoch=epoch,
                    )
                    status = metrics.push()

                    if (metrics.global_step % save_interval == 0):
                        self.save_checkpoint(
                            run_dir=metrics.run_dir,
                            global_step=metrics.global_step,
                            epoch=epoch,
                            train_loss=loss.item(),
                            only_trainable=not save_full_model
                        )
                        dist.barrier()
                

                    progress.set_description(status)
                    progress.update()
                else:
                    metrics.commit(global_step=metrics.global_step + 1)


                    