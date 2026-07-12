DEBUG = False

import os
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from time import time
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    HfArgumentParser,
)

if not DEBUG:
    import torch.distributed as dist

from model import LT_Tuning_Model
from utils import Config, set_seed, StageManager, load_yaml_config
from dataset import (
    get_dataset, 
    get_cot_latent_dataset, 
    get_question_dataset,
    build_thinking_strategy,
    MyCollator,
)

# Data update mode options: 'every_epoch', 'on_stage_change', 'post_stage0_once'
DATA_UPDATE_MODE = 'on_stage_change'


@dataclass
class CustomizedArguments:
    """All custom arguments for LT_Tuning training (non-HF TrainingArguments)."""
    # Project & paths
    project: str = field(default="LT_Tuning")
    name: str = field(default="LT_Tuning-experiment")
    save_path: str = field(default="models")
    save_dataset: bool = field(default=True)
    dataset_save_path: Optional[str] = field(default=None)
    train_path: str = field(default="data/gsm_train.json")
    val_path: str = field(default="data/gsm_test.json")
    
    # Model
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B")
    load_model_path: Optional[str] = field(default=None)
    use_flash_attention: bool = field(default=True)
    
    # Mode flags
    only_eval: bool = field(default=False)
    dihr: bool = field(default=True)
    no_thoughts: bool = field(default=False)
    
    # Multi-stage training
    stage_epochs: List[int] = field(default_factory=lambda: [3, 1, 6])
    labels_per_stage: List[int] = field(default_factory=lambda: [0, 6, 8])
    stage_names: List[str] = field(default_factory=lambda: ["stage0-cot", "stage1-hidden", "stage2-fusion"])
    stage_modes: List[str] = field(default_factory=lambda: ["common", "hidden_state", "soft_fusion"])
    
    # Thinking token config
    thinking_token: str = field(default="<thinking>")
    use_unk_for_thinking: bool = field(default=False)
    thinking_strategy: str = field(default="policy")
    thinking_hidden_state_layer: int = field(default=-1)
    thinking_insertion_prob: List[float] = field(default_factory=lambda: [0.0, 0.85, 0.95])
    thinking_secondary_insertion_prob: List[float] = field(default_factory=lambda: [0.0, 0.15, 0.2])
    thinking_operator_regex: str = field(default="[0-9]+|[+\\-*/=]")
    thinking_prompt_tokens: int = field(default=0)
    
    # MLP config
    thinking_use_mlp: bool = field(default=False)
    thinking_mlp_hidden_dim: int = field(default=1024)
    thinking_mlp_activation: str = field(default="gelu")
    
    # Policy thresholds
    reinforce_prob_threshold: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.2])
    reinforce_max_eval_length: int = field(default=2048)
    
    # Soft fusion params
    fusion_alpha: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.6])
    fusion_top_p: float = field(default=0.9)
    fusion_temperature: float = field(default=1.0)
    
    # Checkpoint
    resume: int = field(default=0)
    reset_optimizer: bool = field(default=False)
    save_only_improve: bool = field(default=False)
    max_epoch_checkpoints: Optional[int] = field(default=10)
    max_step_checkpoints: Optional[int] = field(default=None)
    save_every_n_steps: int = field(default=0)
    
    # Optimizer transition on stage change
    stage_optimizer_strategy: str = field(default="none")
    # Options: "none", "reset", "reset_momentum", "reduce_lr", "warmup_lr"
    stage_lr_warmup_steps: int = field(default=100)
    stage_lr_reduction_factor: float = field(default=0.1)


@dataclass
class LTTuningTrainingArguments(transformers.TrainingArguments):
    """HF TrainingArguments with custom defaults for LT_Tuning."""
    output_dir: str = field(default="models/lt_tuning")
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)
    num_train_epochs: float = field(default=10.0)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")
    bf16: bool = field(default=True)
    logging_steps: int = field(default=10)  # Log every step for loss curve
    logging_first_step: bool = field(default=True)  # Log the first step
    save_strategy: str = field(default="epoch")
    eval_strategy: str = field(default="epoch")
    report_to: List[str] = field(default_factory=lambda: ["wandb"])  # Must be list for Trainer
    remove_unused_columns: bool = field(default=False)
    seed: int = field(default=42)
    deepspeed: Optional[str] = field(default=None)


def parse_args_from_yaml(config_path: str):
    """Parse YAML config into CustomizedArguments and LTTuningTrainingArguments."""
    config_dict = load_yaml_config(config_path)
    
    # In DEBUG mode, remove deepspeed config to avoid initialization
    if DEBUG and 'deepspeed' in config_dict:
        del config_dict['deepspeed']
    
    # Split config into custom and training arguments
    training_keys = {f.name for f in LTTuningTrainingArguments.__dataclass_fields__.values()}
    custom_dict, training_dict = {}, {}
    
    for k, v in config_dict.items():
        if k in training_keys:
            training_dict[k] = v
        else:
            custom_dict[k] = v
    
    # Use HfArgumentParser for proper parsing
    parser = HfArgumentParser((CustomizedArguments, LTTuningTrainingArguments))
    custom_args, training_args = parser.parse_dict({**custom_dict, **training_dict}, allow_extra_keys=True)
    
    # Set output_dir from custom args
    training_args.output_dir = os.path.join(custom_args.save_path, custom_args.name)
    training_args.run_name = custom_args.name
    
    return custom_args, training_args


def get_rank() -> int:
    if DEBUG or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def print_rank_0(msg: str):
    if is_main_process():
        print(msg)


def load_model_and_tokenizer(custom_args: CustomizedArguments, training_args: LTTuningTrainingArguments) -> Tuple:
    """Load base model and tokenizer, setup thinking token, wrap with LT_Tuning."""
    # Determine attention implementation and dtype
    attn_impl = "flash_attention_2" if custom_args.use_flash_attention else "eager"
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    
    # Load base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            custom_args.model_name_or_path,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        print_rank_0(f"Loaded model with {attn_impl} and torch_dtype={torch_dtype}")
    except Exception as e:
        print_rank_0(f"Failed to load with {attn_impl} and torch_dtype={torch_dtype}, falling back: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            custom_args.model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(custom_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Setup thinking token
    if custom_args.use_unk_for_thinking:
        if tokenizer.unk_token_id is None:
            raise ValueError("Tokenizer has no unk_token but use_unk_for_thinking=True")
        thinking_token_id = tokenizer.unk_token_id
    else:
        newly_added = tokenizer.add_tokens([custom_args.thinking_token])
        thinking_token_id = tokenizer.convert_tokens_to_ids(custom_args.thinking_token)
        if newly_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            # Initialize thinking token embedding from a reference token
            ref_token_id = tokenizer.convert_tokens_to_ids("emm")
            if ref_token_id == -1 or ref_token_id is None:
                ref_token_id = tokenizer.eos_token_id
            embeddings = model.get_input_embeddings()
            embeddings.weight.data[thinking_token_id] = embeddings.weight.data[ref_token_id].clone()
            if hasattr(model, "lm_head") and model.lm_head.weight.shape[0] == len(tokenizer):
                model.lm_head.weight.data[thinking_token_id] = model.lm_head.weight.data[ref_token_id].clone()
            print_rank_0(f"Added thinking token '{custom_args.thinking_token}' (id={thinking_token_id})")
    
    lt_tuning_model = LT_Tuning_Model(
        base_causallm=model,
        thinking_token_id=thinking_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_thinking_mlp=custom_args.thinking_use_mlp,
        mlp_hidden_dim=custom_args.thinking_mlp_hidden_dim,
        mlp_activation=custom_args.thinking_mlp_activation,
        hidden_state_layer_index=custom_args.thinking_hidden_state_layer,
        stage_mode="common",
        fusion_alpha=custom_args.fusion_alpha[0] if custom_args.fusion_alpha else 0.5,
        fusion_top_p=custom_args.fusion_top_p,
        fusion_temperature=custom_args.fusion_temperature,
    )
    
    # Load checkpoint if specified
    if custom_args.load_model_path:
        state_dict = torch.load(custom_args.load_model_path, map_location="cpu")
        lt_tuning_model.load_state_dict(state_dict, strict=False)
        print_rank_0(f"Loaded checkpoint from {custom_args.load_model_path}")
    
    return lt_tuning_model, tokenizer, thinking_token_id


# ==============================================================================
# Callbacks
# ==============================================================================

class StageUpdateCallback(TrainerCallback):
    """Callback to handle multi-stage training: update model config and regenerate datasets."""
    
    def __init__(self, stage_manager, config, model, tokenizer, thinking_token_id,
                 train_dataset_raw, collator_factory, trainer_ref, debug_num=None,
                 save_dataset=False, dataset_save_path=None):
        self.stage_manager = stage_manager
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.thinking_token_id = thinking_token_id
        self.train_dataset_raw = train_dataset_raw
        self.collator_factory = collator_factory
        self.trainer_ref = trainer_ref  # Mutable list [trainer] to update after creation
        self.last_stage_mode = stage_manager.current_mode
        self.debug_num = debug_num  # Limit dataset size in DEBUG mode
        self.save_dataset = save_dataset
        self.dataset_save_path = dataset_save_path
        
        # Optimizer transition strategy
        self.optimizer_strategy = getattr(config, 'stage_optimizer_strategy', 'none')
        self.lr_warmup_steps = getattr(config, 'stage_lr_warmup_steps', 100)
        self.lr_reduction_factor = getattr(config, 'stage_lr_reduction_factor', 0.1)
        self.warmup_step_counter = 0
        self.original_lr = None
        self.in_warmup = False
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Check for stage change at the beginning of each epoch."""
        epoch = int(state.epoch) if state.epoch else 0
        stage_changed = self.stage_manager.update_for_epoch(epoch)
        
        if stage_changed:
            print_rank_0(f"\n[Callback] Stage changed to: {self.stage_manager.describe()}")
            self._update_model_config()
            if DATA_UPDATE_MODE == 'on_stage_change':
                self._regenerate_dataset()
            # Apply optimizer transition strategy
            self._apply_optimizer_transition()
            self.last_stage_mode = self.stage_manager.current_mode
        return control
    
    def _update_model_config(self):
        """Update model stage configuration."""
        unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        unwrapped.update_stage_config(
            stage_mode=self.config.current_stage_mode,
            fusion_alpha=self.config.fusion_alpha,
            fusion_top_p=self.config.fusion_top_p,
            fusion_temperature=self.config.fusion_temperature,
        )
    
    def _regenerate_dataset(self):
        """Regenerate training dataset for new stage."""
        print_rank_0(f"[Callback] Regenerating dataset for stage: {self.stage_manager.current_mode}")
        
        thinking_strategy = build_thinking_strategy(
            configs=self.config,
            tokenizer=self.tokenizer,
            thinking_token_id=self.thinking_token_id,
            model=self.model.base_causallm if hasattr(self.model, 'base_causallm') else self.model,
        )
        
        new_dataset = get_cot_latent_dataset(
            stage_type=self.stage_manager.current_mode,
            base_dataset=self.train_dataset_raw,
            configs=self.config,
            strategy=thinking_strategy,
            tokenizer=self.tokenizer,
            shuffle=True,
            debug_num=self.debug_num,  # Respect DEBUG mode limit
        )
        
        if self.trainer_ref[0] is not None:
            trainer = self.trainer_ref[0]
            
            # Create new collator for this stage
            new_collator = self.collator_factory(
                self.stage_manager.current_mode != "common"
            )
            
            if isinstance(trainer.train_dataset, DynamicDatasetProxy):
                trainer.train_dataset.update_dataset(new_dataset)
                print_rank_0(f"[Callback] Updated DynamicDatasetProxy for stage: {self.stage_manager.current_mode}")
            else:
                trainer.train_dataset = new_dataset
                print_rank_0(f"[Callback] WARNING: train_dataset is not DynamicDatasetProxy")
            
            if isinstance(trainer.data_collator, DynamicCollatorProxy):
                trainer.data_collator.update_collator(new_collator)
                print_rank_0(f"[Callback] Updated DynamicCollatorProxy")
            else:
                trainer.data_collator = new_collator
                print_rank_0(f"[Callback] WARNING: data_collator is not DynamicCollatorProxy")
            
            # Force garbage collection to release old dataset memory
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print_rank_0(f"[Callback] Cleared memory cache")

            print_rank_0(f"[Callback] New dataset samples:")
            # for item in new_dataset[:5]:  # Print first 5 samples
                # input_str = self.tokenizer.decode(item['input_ids'], skip_special_tokens=False)
                # print_rank_0(f"⭐⭐⭐⭐ {input_str}")
            trainer._train_dataloader = None
        
        # Save dataset if configured (check if dataset has save_to_disk method)
        if self.save_dataset and self.dataset_save_path and is_main_process():
            if hasattr(new_dataset, 'save_to_disk'):
                dataset_dir = os.path.join(self.dataset_save_path, self.stage_manager.current_label)
                os.makedirs(dataset_dir, exist_ok=True)
                new_dataset.save_to_disk(os.path.join(dataset_dir, "train"))
                print_rank_0(f"[Callback] Saved dataset to {dataset_dir}")
        
        print_rank_0(f"[Callback] Dataset regenerated: {len(new_dataset)} samples")
    
    def _apply_optimizer_transition(self):
        """Apply optimizer transition strategy when stage changes."""
        if self.trainer_ref[0] is None:
            return
        
        trainer = self.trainer_ref[0]
        if trainer.optimizer is None:
            return
        
        strategy = self.optimizer_strategy
        print_rank_0(f"[Callback] Applying optimizer transition strategy: '{strategy}'")
        
        if strategy == "reset":
            # Complete optimizer reset - recreate optimizer from scratch
            self._reset_optimizer_completely(trainer)
        
        elif strategy == "reset_momentum":
            # Reset momentum/adaptive states but keep param groups
            self._reset_optimizer_momentum(trainer)
        
        elif strategy == "reduce_lr":
            # Reduce learning rate temporarily
            self._reduce_learning_rate(trainer)
        
        elif strategy == "warmup_lr":
            # Start with low LR and warmup
            self._setup_lr_warmup(trainer)
        
        elif strategy == "none":
            print_rank_0(f"[Callback] No optimizer transition applied")
        
        else:
            print_rank_0(f"[Callback] Warning: Unknown optimizer strategy '{strategy}'")
    
    def _get_actual_optimizer(self, optimizer):
        """Unwrap optimizer to get the actual PyTorch optimizer."""
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        return optimizer
    
    def _reset_optimizer_completely(self, trainer):
        """Reset optimizer by clearing its state (simple and robust)."""
        print_rank_0("[Callback] Resetting optimizer state...")
        
        # Get the actual optimizer
        actual_optimizer = self._get_actual_optimizer(trainer.optimizer)
        
        # Simple and brutal: clear all optimizer state
        if hasattr(actual_optimizer, 'state'):
            actual_optimizer.state.clear()
            print_rank_0("[Callback] Cleared optimizer state dict")
        
        # Reset step count if it exists
        for param_group in getattr(actual_optimizer, 'param_groups', []):
            for param in param_group.get('params', []):
                if param in getattr(actual_optimizer, 'state', {}):
                    actual_optimizer.state[param].clear()
        
        print_rank_0("[Callback] Optimizer state reset complete")
    
    def _reset_optimizer_momentum(self, trainer):
        """Reset momentum/adaptive states but keep learning rates."""
        print_rank_0("[Callback] Resetting optimizer momentum states...")
        
        # Get the actual optimizer
        optimizer = self._get_actual_optimizer(trainer.optimizer)
        
        # Reset momentum states
        for param_group in getattr(optimizer, 'param_groups', []):
            for param in param_group.get('params', []):
                if hasattr(optimizer, 'state') and param in optimizer.state:
                    state = optimizer.state[param]
                    # Zero out momentum buffers but keep step count
                    if 'exp_avg' in state:
                        state['exp_avg'].zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'].zero_()
                    if 'momentum_buffer' in state:
                        state['momentum_buffer'].zero_()
        
        print_rank_0("[Callback] Momentum states reset")
    
    def _reduce_learning_rate(self, trainer):
        """Temporarily reduce learning rate."""
        print_rank_0(f"[Callback] Reducing learning rate by factor {self.lr_reduction_factor}...")
        
        # Get the actual optimizer
        optimizer = self._get_actual_optimizer(trainer.optimizer)
        
        for param_group in getattr(optimizer, 'param_groups', []):
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
            param_group['lr'] = param_group['initial_lr'] * self.lr_reduction_factor
        
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
            print_rank_0(f"[Callback] Learning rate reduced to {optimizer.param_groups[0]['lr']:.2e}")
    
    def _setup_lr_warmup(self, trainer):
        """Setup learning rate warmup."""
        print_rank_0(f"[Callback] Setting up LR warmup for {self.lr_warmup_steps} steps...")
        
        # Get the actual optimizer
        optimizer = self._get_actual_optimizer(trainer.optimizer)
        
        # Save original learning rates
        if self.original_lr is None and hasattr(optimizer, 'param_groups'):
            self.original_lr = [group['lr'] for group in optimizer.param_groups]
        
        # Set to very low initial LR
        for i, param_group in enumerate(getattr(optimizer, 'param_groups', [])):
            param_group['lr'] = self.original_lr[i] * 0.01
        
        self.warmup_step_counter = 0
        self.in_warmup = True
        
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
            print_rank_0(f"[Callback] Starting warmup from LR {optimizer.param_groups[0]['lr']:.2e}")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Handle learning rate warmup during steps."""
        if self.in_warmup and self.trainer_ref[0] is not None:
            trainer = self.trainer_ref[0]
            optimizer = self._get_actual_optimizer(trainer.optimizer)
            
            self.warmup_step_counter += 1
            
            if self.warmup_step_counter <= self.lr_warmup_steps:
                warmup_factor = self.warmup_step_counter / self.lr_warmup_steps
                current_factor = 0.01 + (1.0 - 0.01) * warmup_factor
                
                for i, param_group in enumerate(getattr(optimizer, 'param_groups', [])):
                    if self.original_lr and i < len(self.original_lr):
                        param_group['lr'] = self.original_lr[i] * current_factor
                
                if self.warmup_step_counter % 20 == 0 and hasattr(optimizer, 'param_groups'):
                    print_rank_0(f"[Callback] Warmup step {self.warmup_step_counter}/{self.lr_warmup_steps}, "
                               f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            else:
                for i, param_group in enumerate(getattr(optimizer, 'param_groups', [])):
                    if self.original_lr and i < len(self.original_lr):
                        param_group['lr'] = self.original_lr[i]
                
                self.in_warmup = False
                if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                    print_rank_0(f"[Callback] Warmup complete, LR restored to {optimizer.param_groups[0]['lr']:.2e}")
        
        return control


class WandbLoggingCallback(TrainerCallback):
    """Callback for custom wandb logging."""
    
    def __init__(self, stage_manager, config):
        self.stage_manager = stage_manager
        self.config = config
        self.dataset_update_events = []  # Buffer for dataset update messages
    
    def log_dataset_update(self, stage: str, reason: str):
        """Buffer a dataset update event to log to wandb."""
        import datetime
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "stage": stage,
            "reason": reason,
        }
        self.dataset_update_events.append(event)
        print_rank_0(f"[WandB] Dataset update logged: stage={stage}, reason={reason}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add stage info to logs."""
        if logs is not None and is_main_process():
            logs["stage"] = self.stage_manager.current_label
            logs["stage_mode"] = self.stage_manager.current_mode
            logs["stage_idx"] = self.stage_manager.stage_idx
            if isinstance(self.config.fusion_alpha, (list, tuple)):
                logs["fusion_alpha"] = self.config.fusion_alpha[self.stage_manager.stage_idx] if self.stage_manager.stage_idx < len(self.config.fusion_alpha) else self.config.fusion_alpha[-1]
            else:
                logs["fusion_alpha"] = self.config.fusion_alpha
            
            # Log any pending dataset update events
            if self.dataset_update_events:
                logs["dataset_updates"] = len(self.dataset_update_events)
                self.dataset_update_events.clear()
        return control


class GenerationEvalCallback(TrainerCallback):
    """Callback to run generation-based evaluation at the end of each epoch.
    
    Supports distributed evaluation: each GPU evaluates a subset of samples,
    then results are aggregated using all_reduce.
    
    Args:
        evaluation_strategy: 'epoch' (eval every epoch) or 'final_only' (eval only at last epoch)
        output_dir: Directory to save detailed evaluation results JSON file
        save_detailed_results: Whether to save detailed results to JSON (default: True for final_only)
    """
    
    def __init__(self, model, tokenizer, val_dataset_raw, stage_manager, 
                 thinking_token_id, trainer_ref=None, max_new_tokens=1024, debug_num=None,
                 evaluation_strategy='epoch', output_dir=None, save_detailed_results=None):
        self.model = model
        self.tokenizer = tokenizer
        self.val_dataset_raw = val_dataset_raw
        self.stage_manager = stage_manager
        self.thinking_token_id = thinking_token_id
        self.trainer_ref = trainer_ref
        self.max_new_tokens = max_new_tokens
        self.debug_num = debug_num
        self.best_acc = 0.0
        self.evaluation_strategy = evaluation_strategy
        self.output_dir = output_dir
        # Auto-enable detailed results for final_only strategy
        self.save_detailed_results = save_detailed_results if save_detailed_results is not None else (evaluation_strategy == 'final_only')
        
        # Prepare ground truth answers
        self.questions = [d["question"] for d in val_dataset_raw]
        self.answers = [d["answer"].replace(",", "").strip() for d in val_dataset_raw]
        self.cots = [d.get("reasoning_chain", "") for d in val_dataset_raw]
    
    def _get_world_size(self) -> int:
        """Get number of processes in distributed setting."""
        if DEBUG or not dist.is_initialized():
            return 1
        return dist.get_world_size()
    
    def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum tensor across all processes."""
        if DEBUG or not dist.is_initialized():
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor
    
    def _gather_results_from_all_ranks(self, local_results: List[Dict]) -> List[Dict]:
        """Gather results from all ranks to rank 0."""
        rank = get_rank()
        world_size = self._get_world_size()
        
        if world_size == 1:
            return local_results
        
        if not DEBUG and dist.is_initialized():
            # Gather all results to rank 0
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_results)
            
            if rank == 0:
                # Flatten the list of lists
                all_results = []
                for results in gathered_results:
                    all_results.extend(results)
                return all_results
        
        return local_results
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Run distributed generation evaluation at the end of each epoch."""
        return control # TEMP DISABLE
        epoch = int(state.epoch) if state.epoch else 0
        total_epochs = int(args.num_train_epochs) if args.num_train_epochs else 1
        is_final_epoch = (epoch >= total_epochs)
        
        # Check if we should run evaluation based on strategy
        if self.evaluation_strategy == 'final_only' and not is_final_epoch:
            # Skip evaluation if not final epoch
            return control
        
        print_rank_0(f"\n===== Generation Evaluation (Epoch {epoch}/{total_epochs}) =====")
        begin_time = time()
        # Get model in eval mode
        unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        unwrapped.eval()
        device = next(unwrapped.parameters()).device
        
        # Prepare eval dataset
        eval_dataset = get_question_dataset(
            stage_type=self.stage_manager.current_mode,
            base_dataset_valid=self.val_dataset_raw,
            tokenizer=self.tokenizer,
        )
        
        # Limit samples in debug mode
        total_samples = len(eval_dataset)
        if self.debug_num:
            total_samples = min(total_samples, self.debug_num)
        
        # Distributed: split samples across GPUs
        rank = get_rank()
        world_size = self._get_world_size()
        
        # Calculate indices for this rank
        samples_per_rank = (total_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_rank
        end_idx = min(start_idx + samples_per_rank, total_samples)
        local_indices = list(range(start_idx, end_idx))
        
        print_rank_0(f"World size: {world_size}, evaluating {total_samples} samples")
        if world_size > 1:
            print(f"[Rank {rank}] Evaluating samples {start_idx} to {end_idx-1} ({len(local_indices)} samples)")
        
        local_correct = 0
        local_correct_cot = 0
        local_total = len(local_indices)
        local_detailed_results = []  # Store detailed results for JSON export
        
        from tqdm import tqdm
        pbar = tqdm(local_indices, desc=f"[Rank {rank}] Evaluating", disable=(rank != 0))
        
        with torch.no_grad():
            for idx in pbar:
                sample = eval_dataset[idx]
                input_ids = torch.tensor([sample["input_ids"]], device=device)
                attention_mask = torch.tensor([sample["attention_mask"]], device=device)
                
                # Generate
                outputs = unwrapped.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode and extract answer
                text_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                if "<thinking>" in text_output:
                    print_rank_0(f"[Rank {rank}] Info: '<thinking>' token found in output for sample {idx}")
                # Extract answer after "###" or "#"
                answer_output = text_output.split("#")[-1].replace(",", "").replace("<thinking>", "").strip()
                
                # Compare with ground truth
                gt_answer = self.answers[idx]
                is_correct = (answer_output == gt_answer)
                if is_correct:
                    local_correct += 1
                
                # Store detailed result if needed
                if self.save_detailed_results:
                    # Get input text
                    input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    # Get output tokens (only the generated part)
                    output_tokens = outputs[0][len(input_ids[0]):].tolist()
                    
                    detailed_result = {
                        "sample_idx": idx,
                        "question": self.questions[idx],
                        "ground_truth_answer": gt_answer,
                        "predicted_answer": answer_output,
                        "is_correct": is_correct,
                        "input_text": input_text,
                        "input_token_ids": input_ids[0].tolist(),
                        "output_text": text_output,
                        "output_token_ids": outputs[0].tolist(),
                        "generated_token_ids": output_tokens,
                        "rank": rank,
                    }
                    local_detailed_results.append(detailed_result)
                
                # CoT match (optional)
                # cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                # if cot_output == self.cots[idx]:
                #     local_correct_cot += 1
                
                # Print some examples (only rank 0)
                if rank == 0 and idx < 10:
                    print_rank_0(f"\n[Sample {idx}] GT: '{gt_answer}' | Pred: '{answer_output}'")
                    print_rank_0(f"Full output: {text_output}")
                
                if rank == 0:
                    pbar.set_description(f"[Rank 0] Acc: {local_correct}/{idx - start_idx + 1}")
        
        # Aggregate results across all ranks
        correct_tensor = torch.tensor([local_correct], device=device, dtype=torch.float32)
        # correct_cot_tensor = torch.tensor([local_correct_cot], device=device, dtype=torch.float32)
        total_tensor = torch.tensor([local_total], device=device, dtype=torch.float32)
        
        correct_tensor = self._all_reduce_sum(correct_tensor)
        # correct_cot_tensor = self._all_reduce_sum(correct_cot_tensor)
        total_tensor = self._all_reduce_sum(total_tensor)
        
        total_correct = int(correct_tensor.item())
        # total_correct_cot = int(correct_cot_tensor.item())
        total_evaluated = int(total_tensor.item())
        
        acc = total_correct / total_evaluated if total_evaluated > 0 else 0
        # cot_em = total_correct_cot / total_evaluated if total_evaluated > 0 else 0
        
        print_rank_0(f"Epoch {epoch} - Accuracy: {total_correct}/{total_evaluated} = {acc:.4f}")
        # print_rank_0(f"Epoch {epoch} - CoT EM: {total_correct_cot}/{total_evaluated} = {cot_em:.4f}")
        
        # Save detailed results to JSON if enabled
        if self.save_detailed_results and self.output_dir:
            # Gather all results from all ranks to rank 0
            all_results = self._gather_results_from_all_ranks(local_detailed_results)
            
            if is_main_process():
                # Save to JSON file
                import json
                os.makedirs(self.output_dir, exist_ok=True)
                output_file = os.path.join(self.output_dir, f"eval_results_epoch{epoch}.json")
                
                # Sort by sample_idx for consistent ordering
                all_results.sort(key=lambda x: x["sample_idx"])
                
                # Create summary
                summary = {
                    "epoch": epoch,
                    "total_samples": total_evaluated,
                    "correct": total_correct,
                    "accuracy": acc,
                    "stage": self.stage_manager.current_label,
                    "stage_mode": self.stage_manager.current_mode,
                    "evaluation_strategy": self.evaluation_strategy,
                }
                
                output_data = {
                    "summary": summary,
                    "detailed_results": all_results,
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print_rank_0(f"Detailed results saved to: {output_file}")
                print_rank_0(f"Summary: {summary}")
        
        print_rank_0("===== End Evaluation =====\n")
        
        # Log to wandb via trainer's log method
        if self.trainer_ref is not None and self.trainer_ref[0] is not None:
            trainer = self.trainer_ref[0]
            trainer.log({
                "eval/generation_acc": acc,
                # "eval/cot_em": cot_em,
                "eval/best_acc": max(acc, self.best_acc),
            })
        
        # Update best accuracy
        if acc > self.best_acc:
            self.best_acc = acc
            print_rank_0(f"New best accuracy: {acc:.4f}")
        
        # Set model back to training mode
        unwrapped.train()
        time_elapsed = time() - begin_time
        print_rank_0(f"Evaluation completed in {time_elapsed:.2f} seconds")
        return control


def create_collator_factory(tokenizer, thinking_token_id):
    """Create a factory function for data collators."""
    def factory(use_thinking: bool):
        return MyCollator(
            tokenizer=tokenizer,
            thinking_id=thinking_token_id if use_thinking else None,
            label_pad_token_id=-100,
        )
    return factory


class DynamicDatasetProxy(torch.utils.data.Dataset):
    """A proxy dataset that delegates to an underlying dataset which can be swapped at runtime.
    
    This solves the problem of HuggingFace Trainer creating DataLoader once at the start
    of training and not updating it. By using this proxy, we can change the underlying
    dataset without needing to recreate the DataLoader.
    """
    
    def __init__(self, initial_dataset):
        self._dataset = initial_dataset
        self._version = 0  # Track updates for debugging
    
    def update_dataset(self, new_dataset):
        """Update the underlying dataset. Thread-safe for single-writer scenarios."""
        self._dataset = new_dataset
        self._version += 1
        print_rank_0(f"[DynamicDatasetProxy] Dataset updated to version {self._version}, size={len(new_dataset)}")
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        return self._dataset[idx]
    
    @property
    def current_dataset(self):
        return self._dataset
    
    @property
    def version(self):
        return self._version
    
    # Forward common dataset attributes
    def __getattr__(self, name):
        # Avoid infinite recursion for _dataset
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._dataset, name)


class DynamicCollatorProxy:
    """A proxy collator that delegates to an underlying collator which can be swapped at runtime."""
    
    def __init__(self, initial_collator):
        self._collator = initial_collator
    
    def update_collator(self, new_collator):
        """Update the underlying collator."""
        self._collator = new_collator
        print_rank_0(f"[DynamicCollatorProxy] Collator updated")
    
    def __call__(self, features):
        return self._collator(features)


class LTTuningTrainer(Trainer):
    """Custom Trainer for LT_Tuning model with special forward handling."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to handle LT_Tuning model outputs."""
        # Remove idx from inputs if present
        inputs = {k: v for k, v in inputs.items() if k != "idx"}

        # ====== 🔍 DEBUG: decode inputs to text ======
        # if "input_ids" in inputs:
        #     input_ids = inputs["input_ids"]

        #     # batch-safe
        #     decoded = self.tokenizer.batch_decode(
        #         input_ids,
        #         skip_special_tokens=True
        #     )

        #     for i, text in enumerate(decoded):

        #         print(f"[LTTuningTrainer: Rank{self.args.local_rank}] input[{i}]:\n{text}\n")
        # ===========================================

        # Forward pass
        outputs = model(**inputs)
        
        # LT_Tuning returns namedtuple with loss
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        return (loss, outputs) if return_outputs else loss


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="LT_Tuning Training")
    parser.add_argument("config_file", type=str, help="Path to YAML config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (passed by DeepSpeed)")
    args = parser.parse_args()
    
    # Parse arguments from YAML
    custom_args, training_args = parse_args_from_yaml(args.config_file)
    
    # Pass local_rank from command line to training_args (for DeepSpeed)
    if args.local_rank != -1:
        training_args.local_rank = args.local_rank
    
    # Set seed
    set_seed(training_args.seed)
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Setup wandb environment variables for better logging
    if not DEBUG and is_main_process():
        os.environ.setdefault("WANDB_PROJECT", custom_args.project)
        os.environ.setdefault("WANDB_WATCH", "all")  # Log gradients and model topology
        os.environ.setdefault("WANDB_LOG_MODEL", "false")  # Don't auto-upload model artifacts
        os.environ.setdefault("WANDB_NAME", custom_args.name)
    
    # Convert to Config object for StageManager compatibility
    config = Config({**vars(custom_args), **{k: v for k, v in vars(training_args).items()}})
    
    # Initialize StageManager
    stage_manager = StageManager(config)
    print_rank_0(f"Initialized StageManager: {stage_manager.describe()}")
    
    # =========== Part 2: Load Model ===========
    model, tokenizer, thinking_token_id = load_model_and_tokenizer(custom_args, training_args)
    print_rank_0(f"Model loaded: {type(model).__name__}, thinking_token_id={thinking_token_id}, dtype={next(model.parameters()).dtype}")
    
    # Update model stage config from stage_manager
    model.update_stage_config(
        stage_mode=stage_manager.current_mode,
        fusion_alpha=config.fusion_alpha,
        fusion_top_p=config.fusion_top_p,
        fusion_temperature=config.fusion_temperature,
    )
    
    # =========== Part 3: Load Datasets ===========
    print_rank_0("Loading datasets...")
    
    # Load raw datasets
    train_dataset_raw = get_dataset(custom_args.train_path, dataset_name="gsm8k")
    val_dataset_raw = get_dataset(custom_args.val_path, dataset_name="gsm8k")
    print_rank_0(f"Raw datasets loaded: train={len(train_dataset_raw)}, val={len(val_dataset_raw)}")
    
    # Build thinking strategy (None for stage0/common mode)
    thinking_strategy = None
    if stage_manager.current_mode != "common":
        thinking_strategy = build_thinking_strategy(
            configs=config,
            tokenizer=tokenizer,
            thinking_token_id=thinking_token_id,
            model=model.base_causallm if hasattr(model, 'base_causallm') else model,
        )
        print_rank_0(f"Built thinking strategy: {type(thinking_strategy).__name__}")
    
    # Process training dataset with thinking tokens
    train_dataset = get_cot_latent_dataset(
        stage_type=stage_manager.current_mode,
        base_dataset=train_dataset_raw,
        configs=config,
        strategy=thinking_strategy,
        tokenizer=tokenizer,
        shuffle=True,
        debug_num=100 if DEBUG else None,  # Limit samples in debug mode
    )
    
    # Process validation dataset (question only for generation)
    val_dataset = get_question_dataset(
        stage_type=stage_manager.current_mode,
        base_dataset_valid=val_dataset_raw,
        tokenizer=tokenizer,
    )
    print_rank_0(f"Processed datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # Save dataset if configured
    if custom_args.save_dataset and custom_args.dataset_save_path and is_main_process():
        dataset_dir = os.path.join(custom_args.dataset_save_path, stage_manager.current_label)
        os.makedirs(dataset_dir, exist_ok=True)
        train_dataset.save_to_disk(os.path.join(dataset_dir, "train"))
        val_dataset.save_to_disk(os.path.join(dataset_dir, "val"))
        print_rank_0(f"Saved datasets to {dataset_dir}")
    
    # Create data collator
    data_collator = MyCollator(
        tokenizer=tokenizer,
        thinking_id=thinking_token_id if stage_manager.current_mode != "common" else None,
        label_pad_token_id=-100,
    )
    
    # ================== WRAP WITH PROXIES ==================
    # Wrap dataset and collator with dynamic proxies so they can be
    # updated at runtime during stage changes without recreating DataLoader
    # =========================================================
    train_dataset = DynamicDatasetProxy(train_dataset)
    data_collator = DynamicCollatorProxy(data_collator)
    print_rank_0(f"Wrapped train_dataset with DynamicDatasetProxy and data_collator with DynamicCollatorProxy")
    
    # =========== Debug: Test data loading ===========
    if DEBUG:
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            train_dataset, 
            batch_size=2, 
            collate_fn=data_collator,
            shuffle=False,
        )
        batch = next(iter(test_loader))
        print_rank_0("\n===== Debug: Sample Batch =====")
        print_rank_0(f"Batch keys: {batch.keys()}")
        print_rank_0(f"input_ids shape: {batch['input_ids'].shape}")
        print_rank_0(f"labels shape: {batch['labels'].shape}")
        print_rank_0(f"attention_mask shape: {batch['attention_mask'].shape}")
        # Show first sample
        sample_ids = batch['input_ids'][0]
        sample_labels = batch['labels'][0]
        print_rank_0(f"\nFirst sample tokens: {tokenizer.decode(sample_ids[:50])}...")
        label_tokens = [t for t in sample_labels.tolist() if t != -100]
        print_rank_0(f"First sample labels: {tokenizer.decode(label_tokens[:30])}...")
        print_rank_0("===== End Debug =====\n")
    
    # =========== Part 4: Callbacks ===========
    trainer_ref = [None]  # Mutable container to pass trainer reference
    collator_factory = create_collator_factory(tokenizer, thinking_token_id)
    debug_num = 100 if DEBUG else None  # Consistent limit for DEBUG mode
    
    stage_callback = StageUpdateCallback(
        stage_manager=stage_manager,
        config=config,
        model=model,
        tokenizer=tokenizer,
        thinking_token_id=thinking_token_id,
        train_dataset_raw=train_dataset_raw,
        collator_factory=collator_factory,
        trainer_ref=trainer_ref,
        debug_num=debug_num,
        save_dataset=custom_args.save_dataset,
        dataset_save_path=custom_args.dataset_save_path,
    )
    
    wandb_callback = WandbLoggingCallback(
        stage_manager=stage_manager,
        config=config,
    )
    callbacks = [stage_callback, wandb_callback]
    # Generation evaluation callback
    # If eval_strategy is 'no', use 'final_only' to evaluate at the end with detailed results
    # If eval_strategy is 'epoch', evaluate every epoch
    if training_args.eval_strategy == "no":
        evaluation_strategy = 'final_only'
        print_rank_0("eval_strategy is 'no', using 'final_only' - will evaluate only at final epoch with detailed JSON output")
    else:
        evaluation_strategy = 'epoch'
        print_rank_0(f"eval_strategy is '{training_args.eval_strategy}', using 'epoch' - will evaluate every epoch")
    
    eval_callback = GenerationEvalCallback(
        model=model,
        tokenizer=tokenizer,
        val_dataset_raw=val_dataset_raw,
        stage_manager=stage_manager,
        thinking_token_id=thinking_token_id,
        trainer_ref=trainer_ref,
        max_new_tokens=1024,
        debug_num=10 if DEBUG else None,  # Fewer samples for eval in DEBUG mode
        evaluation_strategy=evaluation_strategy,
        output_dir=training_args.output_dir,
        save_detailed_results=None,  # Auto-enable for final_only
    )
    callbacks.append(eval_callback)
    
    print_rank_0(f"Created {len(callbacks)} callbacks.")
    
    # =========== Part 5: Training ===========
    # Disable wandb and deepspeed in DEBUG mode
    if DEBUG:
        training_args.report_to = []
        training_args.deepspeed = None  # Disable DeepSpeed in DEBUG mode
        training_args.logging_steps = 10
        training_args.save_strategy = "no"
        training_args.eval_strategy = "no"
        print_rank_0(f"DEBUG mode: wandb/deepspeed disabled, running {int(training_args.num_train_epochs)} epochs")
    
    # Create trainer
    trainer = LTTuningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        tokenizer=tokenizer,
    )
    
    # Set trainer reference for callbacks
    trainer_ref[0] = trainer
    
    # Run training or evaluation
    if custom_args.only_eval:
        print_rank_0("Running evaluation only...")
        metrics = trainer.evaluate()
        print_rank_0(f"Evaluation results: {metrics}")
    else:
        print_rank_0("Starting training...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        # Save final model (save base_causallm to avoid shared tensor issues)
        if is_main_process():
            print_rank_0(f"Saving final model to {training_args.output_dir}")
            unwrapped_model = model.module if hasattr(model, 'module') else model
            # Save the base causal LM instead of the wrapper
            unwrapped_model.base_causallm.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
    
    print_rank_0("Done!")


if __name__ == "__main__":
    main()
