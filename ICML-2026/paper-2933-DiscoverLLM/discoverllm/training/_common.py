"""
Shared scaffolding for the trainers in :mod:`discoverllm.training.trainers`.

The four trainer scripts (sft, offline_dpo, online_dpo, grpo) share most of
their boilerplate: CLI flags, model + LoRA loading, BNB quantization, the
DeepSpeed ZeRO-2 config, distributed-training init, and W&B setup. This
module collects all of it so each trainer reduces to its own algorithm-
specific call into TRL.

The ``parse_args()`` of each trainer typically does:

    p = make_base_argparser("my-trainer")
    p.add_argument("--my_extra", ...)
    return apply_config_file_overrides(p.parse_args())
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def make_base_argparser(prog: str) -> argparse.ArgumentParser:
    """ArgumentParser pre-populated with the flags shared by every trainer."""
    p = argparse.ArgumentParser(prog)

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)
    p.add_argument("--eval_ratio", type=float, default=0.1)

    # Base / adapter models
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--peft_r", type=int, default=32)
    p.add_argument("--peft_alpha", type=int, default=16)
    p.add_argument("--peft_dropout", type=float, default=0.1)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    # Optim & schedule
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=1)

    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)
    p.add_argument(
        "--system_prompt_type",
        type=str,
        default=None,
        choices=["ours"],
        help="Which system prompt to bake into the dataset "
             "(see discoverllm/training/prompts/).",
    )

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity", type=str)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hf_org", type=str)

    # Optional JSON/YAML override
    p.add_argument("--config_file", type=str)

    return p


def apply_config_file_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """If ``--config_file`` was passed, merge its keys into ``args``."""
    if not getattr(args, "config_file", None):
        return args
    path = args.config_file
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            override = json.load(f)
        else:
            import yaml  # only needed for YAML configs

            override = yaml.safe_load(f)
    for key, value in (override or {}).items():
        setattr(args, key, value)
    return args


# --------------------------------------------------------------------------- #
# Distributed                                                                 #
# --------------------------------------------------------------------------- #
def init_distributed() -> int:
    """``dist.init_process_group`` + bind this rank to its CUDA device.

    Reads ``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` /
    ``MASTER_PORT`` from the environment, so the trainer is normally launched
    via ``torchrun`` (or ``accelerate launch``)::

        torchrun --standalone --nproc_per_node 1 \\
            -m discoverllm.training.trainers.sft ...

    For a quick single-GPU smoke run, plain ``python -m
    discoverllm.training.trainers.*`` also works: when ``LOCAL_RANK`` is
    unset we synthesize a 1-process world and bind to a free port.

    For Online DPO / GRPO under torchrun, additionally export
    ``MASTER_PORT=auto`` so vLLM's in-process ``init_process_group`` doesn't
    fight torchrun for port 29500.
    """
    if "LOCAL_RANK" not in os.environ:
        # Single-process fallback: pick a free port and seed the standard
        # torchrun env vars so HuggingFace + DeepSpeed + vLLM see a coherent
        # 1-rank world without the user having to wrap in torchrun.
        import socket

        with socket.socket() as s:
            s.bind(("", 0))
            free_port = s.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(free_port))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method=None)
    torch.cuda.set_device(local_rank)
    dist.barrier()
    return local_rank


# --------------------------------------------------------------------------- #
# BNB / LoRA configs                                                          #
# --------------------------------------------------------------------------- #
def make_bnb_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if not getattr(args, "use_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def make_lora_config(args: argparse.Namespace) -> Optional[LoraConfig]:
    if not getattr(args, "use_lora", False):
        return None
    return LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=args.target_modules.split(","),
    )


# --------------------------------------------------------------------------- #
# Model / tokenizer                                                           #
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    *,
    bnb_cfg: Optional[BitsAndBytesConfig] = None,
    device: str = "cuda",
    is_eval: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a HF causal LM + tokenizer. If ``model_name`` points at a saved PEFT
    adapter, attaches the adapter to the underlying base model. Otherwise
    loads the model directly.

    NOTE: do *not* wrap the returned model with ``get_peft_model`` here when
    the caller is going to pass ``peft_config=...`` to TRL's trainer — TRL
    rejects an already-wrapped PeftModel + non-None peft_config.
    """
    try:
        pc = PeftConfig.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(
            pc.base_model_name_or_path,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_name, is_trainable=not is_eval)
        tok = AutoTokenizer.from_pretrained(pc.base_model_name_or_path, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    tok.padding_side = "left" if is_eval else "right"
    tok.pad_token = tok.eos_token

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")
    return model, tok


# --------------------------------------------------------------------------- #
# DeepSpeed ZeRO-2                                                            #
# --------------------------------------------------------------------------- #
def deepspeed_zero2_config(args: argparse.Namespace) -> Dict[str, Any]:
    """The DeepSpeed config our trainers use everywhere — no offload, ZeRO-2."""
    return {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": False,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": 200,
    }


# --------------------------------------------------------------------------- #
# Mixed precision                                                             #
# --------------------------------------------------------------------------- #
def precision_kwargs() -> Dict[str, bool]:
    """Pick bf16 vs fp16 based on hardware capabilities."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


# --------------------------------------------------------------------------- #
# W&B                                                                         #
# --------------------------------------------------------------------------- #
def init_wandb_if_configured(args: argparse.Namespace, run_config: Dict[str, Any]) -> None:
    """Init W&B on rank-0 only, when --wandb_project is set."""
    if not getattr(args, "wandb_project", None):
        return
    if os.environ.get("LOCAL_RANK", "0") != "0":
        return
    import wandb  # imported lazily so non-training envs can import this module

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.output_dir.replace("/", "_"),
        config=run_config,
        save_code=True,
        job_type="train",
    )


def finish_wandb_if_configured(args: argparse.Namespace) -> None:
    if not getattr(args, "wandb_project", None):
        return
    import wandb

    wandb.finish()


# --------------------------------------------------------------------------- #
# DeepSpeed precondition                                                      #
# --------------------------------------------------------------------------- #
def assert_has_trainable_params(model: torch.nn.Module, *, hint: str) -> None:
    """
    DeepSpeed ZeRO crashes with ``IndexError: list index out of range`` when
    the optimizer's param_groups are empty. Fail fast with a helpful message
    instead. Call this right after ``load_model_and_tokenizer`` in any
    online-RL trainer.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(
            "Model has 0 trainable parameters; DeepSpeed ZeRO will crash.\n"
            f"Hint: {hint}\n"
            "Common causes:\n"
            "  - Loaded a PEFT adapter with is_trainable=False.\n"
            "  - Activated an inference-only adapter (e.g. a 'ref' adapter).\n"
            "  - Froze all params while expecting LoRA to be trainable."
        )
