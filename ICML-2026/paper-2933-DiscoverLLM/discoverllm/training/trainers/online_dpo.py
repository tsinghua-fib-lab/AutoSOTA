#!/usr/bin/env python3
"""
Online DPO trainer.

On every step the trainer samples completions from the current policy (via
TRL's native vLLM colocation) and scores each one with
:class:`discoverllm.training.reward.DesignLLMRewardComputer`. TRL's
``OnlineDPOTrainer`` then forms preference pairs from those scalar rewards
and drives the DPO loss as usual.

Compared to TRL's stock OnlineDPOTrainer this script:

* uses our criteria-aware multi-turn reward fn instead of a single-shot LLM judge,
* runs the policy and vLLM in the same process (avoids stale-weights bugs and
  GPU-memory fighting from a separate generator process),
* enables DeepSpeed CPU offload so 8B models fit on 2x80GB.

TRL-1.4 compatibility notes (see `docs/E2E_TRAJECTORY.md` for context):

* ``trl.experimental.judges.BasePairwiseJudge`` was removed in TRL 1.4 and
  ``OnlineDPOTrainer(judge=...)`` no longer exists. This module now uses the
  scalar ``reward_funcs=[fn]`` API: the wrapped reward fn receives
  ``(prompts, completions, ...)`` and returns one float per completion.
* See :class:`MultiturnRewardFunc` below for the wrapper around
  ``DesignLLMRewardComputer.compute_rewards``.

Run as a module (note: needs ``torchrun`` so ``LOCAL_RANK`` etc. are set;
plain ``python -m`` will ``KeyError`` in ``init_distributed``):

    torchrun --standalone --nproc_per_node 1 \\
        -m discoverllm.training.trainers.online_dpo \\
        --dataset_repo path/to/processed_dataset \\
        --model_name path/to/sft_or_dpo_checkpoint \\
        --output_dir outputs/online_dpo/my_run \\
        --use_lora --system_prompt_type ours

See ``scripts/train/online_dpo.sh`` for a full launch example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from peft import PeftModel

# OnlineDPOTrainer moved under ``trl.experimental.online_dpo`` in TRL 1.x;
# in TRL 0.13–0.16 it was top-level. ``BasePairwiseJudge`` was removed in
# TRL 1.4 entirely — we now use the ``reward_funcs=`` scalar-reward interface.
try:
    from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer
except ImportError:
    from trl import OnlineDPOConfig, OnlineDPOTrainer

from discoverllm.training._common import (
    apply_config_file_overrides,
    assert_has_trainable_params,
    deepspeed_zero2_config,
    finish_wandb_if_configured,
    init_distributed,
    init_wandb_if_configured,
    load_model_and_tokenizer,
    make_base_argparser,
    make_bnb_config,
    make_lora_config,
    precision_kwargs,
)
from discoverllm.training.datasets.multiturn import MultiturnDataset
from discoverllm.training.reward import DesignLLMRewardComputer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = make_base_argparser("multiturn online-DPO trainer")
    # Online-DPO specific: vLLM colocation knobs + judge config.
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}",
                   help='JSON dict; e.g. {"model": "outputs/sft/foo", "temperature": 0.6}')
    p.add_argument("--max_new_turns", type=int, default=0,
                   help="Future-turn lookahead used by DesignLLMRewardComputer.")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_metric_workers", type=int, default=4,
                   help="Parallel workers used inside DesignLLMRewardComputer.")
    p.add_argument("--user_simulator_config", type=json.loads, default="{}",
                   help='JSON dict for the user simulator the reward computer drives; '
                        'e.g. {"model_name": "claude-haiku-4-5", "temperature": 0.3}. '
                        'Defaults to claude-haiku-4-5.')
    p.add_argument("--gpu_memory_utilization", type=float, default=0.3,
                   help="vLLM gpu_memory_utilization (kept low to leave room for trainer).")
    p.add_argument("--log_completions", action="store_true", default=False)
    p.add_argument("--log_completions_steps", type=int, default=100)
    p.add_argument("--log_completions_file", type=str, default=None)
    return apply_config_file_overrides(p.parse_args())


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    init_distributed()

    ds = MultiturnDataset(
        args.dataset_repo, system_prompt_type=args.system_prompt_type,
    ).to_inputs_dataset(eval_ratio=0.0)

    bnb_cfg = make_bnb_config(args)
    lora_cfg = make_lora_config(args)
    model, tok = load_model_and_tokenizer(
        args.model_name, bnb_cfg=bnb_cfg, device=args.device, is_eval=False,
    )
    assert_has_trainable_params(
        model,
        hint="Right after loading. Make sure your policy adapter is trainable.",
    )
    peft_config_for_trainer = None if isinstance(model, PeftModel) else lora_cfg

    gen_kwargs = args.assistant_generation_kwargs
    train_cfg = OnlineDPOConfig(
        output_dir=args.output_dir,
        beta=0.1,
        loss_type="sigmoid",
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="wandb" if args.wandb_project else "none",
        do_eval=False,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=1,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        deepspeed=deepspeed_zero2_config(args),
        # vLLM colocation: TRL spins up vLLM in-process, sharing GPU memory.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=gen_kwargs.get("temperature", 0.7),
        remove_unused_columns=False,
        **precision_kwargs(),
    )
    init_wandb_if_configured(args, train_cfg.to_dict())

    # ------------------------------------------------------------------ #
    # Judge: hash each templated prompt so the reward computer can find  #
    # the original (criteria_history, prompt) pair to score against.     #
    # ------------------------------------------------------------------ #
    def compute_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    str_prompt_to_data: dict = {}

    def process(row):
        templated = tok.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        str_prompt_to_data.setdefault(
            compute_hash(templated),
            {k: row[k] for k in ("criteria_history", "prompt")},
        )
        row["prompt"] = templated
        return row

    ds["train"] = ds["train"].map(process, load_from_cache_file=False)

    reward_computer = DesignLLMRewardComputer(
        reward_assistant_config=args.assistant_generation_kwargs,
        window_size=args.max_new_turns,
        max_workers=args.max_metric_workers,
        verbose=False,
        user_simulator_config=args.user_simulator_config,
    )

    reward_func = MultiturnRewardFunc(
        reward_computer=reward_computer,
        prompt_to_data_map=str_prompt_to_data,
        compute_hash_fn=compute_hash,
        log_completions=args.log_completions,
        log_completions_steps=args.log_completions_steps,
        log_completions_file=args.log_completions_file,
        wandb_project=args.wandb_project,
    )

    trainer = OnlineDPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        train_dataset=ds["train"],
        processing_class=tok,
        peft_config=peft_config_for_trainer,
        args=train_cfg,
    )
    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"online_dpo-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    finish_wandb_if_configured(args)


# --------------------------------------------------------------------------- #
# Scalar reward function wrapping DesignLLMRewardComputer                     #
# --------------------------------------------------------------------------- #
class MultiturnRewardFunc:
    """TRL 1.4 ``reward_funcs=`` callable that delegates to DesignLLMRewardComputer.

    TRL's ``reward_funcs`` API expects callables with signature
    ``(prompts: list[str], completions: list[str], completion_ids=None, **kwargs)
    -> list[float]``. Each completion gets its own scalar reward; OnlineDPO then
    forms preference pairs internally from those scores.
    """

    __name__ = "multiturn_reward"

    def __init__(
        self,
        *,
        reward_computer: DesignLLMRewardComputer,
        prompt_to_data_map: dict,
        compute_hash_fn,
        log_completions: bool = False,
        log_completions_steps: int = 100,
        log_completions_file: str | None = None,
        wandb_project: str | None = None,
    ):
        self.reward_computer = reward_computer
        self.prompt_to_data_map = prompt_to_data_map
        self.compute_hash_fn = compute_hash_fn
        self.log_completions = log_completions
        self.log_completions_steps = log_completions_steps
        self.log_completions_file = log_completions_file
        self.wandb_project = wandb_project
        self._step = 0

    def __call__(self, prompts, completions, completion_ids=None, **kwargs):
        tasks = []
        idx_to_pos = {}
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            data = self.prompt_to_data_map.get(self.compute_hash_fn(prompt))
            if data is None:
                logger.warning(f"No data found for prompt #{i}; emitting zero reward.")
                continue
            idx_to_pos[len(tasks)] = i
            tasks.append({
                "idx": len(tasks),
                "chat_history": data["prompt"],
                "criteria_history": json.loads(data["criteria_history"]),
                "assistant_response": completion,
            })

        rewards = [0.0] * len(completions)
        if tasks:
            results = self.reward_computer.compute_rewards(tasks)
            for r in results:
                rewards[idx_to_pos[r["idx"]]] = float(r["reward"])

        if self.log_completions and self._step % self.log_completions_steps == 0:
            self._log(prompts, completions, rewards, self._step)
        self._step += 1
        return rewards

    def _log(self, prompts, completions, rewards, step: int) -> None:
        entries = [
            {
                "step": step,
                "prompt_idx": i,
                "prompt": (p[:500] + "...") if len(p) > 500 else p,
                "completion": (c[:1000] + "...") if len(c) > 1000 else c,
                "reward": r,
                "timestamp": datetime.now().isoformat(),
            }
            for i, (p, c, r) in enumerate(zip(prompts, completions, rewards))
        ]
        if self.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
            try:
                import wandb

                wandb.log(
                    {f"completions/step_{step}": wandb.Table(
                        columns=["step", "prompt_idx", "prompt", "completion", "reward"],
                        data=[[e["step"], e["prompt_idx"], e["prompt"], e["completion"], e["reward"]]
                              for e in entries],
                    )},
                    step=step,
                )
            except Exception as e:
                logger.warning(f"Failed to log completions to wandb: {e}")
        if self.log_completions_file:
            try:
                os.makedirs(os.path.dirname(self.log_completions_file) or ".", exist_ok=True)
                with open(self.log_completions_file, "a") as f:
                    for entry in entries:
                        f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write completions to file: {e}")
        logger.info(f"Logged {len(entries)} completions at step {step}")


if __name__ == "__main__":
    load_dotenv(".env")
    main()
