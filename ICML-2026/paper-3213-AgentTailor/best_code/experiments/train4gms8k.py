from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to Python path (allow running this file directly)
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments import train_base

PROJECT_ROOT = Path(train_base.PROJECT_ROOT)
MAX_DEFAULT_SAMPLES = 161


def _build_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "agent_names": ["MathSolver", "AnalyzeAgent", "AnalyzeAgent", "AnalyzeAgent", "AnalyzeAgent"],
        "llm_name": "gpt-4o",
        "decision_method": "FinalRefer",
        "optimized_spatial": True,
        "optimized_temporal": True,
        "domain": "gsm8k",
        "num_rounds": 2,
        "lr_actor": None,
        "lr_critic": None,
        "sparsity_weight": None,
        "stage1_sample_count": None,
        "stage2_sample_count": None,
        "stage2_virtual_steps": None,
        "lambda2": None,
        "lambda3": None,
        "stage3_virtual_steps": None,
        "stage3_prune_ratio": None,
        "lock_threshold": None,
        "temperature": None,
        "epn_dropout": None,
        "critic_weight_decay": None,
        "epn_dims": [train_base.epn_concat_input_dim()] + train_base.epn_head_hidden_sizes(),
        "dataset_split": "train",
        "max_training_samples": 40,
        "validation_split": "val",
        "max_validation_samples": 200,
        "gsm8k_shuffle_seed": 888,
    }
    if overrides:
        config.update(overrides)
    return config


import asyncio
import time

from dataset.gsm8k_dataset import gsm_get_predict
from AgentTailor.utils.globals import PromptTokens, CompletionTokens, Cost, ApiCalls


class GSM8KDataset:


    def __init__(
        self,
        split: str = "test",
        data_path: Optional[str] = None,
        seed: int = 888,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        max_samples: Optional[int] = MAX_DEFAULT_SAMPLES,
    ) -> None:
        assert split in {"train", "val", "test"}, f"Unsupported split: {split}"
        path = Path(data_path) if data_path else PROJECT_ROOT / "dataset" / "gsm8k" / "gsm8k.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"GSM8K data not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

        rng = random.Random(seed)
        rng.shuffle(all_records)

        total = len(all_records)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])

        if split == "train":
            selected = all_records[:train_end]
        elif split == "val":
            selected = all_records[train_end:val_end]
        else:
            selected = all_records[val_end:]

        if max_samples is not None:
            selected = selected[:max_samples]

        self.records: List[Dict[str, Any]] = []
        for idx, record in enumerate(selected):
            record = dict(record)
            record.setdefault("id", record.get("question", f"gsm8k_{idx}"))
            record.setdefault("name", f"gsm8k_{record['id']}")
            self.records.append(record)

        self._split = split
        print(f"✅ Loaded {len(self.records)} GSM8K samples (split: {split}) from {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]

    @staticmethod
    def get_domain() -> str:
        return "gsm8k"

    def record_to_input(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task": record["question"],
            "name": record.get("name", ""),
        }

    def postprocess_answer(self, answer: Any) -> str:
        if isinstance(answer, list):
            answer = answer[-1] if answer else ""
        if not isinstance(answer, str):
            answer = str(answer)
        return answer.strip()

    def evaluate_candidate(
        self,
        actor_output: str,
        record: Dict[str, Any],
        timeout: int = 0,
    ) -> Tuple[float, bool, str, Tuple[bool, ...], List[str]]:

        target = self._extract_target(record)
        predicted = gsm_get_predict(actor_output)
        is_correct = predicted == target
        pass_ratio = 1.0 if is_correct else 0.0
        feedback = "Correct" if is_correct else f"Incorrect: expected {target}, got {predicted}"
        state = (is_correct,)
        unit_tests = [f"answer == {target}"]
        return pass_ratio, is_correct, feedback, state, unit_tests

    @staticmethod
    def format_feedback_summary(
        pass_ratio: float,
        test_state: Tuple[bool, ...],
        feedback: str,
    ) -> str:
        passed = sum(test_state)
        total = len(test_state) or 1
        summary = [
            f"pass_ratio={pass_ratio:.3f}",
            f"tests_passed={passed}/{total}",
            "feedback:",
            feedback,
        ]
        return "\n".join(summary)

    @staticmethod
    def _extract_target(record: Dict[str, Any]) -> str:
        raw_answer = record.get("answer", "")
        if "####" in raw_answer:
            normalized = raw_answer.split("####")[-1]
        else:
            normalized = raw_answer
        return normalized.strip().replace(",", "")


async def train_gsm8k_with_stage3_stats(**config: Any) -> Dict[str, Any]:

    config.pop("dataset_split", None)
    validation_split = config.pop("validation_split", "val")
    max_training_samples = config.pop("max_training_samples", None)
    max_validation_samples = config.pop("max_validation_samples", None)
    gsm8k_shuffle_seed = config.pop("gsm8k_shuffle_seed", None)

    result = await train_base.train_all(
        **config,
        max_train_split_samples=max_training_samples,
        max_test_split_samples=max_validation_samples,
        gsm8k_shuffle_seed=gsm8k_shuffle_seed,
    )

    _vseed = int(gsm8k_shuffle_seed) if gsm8k_shuffle_seed is not None else 888

    stage3_actor = result.get("stage3_actor") or result.get("actor")
    if stage3_actor is None:
        print("⚠️ Actor not found; skipping Stage 3 validation")
        return result

    val_dataset = GSM8KDataset(
        split=validation_split,
        max_samples=max_validation_samples,
        seed=_vseed,
    )
    total_val = len(val_dataset.records)
    if total_val == 0:
        print("⚠️ Validation set is empty; skipping Stage 3 validation")
        return result

    print("\n" + "=" * 80)
    print(f"Stage 3 validation: using GSM8K {validation_split} split ({total_val} samples)")
    print("=" * 80)

    stage3_state = train_base.TrainingState()
    stage3_state.reset_token_baseline()

    for idx, record in enumerate(val_dataset.records):
        print(f"\n====== Stage 3 validation sample {idx + 1}/{total_val} ======")
        task_input = val_dataset.record_to_input(record)

        answers, _, _ = await stage3_actor.arun(
            task_input,
            num_rounds=config.get("num_rounds", 3),
            aggregate_mode="all connected",
        )

        final_answer = val_dataset.postprocess_answer(answers[-1] if answers else "")
        pass_ratio, is_passing, feedback, test_state, unit_tests = val_dataset.evaluate_candidate(
            final_answer,
            record,
        )

        stage3_state.update(is_passing)

        status = "✅" if is_passing else "❌"
        print(f"Sample {idx + 1} result {status} | pass_ratio={pass_ratio:.2%}")
        if (idx + 1) % 10 == 0:
            print(
                f"  Current cumulative accuracy: {stage3_state.accuracy:.2%} "
                f"({stage3_state.cumulative_correct}/{stage3_state.cumulative_total})"
            )

    print("\n" + "=" * 80)
    print("[Stage 3 validation cumulative statistics] (GSM8K validation set)")
    print("=" * 80)
    v_prompt, v_completion, v_total = stage3_state.get_token_stats()
    print(
        f"  Cumulative accuracy: {stage3_state.accuracy:.2%} "
        f"({stage3_state.cumulative_correct}/{stage3_state.cumulative_total})"
    )
    print(
        f"  Token stats: Prompt={v_prompt:,} | Completion={v_completion:,} | Total={v_total:,}"
    )
    print("=" * 80)

    result["stage3_validation_accuracy"] = stage3_state.accuracy
    result["stage3_validation_correct"] = stage3_state.cumulative_correct
    result["stage3_validation_total"] = stage3_state.cumulative_total
    result["stage3_validation_prompt_tokens"] = v_prompt
    result["stage3_validation_completion_tokens"] = v_completion
    result["stage3_validation_total_tokens"] = v_total

    return result


def main() -> None:

    config = _build_config()
    start_wall_clock = time.time()
    result = asyncio.run(train_gsm8k_with_stage3_stats(**config))

    print("\n" + "=" * 80)
    print("Token statistics")
    print("=" * 80)
    prompt_tokens = int(result.get("prompt_tokens", int(PromptTokens.instance().value)))
    completion_tokens = int(result.get("completion_tokens", int(CompletionTokens.instance().value)))
    total_tokens = int(result.get("total_tokens", prompt_tokens + completion_tokens))
    cost = float(result.get("estimated_cost_usd", float(Cost.instance().value)))
    api_calls = int(result.get("api_calls", int(ApiCalls.instance().value)))
    wall_clock_seconds = float(result.get("wall_clock_seconds", time.time() - start_wall_clock))
    print(f"Prompt Tokens: {prompt_tokens:,}")
    print(f"Completion Tokens: {completion_tokens:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"API Calls: {api_calls:,}")
    print(f"Wall-clock Time: {wall_clock_seconds:.2f}s")
    print(f"Cost: ${cost:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ GSM8K training interrupted by user.")

