from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to Python path (allow running this file directly)
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments import train_base

PROJECT_ROOT = Path(train_base.PROJECT_ROOT)


def _build_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "agent_names": ["AnalyzeAgent"] * 5,
        "llm_name": "deepseek-chat",
        "decision_method": "FinalRefer",
        "optimized_spatial": True,
        "optimized_temporal": True,
        "domain": "mmlu",
        "num_rounds": 2,
        "lr_actor": 0.01,
        "lr_critic": 0.001,
        "sparsity_weight": 0.2,
        "stage1_sample_count": 20,
        "stage2_sample_count": 20,
        "stage2_virtual_steps": 2,
        "lambda2": 0.5,
        "lambda3": None,
        "stage3_virtual_steps": 1,
        "stage3_prune_ratio": 0.3,
        "lock_threshold": 0.01,
        "temperature": 5.0,
        "epn_dropout": 0.1,
        "critic_weight_decay": 0.01,
        "epn_dims": [train_base.epn_concat_input_dim()] + train_base.epn_head_hidden_sizes(),
        "dataset_split": "dev",
        "max_training_samples": 40,
        "max_validation_samples": 153,
    }
    if overrides:
        config.update(overrides)
    return config


import asyncio
import random
import time
import numpy as np
import pandas as pd
import torch

from experiments_util.textqa_edge_judge import TextQEdgeJudge as TextQJudge
from AgentTailor.utils.globals import PromptTokens, CompletionTokens, Cost, ApiCalls

SEED = 888
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class MMLUDataset:


    def __init__(
        self,
        split: str = "test",
        data_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:


        base_dir = PROJECT_ROOT / "dataset" / "MMLU" / "data"


        if split == "test":

            dev_path = base_dir / "dev"
            if not dev_path.exists():
                raise FileNotFoundError(f"MMLU dev data directory not found: {dev_path}")

            dev_csv_files = sorted(dev_path.glob("*.csv"))
            if not dev_csv_files:
                raise FileNotFoundError(f"No CSV files found in {dev_path}")

            dev_frames = [pd.read_csv(path, header=None, names=["question", "A", "B", "C", "D", "correct_answer"])
                         for path in dev_csv_files]
            dev_merged = pd.concat(dev_frames, ignore_index=True)


            val_path = base_dir / "val"
            if not val_path.exists():
                raise FileNotFoundError(f"MMLU val data directory not found: {val_path}")

            val_csv_files = sorted(val_path.glob("*.csv"))
            if not val_csv_files:
                raise FileNotFoundError(f"No CSV files found in {val_path}")

            val_frames = [pd.read_csv(path, header=None, names=["question", "A", "B", "C", "D", "correct_answer"])
                         for path in val_csv_files]
            val_merged = pd.concat(val_frames, ignore_index=True)


            rng = np.random.default_rng(SEED)


            dev_perm = rng.permutation(len(dev_merged))
            dev_merged = dev_merged.iloc[dev_perm].reset_index(drop=True)

            val_perm = rng.permutation(len(val_merged))
            val_merged = val_merged.iloc[val_perm].reset_index(drop=True)


            dev_merged = dev_merged.iloc[:40].reset_index(drop=True)
            val_merged = val_merged.iloc[:153].reset_index(drop=True)


            merged = pd.concat([dev_merged, val_merged], ignore_index=True)

            if max_samples is not None:
                merged = merged.iloc[:max_samples].reset_index(drop=True)

            self.records = merged.to_dict(orient="records")
            self._split = "test"
            print(f"✅ Loaded {len(self.records)} MMLU samples (dev: 40 + val: 153 = 193 total)")
        else:

            assert split in {"dev", "val"}, f"Unsupported split: {split}"
            data_path = Path(data_dir) if data_dir else base_dir / split
            if not data_path.exists():
                raise FileNotFoundError(f"MMLU data directory not found: {data_path}")

            csv_files = sorted(data_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {data_path}")

            frames = [pd.read_csv(path, header=None, names=["question", "A", "B", "C", "D", "correct_answer"])
                     for path in csv_files]
            merged = pd.concat(frames, ignore_index=True)

            rng = np.random.default_rng(888)
            perm = rng.permutation(len(merged))
            merged = merged.iloc[perm].reset_index(drop=True)

            if max_samples is not None:
                merged = merged.iloc[:max_samples].reset_index(drop=True)

            self.records = merged.to_dict(orient="records")
            self._split = split
            print(f"✅ Loaded {len(self.records)} MMLU samples (split: {split}) from {data_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]

    @staticmethod
    def get_domain() -> str:
        return "mmlu"

    def record_to_input(self, record: Dict[str, Any]) -> Dict[str, Any]:
        prompt = "\n".join(
            [
                record["question"],
                f"A. {record['A']}",
                f"B. {record['B']}",
                f"C. {record['C']}",
                f"D. {record['D']}",
                "Please provide your answer and analysis. The first line should contain only the letter (A, B, C, or D), followed by your reasoning.",
            ]
        )
        return {
            "task": prompt,
            "name": record.get("topic", "mmlu_question"),
        }

    def postprocess_answer(self, answer: Any) -> str:
        if isinstance(answer, list):
            answer = answer[-1] if answer else ""
        if not isinstance(answer, str):
            answer = str(answer)

        cleaned = answer.strip()
        lowered = cleaned.lower()
        for marker in ["the answer is", "answer:", "option"]:
            if marker in lowered:
                idx = lowered.rfind(marker)
                cleaned = cleaned[idx + len(marker):].strip()
                break
        return cleaned[:1]

    def evaluate_candidate(
        self,
        actor_output: str,
        record: Dict[str, Any],
        timeout: int = 0,
    ) -> Tuple[float, bool, str, Tuple[bool, ...], List[str]]:
        predicted = self._normalize_option(actor_output)
        target = str(record["correct_answer"]).strip().upper()
        is_correct = predicted == target
        pass_ratio = 1.0 if is_correct else 0.0
        feedback = "Correct" if is_correct else f"Incorrect: expected {target}, got {predicted}"
        state = (is_correct,)
        unit_tests = [f"choice == {target}"]
        return pass_ratio, is_correct, feedback, state, unit_tests

    @staticmethod
    def format_feedback_summary(
        pass_ratio: float,
        test_state: Tuple[bool, ...],
        feedback: str,
    ) -> str:
        passed = sum(test_state)
        total = len(test_state) or 1
        return "\n".join(
            [
                f"pass_ratio={pass_ratio:.3f}",
                f"tests_passed={passed}/{total}",
                "feedback:",
                feedback,
            ]
        )

    @staticmethod
    def _normalize_option(answer: str) -> str:
        cleaned = answer.strip()
        if not cleaned:
            return ""
        for char in cleaned:
            if char.upper() in {"A", "B", "C", "D"}:
                return char.upper()
        return cleaned[:1].upper()


async def train_mmlu_with_stage3_stats(**config) -> Dict[str, Any]:

    max_train = config.pop("max_training_samples", None)
    max_eval = config.pop("max_validation_samples", None)
    config.pop("dataset_split", None)
    config.pop("validation_split", None)

    result = await train_base.train_all(
        **config,
        max_train_split_samples=max_train,
        max_test_split_samples=max_eval,
    )



    full_accuracy = result.get("cumulative_accuracy", 0.0)
    full_correct = result.get("cumulative_correct", 0)
    full_total = result.get("cumulative_total", 0)
    full_prompt_tokens = result.get("prompt_tokens", 0)
    full_completion_tokens = result.get("completion_tokens", 0)
    full_total_tokens = result.get("total_tokens", 0)


    stage3_stats = result.get("stage3", {})
    stage3_correct = stage3_stats.get("stage3_correct", 0)
    stage3_total = stage3_stats.get("stage3_total", 0)
    stage3_prompt_tokens = stage3_stats.get("stage3_prompt_tokens", 0)
    stage3_completion_tokens = stage3_stats.get("stage3_completion_tokens", 0)
    stage3_total_tokens = stage3_stats.get("stage3_total_tokens", 0)


    print("\n" + "=" * 80)
    print("MMLU training finished: statistics")
    print("=" * 80)


    stage3_accuracy = stage3_correct / max(1, stage3_total)
    print(f"[Stage3 validation] ({stage3_total} samples)")
    print(f"  Accuracy: {stage3_accuracy:.2%} ({stage3_correct}/{stage3_total})")
    print(f"  Token stats: Prompt={stage3_prompt_tokens:,} | Completion={stage3_completion_tokens:,} | Total={stage3_total_tokens:,}")


    print(f"\n[Overall cumulative results] ({full_total} samples)")
    print(f"  Accuracy: {full_accuracy:.2%} ({full_correct}/{full_total})")
    print(f"  Token stats: Prompt={full_prompt_tokens:,} | Completion={full_completion_tokens:,} | Total={full_total_tokens:,}")
    print("=" * 80)


    result["stage3_validation_accuracy"] = stage3_accuracy
    result["stage3_validation_correct"] = stage3_correct
    result["stage3_validation_total"] = stage3_total
    result["stage3_validation_prompt_tokens"] = stage3_prompt_tokens
    result["stage3_validation_completion_tokens"] = stage3_completion_tokens
    result["stage3_validation_total_tokens"] = stage3_total_tokens

    return result


def main() -> None:
    config = _build_config({"edge_judge": TextQJudge()})
    start_wall_clock = time.time()
    result = asyncio.run(train_mmlu_with_stage3_stats(**config))

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
        print("\n⚠️ MMLU training interrupted by user.")

