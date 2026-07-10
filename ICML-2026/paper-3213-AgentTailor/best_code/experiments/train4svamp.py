from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to Python path (allow running this file directly)
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments import train_base


def _build_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "agent_names": ["MathSolver"] * 5,
        "llm_name": "gpt-4o",
        "decision_method": "FinalRefer",
        "optimized_spatial": True,
        "optimized_temporal": True,
        "domain": "svamp",
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
        "max_training_samples": 40,
        "max_validation_samples": 200,
    }
    if overrides:
        config.update(overrides)
    return config


import asyncio
import random
import time
import numpy as np
import torch
from AgentTailor.utils.globals import PromptTokens, CompletionTokens, Cost, ApiCalls

SEED = 888
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def main() -> None:

    config = _build_config()
    start_wall_clock = time.time()
    result = asyncio.run(train_svamp_with_stage3_stats(**config))

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


async def train_svamp_with_stage3_stats(**config: Any) -> Dict[str, Any]:
    max_train = config.pop("max_training_samples", None)
    max_eval = config.pop("max_validation_samples", None)
    config.pop("dataset_split", None)
    config.pop("validation_split", None)
    return await train_base.train_all(
        **config,
        max_train_split_samples=max_train,
        max_test_split_samples=max_eval,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Svamp training interrupted by user.")
