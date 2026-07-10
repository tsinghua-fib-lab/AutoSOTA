"""
discoverllm.training
~~~~~~~~~~~~~~~~~~~~

Training entrypoints (SFT, offline DPO, online DPO, GRPO) for fine-tuning
LLMs against the DiscoverLLM user-simulator reward.

The four trainer scripts live under :mod:`discoverllm.training.trainers`:

* ``sft.py``         — supervised fine-tuning on multi-turn chat data
* ``offline_dpo.py`` — direct preference optimization on synthesised pairs
* ``online_dpo.py``  — DPO with on-policy rollouts judged by the user simulator
* ``grpo.py``        — group-relative policy optimization with the same judge

Online DPO and GRPO depend on :class:`discoverllm.training.reward.DesignLLMRewardComputer`,
which in turn calls into :mod:`discoverllm.pipeline` to run the user simulator.

Attribution
-----------
The four trainers, the multi-turn dataset loader, and the system-prompt
scaffolding in this package are adapted from **CollabLLM** (Wu et al., 2024 —
https://github.com/Wuyxin/collabllm). The user-simulator reward
(:class:`DesignLLMRewardComputer`) is original to this work.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Quiet LiteLLM by default — its INFO logs are extremely chatty during training.
# Users who want them back can call ``logging.getLogger("LiteLLM").setLevel(...)``.
try:
    import litellm

    litellm.disable_cache()
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
except ImportError:
    # litellm is an optional runtime dependency for some training paths.
    pass


# Per-user runtime directory used by some trainers for tmp files and checkpoints.
# Honours ``RUN_USER_DIR`` if set; otherwise falls back to ~/.cache/discoverllm.
def _resolve_run_user_dir() -> Path:
    env_val = os.environ.get("RUN_USER_DIR")
    if env_val:
        return Path(env_val).expanduser()
    return Path.home() / ".cache" / "discoverllm"


RUN_USER_DIR: Path = _resolve_run_user_dir()
RUN_USER_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("RUN_USER_DIR", str(RUN_USER_DIR))
