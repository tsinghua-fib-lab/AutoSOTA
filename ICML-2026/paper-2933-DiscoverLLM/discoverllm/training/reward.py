"""
discoverllm.training.reward
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bridge between the trainers in :mod:`discoverllm.training.trainers` and the
user-simulator-based reward defined in :mod:`discoverllm.pipeline`.

The single public class, :class:`DesignLLMRewardComputer`, exposes one entry
point used by both online RL algorithms:

* :meth:`compute_rewards` — score a batch of (chat_history, response) tasks
  in parallel; consumed by both GRPO (per-rollout reward) and Online DPO
  (TRL 1.4's scalar ``reward_funcs=`` API forms preference pairs internally).

It delegates to :func:`discoverllm.pipeline.rewards.multiturn_reward`, which
runs the user simulator forward and reports awareness / satisfaction deltas.
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from discoverllm.pipeline.rewards import multiturn_reward
from discoverllm.pipeline.user_simulator import UserSimulator
from discoverllm.simulate.config import AssistantConfig

logger = logging.getLogger(__name__)


# Default simulator. Matches examples/configs/user.json so a freshly-cloned
# repo can run the online trainers without Gemini access.
_DEFAULT_USER_SIMULATOR_CONFIG: Dict[str, Any] = {
    "model_name": "claude-haiku-4-5",
    "temperature": 0.3,
}


class DesignLLMRewardComputer:
    """
    Criteria-aware reward computer used by online RL trainers.

    :meth:`compute_rewards` runs the user simulator forward on each candidate
    response and asks :func:`discoverllm.pipeline.rewards.multiturn_reward`
    for the reward.

    Parameters
    ----------
    reward_assistant_config
        Generation kwargs (``model``, ``temperature``, ``system_prompt``) used
        when the simulator needs to roll out future turns to score the current
        response. This is the *assistant* under evaluation, not the user
        simulator itself.
    user_simulator_config
        Generation kwargs for the user simulator. Defaults to
        ``{"model_name": "claude-haiku-4-5", "temperature": 0.3}``. Pass via
        the trainer CLI as ``--user_simulator_config '{"model_name": "..."}'``
        when training requires a different simulator (e.g. when the default
        provider is rate-limited).
    """

    def __init__(
        self,
        reward_assistant_config: Dict[str, Any],
        window_size: int = 0,
        max_workers: int = 4,
        verbose: bool = False,
        user_simulator_config: Optional[Dict[str, Any]] = None,
    ):
        self.reward_assistant_config = reward_assistant_config
        self.window_size = window_size
        self.max_workers = max_workers
        self.verbose = verbose
        merged = dict(_DEFAULT_USER_SIMULATOR_CONFIG)
        if user_simulator_config:
            merged.update(user_simulator_config)
        self.user_simulator_config = merged

    def _score_one(
        self,
        chat_history: List[Dict[str, str]],
        criteria_history: List[List[Dict[str, Any]]],
        assistant_response: str,
    ) -> Dict[str, Any]:
        """
        Run the user simulator + ``multiturn_reward`` for a single candidate.

        Returns a dict with ``reward``, ``delta_awareness``,
        ``delta_satisfaction``, ``full_results``, ``updated_criteria_objs``,
        and ``future_trajectory``. Raises if the inputs are empty.
        """
        if not chat_history:
            raise ValueError("chat_history cannot be empty")
        if not criteria_history:
            raise ValueError("criteria_history cannot be empty")

        sim_cfg = self.user_simulator_config
        user_sim = UserSimulator(
            initial_chat_history=copy.deepcopy(chat_history),
            initial_criteria_history=copy.deepcopy(criteria_history),
            model_name=sim_cfg["model_name"],
            temperature=sim_cfg.get("temperature", 0.3),
            verbose=self.verbose,
        )
        reward_asst_cfg = AssistantConfig(
            model_name=self.reward_assistant_config.get("model", "gpt-4o-mini"),
            system_prompt=self.reward_assistant_config.get("system_prompt", ""),
            temperature=self.reward_assistant_config.get("temperature", 0.7),
        )
        result = multiturn_reward(
            user_simulator=user_sim,
            assistant_response=assistant_response,
            reward_assistant_config=reward_asst_cfg,
            window_size=self.window_size,
            verbose=self.verbose,
        )
        return {
            "reward": result["reward"],
            "delta_awareness": result["delta_awareness"],
            "delta_satisfaction": result["delta_satisfaction"],
            "full_results": result["full_results"],
            "updated_criteria_objs": result["updated_criteria_objs"],
            "future_trajectory": result.get("future_trajectory"),
        }

    def _compute_single(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single task; on failure return a zero-reward placeholder."""
        try:
            result = self._score_one(
                chat_history=task["chat_history"],
                criteria_history=task["criteria_history"],
                assistant_response=task["assistant_response"],
            )
            return {
                "idx": task["idx"],
                "reward": result["reward"],
                "delta_awareness": result.get("delta_awareness"),
                "delta_satisfaction": result.get("delta_satisfaction"),
            }
        except Exception as e:
            logger.error(f"Error computing reward for task {task.get('idx')}: {e}")
            return {
                "idx": task["idx"],
                "reward": 0.0,
                "delta_awareness": 0.0,
                "delta_satisfaction": 0.0,
            }

    def compute_rewards(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Compute rewards for a batch of tasks in parallel.

        Parameters
        ----------
        tasks : List[Dict[str, Any]]
            Each task must have: idx, chat_history, criteria_history, assistant_response

        Returns
        -------
        List[Dict[str, Any]]
            Results sorted by idx, each with 'idx' and 'reward' keys.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._compute_single, tasks))
        return sorted(results, key=lambda x: x["idx"])
