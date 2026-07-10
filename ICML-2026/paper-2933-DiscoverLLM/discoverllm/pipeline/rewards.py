import copy
from typing import Any, Dict, List, Tuple

from discoverllm.metrics import calculate_score_delta
from discoverllm.pipeline.user_simulator import UserSimulator
from discoverllm.simulate.config import AssistantConfig, make_assistant_from_config
from discoverllm.utils import count_tokens_in_conversation


def calculate_reward_and_deltas(
    new_criteria_objs: List[Dict[str, Any]],
    prev_criteria_objs: List[Dict[str, Any]],
    conversation_slice: List[Dict[str, Any]]
) -> Tuple[float, float, float]:
    delta_awareness = calculate_score_delta(new_criteria_objs, prev_criteria_objs, "awareness")
    delta_satisfaction = calculate_score_delta(new_criteria_objs, prev_criteria_objs, "satisfaction")

    # Token penalty: kicks in after 250 tokens, capped at 1.0
    LAMBDA = 1e-3
    TOKEN_THRESHOLD = 1 / 4 / LAMBDA # 250.0 for LAMBDA=1e-3
    MAX_TOKEN_PENALTY = 1.0
    token_count = count_tokens_in_conversation(conversation_slice)
    token_penalty = min(LAMBDA * max(0, token_count - TOKEN_THRESHOLD), MAX_TOKEN_PENALTY)

    # Reward formula (delta_awareness ∈ [0, 1]):
    #   reward = delta_awareness - token_penalty
    #
    # An earlier variant added a constant AWARENESS_FLOOR (= MAX_TOKEN_PENALTY +
    # 0.01) when delta_awareness > 0, guaranteeing that any awareness gain
    # outranked any token-penalty-only response. We dropped it because the
    # current formula's [-1, 1] range is more directly comparable across runs;
    # see git blame for the previous form if you want to revert.
    reward = delta_awareness - token_penalty

    return {
        "delta_awareness": delta_awareness,
        "delta_satisfaction": delta_satisfaction,
        "reward": reward,
    }

def calculate_future_reward_and_deltas(
    initial_criteria_objs: List[Dict[str, Any]],
    future_criteria_history: List[List[Dict[str, Any]]],
    future_chat_history: List[Dict[str, Any]]
) -> Tuple[float, float, float]:
    singleturn_result = calculate_reward_and_deltas(
        future_criteria_history[0],
        initial_criteria_objs,
        future_chat_history[:1]
    )
    multiturn_result = singleturn_result
    if len(future_criteria_history) > 1:
        multiturn_result = calculate_reward_and_deltas(
            future_criteria_history[-1],
            initial_criteria_objs,
            future_chat_history
        )

    return {
        "singleturn": singleturn_result,
        "multiturn": multiturn_result
    }


def multiturn_reward(
    user_simulator: UserSimulator,
    assistant_response: str,
    reward_assistant_config: AssistantConfig,
    window_size: int,
    verbose: bool,
) -> Dict[str, Any]:

    # Current turn simulation
    clone_user = user_simulator.clone()
    clone_user.update_criteria_only(assistant_response)

    # Future turns simulations
    reward_assistant = make_assistant_from_config(clone_user.chat_history, reward_assistant_config, verbose)
    future_user = clone_user.clone()
    for _ in range(window_size):
        user_response = future_user.generate_next_user_response()
        temp_assistant_response = reward_assistant(user_response)
        future_user.update_criteria_only(temp_assistant_response)


    future_chat_history = copy.deepcopy(future_user.chat_history[-1 - (2 * window_size):])
    future_criteria_history = copy.deepcopy(future_user.criteria_history[-1 - window_size:])
    future_trajectory = {
        "criteria_history": copy.deepcopy(future_criteria_history),
        "chat_history": copy.deepcopy(future_chat_history),
    }

    # Calculate all rewards and deltas (incl. current turn and future turns)
    full_results = calculate_future_reward_and_deltas(
        initial_criteria_objs=copy.deepcopy(user_simulator.criteria_history[-1]),
        future_criteria_history=future_criteria_history,
        future_chat_history=future_chat_history
    )

    return {
        "assistant_response": assistant_response,
        "trial_sim": clone_user,
        "updated_criteria_objs": clone_user.criteria_history[-1],
        "full_results": full_results,
        "reward": full_results["multiturn"]["reward"],
        "delta_awareness": full_results["multiturn"]["delta_awareness"],
        "delta_satisfaction": full_results["multiturn"]["delta_satisfaction"],
        "future_trajectory": future_trajectory
    }

