from typing import Any, Dict, List, Optional


def score_hierarchy(hierarchy: List[Dict[str, Any]], type: str) -> float:
    type_mapping = {
        "awareness": "aware",
        "aware": "aware",
        "satisfaction": "satisfied",
        "satisfied": "satisfied"
    }
    total_score = 0
    curr_score = 0
    total_count = 0
    for item in hierarchy:
        if "children" in item:
            all_descendants_score, children_score, count = score_hierarchy(item["children"], type)
            total_score += all_descendants_score
            total_count += count

        if children_score == len(item["children"]) and len(item["children"]) > 0:
            curr_score += 1
        else:
            curr_score += item[type_mapping[type]]

    total_score += curr_score
    total_count += len(hierarchy)
    return total_score, curr_score, total_count

def get_criterion_level(criteria_obj: Dict[str, Any], type: str) -> float:
    """
    Calculates the "awareness" or "satisfaction" level for a criterion.
    """
    total_score, _, total_count = score_hierarchy(criteria_obj["hierarchy"], type)
    if total_count == 0:
        return 0.0
    return total_score / total_count

def get_overall_level(criteria_objs: List[Dict[str, Any]], type: str) -> float:
    """
    Calculates the overall "awareness" or "satisfaction" level for a set of criteria.
    """
    if len(criteria_objs) == 0:
        return 0.0
    return sum(get_criterion_level(criteria_obj, type) for criteria_obj in criteria_objs) / len(criteria_objs)

def get_turnwise_scores(turn: int, criterion_objs_history: List[Dict[str, Any]], type: str) -> float:
    """
    Calculates the "awareness" or "satisfaction" level for a set of criteria at a given turn.
    """
    prev_level = get_overall_level(criterion_objs_history[turn - 1], type)
    curr_level = get_overall_level(criterion_objs_history[turn], type)
    return curr_level - prev_level

def get_session_score(chat_history: List[Dict[str, str]], criterion_objs_history: List[Dict[str, Any]], type: str) -> float:
    """
    Calculates the "awareness" or "satisfaction" level for a set of criteria over the entire session.
    """
    initial_level = get_overall_level(criterion_objs_history[0], type)
    final_level = get_overall_level(criterion_objs_history[-1], type)
    num_assistant_turns = len([msg for msg in chat_history if msg["role"] == "assistant"])
    if num_assistant_turns == 0:
        return 0.0
    return (final_level - initial_level) / num_assistant_turns

def calculate_criterion_score_delta(new_criteria_obj: Dict[str, Any], prev_criteria_obj: Dict[str, Any], type: str, weight_step: float = 0) -> Optional[float]:
    """
    Calculates the change in the "awareness" or "satisfaction" level for a criterion.
    """
    prev_score, _, _ = score_hierarchy(prev_criteria_obj["hierarchy"], type)
    new_score, _, total_count = score_hierarchy(new_criteria_obj["hierarchy"], type)

    # denominator = total_count - prev_score
    # if denominator == 0:
    #     return None
    return (new_score - prev_score) # / denominator

def calculate_score_delta(new_criteria_objs: List[Dict[str, Any]], prev_criteria_objs: List[Dict[str, Any]], type: str, weight_step: float = 0) -> float:
    """
    Calculates the change in the "awareness" or "satisfaction" level for a set of criteria.
    Returns normalized value (divided by number of criteria).
    """
    total_change = 0
    num_valid_criteria = 0
    for i in range(len(new_criteria_objs)):
        criterion_change = calculate_criterion_score_delta(new_criteria_objs[i], prev_criteria_objs[i], type, weight_step)
        if criterion_change is None:
            continue
        total_change += criterion_change
        num_valid_criteria += 1
    if num_valid_criteria == 0:
        return 0
    return total_change / num_valid_criteria


def calculate_score_delta_raw(new_criteria_objs: List[Dict[str, Any]], prev_criteria_objs: List[Dict[str, Any]], type: str) -> float:
    """
    Calculates the raw (unnormalized) change in the "awareness" or "satisfaction" level.
    Returns the sum of all score changes without dividing by the number of criteria.
    """
    total_change = 0
    for i in range(len(new_criteria_objs)):
        criterion_change = calculate_criterion_score_delta(new_criteria_objs[i], prev_criteria_objs[i], type)
        if criterion_change is not None:
            total_change += criterion_change
    return total_change


def get_overall_level_raw(criteria_objs: List[Dict[str, Any]], type: str) -> float:
    """
    Calculates the raw (unnormalized) total "awareness" or "satisfaction" score.
    Returns the sum of all scores without dividing by total count.
    """
    total_score = 0
    for criteria_obj in criteria_objs:
        score, _, _ = score_hierarchy(criteria_obj["hierarchy"], type)
        total_score += score
    return total_score
