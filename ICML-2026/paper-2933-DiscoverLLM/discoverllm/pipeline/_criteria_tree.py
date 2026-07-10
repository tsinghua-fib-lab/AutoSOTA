"""
Pure tree-navigation helpers for the criteria-objects structure used by
:class:`discoverllm.pipeline.user_simulator.UserSimulator`.

A criteria_objs payload is a list of criterion dicts; each has a
``hierarchy`` of root nodes; each root has nested ``children`` nodes. The
simulator traverses this tree by *flat* root-node position so it can
advance through one root at a time, regardless of which top-level criterion
the root belongs to. The functions in this module manipulate that
flat-position view without making any LLM calls — they're cheap, pure, and
straightforward to test in isolation.

Names are underscore-prefixed because they're considered internal to the
pipeline package; callers should go through ``UserSimulator``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple


def _is_root_node_tree_satisfied(node: Dict[str, Any]) -> bool:
    """Whether every leaf in this subtree has ``satisfied == 1``.

    Intermediate nodes' own ``satisfied`` flag is irrelevant — only the
    leaves count.
    """
    children = node.get("children", [])
    if not children:
        return node.get("satisfied", 0) == 1
    return all(_is_root_node_tree_satisfied(child) for child in children)


def _count_total_root_nodes(criteria_objs: List[Dict[str, Any]]) -> int:
    """Total root-node count across every criterion."""
    return sum(len(c.get("hierarchy", [])) for c in criteria_objs)


def _get_root_node_at_position(
    criteria_objs: List[Dict[str, Any]],
    position: int,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Map a flat position back to ``(criterion_idx, root_idx, root_node)``.

    Raises ``IndexError`` if ``position`` is outside the flat range.
    """
    current_pos = 0
    for criterion_idx, criterion_obj in enumerate(criteria_objs):
        for root_idx, root_node in enumerate(criterion_obj.get("hierarchy", [])):
            if current_pos == position:
                return criterion_idx, root_idx, root_node
            current_pos += 1
    raise IndexError(
        f"Position {position} out of range for {current_pos} total root nodes"
    )


def _extract_root_nodes_up_to_position(
    criteria_objs: List[Dict[str, Any]],
    end_position: int,
) -> List[Dict[str, Any]]:
    """
    Build a one-criterion view containing every root from positions 0…end.

    Used during response generation: the simulator wants to see all
    completed roots plus the current root, but no future ones.
    """
    if end_position < 0:
        return []
    extracted_roots: List[Dict[str, Any]] = []
    base_criterion: Dict[str, Any] | None = None
    current_pos = 0
    for criterion_obj in criteria_objs:
        for root_node in criterion_obj.get("hierarchy", []):
            if current_pos <= end_position:
                if base_criterion is None:
                    base_criterion = copy.deepcopy(criterion_obj)
                    base_criterion["hierarchy"] = []
                extracted_roots.append(copy.deepcopy(root_node))
            current_pos += 1
    if base_criterion is None:
        return []
    base_criterion["hierarchy"] = extracted_roots
    return [base_criterion]


def _extract_root_nodes_at_positions(
    criteria_objs: List[Dict[str, Any]],
    positions: List[int],
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """
    Build a one-criterion view containing only the roots at ``positions``.

    Returns ``(view, mapping)`` where ``mapping[i]`` is the original
    ``(criterion_idx, root_idx)`` of the i-th extracted root — needed so the
    caller can merge updated roots back into the full structure.
    """
    if not positions:
        return [], []
    extracted_roots: List[Dict[str, Any]] = []
    position_mapping: List[Tuple[int, int]] = []
    base_criterion: Dict[str, Any] | None = None
    current_pos = 0
    for criterion_idx, criterion_obj in enumerate(criteria_objs):
        for root_idx, root_node in enumerate(criterion_obj.get("hierarchy", [])):
            if current_pos in positions:
                if base_criterion is None:
                    base_criterion = copy.deepcopy(criterion_obj)
                    base_criterion["hierarchy"] = []
                extracted_roots.append(copy.deepcopy(root_node))
                position_mapping.append((criterion_idx, root_idx))
            current_pos += 1
    if base_criterion is None:
        return [], []
    base_criterion["hierarchy"] = extracted_roots
    return [base_criterion], position_mapping


def _merge_root_node_back(
    criteria_objs: List[Dict[str, Any]],
    updated_root_node: Dict[str, Any],
    criterion_idx: int,
    root_idx: int,
) -> List[Dict[str, Any]]:
    """
    Return a deep-copied criteria_objs with one root node replaced.
    """
    criteria_objs = copy.deepcopy(criteria_objs)
    criteria_objs[criterion_idx]["hierarchy"][root_idx] = updated_root_node
    return criteria_objs


def _find_first_unsatisfied_root_position(criteria_objs: List[Dict[str, Any]]) -> int:
    """
    Find the flat position of the first unsatisfied root.
    Returns the total root count if every root is satisfied.
    """
    position = 0
    for criterion_obj in criteria_objs:
        for root_node in criterion_obj.get("hierarchy", []):
            if not _is_root_node_tree_satisfied(root_node):
                return position
            position += 1
    return position
