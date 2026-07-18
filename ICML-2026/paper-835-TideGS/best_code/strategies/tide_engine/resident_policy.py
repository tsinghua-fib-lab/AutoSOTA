from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ResidentTransition:
    """Explicit resident-set transition for paper mode."""

    current_active_blocks: List[int]
    next_active_blocks: List[int]
    current_resident_blocks: List[int]
    candidate_blocks: List[int]
    next_resident_blocks: List[int]
    keep_resident_blocks: List[int]
    stream_in_blocks: List[int]
    evict_blocks: List[int]
    active_overlap_blocks: List[int]
    active_delta_plus_blocks: List[int]
    active_delta_minus_blocks: List[int]
    next_camera_ids: List[int]
    resident_selection_policy: str = "passthrough_active_set"
    resident_capacity_blocks: int = -1
    requested_resident_capacity_blocks: int = -1
    resident_capacity_bytes: int = -1
    current_resident_source: str = "current_active_fallback"
    resident_lambda_weight: float = 1.0
    resident_recency_decay: float = 1.0
    balanced_seed_fraction: float = 1.0
    topc_cutoff_score: float = 0.0
    next_active_coverage: int = 0
    next_camera_coverage: int = 0
    next_camera_total: int = 0
    enforce_next_active_coverage: bool = False
    updated_recency_scores: Optional[Dict[int, float]] = None
    optional_selected_blocks: Optional[List[int]] = None
    camera_seed_blocks: Optional[List[int]] = None

    def __post_init__(self):
        if self.updated_recency_scores is None:
            self.updated_recency_scores = {}
        if self.optional_selected_blocks is None:
            self.optional_selected_blocks = []
        if self.camera_seed_blocks is None:
            self.camera_seed_blocks = []

    def to_dict(self) -> Dict[str, object]:
        return {
            "current_blocks": self.current_active_blocks,
            "next_blocks": self.next_active_blocks,
            "omega_blocks": self.active_overlap_blocks,
            "delta_plus_blocks": self.active_delta_plus_blocks,
            "delta_minus_blocks": self.active_delta_minus_blocks,
            "current_resident_blocks": self.current_resident_blocks,
            "candidate_blocks": self.candidate_blocks,
            "next_resident_blocks": self.next_resident_blocks,
            "keep_resident_blocks": self.keep_resident_blocks,
            "stream_in_blocks": self.stream_in_blocks,
            "evict_blocks": self.evict_blocks,
            "next_camera_ids": self.next_camera_ids,
            "resident_selection_policy": self.resident_selection_policy,
            "resident_capacity_blocks": self.resident_capacity_blocks,
            "requested_resident_capacity_blocks": self.requested_resident_capacity_blocks,
            "resident_capacity_bytes": self.resident_capacity_bytes,
            "current_resident_source": self.current_resident_source,
            "resident_lambda_weight": self.resident_lambda_weight,
            "resident_recency_decay": self.resident_recency_decay,
            "balanced_seed_fraction": self.balanced_seed_fraction,
            "topc_cutoff_score": self.topc_cutoff_score,
            "next_active_coverage": self.next_active_coverage,
            "next_camera_coverage": self.next_camera_coverage,
            "next_camera_total": self.next_camera_total,
            "enforce_next_active_coverage": self.enforce_next_active_coverage,
            "updated_recency_scores": self.updated_recency_scores,
            "optional_selected_blocks": self.optional_selected_blocks,
            "camera_seed_blocks": self.camera_seed_blocks,
        }


def _sorted_unique_ints(block_ids: Optional[List[int]]) -> List[int]:
    if not block_ids:
        return []
    return sorted(set(int(block_id) for block_id in block_ids))


def _resolve_passthrough_current_resident(
    current_active_blocks: List[int],
    current_resident_blocks: Optional[List[int]],
):
    current_active = _sorted_unique_ints(current_active_blocks)
    if current_resident_blocks:
        requested_current_resident = _sorted_unique_ints(current_resident_blocks)
        if set(requested_current_resident) == set(current_active):
            return requested_current_resident, "expected_resident_match"
    return current_active, "current_active_fallback"


def _resolve_scored_current_resident(
    current_active_blocks: List[int],
    current_resident_blocks: Optional[List[int]],
):
    current_active = _sorted_unique_ints(current_active_blocks)
    if current_resident_blocks:
        resolved = _sorted_unique_ints(current_resident_blocks)
        if set(resolved) == set(current_active):
            return resolved, "expected_resident_match"
        return resolved, "loaded_resident_state"
    return current_active, "current_active_fallback"


def _update_recency_scores(
    previous_recency_scores: Optional[Dict[int, float]],
    accessed_blocks: List[int],
    recency_decay: float,
) -> Dict[int, float]:
    updated: Dict[int, float] = {}
    for block_id, score in (previous_recency_scores or {}).items():
        decayed = float(score) * float(recency_decay)
        if decayed > 1e-6:
            updated[int(block_id)] = decayed

    for block_id in _sorted_unique_ints(accessed_blocks):
        updated[int(block_id)] = 1.0

    return updated


def _normalize_camera_blocks(
    camera_blocks: Optional[Dict[int, List[int]]],
    allowed_blocks: Optional[Set[int]] = None,
) -> List[Tuple[int, List[int]]]:
    if not camera_blocks:
        return []

    normalized: List[Tuple[int, List[int]]] = []
    for camera_id in sorted(camera_blocks):
        blocks = _sorted_unique_ints(camera_blocks.get(camera_id, []))
        if allowed_blocks is not None:
            blocks = [block_id for block_id in blocks if block_id in allowed_blocks]
        if blocks:
            normalized.append((int(camera_id), blocks))
    return normalized


def _camera_coverage_counts(
    camera_blocks: Optional[Dict[int, List[int]]],
    selected_blocks: Set[int],
    allowed_blocks: Optional[Set[int]] = None,
) -> Tuple[int, int]:
    normalized = _normalize_camera_blocks(camera_blocks, allowed_blocks)
    total = len(normalized)
    if total == 0:
        return 0, 0
    covered = sum(
        1 for _, blocks in normalized
        if any(block_id in selected_blocks for block_id in blocks)
    )
    return covered, total


def _select_camera_balanced_seed_blocks(
    camera_blocks: Optional[Dict[int, List[int]]],
    candidate_blocks: Set[int],
    scores: Dict[int, float],
    current_resident_set: Set[int],
    capacity: int,
) -> List[int]:
    """Pick a capacity-bounded seed set that gives each camera a fair chance.

    When TopC capacity is smaller than the full active set, ranking all blocks
    globally can accidentally select blocks for only a few cameras.  This helper
    keeps the same score signal, but applies it inside a round-robin pass across
    cameras first.  The remaining capacity is still filled by global TopC.
    """
    if capacity <= 0:
        return []

    normalized = _normalize_camera_blocks(camera_blocks, candidate_blocks)
    if not normalized:
        return []

    frequency: Dict[int, int] = {}
    for _, blocks in normalized:
        for block_id in blocks:
            frequency[block_id] = frequency.get(block_id, 0) + 1

    def camera_rank_key(block_id: int):
        return (
            -float(scores.get(block_id, 0.0)),
            -int(frequency.get(block_id, 0)),
            -(1 if block_id in current_resident_set else 0),
            block_id,
        )

    ordered_per_camera = [
        sorted(blocks, key=camera_rank_key)
        for _, blocks in normalized
    ]
    positions = [0 for _ in ordered_per_camera]
    selected: List[int] = []
    selected_set: Set[int] = set()

    while len(selected) < capacity:
        made_progress = False
        for i, blocks in enumerate(ordered_per_camera):
            while positions[i] < len(blocks) and blocks[positions[i]] in selected_set:
                positions[i] += 1
            if positions[i] >= len(blocks):
                continue

            block_id = blocks[positions[i]]
            positions[i] += 1
            selected.append(block_id)
            selected_set.add(block_id)
            made_progress = True

            if len(selected) >= capacity:
                break

        if not made_progress:
            break

    return selected


def _resolve_balanced_seed_capacity(
    camera_blocks: Optional[Dict[int, List[int]]],
    candidate_blocks: Set[int],
    capacity: int,
    seed_fraction: float,
) -> int:
    if capacity <= 0:
        return 0
    fraction = max(0.0, min(1.0, float(seed_fraction)))
    if fraction >= 1.0:
        return int(capacity)
    if fraction <= 0.0:
        return 0

    camera_count = len(_normalize_camera_blocks(camera_blocks, candidate_blocks))
    quota = int(capacity * fraction)
    if quota <= 0:
        quota = 1
    if camera_count > 0:
        quota = max(quota, min(int(capacity), camera_count))
    return min(int(capacity), quota)


def compute_passthrough_resident_transition(
    current_active_blocks: List[int],
    next_active_blocks: List[int],
    current_resident_blocks: Optional[List[int]] = None,
    next_camera_ids: Optional[List[int]] = None,
    next_camera_blocks: Optional[Dict[int, List[int]]] = None,
    previous_recency_scores: Optional[Dict[int, float]] = None,
    recency_decay: float = 0.95,
) -> ResidentTransition:
    current_active = _sorted_unique_ints(current_active_blocks)
    next_active = _sorted_unique_ints(next_active_blocks)
    current_resident, current_resident_source = _resolve_passthrough_current_resident(
        current_active,
        current_resident_blocks,
    )
    next_resident = list(next_active)

    current_resident_set = set(current_resident)
    next_active_set = set(next_active)
    next_resident_set = set(next_resident)
    current_active_set = set(current_active)

    candidate_blocks = sorted(current_resident_set | next_active_set)
    keep_resident_blocks = sorted(current_resident_set & next_resident_set)
    stream_in_blocks = sorted(next_resident_set - current_resident_set)
    evict_blocks = sorted(current_resident_set - next_resident_set)

    active_overlap_blocks = sorted(current_active_set & next_active_set)
    active_delta_plus_blocks = sorted(next_active_set - current_active_set)
    active_delta_minus_blocks = sorted(current_active_set - next_active_set)
    updated_recency_scores = _update_recency_scores(
        previous_recency_scores,
        accessed_blocks=current_active,
        recency_decay=recency_decay,
    )
    next_camera_coverage, next_camera_total = _camera_coverage_counts(
        next_camera_blocks,
        next_resident_set,
        next_active_set,
    )

    return ResidentTransition(
        current_active_blocks=current_active,
        next_active_blocks=next_active,
        current_resident_blocks=current_resident,
        candidate_blocks=candidate_blocks,
        next_resident_blocks=next_resident,
        keep_resident_blocks=keep_resident_blocks,
        stream_in_blocks=stream_in_blocks,
        evict_blocks=evict_blocks,
        active_overlap_blocks=active_overlap_blocks,
        active_delta_plus_blocks=active_delta_plus_blocks,
        active_delta_minus_blocks=active_delta_minus_blocks,
        next_camera_ids=list(next_camera_ids or []),
        current_resident_source=current_resident_source,
        resident_lambda_weight=1.0,
        resident_recency_decay=recency_decay,
        topc_cutoff_score=1.0 if next_resident else 0.0,
        next_active_coverage=len(next_active),
        next_camera_coverage=next_camera_coverage,
        next_camera_total=next_camera_total,
        updated_recency_scores=updated_recency_scores,
    )


def compute_topc_resident_transition(
    current_active_blocks: List[int],
    next_active_blocks: List[int],
    current_resident_blocks: Optional[List[int]] = None,
    next_camera_ids: Optional[List[int]] = None,
    next_camera_blocks: Optional[Dict[int, List[int]]] = None,
    previous_recency_scores: Optional[Dict[int, float]] = None,
    lambda_weight: float = 0.7,
    recency_decay: float = 0.95,
    resident_capacity_blocks: int = -1,
    balanced_camera_seeds: bool = False,
    balanced_seed_fraction: float = 1.0,
) -> ResidentTransition:
    current_active = _sorted_unique_ints(current_active_blocks)
    next_active = _sorted_unique_ints(next_active_blocks)
    current_resident, current_resident_source = _resolve_scored_current_resident(
        current_active,
        current_resident_blocks,
    )

    updated_recency_scores = _update_recency_scores(
        previous_recency_scores,
        accessed_blocks=current_active,
        recency_decay=recency_decay,
    )

    current_resident_set = set(current_resident)
    next_active_set = set(next_active)
    current_active_set = set(current_active)
    candidate_blocks = sorted(current_resident_set | next_active_set)

    requested_capacity = int(resident_capacity_blocks)
    if requested_capacity < 0:
        effective_capacity = len(candidate_blocks)
    else:
        effective_capacity = max(0, requested_capacity)
    effective_capacity = min(effective_capacity, len(candidate_blocks))

    scores: Dict[int, float] = {}
    for block_id in candidate_blocks:
        next_step_useful = 1.0 if block_id in next_active_set else 0.0
        recency_score = float(updated_recency_scores.get(block_id, 0.0))
        scores[block_id] = float(lambda_weight) * next_step_useful + (1.0 - float(lambda_weight)) * recency_score

    ranking_key = lambda block_id: (
        -scores[block_id],
        -float(updated_recency_scores.get(block_id, 0.0)),
        -(1 if block_id in current_resident_set else 0),
        block_id,
    )

    camera_seed_blocks: List[int] = []
    if balanced_camera_seeds and effective_capacity > 0:
        seed_capacity = _resolve_balanced_seed_capacity(
            camera_blocks=next_camera_blocks,
            candidate_blocks=next_active_set,
            capacity=effective_capacity,
            seed_fraction=balanced_seed_fraction,
        )
        camera_seed_blocks = _select_camera_balanced_seed_blocks(
            camera_blocks=next_camera_blocks,
            candidate_blocks=next_active_set,
            scores=scores,
            current_resident_set=current_resident_set,
            capacity=seed_capacity,
        )

    selected_blocks = list(camera_seed_blocks)
    selected_set = set(selected_blocks)
    remaining_capacity = max(0, effective_capacity - len(selected_blocks))
    if remaining_capacity > 0:
        selected_blocks.extend([
            block_id
            for block_id in sorted(candidate_blocks, key=ranking_key)
            if block_id not in selected_set
        ][:remaining_capacity])
    selected_blocks = selected_blocks[:effective_capacity]
    next_resident_set = set(selected_blocks)
    optional_selected_blocks = sorted(
        block_id for block_id in selected_blocks if block_id not in next_active_set
    )
    enforce_next_active_coverage = False

    next_resident = sorted(next_resident_set)

    keep_resident_blocks = sorted(current_resident_set & next_resident_set)
    stream_in_blocks = sorted(next_resident_set - current_resident_set)
    evict_blocks = sorted(current_resident_set - next_resident_set)

    active_overlap_blocks = sorted(current_active_set & next_active_set)
    active_delta_plus_blocks = sorted(next_active_set - current_active_set)
    active_delta_minus_blocks = sorted(current_active_set - next_active_set)

    selected_scores = [scores[block_id] for block_id in next_resident]
    topc_cutoff_score = min(selected_scores) if selected_scores else 0.0
    next_camera_coverage, next_camera_total = _camera_coverage_counts(
        next_camera_blocks,
        next_resident_set,
        next_active_set,
    )

    return ResidentTransition(
        current_active_blocks=current_active,
        next_active_blocks=next_active,
        current_resident_blocks=current_resident,
        candidate_blocks=candidate_blocks,
        next_resident_blocks=next_resident,
        keep_resident_blocks=keep_resident_blocks,
        stream_in_blocks=stream_in_blocks,
        evict_blocks=evict_blocks,
        active_overlap_blocks=active_overlap_blocks,
        active_delta_plus_blocks=active_delta_plus_blocks,
        active_delta_minus_blocks=active_delta_minus_blocks,
        next_camera_ids=list(next_camera_ids or []),
            resident_selection_policy="topc_balanced" if balanced_camera_seeds else "topc_strict",
        resident_capacity_blocks=effective_capacity,
        requested_resident_capacity_blocks=requested_capacity,
        current_resident_source=current_resident_source,
        resident_lambda_weight=float(lambda_weight),
        resident_recency_decay=float(recency_decay),
        balanced_seed_fraction=float(balanced_seed_fraction) if balanced_camera_seeds else 0.0,
        topc_cutoff_score=float(topc_cutoff_score),
        next_active_coverage=len(next_active_set & next_resident_set),
        next_camera_coverage=next_camera_coverage,
        next_camera_total=next_camera_total,
        enforce_next_active_coverage=enforce_next_active_coverage,
        updated_recency_scores=updated_recency_scores,
        optional_selected_blocks=optional_selected_blocks,
        camera_seed_blocks=camera_seed_blocks,
    )
