"""Release-only Pure SSD/Tide training facade.

This module is the narrow entry point used by ``train_tidegs.py``.
It keeps release training constrained to the TideGS out-of-core configuration.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from storage.schedule_utils import get_current_and_next_camera_batches

from .resident_policy import (
    compute_passthrough_resident_transition,
    compute_topc_resident_transition,
)


def _require_attr(args: Any, name: str, expected: Any) -> None:
    value = getattr(args, name, None)
    if value != expected:
        raise RuntimeError(
            f"Pure SSD/Tide release path requires --{name} {expected!r}; got {value!r}."
        )


def _require_lower(args: Any, name: str, expected: str) -> None:
    value = str(getattr(args, name, "")).lower()
    if value != expected:
        raise RuntimeError(
            f"Pure SSD/Tide release path requires --{name} {expected}; got {value!r}."
        )


def validate_tide_runtime_args(args: Any) -> None:
    """Validate the release engine contract before entering the shared execution core."""
    _require_attr(args, "pure_ssd_offload", True)
    _require_attr(args, "use_ssd_offload", True)
    _require_attr(args, "clm_offload", True)
    if getattr(args, "naive_offload", False) or getattr(args, "no_offload", False):
        raise RuntimeError("Pure SSD/Tide release path does not support naive/no-offload modes.")

    _require_lower(args, "ssd_execution_mode", "paper")
    _require_lower(args, "paper_block_reader_backend", "tiered_cache")
    _require_lower(args, "paper_optimizer_backend", "gpu_resident")
    _require_lower(args, "paper_optimizer_state_mode", "resident_blocks")
    _require_lower(args, "paper_optimizer_deferred_mode", "off")
    _require_attr(args, "paper_free_unified_params", True)
    _require_attr(args, "disable_auto_densification", True)


def paper_debug_logging_enabled(args: Any, is_paper_ssd_mode: bool = True) -> bool:
    """Return whether verbose development diagnostics should be emitted."""
    return bool(is_paper_ssd_mode and getattr(args, "paper_debug_logging", False))


def run_stage0_schedule_interaction(
    *,
    storage_adapter,
    training_schedule,
    iteration: int,
    batch_size: int,
    schedule_ordering: str,
) -> str:
    """Handle Stage0 schedule interaction without reloading full-K.

    TideGS selects the resident set from the camera schedule and materializes it
    on demand via TieredCacheBlockReader during Stage1.
    """
    execution_mode = (
        getattr(storage_adapter, "execution_mode", "fast_ram")
        if storage_adapter is not None
        else "fast_ram"
    )
    if storage_adapter is not None and training_schedule is not None and execution_mode != "paper":
        storage_adapter.prefetch_for_next_iteration(
            iteration=iteration,
            batch_size=batch_size,
            training_schedule=training_schedule,
            schedule_ordering=schedule_ordering,
        )
    return str(execution_mode)


def resolve_paper_stage0_active_blocks(
    *,
    is_paper_ssd_mode: bool,
    iteration: int,
    should_log_paper_sets: bool,
    log_file=None,
) -> Tuple[Optional[Dict[int, torch.Tensor]], bool]:
    """Resolve Stage0 output for the TideGS runtime.

    TideGS has no Stage0 full-K active block payload.  Returning ``None`` is
    intentional: Stage1 loads the resident set from the tiered block reader.
    """
    if not is_paper_ssd_mode:
        return None, False
    if should_log_paper_sets:
        write_paper_phase1_log(
            f"[PAPER PIPELINE] Iter {iteration}: skipped Stage0 full-K wait; "
            "current blocks will be served by TieredCacheBlockReader on demand.\n",
            log_file=log_file,
        )
    return None, True


def log_paper_batch_debug(
    *,
    enabled: bool,
    iteration: int,
    batched_cameras: List[Any],
    current_camera_ids: List[int],
) -> None:
    if not (enabled and iteration == 1):
        return
    print(f"\n[BATCH DEBUG] Iter {iteration}: Batch contains {len(batched_cameras)} cameras")
    print(f"  Global camera IDs: {current_camera_ids}")


def log_paper_block_visibility_debug(
    *,
    enabled: bool,
    iteration: int,
    cam_to_blocks: Dict[int, int],
    visible_block_ids: List[int],
) -> None:
    if not (enabled and iteration == 1):
        return
    print("[BLOCK VISIBILITY] Per-camera stats:")
    for cam_id, num_blocks in sorted(cam_to_blocks.items())[:10]:
        print(f"  Camera {cam_id}: {num_blocks} blocks")
    if len(cam_to_blocks) > 10:
        print(f"  ... (showing 10/{len(cam_to_blocks)} cameras)")
    print(f"  Total unique blocks across batch: {len(visible_block_ids)}")


def log_ssd_stage1_debug(
    *,
    enabled: bool,
    log_file,
    iteration: int,
    ssd_execution_mode: str,
    n_gaussians: int,
    block_size: int,
    max_valid_block_id: int,
    active_blocks_ram,
    is_paper_ssd_mode: bool,
    visible_block_ids: List[int],
) -> None:
    if not (enabled and iteration == 1 and log_file is not None):
        return
    log_file.write(
        f"[SSD DEBUG] execution_mode={ssd_execution_mode}, "
        f"n_gaussians={n_gaussians:,}, block_size={block_size}\n"
    )
    log_file.write(f"[SSD DEBUG] Valid block range: [0, {max_valid_block_id}]\n")
    if active_blocks_ram is not None:
        log_file.write(
            f"[SSD DEBUG] Loaded {len(active_blocks_ram)} blocks from RAM cache: "
            f"{sorted(active_blocks_ram.keys())[:20]}\n"
        )
    elif is_paper_ssd_mode:
        log_file.write(
            f"[SSD DEBUG] Paper mode uses TieredCacheBlockReader on demand for "
            f"{len(visible_block_ids)} visible blocks\n"
        )
    else:
        log_file.write(
            f"[SSD DEBUG] Using unified_params fast path, {len(visible_block_ids)} visible blocks\n"
        )


def resolve_stage2_loaded_gaussian_ids(
    *,
    gaussians,
    loaded_gaussian_mask,
    total_n_gaussians: int,
    is_paper_ssd_mode: bool,
    iteration: int,
    log_file,
    ensure_local_to_global_mapping_fn: Callable,
) -> torch.Tensor:
    """Return loaded Gaussian ids without constructing a full mask in paper mode."""
    if is_paper_ssd_mode and getattr(gaussians, "use_gpu_features", False):
        return ensure_local_to_global_mapping_fn(
            gaussians,
            total_n_gaussians,
            log_file=log_file,
            context=f"stage2_loaded_ids_iter_{iteration}",
        )

    if loaded_gaussian_mask is None:
        raise RuntimeError(
            "Non-Tide SSD path expected loaded_gaussian_mask, but it was not populated"
        )
    return torch.nonzero(loaded_gaussian_mask).squeeze(1)


def log_empty_stage2_loaded_gaussians(
    *,
    num_loaded: int,
    iteration: int,
    visible_block_ids: List[int],
    active_blocks_ram,
    ssd_execution_mode: str,
    bsz: int,
    log_file,
) -> Optional[Tuple[List[Any], List[int], float]]:
    """Emit the existing empty-load warning and return the graceful skip payload."""
    if int(num_loaded) != 0:
        return None
    ram_count = (
        len(active_blocks_ram)
        if active_blocks_ram is not None
        else f"N/A ({ssd_execution_mode} mode)"
    )
    if log_file is not None:
        log_file.write(
            f"\n[CRITICAL WARNING] Iter {iteration}: No Gaussians loaded from SSD!\n"
            f"  This iteration will be skipped.\n"
            f"  Root cause analysis:\n"
            f"    - visible_block_ids: {len(visible_block_ids)}\n"
            f"    - active_blocks_ram: {ram_count}\n"
            f"  Check FrustumCuller visibility or SSD I/O.\n"
        )
    return [], list(range(bsz)), 0.0


def log_paper_gpu_working_set_parameters(
    *,
    iteration: int,
    xyz_compact: torch.Tensor,
    opacity_compact: torch.Tensor,
    scaling_compact: torch.Tensor,
    rotation_compact: torch.Tensor,
    batched_cameras: List[Any],
    log_file,
) -> None:
    """Log the compact GPU working set once, keeping release terminal output quiet."""
    if not (iteration == 1 and log_file is not None):
        return
    log_file.write("\n[PAPER MODE] Parameters in GPU working set:\n")
    log_file.write(f"  xyz: {xyz_compact.device}, shape: {xyz_compact.shape}\n")
    log_file.write(f"  opacity: {opacity_compact.device}, shape: {opacity_compact.shape}\n")
    log_file.write(f"  scaling: {scaling_compact.device}, shape: {scaling_compact.shape}\n")
    log_file.write(f"  rotation: {rotation_compact.device}, shape: {rotation_compact.shape}\n")
    if len(batched_cameras) > 0:
        cam0 = batched_cameras[0]
        log_file.write(f"  Cam[0].world_view_transform: {cam0.world_view_transform.device}\n")


def map_compact_filters_to_global(
    *,
    filters_compact: List[torch.Tensor],
    local_to_global: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Map projection-local filter ids to global ids while preserving local filters."""
    filters_local: List[torch.Tensor] = []
    filters_global: List[torch.Tensor] = []
    for filter_compact in filters_compact:
        if filter_compact.numel() > 0:
            filters_local.append(filter_compact)
            filters_global.append(local_to_global[filter_compact])
        else:
            empty = torch.empty(
                (0,),
                dtype=torch.int64,
                device=local_to_global.device,
            )
            filters_local.append(empty)
            filters_global.append(empty)
    return filters_local, filters_global


def log_empty_paper_projection_cameras(
    *,
    is_paper_ssd_mode: bool,
    iteration: int,
    current_camera_ids: List[int],
    filters_global: List[torch.Tensor],
    log_file,
) -> None:
    if not is_paper_ssd_mode:
        return
    empty_projection_cameras = [
        int(current_camera_ids[i])
        for i, filter_global in enumerate(filters_global)
        if filter_global.numel() == 0
    ]
    if empty_projection_cameras:
        write_paper_phase1_log(
            f"[PAPER RESIDENT CAMERA] Iter {iteration}: "
            f"{len(empty_projection_cameras)}/{len(filters_global)} cameras have zero Gaussian-level projection "
            f"after resident block selection; camera_ids_first8={empty_projection_cameras[:8]}\n",
            log_file=log_file,
        )


def collect_camera_visible_blocks(storage_adapter, camera_ids: List[int]) -> Dict[int, List[int]]:
    camera_blocks: Dict[int, List[int]] = {}
    for cam_id in camera_ids:
        camera_blocks[int(cam_id)] = list(storage_adapter.get_visible_blocks(int(cam_id)))
    return camera_blocks


def union_camera_visible_blocks(camera_blocks: Dict[int, List[int]]) -> List[int]:
    block_ids = set()
    for blocks in camera_blocks.values():
        block_ids.update(int(block_id) for block_id in blocks)
    return sorted(block_ids)


def compute_paper_block_sets(
    storage_adapter,
    training_schedule,
    iteration: int,
    batch_size: int,
    current_block_ids: List[int],
    schedule_ordering: str = "trajectory",
    current_resident_blocks: Optional[List[int]] = None,
    current_resident_recency_scores: Optional[Dict[int, float]] = None,
    resident_selection_policy: str = "passthrough_active_set",
    resident_lambda: float = 0.7,
    resident_recency_decay: float = 0.95,
    resident_capacity_blocks: int = -1,
    balanced_seed_fraction: float = 1.0,
) -> Dict[str, object]:
    current_blocks = sorted(set(int(block_id) for block_id in current_block_ids))

    if training_schedule is None or len(training_schedule) == 0:
        next_camera_ids: List[int] = []
        next_camera_blocks: Dict[int, List[int]] = {}
        next_blocks: List[int] = []
    else:
        _, next_batch = get_current_and_next_camera_batches(
            training_schedule=training_schedule,
            iteration=iteration,
            batch_size=batch_size,
            schedule_ordering=schedule_ordering,
        )
        next_camera_ids = list(next_batch.batch_indices)
        next_camera_blocks = collect_camera_visible_blocks(storage_adapter, next_camera_ids)
        next_blocks = union_camera_visible_blocks(next_camera_blocks)

    if resident_selection_policy in {"topc", "topc_strict", "topc_balanced"}:
        transition = compute_topc_resident_transition(
            current_active_blocks=current_blocks,
            next_active_blocks=next_blocks,
            current_resident_blocks=current_resident_blocks,
            next_camera_ids=next_camera_ids,
            next_camera_blocks=next_camera_blocks,
            previous_recency_scores=current_resident_recency_scores,
            lambda_weight=resident_lambda,
            recency_decay=resident_recency_decay,
            resident_capacity_blocks=resident_capacity_blocks,
            balanced_camera_seeds=(resident_selection_policy == "topc_balanced"),
            balanced_seed_fraction=balanced_seed_fraction,
        )
    else:
        transition = compute_passthrough_resident_transition(
            current_active_blocks=current_blocks,
            next_active_blocks=next_blocks,
            current_resident_blocks=current_resident_blocks,
            next_camera_ids=next_camera_ids,
            next_camera_blocks=next_camera_blocks,
            previous_recency_scores=current_resident_recency_scores,
            recency_decay=resident_recency_decay,
        )
    return transition.to_dict()


def compute_batch_kt_metrics(
    current_camera_blocks: Dict[int, List[int]],
    visible_block_ids: List[int],
    resident_blocks: Optional[List[int]] = None,
) -> Dict[str, float]:
    per_camera_counts = [len(blocks) for blocks in current_camera_blocks.values()]
    camera_count = len(per_camera_counts)
    union_count = len(visible_block_ids)
    sum_count = int(sum(per_camera_counts))
    mean_count = (float(sum_count) / camera_count) if camera_count > 0 else 0.0
    min_count = int(min(per_camera_counts)) if per_camera_counts else 0
    max_count = int(max(per_camera_counts)) if per_camera_counts else 0
    overlap_ratio = 0.0
    if sum_count > 0 and camera_count > 1:
        overlap_ratio = 1.0 - (float(union_count) / float(sum_count))

    resident_count = len(resident_blocks or [])
    resident_intersection = 0
    resident_coverage = 0.0
    if union_count > 0 and resident_blocks is not None:
        visible_set = set(int(b) for b in visible_block_ids)
        resident_intersection = len(visible_set.intersection(int(b) for b in resident_blocks))
        resident_coverage = float(resident_intersection) / float(union_count)

    return {
        "camera_count": camera_count,
        "per_camera_min": min_count,
        "per_camera_mean": mean_count,
        "per_camera_max": max_count,
        "sum_blocks": sum_count,
        "union_blocks": union_count,
        "overlap_ratio": overlap_ratio,
        "resident_blocks": resident_count,
        "resident_intersection": resident_intersection,
        "resident_coverage": resident_coverage,
    }


def resolve_current_iteration_resident_blocks(
    gaussians,
    args,
    visible_block_ids: List[int],
    num_total_blocks: int,
    current_camera_blocks: Optional[Dict[int, List[int]]] = None,
) -> Tuple[List[int], str]:
    policy = str(getattr(args, "paper_resident_selection_policy", "passthrough_active_set")).lower()
    requested_capacity = int(getattr(args, "paper_resident_capacity_blocks", -1))
    visible_set = set(int(b) for b in visible_block_ids if 0 <= int(b) < num_total_blocks)

    if policy not in {"topc", "topc_strict", "topc_balanced"}:
        return sorted(visible_set), "passthrough_active_set"

    expected = [
        int(b) for b in (getattr(gaussians, "_paper_expected_resident_blocks", []) or [])
        if 0 <= int(b) < num_total_blocks
    ]
    if expected:
        return sorted(set(expected)), "expected_from_prev_iter"

    recency_scores = dict(getattr(gaussians, "_paper_resident_recency_scores", {}) or {})
    transition = compute_topc_resident_transition(
        current_active_blocks=sorted(visible_set),
        next_active_blocks=sorted(visible_set),
        current_resident_blocks=[],
        next_camera_blocks=current_camera_blocks,
        previous_recency_scores=recency_scores,
        lambda_weight=float(getattr(args, "paper_resident_lambda", 0.7)),
        recency_decay=float(getattr(args, "paper_resident_recency_decay", 0.95)),
        resident_capacity_blocks=requested_capacity,
        balanced_camera_seeds=(policy == "topc_balanced"),
        balanced_seed_fraction=float(getattr(args, "paper_balanced_seed_fraction", 1.0)),
    )
    return sorted(int(b) for b in transition.next_resident_blocks), "bootstrap_topc_over_k1"


_paper_phase1_log_file = None


def _get_paper_phase1_log_file():
    global _paper_phase1_log_file
    if _paper_phase1_log_file is None:
        import utils.general_utils as utils

        args = utils.get_args()
        model_path = getattr(args, "model_path", None)
        if not model_path:
            return None
        os.makedirs(model_path, exist_ok=True)
        _paper_phase1_log_file = open(
            os.path.join(model_path, "paper_phase1.log"),
            "a",
            buffering=1,
        )
    return _paper_phase1_log_file


def write_paper_phase1_log(message: str, log_file=None):
    if log_file is not None:
        log_file.write(message)

    paper_log = _get_paper_phase1_log_file()
    if paper_log is not None:
        paper_log.write(message)


def initialize_paper_mode_runtime_state(
    *,
    gaussians,
    args,
    is_paper_ssd_mode: bool,
    iteration: int,
    log_file=None,
) -> Tuple[str, str]:
    """Initialize TideGS runtime flags and state on the Gaussian model."""
    paper_optimizer_deferred_mode = (
        str(getattr(args, "paper_optimizer_deferred_mode", "off")).lower()
        if is_paper_ssd_mode
        else "off"
    )
    paper_optimizer_backend = (
        str(getattr(args, "paper_optimizer_backend", "cpu")).lower()
        if is_paper_ssd_mode
        else "cpu"
    )

    if not is_paper_ssd_mode:
        return paper_optimizer_deferred_mode, paper_optimizer_backend

    if not gaussians.use_gpu_features:
        raise RuntimeError("Paper SSD release path requires gpu working-set features.")
    if paper_optimizer_backend != "gpu_resident":
        raise RuntimeError("Paper SSD release path requires --paper_optimizer_backend gpu_resident.")
    if paper_optimizer_deferred_mode != "off":
        raise RuntimeError("Paper SSD release path requires --paper_optimizer_deferred_mode off.")

    paper_optimizer_state_mode = str(
        getattr(args, "paper_optimizer_state_mode", "resident_blocks")
    ).lower()
    if paper_optimizer_state_mode != "resident_blocks":
        raise RuntimeError("Paper SSD release path requires --paper_optimizer_state_mode resident_blocks.")

    gaussians._paper_optimizer_state_mode = paper_optimizer_state_mode
    gaussians._paper_optimizer_block_size = int(getattr(args, "gaussian_block_size", 4096))
    gaussians._paper_optimizer_backend = paper_optimizer_backend
    if not hasattr(gaussians, "_paper_pending_writeback"):
        gaussians._paper_pending_writeback = None
    if not hasattr(gaussians, "_paper_ab_runtime_stats"):
        gaussians._paper_ab_runtime_stats = {
            "prefetch_hits": 0,
            "cold_starts": 0,
            "prefetch_misses": 0,
            "sync_fallbacks": 0,
        }

    if hasattr(gaussians, "block_cache_state"):
        gaussians.block_cache_state["last_shs"] = None
        gaussians.block_cache_state["last_filter"] = None
        gaussians.block_cache_state["last_retention_vec"] = None
        if iteration == 1:
            write_paper_phase1_log(
                "[PAPER MODE] Disabled inter-batch SH hotspot cache state.\n",
                log_file=log_file,
            )

    if iteration == 1:
        write_paper_phase1_log(
            f"[PAPER MODE] paper_optimizer_deferred_mode={paper_optimizer_deferred_mode}\n",
            log_file=log_file,
        )
        write_paper_phase1_log(
            f"[PAPER MODE] paper_optimizer_state_mode={paper_optimizer_state_mode}\n",
            log_file=log_file,
        )

    return paper_optimizer_deferred_mode, paper_optimizer_backend


def log_batch_kt_metrics(
    log_file,
    iteration: int,
    current_camera_blocks: Dict[int, List[int]],
    visible_block_ids: List[int],
    resident_blocks: Optional[List[int]] = None,
) -> None:
    metrics = compute_batch_kt_metrics(
        current_camera_blocks=current_camera_blocks,
        visible_block_ids=visible_block_ids,
        resident_blocks=resident_blocks,
    )
    write_paper_phase1_log(
        f"[PAPER K_T METRICS] Iter {iteration}: "
        f"cameras={int(metrics['camera_count'])} "
        f"per_camera_min={int(metrics['per_camera_min'])} "
        f"per_camera_mean={metrics['per_camera_mean']:.1f} "
        f"per_camera_max={int(metrics['per_camera_max'])} "
        f"sum_blocks={int(metrics['sum_blocks'])} "
        f"union_blocks={int(metrics['union_blocks'])} "
        f"overlap_ratio={metrics['overlap_ratio']:.4f} "
        f"resident_blocks={int(metrics['resident_blocks'])} "
        f"resident_intersection={int(metrics['resident_intersection'])} "
        f"resident_coverage={metrics['resident_coverage']:.4f}\n",
        log_file=log_file,
    )


def log_paper_block_sets(
    log_file,
    iteration: int,
    current_camera_ids: List[int],
    block_sets: Dict[str, object],
):
    current_blocks = block_sets["current_blocks"]
    next_blocks = block_sets["next_blocks"]
    omega_blocks = block_sets["omega_blocks"]
    delta_plus_blocks = block_sets["delta_plus_blocks"]
    delta_minus_blocks = block_sets["delta_minus_blocks"]
    current_resident_blocks = block_sets["current_resident_blocks"]
    candidate_blocks = block_sets["candidate_blocks"]
    next_resident_blocks = block_sets["next_resident_blocks"]
    stream_in_blocks = block_sets["stream_in_blocks"]
    evict_blocks = block_sets["evict_blocks"]
    next_camera_ids = block_sets["next_camera_ids"]
    current_resident_source = block_sets["current_resident_source"]
    resident_selection_policy = block_sets["resident_selection_policy"]
    resident_capacity_blocks = block_sets["resident_capacity_blocks"]
    requested_capacity_blocks = block_sets.get("requested_resident_capacity_blocks", resident_capacity_blocks)
    resident_lambda_weight = float(block_sets.get("resident_lambda_weight", 1.0))
    resident_recency_decay = float(block_sets.get("resident_recency_decay", 1.0))
    balanced_seed_fraction = float(block_sets.get("balanced_seed_fraction", 0.0))
    topc_cutoff_score = float(block_sets.get("topc_cutoff_score", 0.0))
    next_active_coverage = int(block_sets.get("next_active_coverage", len(next_blocks)))
    next_camera_coverage = int(block_sets.get("next_camera_coverage", 0))
    next_camera_total = int(block_sets.get("next_camera_total", 0))
    enforce_next_active_coverage = bool(block_sets.get("enforce_next_active_coverage", False))
    optional_selected_blocks = block_sets.get("optional_selected_blocks", []) or []
    camera_seed_blocks = block_sets.get("camera_seed_blocks", []) or []

    write_paper_phase1_log(
        f"[PAPER WORKING SET] Iter {iteration}: "
        f"|K_t|={len(current_blocks)} "
        f"|K_t+1|={len(next_blocks)} "
        f"|Omega_t|={len(omega_blocks)} "
        f"|Delta_t+|={len(delta_plus_blocks)} "
        f"|Delta_t-|={len(delta_minus_blocks)}\n",
        log_file=log_file,
    )

    requested_desc = "auto" if requested_capacity_blocks < 0 else str(requested_capacity_blocks)
    effective_desc = "unbounded" if resident_capacity_blocks < 0 else str(resident_capacity_blocks)
    write_paper_phase1_log(
        f"[PAPER RESIDENT SET] Iter {iteration}: "
        f"|R_t|={len(current_resident_blocks)} "
        f"|C_t|={len(candidate_blocks)} "
        f"|R_t+1|={len(next_resident_blocks)} "
        f"|S_t+|={len(stream_in_blocks)} "
        f"|S_t-|={len(evict_blocks)} "
        f"policy={resident_selection_policy} "
        f"requested_capacity_blocks={requested_desc} "
        f"effective_capacity_blocks={effective_desc} "
        f"source={current_resident_source}\n",
        log_file=log_file,
    )

    if resident_selection_policy in {"topc", "topc_strict", "topc_balanced"}:
        coverage_desc = f"{next_active_coverage}/{len(next_blocks)}" if next_blocks else "0/0"
        camera_coverage_desc = (
            f"{next_camera_coverage}/{next_camera_total}"
            if next_camera_total > 0
            else "0/0"
        )
        write_paper_phase1_log(
            f"[PAPER RESIDENT SCORE] Iter {iteration}: "
            f"lambda={resident_lambda_weight:.3f} "
            f"decay={resident_recency_decay:.3f} "
            f"seed_fraction={balanced_seed_fraction:.3f} "
            f"cutoff={topc_cutoff_score:.4f} "
            f"next_active_coverage={coverage_desc} "
            f"next_camera_coverage={camera_coverage_desc} "
            f"camera_seed_blocks={len(camera_seed_blocks)} "
            f"optional_retained={len(optional_selected_blocks)} "
            f"guarded_next_active={str(enforce_next_active_coverage)}\n",
            log_file=log_file,
        )

    if iteration == 1:
        write_paper_phase1_log(
            f"[PAPER WORKING SET] Cameras_t (first 8): {current_camera_ids[:8]}\n",
            log_file=log_file,
        )
        if next_camera_ids:
            write_paper_phase1_log(
                f"[PAPER WORKING SET] Cameras_t+1 (first 8): {next_camera_ids[:8]}\n",
                log_file=log_file,
            )
        if resident_selection_policy == "topc":
            write_paper_phase1_log(
                "[PAPER RESIDENT SET] P2 TopC policy enabled: Eq.(5) scores candidate blocks in C_t = R_t ∪ K_t+1 and keeps the highest-scoring blocks under the configured resident capacity. When capacity is smaller than |K_t+1|, a camera-balanced seed pass first spreads the bounded resident set across cameras, then remaining slots follow the global TopC ranking.\n",
                log_file=log_file,
            )
        else:
            write_paper_phase1_log(
                "[PAPER RESIDENT SET] P1 passthrough policy enabled: R_t follows the current active set and R_t+1 follows the next active set until scoring/TopC is introduced.\n",
                log_file=log_file,
            )


def snapshot_paper_warm_layer_metrics(storage_adapter) -> Optional[Dict[str, float]]:
    if storage_adapter is None:
        return None

    metrics: Dict[str, float] = {}

    pipeline = getattr(storage_adapter, "pipeline", None)
    if pipeline is not None:
        pipeline_stats = dict(getattr(pipeline, "stats", {}))
        metrics.update({
            "urgent_queue": pipeline.urgent_prefetch_queue.qsize() if hasattr(pipeline, "urgent_prefetch_queue") else -1,
            "future_queue": pipeline.future_prefetch_queue.qsize() if hasattr(pipeline, "future_prefetch_queue") else -1,
            "completed_queue": pipeline.completed_queue.qsize() if hasattr(pipeline, "completed_queue") else -1,
            "urgent_prefetch_jobs": int(pipeline_stats.get("urgent_prefetch_jobs", 0)),
            "urgent_prefetch_blocks": int(pipeline_stats.get("urgent_prefetch_blocks", 0)),
            "urgent_prefetch_time_ms": float(pipeline_stats.get("urgent_prefetch_time", 0.0) * 1000.0),
            "urgent_wait_time_ms": float(pipeline_stats.get("urgent_wait_time", 0.0) * 1000.0),
            "urgent_wait_calls": int(pipeline_stats.get("urgent_wait_calls", 0)),
            "urgent_wait_timeouts": int(pipeline_stats.get("urgent_wait_timeouts", 0)),
            "future_prefetch_jobs": int(pipeline_stats.get("future_prefetch_jobs", 0)),
            "future_prefetch_blocks": int(pipeline_stats.get("future_prefetch_blocks", 0)),
            "future_prefetch_time_ms": float(pipeline_stats.get("future_prefetch_time", 0.0) * 1000.0),
            "load_service_time_ms": float(pipeline_stats.get("load_time", 0.0) * 1000.0),
        })

    cache = getattr(storage_adapter, "cache", None)
    if cache is not None:
        cache_stats = cache.get_stats()
        metrics.update({
            "cache_size": int(cache_stats.get("cache_size", 0)),
            "dirty_blocks": int(cache_stats.get("dirty_blocks", 0)),
            "flushing_blocks": int(cache_stats.get("flushing_blocks", 0)),
            "flush_queue_size": int(cache_stats.get("flush_queue_size", 0)),
            "cache_hits": int(cache_stats.get("cache_hits", 0)),
            "cache_misses": int(cache_stats.get("cache_misses", 0)),
            "prefetches": int(cache_stats.get("prefetches", 0)),
            "hit_rate": float(cache_stats.get("hit_rate", 0.0)),
            "async_flush_requests": int(cache_stats.get("async_flush_requests", 0)),
            "flush_queue_drops": int(cache_stats.get("flush_queue_drops", 0)),
            "async_flush_jobs": int(cache_stats.get("async_flush_jobs", 0)),
            "async_flush_blocks": int(cache_stats.get("async_flush_blocks", 0)),
            "async_flush_time_ms": float(cache_stats.get("async_flush_time", 0.0) * 1000.0),
            "sync_flush_jobs": int(cache_stats.get("sync_flush_jobs", 0)),
            "sync_flush_blocks": int(cache_stats.get("sync_flush_blocks", 0)),
            "sync_flush_time_ms": float(cache_stats.get("sync_flush_time", 0.0) * 1000.0),
            "cache_urgent_prefetch_calls": int(cache_stats.get("urgent_prefetch_calls", 0)),
            "cache_urgent_prefetch_blocks": int(cache_stats.get("urgent_prefetch_blocks", cache_stats.get("urgent_prefetch_miss_blocks", 0))),
            "cache_urgent_prefetch_miss_blocks": int(cache_stats.get("urgent_prefetch_miss_blocks", 0)),
            "cache_urgent_prefetch_time_ms": float(cache_stats.get("urgent_prefetch_time", 0.0) * 1000.0),
            "cache_future_prefetch_submitted": int(cache_stats.get("future_prefetch_submitted", 0)),
            "cache_future_prefetch_skipped": int(cache_stats.get("future_prefetch_skipped", 0)),
            "cache_future_prefetch_dropped": int(cache_stats.get("future_prefetch_dropped", 0)),
            "cache_future_prefetch_jobs": int(cache_stats.get("future_prefetch_jobs", 0)),
            "cache_future_prefetch_blocks": int(cache_stats.get("future_prefetch_blocks", 0)),
            "cache_future_prefetch_time_ms": float(cache_stats.get("future_prefetch_time", 0.0) * 1000.0),
            "cache_future_prefetch_errors": int(cache_stats.get("future_prefetch_errors", 0)),
            "cache_future_queue": int(cache_stats.get("future_prefetch_queue_size", cache_stats.get("future_queue_size", 0))),
            "cache_future_pending": int(cache_stats.get("future_prefetch_pending", cache_stats.get("future_pending_blocks", 0))),
            "ram_usage_mb": float(cache_stats.get("ram_usage_mb", 0.0)),
            "max_ram_mb": float(cache_stats.get("max_ram_mb", 0.0)),
        })

    return metrics


def log_paper_warm_layer_metrics(storage_adapter, iteration: int, stage: str, log_file=None):
    metrics = snapshot_paper_warm_layer_metrics(storage_adapter)
    if not metrics:
        return

    write_paper_phase1_log(
        f"[PAPER WARM LAYER] Iter {iteration} @ {stage}: "
        f"urgent_q={metrics.get('urgent_queue', -1)} "
        f"completed_q={metrics.get('completed_queue', -1)} "
        f"urgent_jobs={metrics.get('urgent_prefetch_jobs', 0)} "
        f"urgent_blocks={metrics.get('urgent_prefetch_blocks', 0)} "
        f"urgent_service={metrics.get('urgent_prefetch_time_ms', 0.0):.1f}ms "
        f"urgent_wait={metrics.get('urgent_wait_time_ms', 0.0):.1f}ms "
        f"wait_calls={metrics.get('urgent_wait_calls', 0)} "
        f"wait_timeouts={metrics.get('urgent_wait_timeouts', 0)} "
        f"future_q={metrics.get('future_queue', -1)} "
        f"future_jobs={metrics.get('future_prefetch_jobs', 0)} "
        f"future_blocks={metrics.get('future_prefetch_blocks', 0)} "
        f"future_service={metrics.get('future_prefetch_time_ms', 0.0):.1f}ms\n",
        log_file=log_file,
    )
    write_paper_phase1_log(
        f"[PAPER WARM LAYER] Cache @ {stage}: "
        f"size={metrics.get('cache_size', 0)} "
        f"dirty={metrics.get('dirty_blocks', 0)} "
        f"flushing={metrics.get('flushing_blocks', 0)} "
        f"flush_q={metrics.get('flush_queue_size', 0)} "
        f"async_jobs={metrics.get('async_flush_jobs', 0)} "
        f"async_blocks={metrics.get('async_flush_blocks', 0)} "
        f"async_time={metrics.get('async_flush_time_ms', 0.0):.1f}ms "
        f"sync_jobs={metrics.get('sync_flush_jobs', 0)} "
        f"sync_blocks={metrics.get('sync_flush_blocks', 0)} "
        f"sync_time={metrics.get('sync_flush_time_ms', 0.0):.1f}ms "
        f"hits={metrics.get('cache_hits', 0)} "
        f"misses={metrics.get('cache_misses', 0)} "
        f"prefetches={metrics.get('prefetches', 0)} "
        f"hit_rate={metrics.get('hit_rate', 0.0) * 100.0:.1f}% "
        f"async_flush_req={metrics.get('async_flush_requests', 0)} "
        f"flush_drops={metrics.get('flush_queue_drops', 0)} "
        f"ram={metrics.get('ram_usage_mb', 0.0):.1f}/{metrics.get('max_ram_mb', 0.0):.1f}MB\n",
        log_file=log_file,
    )
    write_paper_phase1_log(
        f"[PAPER PIPELINE CACHE] Iter {iteration} @ {stage}: "
        f"urgent_calls={metrics.get('cache_urgent_prefetch_calls', 0)} "
        f"urgent_blocks={metrics.get('cache_urgent_prefetch_blocks', 0)} "
        f"urgent_misses={metrics.get('cache_urgent_prefetch_miss_blocks', 0)} "
        f"urgent_time={metrics.get('cache_urgent_prefetch_time_ms', 0.0):.1f}ms "
        f"future_submitted={metrics.get('cache_future_prefetch_submitted', 0)} "
        f"future_skipped={metrics.get('cache_future_prefetch_skipped', 0)} "
        f"future_dropped={metrics.get('cache_future_prefetch_dropped', 0)} "
        f"future_jobs={metrics.get('cache_future_prefetch_jobs', 0)} "
        f"future_blocks={metrics.get('cache_future_prefetch_blocks', 0)} "
        f"future_time={metrics.get('cache_future_prefetch_time_ms', 0.0):.1f}ms "
        f"future_q={metrics.get('cache_future_queue', 0)} "
        f"future_pending={metrics.get('cache_future_pending', 0)} "
        f"future_errors={metrics.get('cache_future_prefetch_errors', 0)}\n",
        log_file=log_file,
    )


def activate_paper_prefetched_buffer(
    gaussians,
    double_buffer,
    ensure_local_to_global_mapping_fn: Callable,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manager = gaussians.gpu_working_set_manager
    active_buffer = double_buffer.active_buffer

    manager.gpu_xyz = active_buffer.xyz
    manager.gpu_scaling = active_buffer.scaling
    manager.gpu_rotation = active_buffer.rotation
    manager.gpu_opacity = active_buffer.opacity
    manager.gpu_features_dc = active_buffer.features_dc
    manager.gpu_features_rest = active_buffer.features_rest
    manager.local_to_global_idx = active_buffer.local_to_global_idx
    ensure_local_to_global_mapping_fn(
        gaussians,
        manager.num_total,
        context="activate_paper_prefetched_buffer",
    )
    manager.loaded_blocks = list(active_buffer.loaded_blocks)
    manager.previous_blocks = list(active_buffer.loaded_blocks)
    manager.filters_local = list(active_buffer.filters_local)

    manager.block_to_gpu_slice.clear()
    if active_buffer.block_to_local_slice:
        manager.block_to_gpu_slice.update(dict(active_buffer.block_to_local_slice))
    else:
        offset = 0
        for block_id in manager.loaded_blocks:
            start_idx = block_id * manager.block_size
            end_idx = min(start_idx + manager.block_size, manager.num_total)
            block_len = end_idx - start_idx
            manager.block_to_gpu_slice[block_id] = slice(offset, offset + block_len)
            offset += block_len

    num_gaussians = (
        int(active_buffer.local_to_global_idx.numel())
        if active_buffer.local_to_global_idx is not None
        else 0
    )
    memory_mb = num_gaussians * 59 * 4 / (1024 ** 2)

    return {
        "xyz": manager.gpu_xyz,
        "scaling": manager.gpu_scaling,
        "rotation": manager.gpu_rotation,
        "opacity": manager.gpu_opacity,
        "features_dc": manager.gpu_features_dc,
        "features_rest": manager.gpu_features_rest,
    }, {
        "hotspot_count": 0,
        "cold_count": len(manager.loaded_blocks),
        "total_count": len(manager.loaded_blocks),
        "hit_rate": 0.0,
        "memory_mb": memory_mb,
        "num_gaussians": num_gaussians,
        "source": "prefetched_ab_buffer",
        "resident_count": len(active_buffer.resident_blocks),
        "streamed_count": len(active_buffer.streamed_blocks),
        "evicted_count": len(active_buffer.evicted_blocks),
    }


def seed_paper_active_buffer_from_manager(
    gaussians,
    double_buffer,
    ensure_local_to_global_mapping_fn: Callable,
    preserve_resident_metadata: bool = False,
) -> None:
    manager = gaussians.gpu_working_set_manager
    active_buffer = double_buffer.active_buffer

    keep_metadata = (
        preserve_resident_metadata
        and list(active_buffer.loaded_blocks) == list(manager.loaded_blocks)
    )

    active_buffer.xyz = manager.gpu_xyz
    active_buffer.scaling = manager.gpu_scaling
    active_buffer.rotation = manager.gpu_rotation
    active_buffer.opacity = manager.gpu_opacity
    active_buffer.features_dc = manager.gpu_features_dc
    active_buffer.features_rest = manager.gpu_features_rest
    active_buffer.local_to_global_idx = ensure_local_to_global_mapping_fn(
        gaussians,
        manager.num_total,
        context="seed_paper_active_buffer_from_manager",
    )
    active_buffer._block_starts = None
    active_buffer.loaded_blocks = list(manager.loaded_blocks)
    active_buffer.filters_local = list(getattr(manager, "filters_local", []))
    active_buffer.block_to_local_slice = dict(manager.block_to_gpu_slice)
    active_buffer.num_gaussians = (
        int(manager.local_to_global_idx.numel())
        if manager.local_to_global_idx is not None
        else 0
    )

    if not keep_metadata:
        active_buffer.resident_blocks = list(active_buffer.loaded_blocks)
        active_buffer.streamed_blocks = []
        active_buffer.evicted_blocks = []

    gaussians._paper_loaded_resident_blocks = list(active_buffer.loaded_blocks)


def log_ab_buffer_activation(
    *,
    iteration: int,
    double_buffer,
    visible_block_ids: List[int],
    log_file=None,
) -> None:
    active_buf = double_buffer.active_buffer
    write_paper_phase1_log(
        f"[PAPER A/B BUFFER] Iter {iteration}: Activated prefetched buffer "
        f"{double_buffer.get_stats()['active_buffer']} for |R_t|={len(active_buf.loaded_blocks)} "
        f"resident blocks (current |K_t|={len(visible_block_ids)}, "
        f"resident={len(active_buf.resident_blocks)}, "
        f"streamed={len(active_buf.streamed_blocks)}, "
        f"evicted={len(active_buf.evicted_blocks)})\n",
        log_file=log_file,
    )


def log_ab_buffer_sync_fallback(
    *,
    iteration: int,
    miss_reason: str,
    ab_stats: Dict[str, int],
    log_file=None,
) -> None:
    write_paper_phase1_log(
        f"[PAPER A/B BUFFER] Iter {iteration}: {miss_reason}; falling back to synchronous "
        f"RAM->GPU materialization (hits={ab_stats['prefetch_hits']} "
        f"cold={ab_stats['cold_starts']} misses={ab_stats['prefetch_misses']} "
        f"sync_fallbacks={ab_stats['sync_fallbacks']})\n",
        log_file=log_file,
    )


def log_ab_buffer_prefetch_failure(
    *,
    iteration: int,
    error: Exception,
    log_file=None,
) -> None:
    write_paper_phase1_log(
        f"[PAPER A/B BUFFER] Warning: Prefetch failed at iter {iteration}: {error}\n",
        log_file=log_file,
    )


def log_filters_local_reordered(
    *,
    enabled: bool,
    log_file=None,
) -> None:
    if enabled and log_file is not None:
        log_file.write("[PAPER MODE] Reordered filters_local to match camera order\n")


def log_empty_microbatch_camera(
    *,
    micro_idx: int,
    is_paper_ssd_mode: bool,
    log_file=None,
) -> None:
    warning_msg = f"[WARNING] Camera {micro_idx} sees no Gaussians, skipping..."
    if is_paper_ssd_mode:
        write_paper_phase1_log(warning_msg + "\n", log_file=log_file)
    else:
        print(warning_msg)
        if log_file is not None:
            log_file.write(warning_msg + "\n")
            log_file.flush()


def log_double_buffer_stats(
    *,
    iteration: int,
    double_buffer,
    log_file=None,
) -> None:
    if double_buffer is None or log_file is None:
        return
    stats = double_buffer.get_stats()
    log_file.write(f"[DOUBLE BUFFER STATS] Iter {iteration}:\n")
    log_file.write(f"  Buffer swaps: {stats['swaps']}\n")
    log_file.write(
        f"  Prefetch hits: {stats['prefetch_hits']}, misses: {stats['prefetch_misses']}\n"
    )
    log_file.write(f"  Hit rate: {stats['hit_rate']*100:.1f}%\n")
    log_file.write(f"  Avg prefetch time: {stats['avg_prefetch_time_ms']:.1f}ms\n")
    log_file.write(
        f"  Active buffer: {stats['active_buffer']}, Gaussians: {stats['active_gaussians']:,}\n"
    )


def log_delta_stream_prefetch(
    *,
    iteration: int,
    paper_block_sets: Dict[str, object],
    future_submitted: int,
    future_submitted_late: int,
    storage_adapter,
    log_file=None,
) -> None:
    next_camera_ids = paper_block_sets["next_camera_ids"]
    next_active_blocks = paper_block_sets["next_blocks"]
    next_resident_blocks = paper_block_sets["next_resident_blocks"]
    stream_in_blocks = paper_block_sets["stream_in_blocks"]
    keep_resident_blocks = paper_block_sets["keep_resident_blocks"]
    evict_blocks = paper_block_sets["evict_blocks"]

    write_paper_phase1_log(
        f"[PAPER DELTA STREAM] Iter {iteration}: keep |R_t∩R_t+1|={len(keep_resident_blocks)} "
        f"stream |S_t+|={len(stream_in_blocks)} evict |S_t-|={len(evict_blocks)}\n",
        log_file=log_file,
    )
    write_paper_phase1_log(
        f"[PAPER A/B BUFFER] Iter {iteration}: Started async prefetch for R_t+1 "
        f"with |R_t+1|={len(next_resident_blocks)} resident blocks "
        f"(next |K_t+1|={len(next_active_blocks)})\n",
        log_file=log_file,
    )
    write_paper_phase1_log(
        f"[PAPER A/B BUFFER] Iter {iteration}: Next cameras_t+1 (first 8): "
        f"{next_camera_ids[:8]}\n",
        log_file=log_file,
    )
    cache_stats = (
        storage_adapter.cache.get_stats()
        if getattr(storage_adapter, "cache", None) is not None
        else {}
    )
    write_paper_phase1_log(
        f"[PAPER PIPELINE] Iter {iteration}: "
        f"omega={len(keep_resident_blocks)} "
        f"delta_plus={len(stream_in_blocks)} "
        f"evict={len(evict_blocks)} "
        f"next_resident={len(next_resident_blocks)} "
        f"future_submitted={future_submitted} "
        f"future_submitted_late={future_submitted_late} "
        f"future_q={cache_stats.get('future_prefetch_queue_size', cache_stats.get('future_queue_size', 0))} "
        f"future_pending={cache_stats.get('future_prefetch_pending', cache_stats.get('future_pending_blocks', 0))} "
        f"urgent_miss_blocks={cache_stats.get('urgent_prefetch_miss_blocks', 0)} "
        f"hit_rate={float(cache_stats.get('hit_rate', 0.0)) * 100.0:.1f}% "
        f"flush_q={cache_stats.get('flush_queue_size', 0)} "
        f"ram={float(cache_stats.get('ram_usage_mb', 0.0)):.1f}MB\n",
        log_file=log_file,
    )


def log_delta_refresh(
    *,
    iteration: int,
    refreshed_omega: int,
    refresh_context: str,
    deferred: bool = False,
    log_file=None,
) -> None:
    if refreshed_omega <= 0:
        return

    if deferred:
        message = (
            f"[PAPER DELTA STREAM] Iter {iteration}: Refreshed {refreshed_omega} "
            f"deferred kept resident blocks from updated CPU cache {refresh_context}.\n"
        )
    else:
        message = (
            f"[PAPER DELTA STREAM] Iter {iteration}: Refreshed {refreshed_omega} "
            f"kept resident blocks from updated CPU cache at {refresh_context}.\n"
        )
    write_paper_phase1_log(message, log_file=log_file)


def log_early_delta_hint(
    *,
    storage_adapter,
    iteration: int,
    submitted: int,
    requested: int,
    log_file=None,
) -> None:
    cache_stats = (
        storage_adapter.cache.get_stats()
        if getattr(storage_adapter, "cache", None) is not None
        else {}
    )
    write_paper_phase1_log(
        f"[PAPER PIPELINE] Iter {iteration}: early SSD->RAM hint for Delta+ "
        f"submitted={submitted}/{requested} "
        f"future_q={cache_stats.get('future_prefetch_queue_size', cache_stats.get('future_queue_size', 0))} "
        f"future_pending={cache_stats.get('future_prefetch_pending', cache_stats.get('future_pending_blocks', 0))}\n",
        log_file=log_file,
    )


def log_paper_working_set_loaded(
    *,
    iteration: int,
    loaded_blocks: List[int],
    paper_block_sets: Dict[str, object],
    retention_stats: Dict[str, Any],
    load_source: str,
    log_file=None,
) -> None:
    write_paper_phase1_log(
        f"[PAPER GPU WORKING SET] Iter {iteration}: Loaded |R_t|={len(loaded_blocks)} "
        f"resident blocks for |K_t|={len(paper_block_sets['current_blocks'])} visible blocks "
        f"({retention_stats['num_gaussians']:,} Gaussians, {retention_stats['memory_mb']:.2f} MB) "
        f"into the active GPU working set via {load_source}\n",
        log_file=log_file,
    )


def log_paper_resident_camera_coverage(
    *,
    iteration: int,
    current_camera_blocks: Dict[int, List[int]],
    loaded_blocks: List[int],
    should_log: bool,
    log_file=None,
) -> None:
    loaded_block_set = set(int(block_id) for block_id in loaded_blocks)
    camera_total = 0
    camera_covered = 0
    missing_camera_ids: List[int] = []
    for cam_id, blocks in current_camera_blocks.items():
        if not blocks:
            continue
        camera_total += 1
        if any(int(block_id) in loaded_block_set for block_id in blocks):
            camera_covered += 1
        else:
            missing_camera_ids.append(int(cam_id))
    if should_log or missing_camera_ids:
        write_paper_phase1_log(
            f"[PAPER RESIDENT CAMERA] Iter {iteration}: "
            f"current_camera_block_coverage={camera_covered}/{camera_total} "
            f"missing_camera_ids_first8={missing_camera_ids[:8]}\n",
            log_file=log_file,
        )


def log_paper_order_calculation_skip(
    *,
    iteration: int,
    log_file=None,
) -> None:
    write_paper_phase1_log(
        f"[PAPER MODE] Iter {iteration}: skipped order_calculation; "
        "using paper camera schedule order directly.\n",
        log_file=log_file,
    )


def log_paper_sh_indexed(
    *,
    iteration: int,
    filter_len: int,
    log_file=None,
) -> None:
    write_paper_phase1_log(
        f"[PAPER GPU WORKING SET] Iter {iteration}: Indexed {filter_len} SH features "
        "from the active GPU working set\n",
        log_file=log_file,
    )


def log_paper_interbatch_sh_cache_disabled(
    *,
    iteration: int,
    log_file=None,
) -> None:
    write_paper_phase1_log(
        f"[PAPER GPU WORKING SET] Iter {iteration}: Disabled inter-batch SH "
        "carry-over cache; next batch will materialize K_t+1 explicitly\n",
        log_file=log_file,
    )


def log_paper_cpu_source_refs_restored(log_file=None) -> None:
    write_paper_phase1_log(
        "[PAPER GPU WORKING SET] Restored CPU source references after batch\n",
        log_file=log_file,
    )


def log_paper_writeback_staged(
    *,
    iteration: int,
    staged_writeback_blocks: int,
    log_file=None,
) -> None:
    write_paper_phase1_log(
        f"[SSD WRITEBACK][paper] Iter {iteration}: Staged {staged_writeback_blocks} "
        "updated blocks into dirty CPU cache (lazy SSD writeback)\n",
        log_file=log_file,
    )


def log_paper_writeback_finalize(
    *,
    storage_adapter,
    iteration: int,
    optimizer_deferred_mode: str,
    candidate_blocks: int,
    staged_writeback_blocks: int,
    refreshed_omega: int,
    log_file=None,
) -> None:
    if optimizer_deferred_mode == "same_iter":
        write_paper_phase1_log(
            f"[PAPER SSD MODE] Finalized same-iteration optimizer/lazy-writeback "
            f"payload in iter {iteration}: candidate_blocks={candidate_blocks}, "
            f"staged={staged_writeback_blocks}, refreshed_omega={refreshed_omega}\n",
            log_file=log_file,
        )
        log_paper_warm_layer_metrics(storage_adapter, iteration, "writeback_same_iter", log_file=log_file)
    else:
        write_paper_phase1_log(
            f"[PAPER SSD MODE] Finalized immediate optimizer/lazy-writeback payload "
            f"in iter {iteration}: candidate_blocks={candidate_blocks}, "
            f"staged={staged_writeback_blocks}, refreshed_omega={refreshed_omega}\n",
            log_file=log_file,
        )
        log_paper_warm_layer_metrics(storage_adapter, iteration, "writeback", log_file=log_file)


def log_paper_working_set_cleared(log_file=None) -> None:
    write_paper_phase1_log(
        "[PAPER GPU WORKING SET] Cleared GPU working set after direct SSD writeback\n",
        log_file=log_file,
    )


def apply_paper_writeback_payload(
    *,
    storage_adapter,
    args,
    payload_iteration: int,
    current_iteration: int,
    updated_block_ids,
    omega_blocks,
    total_n_gaussians: int,
    original_xyz,
    original_scaling,
    original_rotation,
    original_opacity,
    original_features_dc,
    original_features_rest,
    gpu_working_set_manager=None,
    build_updated_blocks_dict_from_gpu_fn: Callable,
    sync_updated_blocks_from_gpu_views_to_cpu_fn: Callable,
    materialize_updated_blocks_from_cpu_views_fn: Callable,
    get_double_buffer_gpu_fn: Callable,
) -> Tuple[int, int, int]:
    _ = payload_iteration, current_iteration

    use_direct_gpu_path = (
        original_xyz is None
        or original_scaling is None
        or original_rotation is None
        or original_opacity is None
        or original_features_dc is None
        or original_features_rest is None
    )

    if use_direct_gpu_path:
        updated_blocks_dict = build_updated_blocks_dict_from_gpu_fn(
            updated_block_ids=updated_block_ids,
            total_n_gaussians=total_n_gaussians,
            block_size=args.gaussian_block_size,
            gpu_working_set_manager=gpu_working_set_manager,
        )
    else:
        if gpu_working_set_manager is not None and len(updated_block_ids) > 0:
            sync_updated_blocks_from_gpu_views_to_cpu_fn(
                updated_block_ids=updated_block_ids,
                total_n_gaussians=total_n_gaussians,
                block_size=args.gaussian_block_size,
                gpu_working_set_manager=gpu_working_set_manager,
                original_xyz=original_xyz,
                original_scaling=original_scaling,
                original_rotation=original_rotation,
                original_opacity=original_opacity,
                original_features_dc=original_features_dc,
                original_features_rest=original_features_rest,
            )

        updated_blocks_dict = materialize_updated_blocks_from_cpu_views_fn(
            updated_block_ids=updated_block_ids,
            total_n_gaussians=total_n_gaussians,
            block_size=args.gaussian_block_size,
            original_xyz=original_xyz,
            original_scaling=original_scaling,
            original_rotation=original_rotation,
            original_opacity=original_opacity,
            original_features_dc=original_features_dc,
            original_features_rest=original_features_rest,
        )

    staged_writeback_blocks = storage_adapter.sync_cache_from_cpu_views(updated_blocks_dict)

    refreshed_omega = 0
    if len(omega_blocks) > 0:
        omega_refresh_blocks = {
            block_id: updated_blocks_dict[block_id]
            for block_id in omega_blocks
            if block_id in updated_blocks_dict
        }
        missing_omega_blocks = [
            block_id for block_id in omega_blocks
            if block_id not in omega_refresh_blocks
        ]
        if missing_omega_blocks:
            omega_refresh_blocks.update(storage_adapter.cache.prefetch(missing_omega_blocks))

        double_buffer = get_double_buffer_gpu_fn(
            num_total=total_n_gaussians,
            block_size=args.gaussian_block_size,
            device="cuda",
        )
        refreshed_omega = double_buffer.refresh_blocks_from_block_cache(
            block_cache=omega_refresh_blocks,
            block_ids=omega_blocks,
            target="loading",
        )

    return staged_writeback_blocks, refreshed_omega, len(updated_block_ids)


def collect_paper_updated_block_ids(
    *,
    iteration: int,
    optimizer_updated_global_indices,
    filters,
    block_size: int,
    total_n_gaussians: int,
    collect_from_indices_fn: Callable,
    collect_from_filters_fn: Callable,
    should_log: bool,
    log_file=None,
) -> List[int]:
    if optimizer_updated_global_indices is not None:
        updated_block_ids = collect_from_indices_fn(
            optimizer_updated_global_indices,
            block_size,
            total_n_gaussians,
        )
        if should_log:
            write_paper_phase1_log(
                f"[SSD WRITEBACK][paper] Iter {iteration}: Block selection driven by "
                f"optimizer-updated rows={len(optimizer_updated_global_indices)} -> "
                f"updated_blocks={len(updated_block_ids)}\n",
                log_file=log_file,
            )
        return updated_block_ids

    return collect_from_filters_fn(
        filters,
        block_size,
        total_n_gaussians,
    )


def log_paper_perf_profile(
    *,
    iteration: int,
    perf_times: Dict[str, float],
    log_file,
    storage_adapter=None,
) -> None:
    def _dt(start: str, end: str) -> float:
        iter_start = perf_times["iter_start"]
        return (perf_times.get(end, iter_start) - perf_times.get(start, iter_start)) * 1000.0

    perf_message = (
        f"[PERF] Iter {iteration}: "
        f"setup={_dt('iter_start','stage1_setup_done'):.0f}ms  "
        f"ssd_cull+load={_dt('stage1_setup_done','stage1_5_ssd_done'):.0f}ms  "
        f"gauss_cull={_dt('stage1_5_ssd_done','stage2_3_culling_done'):.0f}ms  "
        f"train={_dt('stage2_3_culling_done','stage4_train_done'):.0f}ms  "
        f"optim={_dt('stage5_optim_start','stage5_optim_done'):.0f}ms  "
        f"writeback={_dt('stage5_optim_done','stage5_writeback_done'):.0f}ms  "
        f"cuda_sync={_dt('stage5_writeback_done','stage5_sync_done'):.0f}ms  "
        f"cleanup={_dt('stage5_sync_done','iter_end'):.0f}ms  "
        f"TOTAL={_dt('iter_start','iter_end'):.0f}ms\n"
    )
    log_file.write(perf_message)
    write_paper_phase1_log(perf_message, log_file=None)
    log_paper_warm_layer_metrics(storage_adapter, iteration, "perf", log_file=log_file)


def load_paper_stage1_working_set(
    *,
    gaussians,
    args,
    iteration: int,
    total_n_gaussians: int,
    visible_block_ids: List[int],
    active_blocks_ram,
    current_camera_blocks: Dict[int, List[int]],
    enable_retention: bool,
    use_fast_ram_ssd_path: bool,
    training_schedule,
    storage_adapter,
    should_log: bool,
    get_double_buffer_gpu_fn: Callable,
    ensure_local_to_global_mapping_fn: Callable,
    resolve_current_iteration_resident_blocks_fn: Callable,
    log_file=None,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool, str]:
    used_prefetch_buffer = False
    load_source: Optional[str] = None
    gpu_tensors = None
    retention_stats = None

    if training_schedule is not None:
        double_buffer = get_double_buffer_gpu_fn(
            num_total=total_n_gaussians,
            block_size=args.gaussian_block_size,
            device="cuda",
        )
        if double_buffer.wait_for_prefetch(iteration):
            double_buffer.swap_buffers()
            gpu_tensors, retention_stats = activate_paper_prefetched_buffer(
                gaussians,
                double_buffer,
                ensure_local_to_global_mapping_fn,
            )
            used_prefetch_buffer = True
            load_source = "prefetched_ab_buffer"
            gaussians._paper_ab_runtime_stats["prefetch_hits"] += 1
            if should_log:
                log_ab_buffer_activation(
                    iteration=iteration,
                    double_buffer=double_buffer,
                    visible_block_ids=visible_block_ids,
                    log_file=log_file,
                )
                log_paper_warm_layer_metrics(storage_adapter, iteration, "load", log_file=log_file)

    if used_prefetch_buffer:
        return gpu_tensors, retention_stats, used_prefetch_buffer, str(load_source)

    ab_stats = gaussians._paper_ab_runtime_stats
    if iteration == 1:
        ab_stats["cold_starts"] += 1
        miss_reason = "cold_start"
    else:
        ab_stats["prefetch_misses"] += 1
        miss_reason = "prefetch_miss"
    ab_stats["sync_fallbacks"] += 1
    if should_log:
        log_ab_buffer_sync_fallback(
            iteration=iteration,
            miss_reason=miss_reason,
            ab_stats=ab_stats,
            log_file=log_file,
        )

    sync_visible_block_ids = visible_block_ids
    total_blocks_estimate = (total_n_gaussians + args.gaussian_block_size - 1) // args.gaussian_block_size
    r_t_blocks, r_t_source = resolve_current_iteration_resident_blocks_fn(
        gaussians=gaussians,
        args=args,
        visible_block_ids=visible_block_ids,
        num_total_blocks=int(total_blocks_estimate),
        current_camera_blocks=current_camera_blocks,
    )

    if not use_fast_ram_ssd_path:
        reachable = set(int(b) for b in visible_block_ids)
        if active_blocks_ram is not None:
            reachable.update(int(b) for b in active_blocks_ram.keys())
        r_t_blocks_restricted = [b for b in r_t_blocks if b in reachable]
        if len(r_t_blocks_restricted) != len(r_t_blocks) and should_log:
            write_paper_phase1_log(
                f"[PAPER RESIDENT ENFORCEMENT] Iter {iteration}: "
                f"Dropped {len(r_t_blocks) - len(r_t_blocks_restricted)} R_t blocks "
                f"not reachable from K_t∪RAM cache (source={r_t_source})\n",
                log_file=log_file,
            )
        r_t_blocks = r_t_blocks_restricted

    policy = str(getattr(args, "paper_resident_selection_policy", "passthrough_active_set")).lower()
    if policy in {"topc", "topc_strict", "topc_balanced"} and r_t_blocks:
        sync_visible_block_ids = r_t_blocks
        load_source = f"sync_resident_set({r_t_source})"
        if should_log:
            write_paper_phase1_log(
                f"[PAPER RESIDENT ENFORCEMENT] Iter {iteration}: "
                f"Materializing |R_t|={len(r_t_blocks)} "
                f"(bound by capacity C={args.paper_resident_capacity_blocks}) "
                f"instead of full |K_t|={len(visible_block_ids)} [source={r_t_source}]\n",
                log_file=log_file,
            )
    elif should_log:
        write_paper_phase1_log(
            f"[PAPER A/B BUFFER] Iter {iteration}: "
            f"Loading full |K_t|={len(visible_block_ids)} "
            f"(policy={policy}, capacity_enforcement disabled)\n",
            log_file=log_file,
        )

    active_block_reader = getattr(gaussians, "_block_reader", None)
    gpu_tensors, retention_stats = gaussians.gpu_working_set_manager.load_visible_blocks_with_retention(
        visible_block_ids=sync_visible_block_ids,
        active_blocks_ram=active_blocks_ram,
        enable_retention=enable_retention,
        unified_params=(
            gaussians._unified_params
            if (active_block_reader is None and use_fast_ram_ssd_path)
            else None
        ),
        block_reader=active_block_reader,
    )
    if load_source is None:
        load_source = "sync_ram_to_gpu"
    return gpu_tensors, retention_stats, used_prefetch_buffer, load_source


def configure_gpu_resident_optimizer_state(
    *,
    gaussians,
    args,
    actual_current_resident_blocks: List[int],
    iteration: int,
    get_gpu_resident_optimizer_fn: Callable,
    log_file=None,
) -> None:
    if getattr(gaussians, "_paper_optimizer_state_mode", "full_cpu") != "resident_blocks":
        return
    if args.stop_update_param:
        return

    resident_state_optimizer = get_gpu_resident_optimizer_fn(gaussians, args.bsz)
    resident_state_optimizer.set_resident_blocks(actual_current_resident_blocks)
    if iteration == 1:
        write_paper_phase1_log(
            "[PAPER OPTIMIZER STATE] gpu_resident backend enabled: optimizer moments "
            "exist only for resident blocks on GPU and cold-restart on block re-admission.\n",
            log_file=log_file,
        )


def run_gpu_resident_adam_step(
    *,
    gaussians,
    args,
    iteration: int,
    total_n_gaussians: int,
    sparse_visibility_indices,
    sparse_grad_local_ids,
    sparse_grad_components,
    get_gpu_resident_optimizer_fn: Callable,
    ensure_local_to_global_mapping_fn: Callable,
    log_prefix: str = "[PAPER SSD MODE]",
    log_file=None,
) -> Dict[str, Any]:
    write_paper_phase1_log(
        f"{log_prefix} Using GPU resident Adam for optimization\n",
        log_file=log_file,
    )
    if args.stop_update_param:
        return {}

    torch.cuda.nvtx.range_push("Paper SSD: GPU Resident Adam")
    try:
        gpu_resident_optimizer = get_gpu_resident_optimizer_fn(gaussians, args.bsz)
        optimizer_updated_global_indices = sparse_visibility_indices
        if optimizer_updated_global_indices is None and sparse_grad_local_ids is not None:
            local_to_global = ensure_local_to_global_mapping_fn(
                gaussians,
                total_n_gaussians,
                log_file=log_file,
                context=f"gpu_resident_optimizer_iter_{iteration}",
            )
            optimizer_updated_global_indices = local_to_global[sparse_grad_local_ids].cpu()
        _ = optimizer_updated_global_indices

        step_stats = gpu_resident_optimizer.step(
            iteration=iteration,
            gaussians=gaussians,
            sparse_grad_local_ids=sparse_grad_local_ids,
            sparse_grad_components=sparse_grad_components,
        )
        gaussians._paper_last_gpu_optimizer_step = dict(step_stats)
        write_paper_phase1_log(
            f"{log_prefix} GPU resident Adam updated {step_stats['touched_rows']} rows "
            f"across {step_stats['updated_blocks']} resident blocks "
            f"(cold_rows={step_stats['cold_rows']})\n",
            log_file=log_file,
        )
        return dict(step_stats)
    finally:
        torch.cuda.nvtx.range_pop()


def train_tide_batch(
    *,
    gaussians,
    scene,
    batched_cameras,
    parameters_grad_buffer,
    background,
    pipe_args,
    comm_stream,
    perm_generator,
    storage_adapter,
    training_schedule,
):
    """Train one batch through the TideGS out-of-core engine."""
    args = getattr(gaussians, "args", None)
    if args is None:
        import utils.general_utils as utils

        args = utils.get_args()
    validate_tide_runtime_args(args)

    if storage_adapter is None:
        raise RuntimeError("Pure SSD/Tide release path requires a TideStorageAdapter.")
    if getattr(storage_adapter, "execution_mode", "") != "paper":
        raise RuntimeError("Pure SSD/Tide release path requires storage_adapter.execution_mode == 'paper'.")
    if training_schedule is None:
        raise RuntimeError("Pure SSD/Tide release path requires a camera training schedule.")

    # Keep this import lazy while engine.py remains the shared execution core.
    # engine.py imports this module for TideGS helpers, so a module-level import
    # would create a circular dependency during release cleanup.
    from .engine import clm_offload_train_one_batch

    return clm_offload_train_one_batch(
        gaussians,
        scene,
        batched_cameras,
        parameters_grad_buffer,
        background,
        pipe_args,
        comm_stream,
        perm_generator,
        storage_adapter=storage_adapter,
        training_schedule=training_schedule,
    )
