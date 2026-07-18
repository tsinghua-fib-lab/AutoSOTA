import torch
import math
import utils.general_utils as utils
import clm_kernels
import fast_tsp
import os
import torch.nn as nn
import gc
from typing import Optional, List, Dict, Tuple

from strategies.tide_engine.gpu_resident_optimizer import GPUResidentAdam

def get_gpu_resident_optimizer(gaussians, batch_size):
    desired_block_size = int(
        getattr(
            gaussians,
            '_paper_optimizer_block_size',
            getattr(getattr(gaussians, 'args', None), 'gaussian_block_size', 4096),
        )
    )
    current_optimizer = getattr(gaussians, '_paper_gpu_resident_optimizer', None)
    needs_recreate = (
        current_optimizer is None
        or getattr(current_optimizer, 'batch_size', None) != batch_size
        or getattr(current_optimizer, 'block_size', desired_block_size) != desired_block_size
    )
    if needs_recreate:
        gaussians._paper_gpu_resident_optimizer = GPUResidentAdam(
            batch_size=batch_size,
            block_size=desired_block_size,
            device='cuda',
        )
    return gaussians._paper_gpu_resident_optimizer


def shutdown_gpu_resident_optimizer(gaussians=None):
    if gaussians is None:
        return
    if hasattr(gaussians, '_paper_gpu_resident_optimizer'):
        gaussians._paper_gpu_resident_optimizer = None


def get_total_gaussians_count(gaussians, storage_adapter=None):
    """Return the full-scene Gaussian count from paper metadata or fallbacks."""
    num_total = getattr(gaussians, '_paper_unified_params_num_total', None)
    if num_total is not None:
        return int(num_total)

    unified_params = getattr(gaussians, '_unified_params', None)
    if unified_params is not None:
        return int(unified_params.shape[0])

    gpu_ws = getattr(gaussians, 'gpu_working_set_manager', None)
    if gpu_ws is not None:
        ws_total = getattr(gpu_ws, 'num_total_gaussians', None)
        if ws_total is not None:
            return int(ws_total)

    if storage_adapter is not None and hasattr(storage_adapter, 'total_gaussians'):
        total_gaussians = getattr(storage_adapter, 'total_gaussians', None)
        if total_gaussians is not None:
            return int(total_gaussians)

    xyz = getattr(gaussians, '_xyz', None)
    if xyz is not None:
        return int(xyz.shape[0])

    grad_buffer = getattr(gaussians, 'parameters_grad_buffer', None)
    if grad_buffer is not None:
        return int(grad_buffer.shape[0])

    raise RuntimeError(
        'Unable to determine total Gaussian count: paper-mode tensors were released, '
        'and no fallback metadata was available.'
    )


def ensure_local_to_global_mapping(gaussians, total_n_gaussians: int, *, log_file=None, context: str = ""):
    """Ensure gpu_working_set_manager.local_to_global_idx exists.

    Some paper-mode branches can preserve resident tensors while dropping the
    compact local->global mapping metadata. Rebuild it from block slices when
    needed so later local-index operations remain valid.
    """
    manager = getattr(gaussians, 'gpu_working_set_manager', None)
    if manager is None:
        raise RuntimeError('gpu_working_set_manager is required to build local_to_global_idx')

    local_to_global = getattr(manager, 'local_to_global_idx', None)
    if local_to_global is not None:
        return local_to_global

    block_size = int(getattr(manager, 'block_size', getattr(getattr(gaussians, 'args', None), 'gaussian_block_size', 4096)))
    device = getattr(manager, 'device', torch.device('cuda'))
    rebuilt_global_ids = []

    block_to_gpu_slice = getattr(manager, 'block_to_gpu_slice', {}) or {}
    if block_to_gpu_slice:
        plans = []
        for block_id, local_slice in block_to_gpu_slice.items():
            if local_slice is None:
                continue
            local_len = max(0, int(local_slice.stop) - int(local_slice.start))
            if local_len <= 0:
                continue
            start_idx = int(block_id) * block_size
            max_rows = max(0, min(start_idx + block_size, total_n_gaussians) - start_idx)
            row_count = min(local_len, max_rows)
            if row_count <= 0:
                continue
            plans.append((int(local_slice.start), start_idx, row_count))

        for _, start_idx, row_count in sorted(plans, key=lambda item: item[0]):
            rebuilt_global_ids.extend(range(start_idx, start_idx + row_count))
    else:
        loaded_blocks = getattr(manager, 'loaded_blocks', []) or []
        for block_id in loaded_blocks:
            start_idx = int(block_id) * block_size
            end_idx = min(start_idx + block_size, total_n_gaussians)
            if end_idx > start_idx:
                rebuilt_global_ids.extend(range(start_idx, end_idx))

    manager.local_to_global_idx = torch.tensor(rebuilt_global_ids, dtype=torch.long, device=device)
    if context:
        utils.print_rank_0(
            f"[SSD MAP] Rebuilt local_to_global_idx in {context}: {manager.local_to_global_idx.numel()} entries"
        )
        if log_file is not None:
            log_file.write(
                f"[SSD MAP] Rebuilt local_to_global_idx in {context}: {manager.local_to_global_idx.numel()} entries\n"
            )
    return manager.local_to_global_idx

# Paper/Tide double-buffer GPU residency support.
from strategies.tide_engine.double_buffer_gpu import (
    DoubleBufferGPUWorkingSet,
    get_next_iteration_blocks,
)
from strategies.tide_engine.runtime import (
    apply_paper_writeback_payload as _apply_paper_writeback_payload,
    collect_camera_visible_blocks as _collect_camera_visible_blocks,
    collect_paper_updated_block_ids as _collect_paper_updated_block_ids,
    configure_gpu_resident_optimizer_state as _configure_gpu_resident_optimizer_state,
    compute_paper_block_sets as _compute_paper_block_sets,
    initialize_paper_mode_runtime_state as _initialize_paper_mode_runtime_state,
    log_batch_kt_metrics as _log_batch_kt_metrics,
    log_delta_refresh as _log_delta_refresh,
    log_delta_stream_prefetch as _log_delta_stream_prefetch,
    log_ab_buffer_prefetch_failure as _log_ab_buffer_prefetch_failure,
    log_double_buffer_stats as _log_double_buffer_stats,
    log_empty_microbatch_camera as _log_empty_microbatch_camera,
    log_empty_paper_projection_cameras as _log_empty_paper_projection_cameras,
    log_empty_stage2_loaded_gaussians as _log_empty_stage2_loaded_gaussians,
    log_filters_local_reordered as _log_filters_local_reordered,
    log_paper_block_sets as _log_paper_block_sets,
    log_paper_batch_debug as _log_paper_batch_debug,
    log_paper_block_visibility_debug as _log_paper_block_visibility_debug,
    log_paper_gpu_working_set_parameters as _log_paper_gpu_working_set_parameters,
    log_paper_perf_profile as _log_paper_perf_profile,
    log_paper_warm_layer_metrics as _log_paper_warm_layer_metrics,
    log_ssd_stage1_debug as _log_ssd_stage1_debug,
    log_early_delta_hint as _log_early_delta_hint,
    log_paper_cpu_source_refs_restored as _log_paper_cpu_source_refs_restored,
    log_paper_interbatch_sh_cache_disabled as _log_paper_interbatch_sh_cache_disabled,
    log_paper_order_calculation_skip as _log_paper_order_calculation_skip,
    log_paper_resident_camera_coverage as _log_paper_resident_camera_coverage,
    log_paper_sh_indexed as _log_paper_sh_indexed,
    log_paper_working_set_cleared as _log_paper_working_set_cleared,
    log_paper_working_set_loaded as _log_paper_working_set_loaded,
    log_paper_writeback_finalize as _log_paper_writeback_finalize,
    log_paper_writeback_staged as _log_paper_writeback_staged,
    load_paper_stage1_working_set as _load_paper_stage1_working_set,
    map_compact_filters_to_global as _map_compact_filters_to_global,
    paper_debug_logging_enabled as _paper_debug_logging_enabled,
    resolve_paper_stage0_active_blocks as _resolve_paper_stage0_active_blocks,
    resolve_current_iteration_resident_blocks as _resolve_current_iteration_resident_blocks,
    resolve_stage2_loaded_gaussian_ids as _resolve_stage2_loaded_gaussian_ids,
    run_stage0_schedule_interaction as _run_stage0_schedule_interaction,
    run_gpu_resident_adam_step as _run_gpu_resident_adam_step,
    seed_paper_active_buffer_from_manager as _seed_paper_active_buffer_from_manager,
    write_paper_phase1_log as _write_paper_phase1_log,
)

# 全局双缓冲 GPU 管理器
_double_buffer_gpu: Optional[DoubleBufferGPUWorkingSet] = None

def get_double_buffer_gpu(
    num_total: int,
    block_size: int,
    device: str = 'cuda',
    verbose: bool = False,
) -> DoubleBufferGPUWorkingSet:
    """获取或创建全局双缓冲 GPU 管理器"""
    global _double_buffer_gpu
    if _double_buffer_gpu is None:
        _double_buffer_gpu = DoubleBufferGPUWorkingSet(
            num_total_gaussians=num_total,
            block_size=block_size,
            device=device,
            verbose=verbose,
        )
    return _double_buffer_gpu

def shutdown_double_buffer_gpu():
    """关闭双缓冲 GPU 管理器"""
    global _double_buffer_gpu
    if _double_buffer_gpu is not None:
        _double_buffer_gpu.clear()
        _double_buffer_gpu = None

import numpy as np
from gsplat import (
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)
from clm_kernels import (
    send_shs2gpu_stream,
    send_shs2cpu_grad_buffer_stream,
    send_shs2gpu_stream_retention,
    send_shs2cpu_grad_buffer_stream_retention,
    spherical_harmonics_bwd_inplace,
)
from densification import update_densification_stats_offload_accum_grads
from strategies.base_engine import (
    torch_compiled_loss,
    TILE_SIZE,
    calculate_filters,
    pipeline_forward_one_step,
)




def _collect_updated_block_ids(filters, block_size: int, total_n_gaussians: Optional[int] = None):
    visible_indices_list = []
    for f in filters:
        if f is None or f.numel() == 0:
            continue
        if f.dtype == torch.bool:
            idxs = torch.nonzero(f, as_tuple=False).squeeze(1)
        else:
            idxs = f
        if idxs.device.type != 'cpu':
            idxs = idxs.cpu()
        idxs = idxs.to(dtype=torch.long)
        if total_n_gaussians is not None:
            idxs = idxs[(idxs >= 0) & (idxs < total_n_gaussians)]
        if idxs.numel() > 0:
            visible_indices_list.append(idxs)

    if len(visible_indices_list) == 0:
        return []

    unique_gaussian_ids = torch.cat(visible_indices_list).unique()
    return (unique_gaussian_ids // block_size).unique().cpu().tolist()


def _collect_updated_block_ids_from_indices(
    gaussian_indices: Optional[torch.Tensor],
    block_size: int,
    total_n_gaussians: Optional[int] = None,
):
    if gaussian_indices is None or gaussian_indices.numel() == 0:
        return []

    idxs = gaussian_indices
    if idxs.dtype == torch.bool:
        idxs = torch.nonzero(idxs, as_tuple=False).squeeze(1)
    if idxs.device.type != 'cpu':
        idxs = idxs.cpu()
    idxs = idxs.to(dtype=torch.long)
    if total_n_gaussians is not None:
        idxs = idxs[(idxs >= 0) & (idxs < total_n_gaussians)]
    if idxs.numel() == 0:
        return []
    return torch.unique(torch.div(idxs, block_size, rounding_mode='floor')).tolist()


def _materialize_updated_blocks_from_cpu_views(
    updated_block_ids,
    total_n_gaussians: int,
    block_size: int,
    original_xyz,
    original_scaling,
    original_rotation,
    original_opacity,
    original_features_dc,
    original_features_rest,
):
    updated_blocks_dict = {}
    for block_id in updated_block_ids:
        start_idx = block_id * block_size
        end_idx = min(start_idx + block_size, total_n_gaussians)
        row_count = max(0, end_idx - start_idx)
        if row_count <= 0:
            continue

        xyz_block = original_xyz[start_idx:end_idx].detach()
        scaling_block = original_scaling[start_idx:end_idx].detach()
        rotation_block = original_rotation[start_idx:end_idx].detach()
        opacity_block = original_opacity[start_idx:end_idx].detach()
        f_dc = original_features_dc[start_idx:end_idx].detach()
        f_rest = original_features_rest[start_idx:end_idx].detach()

        updated_blocks_dict[block_id] = torch.cat([
            xyz_block,
            scaling_block,
            rotation_block,
            opacity_block,
            f_dc,
            f_rest,
        ], dim=1)

    return updated_blocks_dict


def _sync_updated_blocks_from_gpu_views_to_cpu(
    updated_block_ids,
    total_n_gaussians: int,
    block_size: int,
    gpu_working_set_manager,
    original_xyz,
    original_scaling,
    original_rotation,
    original_opacity,
    original_features_dc,
    original_features_rest,
):
    """Copy updated-block parameter values from GPU working set back to CPU views.

    Priority 4: Batched D2H writeback.
        The older per-block implementation issued ``6 × len(updated_block_ids)`` separate
        ``.detach().cpu()`` round-trips — each round-trip carried ~300 µs of
        kernel-launch + synchronization overhead, independent of data volume.
        At C=2048 with ~1500 updated blocks per iteration this amounts to
        ``1500 × 6 ≈ 9000`` D2H syncs, dominating writeback latency.

        This implementation collapses the per-block loop into **one** ``torch.cat``
        (single fused GPU kernel) plus **one** ``.cpu()`` DMA **per component**
        (6 total), and restores the original layout on the CPU side with
        contiguous slice copies.  Correctness-preserving:

        * Skips blocks with ``block_to_gpu_slice.get(block_id) is None``
        * Clamps ``row_count = min(block_size, local_slice span, tail rows)``
        * Writes exactly the same rows into ``original_*`` as before
    """
    if gpu_working_set_manager is None or not updated_block_ids:
        return 0

    # Step 1: compute per-block (global_range, local_slice, row_count) tuples.
    plans: List[Tuple[int, int, slice, int]] = []
    for block_id in updated_block_ids:
        local_slice = gpu_working_set_manager.block_to_gpu_slice.get(block_id)
        if local_slice is None:
            continue

        start_idx = block_id * block_size
        end_idx = min(start_idx + block_size, total_n_gaussians)
        row_count = max(0, end_idx - start_idx)
        if row_count <= 0:
            continue

        local_len = int(local_slice.stop - local_slice.start)
        if local_len != row_count:
            row_count = min(row_count, local_len)
            local_slice = slice(local_slice.start, local_slice.start + row_count)
            end_idx = start_idx + row_count

        plans.append((start_idx, end_idx, local_slice, row_count))

    if not plans:
        return 0

    # Step 2+3: per component, gather-via-cat and DMA in a single chained
    # expression.  The intermediate concatenated GPU tensor is a temporary
    # whose reference drops immediately after ``.cpu()`` completes, so the
    # allocator can reuse its VRAM for the next component — peak extra VRAM
    # is bounded by ``max_col_width × total_rows × 4`` (~250 MB for 1.4 M
    # rows × 45 cols of features_rest) rather than the ~1.4 GB that would be
    # needed if we pre-cat all six components first.
    #
    # ``torch.cat`` of N slices lowers to a single fused CUDA kernel, so this
    # produces exactly one DMA source per component.  ``.cpu()`` is a blocking
    # synchronous transfer; six serial calls replace the older
    # ``6 × len(updated_block_ids)`` round-trips.
    local_slices = [plan[2] for plan in plans]
    with torch.no_grad():
        xyz_cpu = torch.cat(
            [gpu_working_set_manager.gpu_xyz[s] for s in local_slices], dim=0
        ).detach().cpu()
        scaling_cpu = torch.cat(
            [gpu_working_set_manager.gpu_scaling[s] for s in local_slices], dim=0
        ).detach().cpu()
        rotation_cpu = torch.cat(
            [gpu_working_set_manager.gpu_rotation[s] for s in local_slices], dim=0
        ).detach().cpu()
        opacity_cpu = torch.cat(
            [gpu_working_set_manager.gpu_opacity[s] for s in local_slices], dim=0
        ).detach().cpu()
        features_dc_cpu = torch.cat(
            [gpu_working_set_manager.gpu_features_dc[s] for s in local_slices], dim=0
        ).detach().cpu()
        features_rest_cpu = torch.cat(
            [gpu_working_set_manager.gpu_features_rest[s] for s in local_slices], dim=0
        ).detach().cpu()

    # Step 4: contiguous slice scatter on CPU — each ``.copy_`` is a fast memcpy
    # into a view over ``_unified_params`` and does not incur any GPU syncs.
    cpu_offset = 0
    synchronized = 0
    for start_idx, end_idx, _, row_count in plans:
        tail = cpu_offset + row_count
        original_xyz[start_idx:end_idx].copy_(xyz_cpu[cpu_offset:tail])
        original_scaling[start_idx:end_idx].copy_(scaling_cpu[cpu_offset:tail])
        original_rotation[start_idx:end_idx].copy_(rotation_cpu[cpu_offset:tail])
        original_opacity[start_idx:end_idx].copy_(opacity_cpu[cpu_offset:tail])
        original_features_dc[start_idx:end_idx].copy_(features_dc_cpu[cpu_offset:tail])
        original_features_rest[start_idx:end_idx].copy_(features_rest_cpu[cpu_offset:tail])
        cpu_offset = tail
        synchronized += 1

    return synchronized


def _build_updated_blocks_dict_from_gpu(
    updated_block_ids,
    total_n_gaussians: int,
    block_size: int,
    gpu_working_set_manager,
) -> Dict[int, torch.Tensor]:
    """Phase 3B: direct GPU → per-block CPU writeback path.

    Collects updated blocks from the GPU working set, performs ONE batched
    ``.cpu()`` transfer per parameter component, and assembles per-block
    ``(rows, 59)`` tensors in the SSD / TieredCache column layout
    (``xyz | scaling | rotation | opacity | features_dc | features_rest``).

    This replaces the older two-stage path that went:
        GPU -> original_xyz (view into _unified_params)
            -> _materialize_updated_blocks_from_cpu_views (per-block torch.cat)

    The round-trip through ``_unified_params`` is eliminated, which both
    halves the wall-clock writeback cost and (more importantly) frees the
    caller from having to keep ``_unified_params`` RAM-resident at all.
    """
    if gpu_working_set_manager is None or not updated_block_ids:
        return {}

    plans: List[Tuple[int, slice, int]] = []
    for block_id in updated_block_ids:
        local_slice = gpu_working_set_manager.block_to_gpu_slice.get(block_id)
        if local_slice is None:
            continue
        start_idx = block_id * block_size
        end_idx = min(start_idx + block_size, total_n_gaussians)
        row_count = max(0, end_idx - start_idx)
        if row_count <= 0:
            continue
        local_len = int(local_slice.stop - local_slice.start)
        if local_len <= 0:
            continue
        if local_len != row_count:
            row_count = min(row_count, local_len)
            if row_count <= 0:
                continue
            local_slice = slice(local_slice.start, local_slice.start + row_count)
        plans.append((int(block_id), local_slice, row_count))

    if not plans:
        return {}

    local_slices = [plan[1] for plan in plans]
    with torch.no_grad():
        # One GPU-side gather + one D2H per component (6 total).
        xyz_cpu = torch.cat(
            [gpu_working_set_manager.gpu_xyz[s] for s in local_slices], dim=0
        ).detach().cpu()
        scaling_cpu = torch.cat(
            [gpu_working_set_manager.gpu_scaling[s] for s in local_slices], dim=0
        ).detach().cpu()
        rotation_cpu = torch.cat(
            [gpu_working_set_manager.gpu_rotation[s] for s in local_slices], dim=0
        ).detach().cpu()
        opacity_cpu = torch.cat(
            [gpu_working_set_manager.gpu_opacity[s] for s in local_slices], dim=0
        ).detach().cpu()
        features_dc_cpu = torch.cat(
            [gpu_working_set_manager.gpu_features_dc[s] for s in local_slices], dim=0
        ).detach().cpu()
        features_rest_cpu = torch.cat(
            [gpu_working_set_manager.gpu_features_rest[s] for s in local_slices], dim=0
        ).detach().cpu()

    # Per-block assembly in CACHE layout (xyz | scaling | rotation | opacity
    # | features_dc | features_rest).  The concatenation is a pure CPU
    # operation on small tensors so it is negligible compared to the D2H.
    updated_blocks_dict: Dict[int, torch.Tensor] = {}
    cpu_offset = 0
    for block_id, _, row_count in plans:
        tail = cpu_offset + row_count
        updated_blocks_dict[block_id] = torch.cat([
            xyz_cpu[cpu_offset:tail],
            scaling_cpu[cpu_offset:tail],
            rotation_cpu[cpu_offset:tail],
            opacity_cpu[cpu_offset:tail],
            features_dc_cpu[cpu_offset:tail],
            features_rest_cpu[cpu_offset:tail],
        ], dim=1)
        if updated_blocks_dict[block_id].numel() == 0:
            updated_blocks_dict.pop(block_id, None)
        cpu_offset = tail

    return updated_blocks_dict


def visualize_frustum_culling_inline(
    batched_cameras,
    block_bounds,
    visible_block_ids,
    iteration,
    save_dir='debug_frustum',
):
    """Frustum HTML visualization is not included in the release package."""
    return None


def pipeline_forward_one_step_shs_inplace(
    filtered_opacity_gpu,
    filtered_scaling_gpu,
    filtered_rotation_gpu,
    filtered_xyz_gpu,
    filtered_shs,
    camera,
    scene,
    gaussians,
    background,
    pipe_args,
    use_autograd_for_sh=False,  # New parameter: whether to use autograd for SH
):
    MICRO_BATCH_SIZE = 1  # NOTE: microbatch here only contains one camera.

    viewmat = camera.world_view_transform.transpose(0, 1)  # why transpose
    # K = camera.create_k_on_gpu() # create K now, which may invoke cpu-gpu transfer
    K = camera.K
    n_selected = filtered_xyz_gpu.shape[0]
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())

    batched_radiis, batched_means2D, batched_depths, batched_conics, _ = (
        fully_fused_projection(
            means=filtered_xyz_gpu,  # (N, 3)
            covars=None,
            quats=filtered_rotation_gpu,
            scales=filtered_scaling_gpu,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=image_width,
            height=image_height,
            packed=False,
        )
    )  # (1, N), (1, N, 2), (1, N), (1, N, 3), (1, N)

    batched_means2D.retain_grad()  # this is only for training.

    sh_degree = gaussians.active_sh_degree
    camtoworlds = camera.camtoworlds
    # camtoworlds = torch.inverse(viewmat.unsqueeze(0)) # (4, 4)
    dirs = filtered_xyz_gpu[None, :, :] - camtoworlds[:, None, :3, 3]
    filtered_shs = filtered_shs.reshape(1, n_selected, 16, 3)

    # ====================================================================
    # Conditional autograd for spherical harmonics.
    # ====================================================================
    if use_autograd_for_sh:
        # GPU-resident path: let PyTorch autograd handle SH gradients.
        # This is cleaner and avoids manual gradient computation
        batched_colors_origin = spherical_harmonics(
            degrees_to_use=sh_degree, dirs=dirs, coeffs=filtered_shs
        )
        batched_colors_detached = batched_colors_origin  # No detach needed!
        batched_colors = torch.clamp_min(batched_colors_origin + 0.5, 0.0)
    else:
        # Archived split-feature path: manual gradient computation for host SH features.
        dirs.retain_grad()  # Need to retain for manual backward
        with torch.no_grad():
            batched_colors_origin = spherical_harmonics(
                degrees_to_use=sh_degree, dirs=dirs, coeffs=filtered_shs
            )
        batched_colors_detached = batched_colors_origin.detach().requires_grad_()
        batched_colors = torch.clamp_min(batched_colors_detached + 0.5, 0.0)
    
    batched_opacities = filtered_opacity_gpu.squeeze(1).unsqueeze(0)  # (N, 1) -> (1, N)

    # NOTE: In the above code, we keep the first batch dimension, even if it is always 1.

    # render
    # Identify intersecting tiles.
    tile_width = math.ceil(image_width / float(TILE_SIZE))
    tile_height = math.ceil(image_height / float(TILE_SIZE))

    # flatten_ids: (C*N)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d=batched_means2D,
        radii=batched_radiis,
        depths=batched_depths,
        tile_size=TILE_SIZE,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, MICRO_BATCH_SIZE, tile_width, tile_height
    )  # (MICRO_BATCH_SIZE, tile_height, tile_width)

    # Rasterize to pixels. batched_rendered_image: (B, image_height, image_width, 3)
    backgrounds = (
        background.repeat(MICRO_BATCH_SIZE, 1) if background is not None else None
    )
    rendered_image, _ = rasterize_to_pixels(
        means2d=batched_means2D,
        conics=batched_conics,
        colors=batched_colors,
        opacities=batched_opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=TILE_SIZE,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
    )

    rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()

    return (
        rendered_image,
        batched_means2D,
        batched_radiis,
        batched_colors_detached,
        dirs,
    )


import queue
import time


def order_calculation(
    filters,
    batched_cameras,
    n_gaussians,
    bsz,
    perm_generator,
    args,
    *,
    build_legacy_update_lists=True,
    build_visibility_mask=True,
):
    # Avoid materializing a full GPU permutation for billion-scale scenes.
    # For the TSP distance estimate we only need n_sampled probe indices, and
    # sampling with replacement has negligible duplicate rate at these scales.
    large_randperm_threshold = 50_000_000

    match bsz:
        case 4 | 8:
            dtype = torch.int8
        case 16:
            dtype = torch.int16
        case 32:
            dtype = torch.int32
        case 64:
            dtype = torch.int64
        case _:
            raise ValueError("Currently supported bsz: (4, 8, 16, 32, 64).")

    torch.cuda.nvtx.range_push("init bitmap and vecs")
    gs_bitmap = torch.zeros((n_gaussians), dtype=dtype, device="cuda")
    # Encode bitmap: MSB->first microbatch; LSB->last microbatch
    for i, f in enumerate(filters):
        clm_kernels.scatter_to_bit(gs_bitmap, f, bsz - 1 - i)

    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("generate distance matrix")
    # Downsample.
    if bsz >= 32:
        n_sampled = n_gaussians // bsz**2
    else:
        n_sampled = n_gaussians // 32
    n_sampled = max(1, min(n_sampled, n_gaussians))

    debug_sampling_cap = None
    if getattr(args, "debug_max_train_cameras", -1) > 0 or getattr(args, "debug_fast_init_scales", False):
        debug_sampling_cap = 1_000_000
        if n_sampled > debug_sampling_cap:
            n_sampled = debug_sampling_cap

    if n_gaussians > large_randperm_threshold:
        if not getattr(order_calculation, "_logged_large_n_sampling", False):
            extra = (
                f" Debug cap applied: {debug_sampling_cap:,}."
                if debug_sampling_cap is not None
                else ""
            )
            print(
                f"[ORDER_CALC] Large-N sampling fallback: n_gaussians={n_gaussians:,}, "
                f"n_sampled={n_sampled:,}. Using torch.randint instead of full randperm "
                f"to avoid GPU OOM.{extra}"
            )
            order_calculation._logged_large_n_sampling = True
        sampled_gaussian_ids = torch.randint(
            n_gaussians,
            (n_sampled,),
            generator=perm_generator,
            device="cuda",
            dtype=torch.int64,
        )
    else:
        sampled_gaussian_ids = torch.randperm(
            n_gaussians, generator=perm_generator, device="cuda"
        )[:n_sampled]
    sampled_bitmap = torch.gather(input=gs_bitmap, dim=0, index=sampled_gaussian_ids)
    # Unzip the bimap.
    unziped = torch.empty((bsz, n_sampled), dtype=torch.uint8, device="cuda")
    for i in range(bsz):
        unziped[bsz - 1 - i] = (sampled_bitmap & 1).to(torch.uint8)
        sampled_bitmap = sampled_bitmap >> 1
    # Compute distance matrix for archived in-batch camera reordering.
    distance_matrix = (
        (unziped.unsqueeze(1) ^ unziped.unsqueeze(0)).sum(dim=-1).tolist()
    )  # intermediate result: (bsz, bsz, n_sampled) = n_gaussians
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("solve order: tsp")
    ordered_cams = fast_tsp.find_tour(distance_matrix, 0.001)
    # find the minimum sparsity camera
    if args.reorder_by_min_sparsity_at_end:
        min_sparsity_i = bsz - 1
        for k in range(0, bsz - 1):
            if len(filters[ordered_cams[k]]) < len(
                filters[ordered_cams[min_sparsity_i]]
            ):
                min_sparsity_i = k
        ordered_cams = (
            ordered_cams[min_sparsity_i + 1 :] + ordered_cams[: min_sparsity_i + 1]
        )

    torch.cuda.nvtx.range_pop()
    batched_cameras = [batched_cameras[i] for i in ordered_cams]
    filters = [filters[i] for i in ordered_cams]
    sparsity = [len(filters[i]) / float(n_gaussians) for i in range(bsz)]

    if (not build_legacy_update_lists) and (not build_visibility_mask):
        return (
            None,
            batched_cameras,
            filters,
            sparsity,
            ordered_cams,
            None,
            None,
            None,
            None,
        )

    torch.cuda.nvtx.range_push("generate cpuadam update ls")
    # Re-encode the bitmap based on given order
    gs_bitmap.zero_()
    for i, f in enumerate(filters):
        clm_kernels.scatter_to_bit(gs_bitmap, f, bsz - 1 - i)


    if not build_legacy_update_lists:
        if build_visibility_mask and args.sparse_adam:
            visibility_mask = torch.zeros((n_gaussians,), dtype=torch.bool, device="cuda")
            for f in filters:
                src = torch.ones((len(f),), dtype=torch.bool, device="cuda")
                visibility_mask.scatter_(dim=0, index=f, src=src)
        else:
            visibility_mask = None
        torch.cuda.nvtx.range_pop()
        return (
            None,
            batched_cameras,
            filters,
            sparsity,
            ordered_cams,
            None,
            None,
            None,
            visibility_mask,
        )

    ffs = torch.empty(n_gaussians, dtype=torch.uint8, device="cuda")
    clm_kernels.extract_ffs(gs_bitmap, ffs)
    sorted_ffs, indices = torch.sort(ffs)
    # indices = indices.to(torch.int32) # convert to int32 because we assume the maximum gaussian index is less than 2^32
    elems, counts = torch.unique_consecutive(sorted_ffs, return_counts=True)
    update_ls = torch.split(indices, counts.tolist(), dim=0)
    update_ls = list(update_ls)
    for i in range(bsz + 1):
        if i not in elems:  # check if there is empty update
            update_ls.insert(i, torch.tensor([], dtype=torch.int64, device="cuda"))
    update_ls = [update_ls[0]] + update_ls[:0:-1]

    if args.sparse_adam:
        not_touched_ids = update_ls[0]
        src = torch.zeros((len(not_touched_ids),), dtype=torch.bool, device="cuda")
        visibility_mask = torch.ones(
            (n_gaussians,), dtype=torch.bool, device="cuda"
        ).scatter_(dim=0, index=not_touched_ids, src=src)
    else:
        visibility_mask = None
    torch.cuda.nvtx.range_pop()

    # HACK: Testing bsz=32/64 for now
    torch.cuda.nvtx.range_push("precompute sums")
    ps_grid_size, ps_blk_size = (64, 256)
    tmp_buffer = torch.empty(
        (bsz - 1, ps_grid_size * ps_blk_size), dtype=torch.int, device="cuda"
    )  # 31 * #t
    clm_kernels.compute_cnt_h(gs_bitmap, tmp_buffer, ps_grid_size, ps_blk_size)
    cnt_d = torch.sum(tmp_buffer, dim=1).flatten()
    filter_len = torch.tensor([len(f) for f in filters], device="cuda")
    cnt_h = filter_len[1:] - cnt_d
    cnt_g = filter_len[:-1] - cnt_d

    torch.cuda.nvtx.range_pop()
    del gs_bitmap, tmp_buffer, filter_len

    torch.cuda.nvtx.range_push("transfer cpuadam update list and sums to cpu")
    cnt_h, cnt_d, cnt_g = (
        cnt_h.to(torch.int64),
        cnt_d.to(torch.int64),
        cnt_g.to(torch.int64),
    )  # Then, cnt_h, cnt_d, cnt_g shares the same types with update_ls.
    data2cpu_ls = update_ls + [
        cnt_h,
        cnt_d,
        cnt_g,
    ]  # update_ls, cnt_h, cnt_d, cnt_g should all be int64.
    cat_data2cpu = torch.cat(data2cpu_ls, dim=0).to(torch.int32)
    cat_data2cpu_h = torch.empty_like(cat_data2cpu, device="cpu", pin_memory=True)
    data2cpu_dim = [len(d) for d in data2cpu_ls]
    cat_data2cpu_h.copy_(cat_data2cpu)
    data2cpu_ls_h = torch.split(cat_data2cpu_h, data2cpu_dim, dim=0)
    assert len(data2cpu_ls_h) == bsz + 4
    update_ls_cpu = data2cpu_ls_h[: bsz + 1]
    cnt_h = data2cpu_ls_h[-3]
    cnt_d = data2cpu_ls_h[-2]
    cnt_g = data2cpu_ls_h[-1]
    torch.cuda.nvtx.range_pop()

    finish_indices_filters = update_ls_cpu

    assert (
        len(finish_indices_filters) == bsz + 1
    ), "len(finish_indices_filters) should be equal to bsz + 1"
    assert (
        sum([len(indicies) for indicies in finish_indices_filters]) == n_gaussians
    ), f"{sum([len(indicies) for indicies in finish_indices_filters])}, {n_gaussians}"
    # Check max index (skip empty tensors)
    non_empty_max = [
        indicies.max().item()
        for indicies in finish_indices_filters
        if len(indicies) > 0
    ]
    if non_empty_max:
        max_index = max(non_empty_max)
        if max_index >= n_gaussians:
            import pdb

            pdb.set_trace()
            print(
                f"WARNING: Found invalid index {max_index} >= {n_gaussians}, will filter it out"
            )
        # assert max_index < n_gaussians, f"max of indices < n_gaussians: {max_index} < {n_gaussians}"

    return (
        finish_indices_filters,
        batched_cameras,
        filters,
        sparsity,
        ordered_cams,
        cnt_h,
        cnt_d,
        cnt_g,
        visibility_mask,
    )


def clm_offload_train_one_batch(
    gaussians,
    scene,
    batched_cameras,
    parameters_grad_buffer,
    background,
    pipe_args,
    comm_stream,
    perm_generator,
    storage_adapter=None,
    training_schedule=None,
):
    args = utils.get_args()
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    # ========================================================================
    # [PERF PROFILE] Lightweight per-iteration stage timer (prints every 50 iters)
    # ========================================================================
    import time as _time
    _perf_t = {}
    # iterations step by bsz (1, 65, 129, ...), so "% 50 == 1" only fires once
    # use a counter that increments per batch instead
    if not hasattr(clm_offload_train_one_batch, '_batch_count'):
        clm_offload_train_one_batch._batch_count = 0
    clm_offload_train_one_batch._batch_count += 1
    _perf_log = (clm_offload_train_one_batch._batch_count % 5 == 1)  # log every 5 batches
    def _ts(name):
        _perf_t[name] = _time.perf_counter()
    _ts('iter_start')

    # ============================================================================
    # STAGE 0: SSD Prefetch (Async, Non_blocking)
    # ============================================================================

    current_camera_ids = [cam.global_idx for cam in batched_cameras]
    bsz = len(batched_cameras)

    stage0_execution_mode = _run_stage0_schedule_interaction(
        storage_adapter=storage_adapter,
        training_schedule=training_schedule,
        iteration=iteration,
        batch_size=bsz,
        schedule_ordering=getattr(args, 'ssd_schedule_ordering', 'trajectory'),
    )

    # ============================================================================
    # STAGE 1: SETUP & PREPROCESSING
    # ============================================================================

    
    # ========================================================================
    # Get total Gaussians count from a paper-safe source.
    # ========================================================================
    # In paper out-of-core mode the full tensors may already be released before
    # Stage 1.5 materializes the current resident set, so we must not assume
    # gaussians._xyz is available here.
    total_n_gaussians = get_total_gaussians_count(gaussians, storage_adapter)

    # Before resident-set materialization, any coarse-grained bookkeeping that
    # still expects "N" should use the full-scene count rather than dereference
    # the released in-memory tensors.
    n_gaussians = total_n_gaussians
    
    # ========================================================================
    # Save original parameter references before the resident-set replacement.
    # ========================================================================
    # These will be used to restore parameters after batch completion
    # MUST be saved BEFORE STAGE 1.5 replaces gaussians._xyz with GPU working set!
    if storage_adapter is not None and gaussians.use_gpu_features:
        original_xyz = gaussians._xyz
        original_scaling = gaussians._scaling
        original_rotation = gaussians._rotation
        original_opacity = gaussians._opacity
        original_features_dc = gaussians._features_dc
        original_features_rest = gaussians._features_rest
    else:
        original_xyz = None
        original_scaling = None
        original_rotation = None
        original_opacity = None
        original_features_dc = None
        original_features_rest = None

    # Paper SSD mode uses compact local->global metadata. Archived paths may
    # still populate loaded_gaussian_mask, but the paper path must not allocate
    # an O(total_n_gaussians) CUDA mask.
    loaded_gaussian_mask = None

    ssd_execution_mode = getattr(storage_adapter, "execution_mode", "fast_ram") if storage_adapter is not None else "fast_ram"
    has_unified_params = hasattr(gaussians, "_unified_params") and gaussians._unified_params is not None
    use_fast_ram_ssd_path = storage_adapter is not None and has_unified_params and ssd_execution_mode == "fast_ram"
    is_paper_ssd_mode = storage_adapter is not None and ssd_execution_mode == "paper"
    paper_debug_logging = _paper_debug_logging_enabled(args, is_paper_ssd_mode)
    paper_optimizer_deferred_mode, paper_optimizer_backend = _initialize_paper_mode_runtime_state(
        gaussians=gaussians,
        args=args,
        is_paper_ssd_mode=is_paper_ssd_mode,
        iteration=iteration,
        log_file=log_file,
    )

    _ts('stage1_setup_done')

    # ============================================================================
    # STAGE 1.5: [SSD HOOK] Block-level Culling & Load from SSD
    # ============================================================================
    if storage_adapter is not None:
        torch.cuda.nvtx.range_push("SSD: block-level culling and loading")

        _log_paper_batch_debug(
            enabled=paper_debug_logging,
            iteration=iteration,
            batched_cameras=batched_cameras,
            current_camera_ids=current_camera_ids,
        )

        # Step 1: collect coarse block visibility for this camera batch.
        visible_block_ids_set = set()
        current_camera_blocks: Dict[int, List[int]] = {}
        cam_to_blocks = {}  # Track per-camera block visibility
        for i, camera in enumerate(batched_cameras):
            cam_global_idx = current_camera_ids[i]
            blocks = storage_adapter.get_visible_blocks(cam_global_idx)
            visible_block_ids_set.update(blocks)
            current_camera_blocks[int(cam_global_idx)] = list(blocks)
            cam_to_blocks[cam_global_idx] = len(blocks)

        visible_block_ids = sorted(list(visible_block_ids_set))
        
        _log_paper_block_visibility_debug(
            enabled=paper_debug_logging,
            iteration=iteration,
            cam_to_blocks=cam_to_blocks,
            visible_block_ids=visible_block_ids,
        )

        # ====================================================================
        # Diagnostic: empty block-level visibility is a geometry/camera issue.
        # ====================================================================
        if len(visible_block_ids) == 0:
            log_file.write(
                f"\n[CRITICAL] Iter {iteration}: NO blocks visible after block-level culling!\n"
                f"  Batch size: {len(batched_cameras)}\n"
                f"  Camera IDs: {current_camera_ids}\n"
                f"  Total blocks in scene: {storage_adapter.culler.num_blocks}\n"
                f"  This suggests either:\n"
                f"    1. Far plane too small\n"
                f"    2. Cameras are far outside scene bounds\n"
                f"    3. Block bounds computed incorrectly\n"
            )
            
            # Include camera positions to diagnose dataset/culling mismatches.
            for i, cam in enumerate(batched_cameras):
                R = np.array(cam.R).reshape(3, 3)
                T = np.array(cam.T).reshape(3, 1)
                cam_pos = (-R.T @ T).flatten()
                log_file.write(f"  Camera {current_camera_ids[i]}: pos={cam_pos}\n")

        # ====================================================================
        # Optional frustum visualization is disabled in the release package.
        # ====================================================================
        if args.debug_frustum and (iteration == 1 or iteration % 500 == 0):
            block_bounds_for_viz = storage_adapter.block_bounds
            visualize_frustum_culling_inline(
                batched_cameras=batched_cameras,
                block_bounds=block_bounds_for_viz,
                visible_block_ids=visible_block_ids,
                iteration=iteration,
                save_dir=os.path.join(args.model_path, 'debug_frustum'),
            )

        # Lightweight visibility summary for python.log.
        if iteration == 1 or iteration % 100 == 0:
            total_blocks = storage_adapter.culler.num_blocks
            vis_ratio = 100.0 * len(visible_block_ids) / total_blocks if total_blocks > 0 else 0.0
            log_file.write(
                f"[SSD] Iter {iteration}: {len(visible_block_ids)}/{total_blocks} blocks visible ({vis_ratio:.1f}%)\n"
            )

        paper_block_sets = None
        paper_ab_buffer_source = None
        paper_delta_future_submitted_early = 0
        should_log_paper_sets = is_paper_ssd_mode and (iteration == 1 or _perf_log or iteration % 500 == 0)
        resident_recency_scores = dict(
            getattr(gaussians, '_paper_resident_recency_scores', {})
        ) if is_paper_ssd_mode else {}

        # Step 2: 获取数据 — execution-mode dependent
        # fast_ram: read cold blocks directly from _unified_params pinned memory
        # paper: wait for the SSD->RAM async pipeline and consume RAM cache blocks
        if use_fast_ram_ssd_path:
            # ================================================================
            # FAST PATH: Skip SSD cache entirely — pass unified_params reference
            # to load_visible_blocks_with_retention which reads directly.
            # ================================================================
            active_blocks_ram = None

            try:
                storage_adapter.pipeline.completed_queue.get_nowait()
            except Exception:
                pass
        elif is_paper_ssd_mode:
            active_blocks_ram, _ = _resolve_paper_stage0_active_blocks(
                is_paper_ssd_mode=is_paper_ssd_mode,
                iteration=iteration,
                should_log_paper_sets=should_log_paper_sets,
                log_file=log_file,
            )
        else:
            prefetch_timeout_s = 180.0 if iteration == 1 else 30.0
            if iteration == 1:
                utils.print_rank_0(
                    f"[SSD PREFETCH] Iter 1 cold-start timeout relaxed to {prefetch_timeout_s:.1f}s"
                )
            active_blocks_ram = storage_adapter.wait_and_load_blocks(
                iteration, timeout=prefetch_timeout_s
            )
        # ====================================================================
        # [DIAGNOSTIC] Check mismatch between visible_block_ids and active_blocks_ram
        # ====================================================================
        if active_blocks_ram is not None:
            missing_blocks = [bid for bid in visible_block_ids if bid not in active_blocks_ram]
            if len(missing_blocks) > 0:
                log_file.write(
                    f"\n[SSD I/O WARNING] Iter {iteration}: {len(missing_blocks)}/{len(visible_block_ids)} "
                    f"blocks NOT loaded from SSD!\n"
                    f"  Missing blocks (first 20): {missing_blocks[:20]}\n"
                    f"  This indicates SSD I/O bottleneck or prefetch failure.\n"
                    f"  Visible blocks requested: {len(visible_block_ids)}\n"
                    f"  Blocks actually in RAM: {len(active_blocks_ram)}\n"
                )
                
                # Check if ALL blocks are missing (severe I/O failure)
                if len(missing_blocks) == len(visible_block_ids):
                    log_file.write(
                        f"[SEVERE] ALL blocks missing! This will cause empty filters.\n"
                        f"  Possible causes:\n"
                        f"    1. Prefetch never started for iteration {iteration}\n"
                        f"    2. Cache evicted all blocks before they could be loaded\n"
                        f"    3. SSD read timeout or hardware issue\n"
                    )

        # Validate block IDs against the full-scene count, not the resident count.
        max_valid_block_id = (total_n_gaussians + args.gaussian_block_size - 1) // args.gaussian_block_size - 1
        block_ids_to_check = active_blocks_ram.keys() if active_blocks_ram is not None else visible_block_ids
        invalid_blocks = [bid for bid in block_ids_to_check if bid > max_valid_block_id or bid < 0]

        if invalid_blocks:
            raise ValueError(
                f"[SSD ERROR] Invalid block IDs found: {invalid_blocks}\n"
                f"Valid range: [0, {max_valid_block_id}]\n"
                f"total_n_gaussians={total_n_gaussians:,}, block_size={args.gaussian_block_size}\n"
                f"Total valid blocks: {max_valid_block_id + 1}"
            )


        _log_ssd_stage1_debug(
            enabled=paper_debug_logging,
            log_file=log_file,
            iteration=iteration,
            ssd_execution_mode=ssd_execution_mode,
            n_gaussians=n_gaussians,
            block_size=args.gaussian_block_size,
            max_valid_block_id=max_valid_block_id,
            active_blocks_ram=active_blocks_ram,
            is_paper_ssd_mode=is_paper_ssd_mode,
            visible_block_ids=visible_block_ids,
        )

        # Step 3: materialize the resident block set on GPU.
        with torch.no_grad():
            if gaussians.use_gpu_features and hasattr(gaussians, 'gpu_working_set_manager'):
                # ================================================================
                # Load the resident working set.
                # Data flow: SSD → RAM → GPU.
                # ================================================================
                torch.cuda.nvtx.range_push("SSD→RAM→GPU: Load resident blocks")
                
                # ============================================================
                # Archived non-paper retention path. Paper mode uses Tide
                # resident-set selection and A/B differential streaming.
                # ============================================================
                enable_retention_flag = getattr(args, 'enable_hotspot_retention', True)
                enable_retention = enable_retention_flag and (iteration > 1) and not is_paper_ssd_mode
                
                # ============================================================
                # Validate that the selected blocks are available before GPU materialization.
                # ============================================================
                if len(visible_block_ids) == 0:
                    log_file.write(
                        f"\n[CRITICAL ERROR] Iter {iteration}: visible_block_ids is EMPTY!\n"
                        f"  This means block-level frustum culling found NO visible blocks.\n"
                        f"  Check:\n"
                        f"    1. Far plane setting\n"
                        f"    2. Camera positions vs scene bounds\n"
                        f"    3. Block bounds computation\n"
                    )
                    # Continue to the RAM-cache checks below before deciding whether to skip.
                
                # Only check active_blocks_ram if NOT using unified_params fast path
                if active_blocks_ram is not None:
                    if len(active_blocks_ram) == 0:
                        log_file.write(
                            f"\n[CRITICAL ERROR] Iter {iteration}: active_blocks_ram is EMPTY!\n"
                            f"  This means SSD prefetch completely failed.\n"
                        )
                    
                    # Check if any visible blocks are actually in RAM
                    available_blocks = [bid for bid in visible_block_ids if bid in active_blocks_ram]
                    if len(available_blocks) == 0:
                        log_file.write(
                            f"\n[CRITICAL ERROR] Iter {iteration}: NO overlap between visible_block_ids and active_blocks_ram!\n"
                            f"  Visible blocks requested: {len(visible_block_ids)}\n"
                            f"  Blocks in RAM cache: {len(active_blocks_ram)}\n"
                        )
                        log_file.write(f"[SKIP ITERATION] Skipping iteration {iteration} due to missing data\n")
                        torch.cuda.nvtx.range_pop()
                        return [], list(range(len(batched_cameras))), 0.0
                
                if is_paper_ssd_mode:
                    gpu_tensors, retention_stats, used_paper_prefetch_buffer, paper_ab_buffer_source = _load_paper_stage1_working_set(
                        gaussians=gaussians,
                        args=args,
                        iteration=iteration,
                        total_n_gaussians=total_n_gaussians,
                        visible_block_ids=visible_block_ids,
                        active_blocks_ram=active_blocks_ram,
                        current_camera_blocks=current_camera_blocks,
                        enable_retention=enable_retention,
                        use_fast_ram_ssd_path=use_fast_ram_ssd_path,
                        training_schedule=training_schedule,
                        storage_adapter=storage_adapter,
                        should_log=should_log_paper_sets,
                        get_double_buffer_gpu_fn=get_double_buffer_gpu,
                        ensure_local_to_global_mapping_fn=ensure_local_to_global_mapping,
                        resolve_current_iteration_resident_blocks_fn=_resolve_current_iteration_resident_blocks,
                        log_file=log_file,
                    )
                else:
                    used_paper_prefetch_buffer = False
                    paper_ab_buffer_source = None
                    active_block_reader = getattr(gaussians, '_block_reader', None)
                    gpu_tensors, retention_stats = gaussians.gpu_working_set_manager.load_visible_blocks_with_retention(
                        visible_block_ids=visible_block_ids,
                        active_blocks_ram=active_blocks_ram,
                        enable_retention=enable_retention,
                        unified_params=gaussians._unified_params if (active_block_reader is None and use_fast_ram_ssd_path) else None,
                        block_reader=active_block_reader,
                    )
                
                # Create nn.Parameters for training (gradients will accumulate here)
                gaussians._xyz = nn.Parameter(gpu_tensors['xyz'].requires_grad_(True))
                gaussians._scaling = nn.Parameter(gpu_tensors['scaling'].requires_grad_(True))
                gaussians._rotation = nn.Parameter(gpu_tensors['rotation'].requires_grad_(True))
                gaussians._opacity = nn.Parameter(gpu_tensors['opacity'].requires_grad_(True))
                gaussians._features_dc = nn.Parameter(gpu_tensors['features_dc'].requires_grad_(True))
                gaussians._features_rest = nn.Parameter(gpu_tensors['features_rest'].requires_grad_(True))
                
                # Update gpu_working_set_manager's GPU tensors (for update_ram_cache)
                gaussians.gpu_working_set_manager.gpu_xyz = gaussians._xyz
                gaussians.gpu_working_set_manager.gpu_scaling = gaussians._scaling
                gaussians.gpu_working_set_manager.gpu_rotation = gaussians._rotation
                gaussians.gpu_working_set_manager.gpu_opacity = gaussians._opacity
                gaussians.gpu_working_set_manager.gpu_features_dc = gaussians._features_dc
                gaussians.gpu_working_set_manager.gpu_features_rest = gaussians._features_rest

                if is_paper_ssd_mode:
                    double_buffer = get_double_buffer_gpu(
                        num_total=total_n_gaussians,
                        block_size=args.gaussian_block_size,
                        device='cuda'
                    )
                    _seed_paper_active_buffer_from_manager(
                        gaussians,
                        double_buffer,
                        ensure_local_to_global_mapping,
                        preserve_resident_metadata=used_paper_prefetch_buffer,
                    )
                    actual_current_resident_blocks = list(gaussians.gpu_working_set_manager.loaded_blocks)
                    paper_block_sets = _compute_paper_block_sets(
                        storage_adapter=storage_adapter,
                        training_schedule=training_schedule,
                        iteration=iteration,
                        batch_size=bsz,
                        current_block_ids=visible_block_ids,
                        schedule_ordering=getattr(args, 'ssd_schedule_ordering', 'trajectory'),
                        current_resident_blocks=actual_current_resident_blocks,
                        current_resident_recency_scores=resident_recency_scores,
                        resident_selection_policy=args.paper_resident_selection_policy,
                        resident_lambda=args.paper_resident_lambda,
                        resident_recency_decay=args.paper_resident_recency_decay,
                        resident_capacity_blocks=args.paper_resident_capacity_blocks,
                        balanced_seed_fraction=float(getattr(args, 'paper_balanced_seed_fraction', 1.0)),
                    )
                    gaussians._paper_loaded_resident_blocks = list(actual_current_resident_blocks)
                    gaussians._paper_expected_resident_blocks = list(
                        paper_block_sets['next_resident_blocks']
                    )
                    gaussians._paper_resident_recency_scores = dict(
                        paper_block_sets.get('updated_recency_scores', {})
                    )
                    stream_in_for_next = list(paper_block_sets.get('stream_in_blocks', []))
                    active_block_reader = getattr(gaussians, '_block_reader', None)
                    if active_block_reader is not None and stream_in_for_next:
                        paper_delta_future_submitted_early = int(active_block_reader.hint_future(stream_in_for_next) or 0)
                        if should_log_paper_sets:
                            _log_early_delta_hint(
                                storage_adapter=storage_adapter,
                                iteration=iteration,
                                submitted=paper_delta_future_submitted_early,
                                requested=len(stream_in_for_next),
                                log_file=log_file,
                            )
                    _configure_gpu_resident_optimizer_state(
                        gaussians=gaussians,
                        args=args,
                        actual_current_resident_blocks=actual_current_resident_blocks,
                        iteration=iteration,
                        get_gpu_resident_optimizer_fn=get_gpu_resident_optimizer,
                        log_file=log_file,
                    )
                    if should_log_paper_sets:
                        _log_paper_block_sets(
                            log_file=log_file,
                            iteration=iteration,
                            current_camera_ids=current_camera_ids,
                            block_sets=paper_block_sets,
                        )
                        _log_batch_kt_metrics(
                            log_file=log_file,
                            iteration=iteration,
                            current_camera_blocks=current_camera_blocks,
                            visible_block_ids=visible_block_ids,
                            resident_blocks=actual_current_resident_blocks,
                        )
                
                # Keep compact global IDs for later bookkeeping. Do not build a
                # full-scene CUDA mask here: at 1B Gaussians a bool mask alone
                # is ~1 GB and forces an O(N) scan before projection.
                local_to_global = ensure_local_to_global_mapping(
                    gaussians,
                    total_n_gaussians,
                    log_file=log_file,
                    context=f"stage1_loaded_ids_iter_{iteration}",
                )
                loaded_gaussian_mask = None
                
                # ============================================================
                # Mode-specific logging: paper working-set semantics vs. archived retention stats.
                # ============================================================
                if is_paper_ssd_mode:
                    if should_log_paper_sets and paper_block_sets is not None:
                        load_source = paper_ab_buffer_source or retention_stats.get('source', 'sync_ram_to_gpu')
                        _log_paper_working_set_loaded(
                            iteration=iteration,
                            loaded_blocks=list(gaussians.gpu_working_set_manager.loaded_blocks),
                            paper_block_sets=paper_block_sets,
                            retention_stats=retention_stats,
                            load_source=load_source,
                            log_file=log_file,
                        )
                    _log_paper_resident_camera_coverage(
                        iteration=iteration,
                        current_camera_blocks=current_camera_blocks,
                        loaded_blocks=list(gaussians.gpu_working_set_manager.loaded_blocks),
                        should_log=should_log_paper_sets,
                        log_file=log_file,
                    )
                else:
                    log_file.write(
                        f"[ARCHIVED RETENTION] Iter {iteration}: "
                        f"Blocks={retention_stats['total_count']} "
                        f"(Retained={retention_stats['hotspot_count']}, "
                        f"Cold={retention_stats['cold_count']}, "
                        f"HitRate={retention_stats['hit_rate']*100:.1f}%)\n"
                    )
                    log_file.write(
                        f"[ARCHIVED RETENTION] GPU Memory: {retention_stats['memory_mb']:.2f} MB, "
                        f"Gaussians: {retention_stats['num_gaussians']:,}\n"
                    )

                    # Periodic summary statistics
                    if iteration % 500 == 0:
                        cumulative_stats = gaussians.gpu_working_set_manager.get_retention_stats()
                        log_file.write(
                            f"\n[ARCHIVED RETENTION SUMMARY] After {cumulative_stats['total_iterations']} iterations:\n"
                            f"  Avg Hit Rate: {cumulative_stats['avg_hit_rate']*100:.1f}%\n"
                            f"  Blocks Loaded: {cumulative_stats['total_blocks_loaded']:,}\n"
                            f"  Blocks Retained: {cumulative_stats['total_blocks_retained']:,}\n"
                            f"  Blocks from RAM: {cumulative_stats['total_blocks_cold']:,}\n"
                            f"  Bandwidth Savings: ~{cumulative_stats['bandwidth_savings_ratio']*100:.1f}%\n\n"
                        )

                torch.cuda.nvtx.range_pop()

                
            else:
                # ================================================================
                # Archived fallback: write into full GPU parameter arrays.
                # ================================================================
                log_file.write(f"[ARCHIVED FULL-GPU PATH] Writing {len(visible_block_ids)} blocks to full GPU arrays\n")
                
                # Track which Gaussians were materialized for archived full-GPU filtering.
                loaded_gaussian_mask = torch.zeros(total_n_gaussians, dtype=torch.bool, device='cuda')

                for block_id in visible_block_ids:
                    if block_id not in active_blocks_ram:
                        continue  # Skip if not prefetched

                    block_tensor = active_blocks_ram[block_id]
                    start_idx = block_id * args.gaussian_block_size
                    end_idx = min(start_idx + args.gaussian_block_size, total_n_gaussians)
                    actual_size = end_idx - start_idx

                    if block_tensor.device.type != 'cuda':
                        block_tensor = block_tensor.cuda()

                    # block_tensor layout: XYZ(3)|Scale(3)|Rot(4)|Opacity(1)|DC(3)|Rest(45)
                    gaussians._xyz[start_idx:end_idx] = block_tensor[:actual_size, 0:3]
                    gaussians._scaling[start_idx:end_idx] = block_tensor[:actual_size, 3:6]
                    gaussians._rotation[start_idx:end_idx] = block_tensor[:actual_size, 6:10]
                    gaussians._opacity[start_idx:end_idx] = block_tensor[:actual_size, 10:11]

                    # ================================================================
                    # Write SH features based on storage mode.
                    # ================================================================
                    feat_dc_src = block_tensor[:actual_size, 11:14]
                    feat_rest_src = block_tensor[:actual_size, 14:59]

                    if gaussians.use_gpu_features:
                        # SSD-backed compatibility path: write SH features directly to GPU parameters.
                        # _features_dc and _features_rest are already GPU tensors.
                        gaussians._features_dc[start_idx:end_idx].copy_(feat_dc_src)
                        gaussians._features_rest[start_idx:end_idx].copy_(feat_rest_src)
                    else:
                        # Archived split-feature path: write SH features to CPU pinned memory.
                        # _parameters is on CPU, need to transfer from GPU block_tensor
                        gaussians._parameters[start_idx:end_idx, :3].copy_(feat_dc_src.cpu())
                        gaussians._parameters[start_idx:end_idx, 3:48].copy_(feat_rest_src.cpu())

                    loaded_gaussian_mask[start_idx:end_idx] = True

        torch.cuda.nvtx.range_pop()

    _ts('stage1_5_ssd_done')

    # ============================================================================
    # STAGE 2: Gaussian-level Culling (精细剔除)
    # ============================================================================
    with torch.no_grad():
        if storage_adapter is not None:
            torch.cuda.nvtx.range_push("compact_and_calculate_filters")

            loaded_gaussian_ids = _resolve_stage2_loaded_gaussian_ids(
                gaussians=gaussians,
                loaded_gaussian_mask=loaded_gaussian_mask,
                total_n_gaussians=total_n_gaussians,
                is_paper_ssd_mode=is_paper_ssd_mode,
                iteration=iteration,
                log_file=log_file,
                ensure_local_to_global_mapping_fn=ensure_local_to_global_mapping,
            )

            num_loaded = loaded_gaussian_ids.shape[0]

            empty_loaded_result = _log_empty_stage2_loaded_gaussians(
                num_loaded=num_loaded,
                iteration=iteration,
                visible_block_ids=visible_block_ids,
                active_blocks_ram=active_blocks_ram,
                ssd_execution_mode=ssd_execution_mode,
                bsz=bsz,
                log_file=log_file,
            )
            if empty_loaded_result is not None:
                torch.cuda.nvtx.range_pop()
                return empty_loaded_result

            # ================================================================
            # Parameters are already on GPU in the compact working set.
            # After load_visible_blocks(), gaussians._xyz/etc point to GPU tensors
            # containing ONLY the visible Gaussians (not all N)
            # ================================================================
            # Use compact resident tensors directly; projection gathers per-camera filters below.
            xyz_compact = gaussians.get_xyz          # GPU tensor, shape (M,3) where M = num visible
            opacity_compact = gaussians.get_opacity  # GPU tensor, shape (M,1)
            scaling_compact = gaussians.get_scaling  # GPU tensor, shape (M,3)
            rotation_compact = gaussians.get_rotation  # GPU tensor, shape (M,4)

            _log_paper_gpu_working_set_parameters(
                iteration=iteration,
                xyz_compact=xyz_compact,
                opacity_compact=opacity_compact,
                scaling_compact=scaling_compact,
                rotation_compact=rotation_compact,
                batched_cameras=batched_cameras,
                log_file=log_file,
            )
            
            # Sanity check
            assert xyz_compact.is_cuda, "[PAPER MODE] xyz should be on GPU in working set!"
            assert opacity_compact.is_cuda, "[PAPER MODE] opacity should be on GPU!"
            assert scaling_compact.is_cuda, "[PAPER MODE] scaling should be on GPU!"
            assert rotation_compact.is_cuda, "[PAPER MODE] rotation should be on GPU!"
            # ==========================================================

            # Run Gaussian-level culling on the compact GPU working set.
            # Input: Visible Gaussians from blocks (M Gaussians)
            # Output: Subset of M that passes per-camera culling
            # Optional debug-frustum diagnostics before calculate_filters.
            if iteration == 1 and getattr(args, "debug_frustum", False):
                print(f"\n[PRE-PROJECTION DEBUG] Gaussians before calculate_filters:")
                print(f"  Number of Gaussians: {xyz_compact.shape[0]}")
                print(f"  XYZ range: min={xyz_compact.min(dim=0).values.cpu().numpy()}, "
                      f"max={xyz_compact.max(dim=0).values.cpu().numpy()}")
                print(f"  Opacity range: min={opacity_compact.min().item():.4f}, "
                      f"max={opacity_compact.max().item():.4f}, "
                      f"mean={opacity_compact.mean().item():.4f}")
                print(f"  Scaling range: min={scaling_compact.min().item():.4f}, "
                      f"max={scaling_compact.max().item():.4f}, "
                      f"mean={scaling_compact.mean().item():.4f}")
                
                # Check how many have valid opacity
                valid_opacity = (opacity_compact > 0.01).sum().item()
                print(f"  Gaussians with opacity > 0.01: {valid_opacity}/{xyz_compact.shape[0]}")
                
                # Check camera positions relative to Gaussians
                cam0_pos = batched_cameras[0].camera_center.cpu().numpy()
                print(f"  Camera[0] position: {cam0_pos}")
                
                # Check if cameras are inside/outside Gaussian bounds
                xyz_min = xyz_compact.min(dim=0).values.cpu().numpy()
                xyz_max = xyz_compact.max(dim=0).values.cpu().numpy()
                inside = (cam0_pos >= xyz_min).all() and (cam0_pos <= xyz_max).all()
                print(f"  Camera[0] inside Gaussian bounds: {inside}")
            
            filters_compact, _, _ = calculate_filters(
                batched_cameras,
                xyz_compact,
                opacity_compact,
                scaling_compact,
                rotation_compact,
            )
            
            # ====================================================================
            # Optional debug-frustum diagnostics after calculate_filters.
            if iteration == 1 and getattr(args, "debug_frustum", False):
                num_visible_per_cam = [len(f) for f in filters_compact]
                print(f"\n[POST-PROJECTION DEBUG] After calculate_filters:")
                print(f"  Gaussians per camera (first 10): {num_visible_per_cam[:10]}")
                print(f"  Total visible entries: {sum(num_visible_per_cam)} / {xyz_compact.shape[0]} unique Gaussians")
                
                # Check for duplicates in ALL cameras with visible Gaussians
                print(f"\n  Per-camera duplicate analysis:")
                for cam_idx in range(min(10, len(filters_compact))):  # Check first 10 cameras
                    if num_visible_per_cam[cam_idx] > 0:
                        unique_count = filters_compact[cam_idx].unique().numel()
                        dup_ratio = num_visible_per_cam[cam_idx] / unique_count
                        print(f"    Camera {cam_idx}: {num_visible_per_cam[cam_idx]} entries, {unique_count} unique → {dup_ratio:.2f}x duplication")
                    else:
                        print(f"    Camera {cam_idx}: 0 Gaussians visible")
                
                # Count cameras with/without Gaussians
                empty_cams = sum(1 for n in num_visible_per_cam if n == 0)
                non_empty_cams = len(num_visible_per_cam) - empty_cams
                
                print(f"\n  Summary:")
                print(f"    Cameras with Gaussians: {non_empty_cams}/{len(num_visible_per_cam)}")
                print(f"    Cameras without Gaussians: {empty_cams}/{len(num_visible_per_cam)}")
                
                if empty_cams > 0:
                    print(f"\n  ⚠️  {empty_cams}/{len(batched_cameras)} cameras see ZERO Gaussians after projection!")
                    print(f"  Possible causes:")
                    print(f"    1. Opacity too low (check mean opacity above)")
                    print(f"    2. Scale too small (projected radius < radius_clip={utils.get_args().radius_clip})")
                    print(f"    3. Depth culling (Gaussians outside near/far planes)")
                    print(f"    4. Camera-Gaussian distance too large")

            # ================================================================
            # Map compact filter indices to global IDs.
            # ================================================================
            # filters_compact contains LOCAL indices (0 to M-1) for GPU working set
            # filters_global contains GLOBAL indices (0 to N-1) for gradient accumulation
            # 
            # We need both index spaces:
            # - filters_local: for indexing gaussians._xyz (GPU working set, size M)
            # - filters_global: for gradient accumulation to full parameter tensors (size N)
            local_to_global = ensure_local_to_global_mapping(
                gaussians,
                total_n_gaussians,
                log_file=log_file,
                context=f"projection_filter_map_iter_{iteration}",
            )
            
            filters_local, filters_global = _map_compact_filters_to_global(
                filters_compact=filters_compact,
                local_to_global=local_to_global,
            )
            
            # Store both for later use
            # filters_local: used in STAGE 4 for gaussians._xyz[this_filter_local]
            # filters_global: used for gradient scatter_add_ to full tensors
            gaussians.gpu_working_set_manager.filters_local = filters_local
            filters = filters_global
            _log_empty_paper_projection_cameras(
                is_paper_ssd_mode=is_paper_ssd_mode,
                iteration=iteration,
                current_camera_ids=current_camera_ids,
                filters_global=filters_global,
                log_file=log_file,
            )

            camera_ids = None
            gaussian_ids = None

            # Cleanup - but keep filters_compact info in filters_local
            del xyz_compact, opacity_compact, scaling_compact, rotation_compact, filters_compact

            torch.cuda.nvtx.range_pop()
        else:
            # Archived split-feature path uses full-tensor Gaussian culling.
            xyz_gpu = gaussians.get_xyz
            opacity_gpu_origin = gaussians.get_opacity
            scaling_gpu_origin = gaussians.get_scaling
            rotation_gpu_origin = gaussians.get_rotation

            torch.cuda.nvtx.range_push("calculate_filters")
            # Filters: list of indices indicating which gaussians are visible per camera
            filters, camera_ids, gaussian_ids = calculate_filters(
                batched_cameras,
                xyz_gpu,
                opacity_gpu_origin,
                scaling_gpu_origin,
                rotation_gpu_origin,
            )
            del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
            torch.cuda.nvtx.range_pop()
    
    # ========================================================================
    # [STATISTICS] Block Visibility Analysis (controlled by --log_block_visibility)
    # ========================================================================
    if args.log_block_visibility and storage_adapter is not None:
        with torch.no_grad():
            # Calculate total loaded Gaussians and visible Gaussians
            if hasattr(gaussians, 'gpu_working_set_manager') and gaussians.gpu_working_set_manager.local_to_global_idx is not None:
                loaded_count = len(gaussians.gpu_working_set_manager.local_to_global_idx)
            elif loaded_gaussian_mask is not None:
                loaded_count = loaded_gaussian_mask.sum().item()
            else:
                loaded_count = total_n_gaussians
            
            # Count visible Gaussians across all cameras
            visible_count = sum(len(f) for f in filters if f is not None)
            
            # Calculate per-block visibility
            if loaded_count > 0:
                visibility_ratio = visible_count / loaded_count
                
                # Detailed per-block analysis (expensive, only do every N iterations)
                if iteration % 100 == 0:
                    # Get all visible Gaussian IDs
                    all_visible_ids = []
                    for f in filters:
                        if f is not None and f.numel() > 0:
                            all_visible_ids.append(f)
                    
                    if len(all_visible_ids) > 0:
                        all_visible_ids = torch.cat(all_visible_ids).unique()
                        
                        # Calculate which blocks have visible Gaussians
                        visible_block_ids_set = set((all_visible_ids // args.gaussian_block_size).cpu().tolist())
                        
                        # Get loaded blocks
                        if hasattr(gaussians, 'gpu_working_set_manager'):
                            loaded_block_ids_set = set(gaussians.gpu_working_set_manager.loaded_blocks)
                        else:
                            # Estimate from visible_block_ids used earlier
                            loaded_block_ids_set = visible_block_ids_set
                        
                        # Calculate per-block statistics
                        block_visibility_stats = {}
                        for block_id in loaded_block_ids_set:
                            start_idx = block_id * args.gaussian_block_size
                            end_idx = min(start_idx + args.gaussian_block_size, total_n_gaussians)
                            block_size = end_idx - start_idx
                            
                            # Count visible Gaussians in this block
                            block_visible = ((all_visible_ids >= start_idx) & (all_visible_ids < end_idx)).sum().item()
                            block_ratio = block_visible / block_size if block_size > 0 else 0
                            
                            block_visibility_stats[block_id] = {
                                'total': block_size,
                                'visible': block_visible,
                                'ratio': block_ratio
                            }
                        
                        # Aggregate statistics
                        if block_visibility_stats:
                            ratios = [s['ratio'] for s in block_visibility_stats.values()]
                            avg_ratio = sum(ratios) / len(ratios)
                            min_ratio = min(ratios)
                            max_ratio = max(ratios)
                            
                            # Categorize blocks
                            high_visibility = sum(1 for r in ratios if r > 0.7)
                            medium_visibility = sum(1 for r in ratios if 0.3 <= r <= 0.7)
                            low_visibility = sum(1 for r in ratios if r < 0.3)
                            
                            log_file.write(
                                f"\n[BLOCK VISIBILITY STATS] Iter {iteration}:\n"
                                f"  Total Loaded Gaussians: {loaded_count:,}\n"
                                f"  Total Visible Gaussians: {visible_count:,}\n"
                                f"  Overall Visibility Ratio: {visibility_ratio*100:.1f}%\n"
                                f"  \n"
                                f"  Loaded Blocks: {len(loaded_block_ids_set)}\n"
                                f"  Blocks with Visible Gaussians: {len(visible_block_ids_set)}\n"
                                f"  \n"
                                f"  Per-Block Visibility:\n"
                                f"    Average: {avg_ratio*100:.1f}%\n"
                                f"    Min: {min_ratio*100:.1f}%\n"
                                f"    Max: {max_ratio*100:.1f}%\n"
                                f"  \n"
                                f"  Block Categories:\n"
                                f"    High (>70%): {high_visibility} blocks\n"
                                f"    Medium (30-70%): {medium_visibility} blocks\n"
                                f"    Low (<30%): {low_visibility} blocks\n"
                                f"  \n"
                            )
                            
                            # Print to console for quick monitoring
                            print(f"[Iter {iteration}] Block Visibility: Overall={visibility_ratio*100:.1f}%, "
                                  f"PerBlock={avg_ratio*100:.1f}% (High={high_visibility}, Med={medium_visibility}, Low={low_visibility})")
                else:
                    # Quick stats every iteration (lightweight)
                    log_file.write(
                        f"[BLOCK VISIBILITY] Iter {iteration}: "
                        f"Loaded={loaded_count:,}, Visible={visible_count:,}, "
                        f"Ratio={visibility_ratio*100:.1f}%\n"
                    )
    
    # ====================================================================
    # ====================================================================
    # Original parameter references were saved in Stage 1.
    # ====================================================================
    # NOTE: original_xyz, original_scaling, etc. were saved BEFORE STAGE 1.5
    # to capture the CPU parameter references before they were replaced with GPU working set.
    # Non-paper paths keep these references for archived full writeback.
    #
    # Paper SSD mode intentionally leaves these references as None after
    # _unified_params has been released.  Filling them with the compact GPU
    # working set would make later block writeback slice a local tensor with
    # global block offsets, which creates empty/partial dirty blocks.
    if original_xyz is None and not (is_paper_ssd_mode and gaussians.use_gpu_features):
        original_xyz = gaussians._xyz
        original_scaling = gaussians._scaling
        original_rotation = gaussians._rotation
        original_opacity = gaussians._opacity
        original_features_dc = gaussians._features_dc
        original_features_rest = gaussians._features_rest
    
    # Prepare archived camera reordering and visibility bookkeeping.
    # ========================================================================
    # Sort cameras to maximize gaussian overlap between consecutive frames (better caching)
    torch.cuda.nvtx.range_push("sort cameras")
    build_legacy_update_lists = not (is_paper_ssd_mode and gaussians.use_gpu_features)
    build_visibility_mask = not (is_paper_ssd_mode and gaussians.use_gpu_features)
    if is_paper_ssd_mode and gaussians.use_gpu_features:
        finish_indices_filters = None
        ordered_cams = list(range(len(batched_cameras)))
        sparsity = [len(f) / float(total_n_gaussians) for f in filters]
        cnt_h = None
        cnt_d = None
        cnt_g = None
        visibility_mask = None
        if should_log_paper_sets:
            _log_paper_order_calculation_skip(iteration=iteration, log_file=log_file)
    else:
        (
            finish_indices_filters,
            batched_cameras,
            filters,
            sparsity,
            ordered_cams,
            cnt_h,
            cnt_d,
            cnt_g,
            visibility_mask,
        ) = order_calculation(
            filters,
            batched_cameras,
            total_n_gaussians,
            bsz,
            perm_generator,
            args,
            build_legacy_update_lists=build_legacy_update_lists,
            build_visibility_mask=build_visibility_mask,
        )
    # cnt_h: count of parameters to HOST (load from CPU)
    # cnt_d: count of parameters to DUPLICATE (retain from previous)
    # cnt_g: count of parameters to GARBAGE (offload to CPU)
    torch.cuda.nvtx.range_pop()
    
    # ========================================================================
    # Reorder filters_local to match reordered filters.
    # ========================================================================
    # order_calculation reorders filters and batched_cameras using ordered_cams
    # We MUST apply the same reordering to filters_local!
    # Otherwise: filters[micro_idx] and filters_local[micro_idx] won't correspond
    if ordered_cams != list(range(len(ordered_cams))) and \
       storage_adapter is not None and hasattr(gaussians, 'gpu_working_set_manager') and \
       gaussians.gpu_working_set_manager.filters_local is not None:
        filters_local_original = gaussians.gpu_working_set_manager.filters_local
        filters_local_reordered = [filters_local_original[i] for i in ordered_cams]
        gaussians.gpu_working_set_manager.filters_local = filters_local_reordered
        _log_filters_local_reordered(
            enabled=is_paper_ssd_mode,
            log_file=log_file,
        )

    # ============================================================================
    # [N+1 PREFETCH] Start async prefetch for NEXT iteration while GPU computes
    # ============================================================================
    # This enables true pipeline parallelism:
    # - GPU Compute Stream: Forward/Backward on current batch
    # - GPU Prefetch Stream: Load next batch's data in background
    if storage_adapter is not None and training_schedule is not None and (use_fast_ram_ssd_path or is_paper_ssd_mode):
        torch.cuda.nvtx.range_push("N+1 Prefetch: Start async load")

        try:
            if use_fast_ram_ssd_path:
                next_camera_ids, next_visible_blocks = get_next_iteration_blocks(
                    storage_adapter=storage_adapter,
                    training_schedule=training_schedule,
                    iteration=iteration,
                    batch_size=bsz,
                    schedule_ordering=getattr(args, 'ssd_schedule_ordering', 'trajectory'),
                )

                if len(next_visible_blocks) > 0:
                    double_buffer = get_double_buffer_gpu(
                        num_total=total_n_gaussians,
                        block_size=args.gaussian_block_size,
                        device='cuda'
                    )

                    active_block_reader = getattr(gaussians, '_block_reader', None)
                    double_buffer.start_prefetch(
                        iteration=iteration + bsz,
                        visible_block_ids=next_visible_blocks,
                        filters_global=filters,
                        ram_cache={},
                        unified_params=gaussians._unified_params if active_block_reader is None else None,
                        block_reader=active_block_reader,
                    )

                    if iteration % 500 == 0:
                        log_file.write(f"[N+1 PREFETCH] Started async prefetch for iter {iteration + bsz}\n")
                        log_file.write(f"  Next cameras: {next_camera_ids[:3]}...\n")
                        log_file.write(f"  Next blocks: {len(next_visible_blocks)} blocks\n")
            elif is_paper_ssd_mode and paper_block_sets is not None:
                next_camera_ids = paper_block_sets['next_camera_ids']
                next_active_blocks = paper_block_sets['next_blocks']
                next_resident_blocks = paper_block_sets['next_resident_blocks']
                stream_in_blocks = paper_block_sets['stream_in_blocks']

                if len(next_resident_blocks) > 0:
                    double_buffer = get_double_buffer_gpu(
                        num_total=total_n_gaussians,
                        block_size=args.gaussian_block_size,
                        device='cuda'
                    )
                    active_block_reader = getattr(gaussians, '_block_reader', None)
                    if active_block_reader is not None:
                        future_submitted_late = int(active_block_reader.hint_future(stream_in_blocks) or 0)
                        future_submitted = int(paper_delta_future_submitted_early) + future_submitted_late
                        next_blocks_ram = None
                    else:
                        future_submitted_late = 0
                        future_submitted = 0
                        next_blocks_ram = storage_adapter.cache.prefetch(next_resident_blocks)
                    double_buffer.start_prefetch(
                        iteration=iteration + bsz,
                        visible_block_ids=next_resident_blocks,
                        filters_global=[],
                        ram_cache={},
                        block_cache=next_blocks_ram if active_block_reader is None else None,
                        resident_block_ids=paper_block_sets['keep_resident_blocks'],
                        evicted_block_ids=paper_block_sets['evict_blocks'],
                        allow_resident_copy=True,
                        block_reader=active_block_reader,
                    )
                    if should_log_paper_sets:
                        _log_delta_stream_prefetch(
                            iteration=iteration,
                            paper_block_sets=paper_block_sets,
                            future_submitted=future_submitted,
                            future_submitted_late=future_submitted_late,
                            storage_adapter=storage_adapter,
                            log_file=log_file,
                        )
                        _log_paper_warm_layer_metrics(storage_adapter, iteration, 'prefetch', log_file=log_file)
        except Exception as e:
            if is_paper_ssd_mode:
                _log_ab_buffer_prefetch_failure(
                    iteration=iteration,
                    error=e,
                    log_file=log_file,
                )
            else:
                log_file.write(f"[N+1 PREFETCH] Warning: Prefetch failed: {e}\n")

        torch.cuda.nvtx.range_pop()

    # ============================================================================
    # STAGE 2: MICRO-BATCH SIGNAL INITIALIZATION
    # ============================================================================
    if not hasattr(gaussians, "signal_tensor_pinned"):
        gaussians.signal_tensor_pinned = torch.zeros(
            bsz, dtype=torch.int32, device="cpu", pin_memory=True
        )
    else:
        gaussians.signal_tensor_pinned.zero_()
    signal_tensor_pinned = gaussians.signal_tensor_pinned
    torch.cuda.synchronize()

    microbatch_idx = 0

    # ========================================================================
    # Get local filter indices for GPU working set indexing.
    # ========================================================================
    if storage_adapter is not None and hasattr(gaussians, 'gpu_working_set_manager'):
        filters_local = gaussians.gpu_working_set_manager.filters_local
        use_local_filters = True
    else:
        filters_local = None
        use_local_filters = False

    # ============================================================================
    # STAGE 3: TRAINING STATE INITIALIZATION
    # ============================================================================
    # In paper SSD mode, allocating .grad for the entire GPU working set can still
    # be too large at 1.1B scale. We therefore accumulate only the Gaussians that
    # are actually touched by this batch and snapshot only that sparse subset.
    use_sparse_gpu_grad_accum = (
        is_paper_ssd_mode and gaussians.use_gpu_features and use_local_filters and filters_local is not None
    )
    sparse_grad_local_ids = None
    sparse_grad_components = None
    sparse_visibility_indices = None

    if use_sparse_gpu_grad_accum:
        with torch.cuda.stream(comm_stream), torch.no_grad():
            touched_local_mask = torch.zeros((gaussians._xyz.shape[0],), dtype=torch.bool, device='cuda')
            for filt_local in filters_local:
                if filt_local is not None and filt_local.numel() > 0:
                    touched_local_mask.scatter_(0, filt_local, True)
            sparse_grad_local_ids = torch.nonzero(touched_local_mask, as_tuple=False).flatten()
            touched_count = int(sparse_grad_local_ids.numel())
            local_to_global = ensure_local_to_global_mapping(
                gaussians,
                total_n_gaussians,
                log_file=log_file,
                context=f"sparse_grad_accum_iter_{iteration}",
            )
            sparse_visibility_indices = local_to_global[sparse_grad_local_ids].cpu()
            sparse_grad_components = {
                'xyz': torch.zeros((touched_count, 3), device='cuda'),
                'opacity': torch.zeros((touched_count, 1), device='cuda'),
                'scaling': torch.zeros((touched_count, 3), device='cuda'),
                'rotation': torch.zeros((touched_count, 4), device='cuda'),
                'features_dc': torch.zeros((touched_count, 3), device='cuda'),
                'features_rest': torch.zeros((touched_count, 45), device='cuda'),
            }
            del touched_local_mask
        utils.print_rank_0(
            f"[SSD GRAD] Touched-only GPU grad buffers: {touched_count}/{gaussians._xyz.shape[0]} working-set Gaussians"
        )
    else:
        gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
        gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
        gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
        gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)

        if gaussians.use_gpu_features:
            gaussians._features_dc.grad = torch.zeros_like(gaussians._features_dc)
            gaussians._features_rest.grad = torch.zeros_like(gaussians._features_rest)

    # Stream management: default_stream for compute, comm_stream for CPU<->GPU transfers
    default_stream = torch.cuda.current_stream()

    # Training loop variables
    num_micro_batches = len(batched_cameras)
    N = gaussians._xyz.shape[0]
    losses = []
    shs_retents = [None for i in range(num_micro_batches)]  # Retained SH coefficients

    # Kernel launch parameters
    grid_size, block_size = args.grid_size_H, 256
    grid_size_D, block_size_D = args.grid_size_D, 256

    # Initialize retention tracking buffers
    with torch.cuda.stream(comm_stream), torch.no_grad():
        this_bit = torch.zeros(
            (N,), dtype=torch.uint8, device="cuda"
        )  # Current iteration visibility
        next_bit = torch.zeros(
            (N,), dtype=torch.uint8, device="cuda"
        )  # Next iteration visibility
        retention_vec = torch.empty(
            (N,), dtype=torch.int32, device="cuda"
        )  # Index mapping

        shs_grad = torch.zeros(
            filters[0].shape[0], 48, device="cuda"
        )  # SH gradient buffer
        shs_grad_init_event = torch.cuda.Event()
        shs_grad_init_event.record(comm_stream)

    # ============================================================================
    _ts('stage2_3_culling_done')

    # STAGE 4: MAIN MICRO-BATCH TRAINING LOOP
    # ============================================================================
    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))
        
        # ====================================================================
        # Use correct filter indices based on mode.
        # ====================================================================
        # this_filter_global: Global Gaussian IDs (0 to N-1) - for gradient accumulation
        # this_filter_local: Local working set indices (0 to M-1) - for GPU tensor indexing
        this_filter_global = filters[micro_idx]  # Always global IDs
        
        if use_local_filters and filters_local is not None:
            this_filter_local = filters_local[micro_idx]  # Local indices for GPU working set
        else:
            this_filter_local = this_filter_global  # In non-SSD mode, they're the same
        
        this_filter_len = this_filter_global.shape[0]

        # ========================================================================
        # Skip cameras whose coarse block visibility has no Gaussian-level hits.
        # ========================================================================
        if this_filter_len == 0:
            _log_empty_microbatch_camera(
                micro_idx=micro_idx,
                is_paper_ssd_mode=is_paper_ssd_mode,
                log_file=log_file,
            )
            
            # Keep the archived optimizer signal index in sync.
            if signal_tensor_pinned is not None:
                clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
            
            microbatch_idx += 1  # Increment even when skipping
            torch.cuda.nvtx.range_pop()  # Close micro_batch_idx range
            continue

        # ------------------------------------------------------------------------
        # 4.1: Load current SH coefficients
        # ------------------------------------------------------------------------
        if micro_idx == 0:
            # ====================================================================
            # Use direct GPU indexing if features are on GPU.
            # ====================================================================
            if gaussians.use_gpu_features:
                # SSD-backed path: index SH features from the GPU working set.
                torch.cuda.nvtx.range_push("gpu_mode_sh_indexing")
                
                with torch.no_grad():
                    # Directly gather SH features from GPU tensors
                    # CRITICAL: Use this_filter_local for GPU working set indexing!
                    shs_dc = gaussians._features_dc[this_filter_local]  # (K, 3)
                    shs_rest = gaussians._features_rest[this_filter_local]  # (K, 45)
                    
                    # Concatenate into single tensor
                    shs = torch.cat([shs_dc, shs_rest], dim=1)  # (K, 48)
                    shs.requires_grad_(True)

                    if is_paper_ssd_mode:
                        shs_retents[micro_idx] = None
                    else:
                        shs_retents[micro_idx] = shs.detach()

                    # Create dummy event for compatibility
                    cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                    cpu2gpu_event.record()

                    log_file = utils.get_log_file()
                    if is_paper_ssd_mode:
                        _log_paper_sh_indexed(
                            iteration=iteration,
                            filter_len=this_filter_len,
                            log_file=log_file,
                        )
                    else:
                        log_file.write(f"[GPU RESIDENT] Iter {iteration}: Indexed {this_filter_len} SH features directly from GPU\n")
                
                torch.cuda.nvtx.range_pop()
            
            # ====================================================================
            # Archived split-feature path: use CPU→GPU SH transfer with retention.
            # ====================================================================
            elif (gaussians.block_cache_state["last_shs"] is not None and
                gaussians.block_cache_state["last_filter"] is not None):

                # WARM START: Reuse cached SH from previous batch's last micro-batch
                torch.cuda.nvtx.range_push("inter_batch_hotspot_retention")

                with torch.cuda.stream(comm_stream), torch.no_grad():
                    last_shs = gaussians.block_cache_state["last_shs"]  # SH from previous batch (GPU)
                    last_filter = gaussians.block_cache_state["last_filter"]  # Indices from previous batch
                    last_len = last_filter.shape[0]

                    # Allocate output buffer for current micro-batch
                    shs = torch.empty(
                        this_filter_len, 48, device="cuda", requires_grad=True
                    )

                    # Compute visibility bitmasks for retention analysis
                    # Similar to intra-batch retention, but across batch boundaries
                    # NOTE: Use separate variable names for inter-batch to avoid conflicts with intra-batch prefetch
                    last_bit_interbatch = torch.zeros((N,), dtype=torch.uint8, device="cuda")
                    this_bit_interbatch = torch.zeros((N,), dtype=torch.uint8, device="cuda")

                    # Mark which gaussians were visible in last batch
                    last_bit_interbatch.scatter_(
                        dim=0,
                        index=last_filter,
                        src=torch.ones(last_len, dtype=torch.uint8, device="cuda"),
                    )

                    # Mark which gaussians are visible in current batch
                    this_bit_interbatch.scatter_(
                        dim=0,
                        index=this_filter_global,
                        src=torch.ones(this_filter_len, dtype=torch.uint8, device="cuda"),
                    )

                    # Setup retention mapping for current batch
                    retention_vec_interbatch = torch.empty((N,), dtype=torch.int32, device="cuda")
                    retention_vec_interbatch.scatter_(
                        dim=0,
                        index=this_filter_global,
                        src=torch.arange(this_filter_len, dtype=torch.int32, device="cuda"),
                    )

                    # Category H: Parameters to load from HOST (CPU)
                    # These are visible now but were NOT visible in previous batch
                    bit_h = ~last_bit_interbatch & this_bit_interbatch
                    # Count how many to load from CPU
                    cnt_h_interbatch = bit_h.sum().item()

                    if cnt_h_interbatch > 0:
                        idx_h = torch.nonzero_static(bit_h, size=cnt_h_interbatch).flatten()
                        host_indices_to_param = idx_h.to(torch.int32)  # Param indices to load
                        param_indices_from_host = torch.gather(retention_vec_interbatch, dim=0, index=idx_h)
                        del idx_h
                    else:
                        host_indices_to_param = torch.empty(0, dtype=torch.int32, device="cuda")
                        param_indices_from_host = torch.empty(0, dtype=torch.int32, device="cuda")

                    del bit_h

                    # Category D: Parameters to DUPLICATE (retain from GPU)
                    # These were visible in previous batch AND are still visible now
                    bit_d = last_bit_interbatch & this_bit_interbatch
                    cnt_d_interbatch = bit_d.sum().item()

                    if cnt_d_interbatch > 0:
                        idx_d = torch.nonzero_static(bit_d, size=cnt_d_interbatch).flatten()
                        param_indices_from_rtnt = torch.gather(retention_vec_interbatch, dim=0, index=idx_d)

                        # Setup mapping from last batch's indexing
                        retention_vec_last = torch.empty((N,), dtype=torch.int32, device="cuda")
                        retention_vec_last.scatter_(
                            dim=0,
                            index=last_filter,
                            src=torch.arange(last_len, dtype=torch.int32, device="cuda"),
                        )
                        rtnt_indices_to_param = torch.gather(retention_vec_last, dim=0, index=idx_d)
                        del idx_d, retention_vec_last
                    else:
                        rtnt_indices_to_param = torch.empty(0, dtype=torch.int32, device="cuda")
                        param_indices_from_rtnt = torch.empty(0, dtype=torch.int32, device="cuda")

                    del bit_d, last_bit_interbatch, this_bit_interbatch

                    # Use the archived retention kernel for mixed GPU/CPU SH sources.
                    send_shs2gpu_stream_retention(
                        shs,                            # Output: SH for current batch
                        gaussians._parameters,          # Input: SH on host (CPU)
                        last_shs,                       # Input: SH retained from previous batch (GPU)
                        host_indices_to_param,          # Category H: param → current mapping
                        rtnt_indices_to_param,          # Category D: previous → current mapping
                        param_indices_from_host,        # Category H: indices for CPU load
                        param_indices_from_rtnt,        # Category D: indices for GPU retention
                        grid_size,
                        block_size,
                        grid_size_D,
                        block_size_D,
                    )

                    shs_retents[micro_idx] = shs.detach()
                    cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                    cpu2gpu_event.record(comm_stream)

                    # Log cache hit statistics
                    log_file = utils.get_log_file()
                    hit_rate = cnt_d_interbatch / this_filter_len if this_filter_len > 0 else 0.0
                    log_file.write(
                        f"[ARCHIVED SH CACHE] Iter {iteration}: Reused {cnt_d_interbatch}/{this_filter_len} "
                        f"({hit_rate*100:.1f}%), Loaded {cnt_h_interbatch} from CPU\n"
                    )

                torch.cuda.nvtx.range_pop()

            else:
                # Archived split-feature cold start: no retained SH data is available.
                with torch.cuda.stream(comm_stream), torch.no_grad():
                    shs = torch.empty(
                        this_filter_len, 48, device="cuda", requires_grad=True
                    )

                    send_shs2gpu_stream(
                        shs,
                        gaussians._parameters,
                        filters[micro_idx],
                        grid_size,
                        block_size,
                    )
                    shs_retents[micro_idx] = shs.detach()
                    cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                    cpu2gpu_event.record(comm_stream)

                    # Log cold start
                    log_file = utils.get_log_file()
                    log_file.write(f"[ARCHIVED SH CACHE] Iter {iteration}: COLD START - Loaded all {this_filter_len} from CPU\n")

        else:
            # ====================================================================
            # Subsequent micro-batches: different strategies based on mode
            # ====================================================================
            if gaussians.use_gpu_features:
                # GPU-resident path: direct SH indexing, no CPU prefetch needed.
                with torch.no_grad():
                    # CRITICAL: Use this_filter_local for GPU working set indexing!
                    shs_dc = gaussians._features_dc[this_filter_local]
                    shs_rest = gaussians._features_rest[this_filter_local]
                    shs = torch.cat([shs_dc, shs_rest], dim=1)
                    shs.requires_grad_(True)
                    if is_paper_ssd_mode:
                        shs_retents[micro_idx] = None
                    else:
                        shs_retents[micro_idx] = shs.detach()

                    cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                    cpu2gpu_event.record()
            else:
                # Archived split-feature path: use prefetched data from previous micro-batch.
                shs = shs_next
                shs_retents[micro_idx] = shs.detach()
                cpu2gpu_event = next_cpu2gpu_event

        # ------------------------------------------------------------------------
        # 4.2: Prefetch NEXT micro-batch SH coefficients (overlapped with compute)
        # Only needed by archived CPU→GPU split-feature transfer.
        # ------------------------------------------------------------------------
        if not gaussians.use_gpu_features:
            # Archived split-feature path: prefetch from CPU for next micro-batch.
            with torch.cuda.stream(comm_stream), torch.no_grad():
                if micro_idx < num_micro_batches - 1:
                    shs_next = torch.empty(
                        filters[micro_idx + 1].shape[0], 48, device="cuda"
                    )

                    # Update visibility bitmasks for current and next iterations
                    if micro_idx == 0:
                        # Initialize both current and next bitmasks
                        this_bit.scatter_(
                            dim=0,
                            index=filters[micro_idx],
                            src=torch.ones(
                                filters[micro_idx].shape[0],
                                dtype=torch.uint8,
                                device="cuda",
                            ),
                        )
                        next_bit.scatter_(
                            dim=0,
                            index=filters[micro_idx + 1],
                            src=torch.ones(
                                filters[micro_idx + 1].shape[0],
                                dtype=torch.uint8,
                                device="cuda",
                            ),
                        )
                    else:
                        # Swap bitmasks and update for next iteration
                        this_bit, next_bit = next_bit, this_bit
                        next_bit.scatter_(
                            dim=0,
                            index=filters[micro_idx - 1],
                            src=torch.zeros(
                                filters[micro_idx - 1].shape[0],
                                dtype=torch.uint8,
                                device="cuda",
                            ),
                        )
                        next_bit.scatter_(
                            dim=0,
                            index=filters[micro_idx + 1],
                            src=torch.ones(
                                filters[micro_idx + 1].shape[0],
                                dtype=torch.uint8,
                                device="cuda",
                            ),
                        )

                    # Compute retention indices: classify parameters into 3 categories
                    # - Category H (Host): parameters not in current but needed in next (~this_bit & next_bit)
                    # - Category D (Duplicate): parameters in both current and next (this_bit & next_bit)
                    # - Category G (Garbage): parameters in current but not in next (this_bit & ~next_bit)

                    # NOTE: Using torch.nonzero_static (torch 2.6+) to avoid device-to-host sync
                    # torch.nonzero() would block CPU waiting for GPU, hurting performance

                    # Setup index mapping for next iteration
                    retention_vec.scatter_(
                        dim=0,
                        index=filters[micro_idx + 1],
                        src=torch.arange(
                            filters[micro_idx + 1].shape[0],
                            dtype=torch.int32,
                            device="cuda",
                        ),
                    )
                    # idx_h = torch.nonzero(~this_bit & next_bit).flatten() # torch.nonzero() blocks cpu!!!
                    bit_h = ~this_bit & next_bit
                    idx_h = torch.empty(
                        (cnt_h[micro_idx],), dtype=torch.int64, device="cuda"
                    )
                    idx_h = torch.nonzero_static(bit_h, size=cnt_h[micro_idx]).flatten()
                    host_indices_to_param = idx_h.to(
                        torch.int32
                    )  # Parameter indices to load from host
                    param_indices_from_host = torch.gather(
                        retention_vec, dim=0, index=idx_h
                    )  # Where to put them
                    del idx_h, bit_h

                    # idx_d = torch.nonzero(this_bit & next_bit).flatten() # overlap # torch.nonzero() blocks cpu!!!
                    bit_d = this_bit & next_bit
                    idx_d = torch.nonzero_static(bit_d, size=cnt_d[micro_idx]).flatten()
                    param_indices_from_rtnt = torch.gather(
                        retention_vec, dim=0, index=idx_d
                    )  # reused in gpu2cpu comm
                    del bit_d

                    # Update retention_vec for current iteration mapping
                    retention_vec.scatter_(
                        dim=0,
                        index=filters[micro_idx],
                        src=torch.arange(
                            filters[micro_idx].shape[0], dtype=torch.int32, device="cuda"
                        ),
                    )
                    rtnt_indices_to_param = torch.gather(
                        retention_vec, dim=0, index=idx_d
                    )  # Current iteration indices
                    del idx_d

                    # Transfer SH coefficients: mix of retained (from GPU) and loaded (from CPU)
                    send_shs2gpu_stream_retention(
                        shs_next,  # Output: SH for next iteration
                        gaussians._parameters,  # Input: SH on host (CPU)
                        shs_retents[
                            micro_idx
                        ],  # Input: SH retained from current iteration (GPU)
                        host_indices_to_param,  # Category H: param → next mapping
                        rtnt_indices_to_param,  # Category D: current → next mapping
                        param_indices_from_host,  # Category H: indices for CPU load
                        param_indices_from_rtnt,  # Category D: indices for GPU retention
                        grid_size,
                        block_size,
                        grid_size_D,
                        block_size_D,
                    )
                    shs_next.requires_grad_(True)
                    del host_indices_to_param, param_indices_from_host

                    next_cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                    next_cpu2gpu_event.record(comm_stream)

        # ------------------------------------------------------------------------
        # 4.3: Forward pass - Render image with filtered gaussian parameters
        # ------------------------------------------------------------------------
        torch.cuda.nvtx.range_push("forward_pass")
        torch.cuda.nvtx.range_push("prepare filtered parameters")

        # ====================================================================
        # Gather filtered parameters from the compact GPU working set. Local
        # indices address resident tensors; global indices address the full model.
        # Clones create independent tensors whose gradients are scattered back.
        filtered_xyz_gpu = gaussians._xyz[this_filter_local].clone().requires_grad_(True)
        _filtered_opacity_gpu = gaussians._opacity[this_filter_local].clone().requires_grad_(True)
        _filtered_scaling_gpu = gaussians._scaling[this_filter_local].clone().requires_grad_(True)
        _filtered_rotation_gpu = gaussians._rotation[this_filter_local].clone().requires_grad_(True)
        
        # Retain gradients for cloned non-leaf tensors until scatter-back.
        filtered_xyz_gpu.retain_grad() # retain_grad(): 虽然不是leaf node，但反向传播结束后, 不要销毁它的梯度，把它留在显存中等待使用
        _filtered_opacity_gpu.retain_grad()
        _filtered_scaling_gpu.retain_grad()
        _filtered_rotation_gpu.retain_grad()
        # Apply activation functions to constrain parameter ranges
        filtered_opacity_gpu = gaussians.opacity_activation(_filtered_opacity_gpu)
        filtered_scaling_gpu = gaussians.scaling_activation(_filtered_scaling_gpu)
        filtered_rotation_gpu = gaussians.rotation_activation(_filtered_rotation_gpu)
        
        torch.cuda.nvtx.range_pop()

        # Wait for SH coefficients. In the GPU-resident path this event has
        # already been recorded after direct GPU indexing.
        cpu2gpu_event.wait(default_stream)
        
        # ====================================================================
        # Handle SH gradient tracking based on mode.
        # ====================================================================
        if gaussians.use_gpu_features:
            # GPU-resident path: autograd backprops to _features_dc/_features_rest.
            filtered_shs = shs.requires_grad_(True)
        else:
            # Archived split-feature path: disable autograd and compute SH gradients manually.
            # Manual backward is needed because SH features are on CPU
            filtered_shs = shs.requires_grad_(False)

        # Render image using filtered Gaussian splatting.
        (
            rendered_image,
            batched_means2D,
            batched_radiis,
            batched_colors_detached,
            dirs,
        ) = pipeline_forward_one_step_shs_inplace(
            filtered_opacity_gpu,
            filtered_scaling_gpu,
            filtered_rotation_gpu,
            filtered_xyz_gpu,
            filtered_shs,
            batched_cameras[micro_idx],
            scene,
            gaussians,
            background,
            pipe_args,
            use_autograd_for_sh=gaussians.use_gpu_features,  # Enable autograd in GPU mode
        )

        # Compute loss
        loss = torch_compiled_loss(
            rendered_image, batched_cameras[micro_idx].original_image
        )
        torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 4.4: Backward pass - Compute gradients
        # ------------------------------------------------------------------------
        torch.cuda.nvtx.range_push("backward_pass")
        
        # ====================================================================
        # Optional first-iteration gradient sanity log.
        # ====================================================================
        log_file = utils.get_log_file()
        if iteration == 1 and micro_idx == 0:
            log_file.write(f"\n[GRAD DEBUG] Before backward (iter {iteration}, micro {micro_idx}):\n")
            log_file.write(f"  filtered_xyz_gpu.requires_grad: {filtered_xyz_gpu.requires_grad}\n")
            log_file.write(f"  filtered_xyz_gpu.is_leaf: {filtered_xyz_gpu.is_leaf}\n")
            log_file.write(f"  filtered_xyz_gpu.grad_fn: {filtered_xyz_gpu.grad_fn}\n")
            log_file.write(f"  _filtered_opacity_gpu.requires_grad: {_filtered_opacity_gpu.requires_grad}\n")
            log_file.write(f"  _filtered_opacity_gpu.grad_fn: {_filtered_opacity_gpu.grad_fn}\n")
            log_file.write(f"  filtered_opacity_gpu.requires_grad: {filtered_opacity_gpu.requires_grad}\n")
            log_file.write(f"  filtered_opacity_gpu.grad_fn: {filtered_opacity_gpu.grad_fn}\n")
            log_file.write(f"  rendered_image.requires_grad: {rendered_image.requires_grad}\n")
            log_file.write(f"  loss.requires_grad: {loss.requires_grad}\n")
            log_file.write(f"  loss.grad_fn: {loss.grad_fn}\n")
        
        # ====================================================================
        # SSD-backed path: one backward is enough; autograd handles gradients.
        # Archived split-feature path: two backward phases handle CPU SH features.
        if gaussians.use_gpu_features:
            # SSD-backed path: simple single backward; autograd handles all gradients.
            loss.backward()
        else:
            # Archived split-feature path: two-step backward for CPU features.
            loss.backward(retain_graph=True)

        # ====================================================================
        # Optional first-iteration gradient sanity log.
        # ====================================================================
        if iteration == 1 and micro_idx == 0:
            log_file.write(f"\n[GRAD DEBUG] After backward:\n")
            log_file.write(f"  filtered_xyz_gpu.grad is None: {filtered_xyz_gpu.grad is None}\n")
            if filtered_xyz_gpu.grad is not None:
                log_file.write(f"  filtered_xyz_gpu.grad.shape: {filtered_xyz_gpu.grad.shape}\n")
                log_file.write(f"  filtered_xyz_gpu.grad.sum(): {filtered_xyz_gpu.grad.sum().item()}\n")
            log_file.write(f"  _filtered_opacity_gpu.grad is None: {_filtered_opacity_gpu.grad is None}\n")
            if _filtered_opacity_gpu.grad is not None:
                log_file.write(f"  _filtered_opacity_gpu.grad.sum(): {_filtered_opacity_gpu.grad.sum().item()}\n")
            log_file.write("  filtered_opacity_gpu.grad: skipped (non-leaf debug tensor)\n")
            log_file.write("  batched_colors_detached.grad: skipped (non-leaf debug tensor)\n")
            log_file.write("  dirs.grad: skipped (non-leaf debug tensor)\n")

        # ====================================================================
        # Conditional SH gradient computation.
        # ====================================================================
        if not gaussians.use_gpu_features:
            # Archived split-feature path: manual backward for spherical harmonics.
            # Wait for shs_grad buffer to be ready
            shs_grad_init_event.wait(default_stream)

            # Manual backward for spherical harmonics (custom gradient computation)
            v_dirs = spherical_harmonics_bwd_inplace(
                degrees_to_use=gaussians.active_sh_degree,
                dirs=dirs,
                coeffs=filtered_shs.reshape(1, -1, 16, 3),
                v_coeffs=shs_grad,
                v_colors=batched_colors_detached.grad,
            )
            dirs.backward(v_dirs)
        else:
            # ================================================================
            # SSD-backed path: autograd already handled everything.
            # ================================================================
            # No manual gradient computation needed because:
            # 1. spherical_harmonics() was called WITH autograd enabled
            # 2. loss.backward() automatically propagated through:
            #    loss → rendered_image → batched_colors → dirs → filtered_xyz_gpu ✓
            # 3. All gradients are already computed and available
            
            if iteration == 1 and micro_idx == 0:
                mode_name = 'PAPER GPU WORKING SET' if is_paper_ssd_mode else 'GPU RESIDENT'
                if is_paper_ssd_mode:
                    _write_paper_phase1_log(
                        f"[{mode_name}] Autograd backward completed (no manual computation)\n",
                        log_file=log_file,
                    )
                else:
                    log_file.write(f"[{mode_name}] Autograd backward completed (no manual computation)\n")
                log_file.write(f"  filtered_xyz_gpu.grad is not None: {filtered_xyz_gpu.grad is not None}\n")
                if filtered_xyz_gpu.grad is not None:
                    log_file.write(f"  filtered_xyz_gpu.grad.sum(): {filtered_xyz_gpu.grad.sum().item()}\n")
        
        torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 4.5: Accumulate gradients back to full parameter tensors
        # ------------------------------------------------------------------------
        with torch.no_grad():
            torch.cuda.nvtx.range_push("scatter gpu grads back to origin")
            
            # ================================================================
            # Optional first-iteration scatter sanity log.
            # ================================================================
            if iteration == 1 and micro_idx == 0:
                log_file.write(f"\n[GRAD DEBUG] Before scatter_add_:\n")
                log_file.write(f"  filtered_xyz_gpu.grad: {filtered_xyz_gpu.grad}\n")
                log_file.write(f"  _filtered_opacity_gpu.grad: {_filtered_opacity_gpu.grad}\n")
                log_file.write(f"  _filtered_scaling_gpu.grad: {_filtered_scaling_gpu.grad}\n")
                log_file.write(f"  _filtered_rotation_gpu.grad: {_filtered_rotation_gpu.grad}\n")
            
            if filtered_xyz_gpu.grad is None:
                log_file.write(f"  [ERROR] filtered_xyz_gpu.grad is None!\n")
                log_file.write(f"  Checking computational graph:\n")
                log_file.write(f"    filtered_xyz_gpu in computation: {filtered_xyz_gpu in loss.grad_fn if hasattr(loss, 'grad_fn') else 'N/A'}\n")
                raise RuntimeError("filtered_xyz_gpu.grad is None - gradient not computed!")
            
            # ================================================================
            # Scatter operates entirely on GPU.
            # ================================================================
            # In SSD-backed mode, all active parameters are in the GPU working set (size M).
            # Use this_filter_local (0 to M-1) for scatter_add_ indices!
            # 
            # In paper SSD mode we accumulate only the Gaussians
            # touched by this batch, not the entire GPU working set.
            if use_sparse_gpu_grad_accum:
                sparse_filter_idx = torch.searchsorted(sparse_grad_local_ids, this_filter_local)
                sparse_grad_components['xyz'].scatter_add_(
                    dim=0,
                    src=filtered_xyz_gpu.grad,
                    index=sparse_filter_idx.reshape(-1, 1).expand(-1, 3),
                )
                sparse_grad_components['opacity'].scatter_add_(
                    dim=0,
                    src=_filtered_opacity_gpu.grad,
                    index=sparse_filter_idx.reshape(-1, 1),
                )
                sparse_grad_components['scaling'].scatter_add_(
                    dim=0,
                    src=_filtered_scaling_gpu.grad,
                    index=sparse_filter_idx.reshape(-1, 1).expand(-1, 3),
                )
                sparse_grad_components['rotation'].scatter_add_(
                    dim=0,
                    src=_filtered_rotation_gpu.grad,
                    index=sparse_filter_idx.reshape(-1, 1).expand(-1, 4),
                )

                if filtered_shs.grad is not None:
                    shs_dc_grad = filtered_shs.grad[:, :3]
                    shs_rest_grad = filtered_shs.grad[:, 3:48]
                    sparse_grad_components['features_dc'].scatter_add_(
                        dim=0,
                        src=shs_dc_grad,
                        index=sparse_filter_idx.reshape(-1, 1).expand(-1, 3),
                    )
                    sparse_grad_components['features_rest'].scatter_add_(
                        dim=0,
                        src=shs_rest_grad,
                        index=sparse_filter_idx.reshape(-1, 1).expand(-1, 45),
                    )
            else:
                gaussians._xyz.grad.scatter_add_(
                    dim=0,
                    src=filtered_xyz_gpu.grad,
                    index=this_filter_local.reshape(-1, 1).expand(-1, 3),
                )
                gaussians._opacity.grad.scatter_add_(
                    dim=0, src=_filtered_opacity_gpu.grad, index=this_filter_local.reshape(-1, 1)
                )
                gaussians._scaling.grad.scatter_add_(
                    dim=0,
                    src=_filtered_scaling_gpu.grad,
                    index=this_filter_local.reshape(-1, 1).expand(-1, 3),
                )
                gaussians._rotation.grad.scatter_add_(
                    dim=0,
                    src=_filtered_rotation_gpu.grad,
                    index=this_filter_local.reshape(-1, 1).expand(-1, 4),
                )

                if gaussians.use_gpu_features and filtered_shs.grad is not None:
                    shs_dc_grad = filtered_shs.grad[:, :3]
                    shs_rest_grad = filtered_shs.grad[:, 3:48]
                    gaussians._features_dc.grad.scatter_add_(
                        dim=0,
                        src=shs_dc_grad,
                        index=this_filter_local.reshape(-1, 1).expand(-1, 3),
                    )
                    gaussians._features_rest.grad.scatter_add_(
                        dim=0,
                        src=shs_rest_grad,
                        index=this_filter_local.reshape(-1, 1).expand(-1, 45),
                    )

            torch.cuda.nvtx.range_pop()

        # Cleanup temporary tensors
        del rendered_image, batched_colors_detached, dirs
        if not gaussians.use_gpu_features:
            del v_dirs  # Only exists in the archived split-feature path.
        shs = None
        del (
            filtered_xyz_gpu,
            filtered_opacity_gpu,
            filtered_scaling_gpu,
            filtered_rotation_gpu,
            filtered_shs,
        )
        del _filtered_opacity_gpu, _filtered_scaling_gpu, _filtered_rotation_gpu

        losses.append(loss.detach())
        del loss

        # Mark completion of GPU computation for this micro-batch
        gpu2cpu_event = torch.cuda.Event(enable_timing=True)
        gpu2cpu_event.record(default_stream)

        # ------------------------------------------------------------------------
        # 4.6: Offload SH gradients back to CPU (GPU → CPU)
        # ------------------------------------------------------------------------
        if gaussians.use_gpu_features:
            # ====================================================================
            # Skip gradient offload; already accumulated on GPU.
            # ====================================================================
            # In SSD-backed mode, SH features (_features_dc/_features_rest) are on GPU.
            # Gradients are directly accumulated to .grad buffers via scatter_add_
            # No CPU offload needed - PyTorch autograd handles everything
            
            if micro_idx < num_micro_batches - 1:
                # Non-final micro-batch: just clean up and signal
                shs_retents[micro_idx] = None
                shs_grad = None  # Will be recreated for next micro-batch
                
                # Keep the archived optimizer signal index in sync.
                clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                microbatch_idx += 1
            else:
                if is_paper_ssd_mode:
                    gaussians.block_cache_state["last_shs"] = None
                    gaussians.block_cache_state["last_filter"] = None
                    gaussians.block_cache_state["last_retention_vec"] = None

                    if iteration == 1 or _perf_log:
                        log_file = utils.get_log_file()
                        _log_paper_interbatch_sh_cache_disabled(
                            iteration=iteration,
                            log_file=log_file,
                        )

                    clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                    microbatch_idx += 1
                else:
                    # Final micro-batch: save archived SH retention cache state.
                    gaussians.block_cache_state["last_shs"] = shs_retents[micro_idx].detach().clone()
                    gaussians.block_cache_state["last_filter"] = filters[-1].clone()

                    log_file = utils.get_log_file()
                    log_file.write(
                        f"[ARCHIVED SH CACHE GPU] Iter {iteration}: Saved {filters[-1].shape[0]} SH features "
                        f"for next batch (GPU memory: {shs_retents[micro_idx].numel() * 4 / 1024**2:.2f} MB)\n"
                    )

                    # Keep the archived optimizer signal index in sync.
                    clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                    microbatch_idx += 1
        
        else:
            # ====================================================================
            # Archived split-feature path: offload gradients using retention.
            # ====================================================================
            if micro_idx < num_micro_batches - 1:
                # Non-final micro-batch: use retention-based selective gradient offloading
                with torch.cuda.stream(comm_stream), torch.no_grad():
                    # Reuse indices from prefetch step (Category D and G)
                    rtnt_indices_from_grad = (
                        param_indices_from_rtnt  # Category D: retained indices
                    )
                    grad_indices_to_rtnt = rtnt_indices_to_param

                    # idx_g = torch.nonzero(this_bit & ~next_bit).flatten() # torch.nonzero() blocks cpu!!!
                    bit_g = this_bit & ~next_bit
                    idx_g = torch.nonzero_static(bit_g, size=cnt_g[micro_idx]).flatten()
                    host_indices_from_grad = idx_g.to(torch.int32)
                    grad_indices_to_host = torch.gather(retention_vec, dim=0, index=idx_g)
                    del idx_g, bit_g

                    # Wait for backward pass to complete
                    gpu2cpu_event.wait(comm_stream)
                    shs_retents[micro_idx] = None
                    shs_grad_next = torch.zeros_like(shs_next, device="cuda")

                    # Offload gradients: mix of retained (keep on GPU) and offloaded (send to CPU)
                    send_shs2cpu_grad_buffer_stream_retention(
                        shs_grad,  # Input: current SH gradients
                        parameters_grad_buffer[:N, :],  # Output: CPU gradient buffer
                        shs_grad_next,  # Output: next iteration SH gradients (retained on GPU)
                        host_indices_from_grad,  # Category G: indices to offload to CPU
                        rtnt_indices_from_grad,  # Category D: indices to retain on GPU
                        grad_indices_to_host,  # Category G: mapping to CPU buffer
                        grad_indices_to_rtnt,  # Category D: mapping to next grad buffer
                        True,
                        grid_size,
                        block_size,
                        grid_size_D,
                        block_size_D,
                    )
                    shs_grad = shs_grad_next
                    shs_grad_init_event.record(comm_stream)

                    # Signal archived optimizer that gradients are ready for this micro-batch.
                    clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                    microbatch_idx += 1

            else:
                # ====================================================================
                # Save archived SH retention state for the next batch.
                # ====================================================================
                # Final micro-batch: offload all gradients to CPU AND save cache state
                with torch.cuda.stream(comm_stream), torch.no_grad():
                    gpu2cpu_event.wait(comm_stream)

                    send_shs2cpu_grad_buffer_stream(
                        shs_grad,
                        parameters_grad_buffer[:N, :],
                        filters[-1],
                        True,
                        grid_size,
                        block_size,
                    )

                    # ================================================================
                    # Detach retained tensors to avoid keeping the computation graph.
                    gaussians.block_cache_state["last_shs"] = shs_retents[micro_idx].detach().clone()
                    gaussians.block_cache_state["last_filter"] = filters[-1].clone()

                    # Optional: also save retention_vec if needed for more advanced retention strategies
                    # gaussians.block_cache_state["last_retention_vec"] = retention_vec.clone()

                    log_file = utils.get_log_file()
                    log_file.write(
                        f"[ARCHIVED SH CACHE] Iter {iteration}: Saved {filters[-1].shape[0]} SH features "
                        f"for next batch (GPU memory: {shs_retents[micro_idx].numel() * 4 / 1024**2:.2f} MB)\n"
                    )

                    # Keep final SH features for archived inter-batch retention.
                    # shs_retents[micro_idx] = None  # <-- REMOVED

                    # Signal archived optimizer that final gradients are ready.
                    clm_kernels.set_signal(signal_tensor_pinned, microbatch_idx, 1)
                    microbatch_idx += 1

        # ====================================================================
        # [GPU WORKING SET] NO gradient sync needed in GPU Adam mode
        # ====================================================================
        # In GPU Adam mode, gradients are used directly on GPU for optimizer.step()
        # No need to sync to RAM's parameters_grad_buffer in SSD-backed mode.
        # Gradient sync has been REMOVED - it was redundant
        
        torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 4.7: Update densification statistics (for adaptive gaussian control)
        # ------------------------------------------------------------------------
        update_densification_stats_offload_accum_grads(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
            filters[micro_idx],
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
        )

        batched_means2D.grad = None
        del batched_means2D, batched_radiis

        # Update visibility mask for sparse Adam optimizer (tracks which parameters received gradients)
        if args.sparse_adam and visibility_mask is not None:
            torch.cuda.nvtx.range_push("update visibility")
            src = torch.ones(
                (len(filters[micro_idx]),), dtype=torch.bool, device="cuda"
            )
            visibility_mask.scatter_(dim=0, index=filters[micro_idx], src=src)
            torch.cuda.nvtx.range_pop()

    _ts('stage4_train_done')

    # ============================================================================
    # STAGE 5: POST-TRAINING OPTIMIZATION & CLEANUP
    # ============================================================================
    _ts('stage5_optim_start')

    optimizer_updated_global_indices = None

    assert microbatch_idx == bsz, f"microbatch_idx should be equal to bsz. Got {microbatch_idx} vs {bsz}"

    # ------------------------------------------------------------------------
    # 5.1: Optimizer step (mode-dependent)
    # ------------------------------------------------------------------------
    if not gaussians.use_gpu_features:
        raise RuntimeError("Pure SSD release path requires GPU working-set features.")

    is_ssd_offload = hasattr(gaussians.optimizer, 'is_ssd_offload_mode') and \
                     gaussians.optimizer.is_ssd_offload_mode

    if is_ssd_offload:
        if not is_paper_ssd_mode:
            raise RuntimeError("Pure SSD release path requires --ssd_execution_mode paper.")
        if paper_optimizer_backend != 'gpu_resident':
            raise RuntimeError(
                "Paper SSD release path only supports GPUResidentAdam. "
                "Set --paper_optimizer_backend gpu_resident."
            )
        _run_gpu_resident_adam_step(
            gaussians=gaussians,
            args=args,
            iteration=iteration,
            total_n_gaussians=total_n_gaussians,
            sparse_visibility_indices=sparse_visibility_indices,
            sparse_grad_local_ids=sparse_grad_local_ids,
            sparse_grad_components=sparse_grad_components,
            get_gpu_resident_optimizer_fn=get_gpu_resident_optimizer,
            ensure_local_to_global_mapping_fn=ensure_local_to_global_mapping,
            log_prefix="[PAPER SSD MODE]",
            log_file=log_file,
        )
    else:
            # ================================================================
            # Non-SSD GPU-resident Adam optimization.
            # ================================================================
            # All parameters are on GPU, use GPU Adam directly.
            for param in gaussians.all_parameters():  # All 6 parameters
                if param.grad is not None:
                    param.grad /= args.bsz

            # Apply optimizer step to all GPU parameters
            if not args.stop_update_param:
                if args.sparse_adam:
                    # Sparse Adam: only update parameters that received gradients
                    # visibility_mask needs to map to GPU working set indices
                    local_visibility = ensure_local_to_global_mapping(
                        gaussians,
                        total_n_gaussians,
                        log_file=log_file,
                        context=f"gpu_adam_visibility_iter_{iteration}",
                    )
                    gaussians.optimizer.gpu_adam.step(visibility=visibility_mask[local_visibility])
                else:
                    # Dense Adam: update all parameters in working set
                    gaussians.optimizer.gpu_adam.step()
            gaussians.optimizer.gpu_adam.zero_grad(set_to_none=True)
            
            # ================================================================
            # Non-SSD GPU working-set writeback to RAM cache.
            # ================================================================
            torch.cuda.nvtx.range_push("Writeback: GPU → RAM")
            log_file.write(f"[GPU Working Set] Writing back updated parameters to RAM cache\n")
            
            # Prepare geometry cache references (will be updated in-place)
            geometry_cache = {
                'xyz': original_xyz.data,       # Original CPU tensor
                'scaling': original_scaling.data,
                'rotation': original_rotation.data,
                'opacity': original_opacity.data
            }
            
            # Writeback all parameters to RAM
            gaussians.gpu_working_set_manager.update_ram_cache(
                geometry_cache=geometry_cache,
                sh_cache=gaussians.parameters_buffer
            )
            
            log_file.write(f"[GPU Working Set] RAM cache updated successfully\n")
            torch.cuda.nvtx.range_pop()

    utils.memory_report("after optimizer step")
    _ts('stage5_optim_done')

    # ============================================================================
    # [GPU WORKING SET] Update RAM cache and cleanup
    # ============================================================================
    defer_gpu_working_set_clear = False
    if gaussians.use_gpu_features and hasattr(gaussians, 'gpu_working_set_manager'):
        # Check whether updates are handled by archived optimizer or GPU optimizer.
        is_ssd_offload = hasattr(gaussians.optimizer, 'is_ssd_offload_mode') and \
                         gaussians.optimizer.is_ssd_offload_mode
        
        if not is_ssd_offload:
            # Non-SSD GPU mode writes the working set back to RAM cache.
            # (In archived SSD offload mode, the optimizer updates _unified_params directly,
            #  so no need to update from GPU working set)
            with torch.no_grad():
                # Prepare geometry cache references (will be updated in-place)
                geometry_cache = {
                    'xyz': original_xyz.data,
                    'scaling': original_scaling.data,
                    'rotation': original_rotation.data,
                    'opacity': original_opacity.data
                }
                gaussians.gpu_working_set_manager.update_ram_cache(
                    geometry_cache=geometry_cache,
                    sh_cache=gaussians.parameters_buffer
                )
                log_file = utils.get_log_file()
                log_file.write(f"[GPU Working Set] Wrote updated parameters back to RAM cache\n")
        
        # Step 2: Clear GPU working set memory after writeback.  Paper
        # gpu_resident writeback reads the updated resident blocks directly
        # from the compact GPU working set, so keep it alive until SSD
        # staging below has consumed it.
        if is_paper_ssd_mode and storage_adapter is not None and paper_optimizer_backend == 'gpu_resident':
            defer_gpu_working_set_clear = True
        else:
            gaussians.gpu_working_set_manager.clear()
        
        # Step 3: Restore original parameter references (ALL parameters)
        gaussians._xyz = original_xyz
        gaussians._scaling = original_scaling
        gaussians._rotation = original_rotation
        gaussians._opacity = original_opacity
        gaussians._features_dc = original_features_dc
        gaussians._features_rest = original_features_rest
        
        log_file = utils.get_log_file()
        if is_paper_ssd_mode:
            _log_paper_cpu_source_refs_restored(log_file=log_file)
        else:
            log_file.write("[GPU WORKING SET] All parameters restored to CPU references\n")

    # ============================================================================
    # [SSD WRITEBACK] Persist updated parameters to SSD after optimizer step
    # ============================================================================
    # In the archived unified-param optimizer path, _unified_params is the source of truth.
    # The optimizer updates _unified_params directly, so blocks are re-read from _unified_params
    # each iteration (fast path above). No need to persist to SSD cache every iteration.
    # Only persist for checkpointing (handled separately).
    is_ssd_offload_mode = hasattr(gaussians, '_unified_params') and gaussians._unified_params is not None and \
                          hasattr(gaussians.optimizer, 'is_ssd_offload_mode') and gaussians.optimizer.is_ssd_offload_mode
    
    if storage_adapter is not None and (not is_ssd_offload_mode or is_paper_ssd_mode):
        with torch.no_grad():
            torch.cuda.nvtx.range_push("SSD: writeback updated blocks")

            if is_paper_ssd_mode:
                updated_block_ids = _collect_paper_updated_block_ids(
                    iteration=iteration,
                    optimizer_updated_global_indices=optimizer_updated_global_indices,
                    filters=filters,
                    block_size=args.gaussian_block_size,
                    total_n_gaussians=total_n_gaussians,
                    collect_from_indices_fn=_collect_updated_block_ids_from_indices,
                    collect_from_filters_fn=_collect_updated_block_ids,
                    should_log=should_log_paper_sets,
                    log_file=log_file,
                )
            else:
                updated_block_ids = _collect_updated_block_ids(
                    filters,
                    args.gaussian_block_size,
                    total_n_gaussians,
                )

            # ====================================================================
            # [MODE-DEPENDENT WRITEBACK] Read updated parameters from correct location
            # ====================================================================
            is_ssd_offload = hasattr(gaussians.optimizer, 'is_ssd_offload_mode') and \
                             gaussians.optimizer.is_ssd_offload_mode

            if is_paper_ssd_mode:
                keep_resident_blocks = list(paper_block_sets['keep_resident_blocks']) if paper_block_sets is not None else []
                staged_writeback_blocks, refreshed_omega, candidate_blocks = _apply_paper_writeback_payload(
                    storage_adapter=storage_adapter,
                    args=args,
                    payload_iteration=iteration,
                    current_iteration=iteration,
                    updated_block_ids=updated_block_ids,
                    omega_blocks=keep_resident_blocks,
                    total_n_gaussians=total_n_gaussians,
                    original_xyz=original_xyz,
                    original_scaling=original_scaling,
                    original_rotation=original_rotation,
                    original_opacity=original_opacity,
                    original_features_dc=original_features_dc,
                    original_features_rest=original_features_rest,
                    gpu_working_set_manager=gaussians.gpu_working_set_manager,
                    build_updated_blocks_dict_from_gpu_fn=_build_updated_blocks_dict_from_gpu,
                    sync_updated_blocks_from_gpu_views_to_cpu_fn=_sync_updated_blocks_from_gpu_views_to_cpu,
                    materialize_updated_blocks_from_cpu_views_fn=_materialize_updated_blocks_from_cpu_views,
                    get_double_buffer_gpu_fn=get_double_buffer_gpu,
                )

                if refreshed_omega > 0:
                    _log_delta_refresh(
                        iteration=iteration,
                        refreshed_omega=refreshed_omega,
                        refresh_context='immediate writeback barrier',
                        log_file=log_file,
                    )

                _log_paper_writeback_staged(
                    iteration=iteration,
                    staged_writeback_blocks=staged_writeback_blocks,
                    log_file=log_file,
                )

                if should_log_paper_sets:
                    _log_paper_writeback_finalize(
                        storage_adapter=storage_adapter,
                        iteration=iteration,
                        optimizer_deferred_mode=paper_optimizer_deferred_mode,
                        candidate_blocks=candidate_blocks,
                        staged_writeback_blocks=staged_writeback_blocks,
                        refreshed_omega=refreshed_omega,
                        log_file=log_file,
                    )
            else:
                updated_blocks_dict = {}
                for block_id in updated_block_ids:
                    start_idx = block_id * args.gaussian_block_size
                    end_idx = min(start_idx + args.gaussian_block_size, total_n_gaussians)

                    if is_ssd_offload:
                        xyz_block = original_xyz[start_idx:end_idx].detach()
                        scaling_block = original_scaling[start_idx:end_idx].detach()
                        rotation_block = original_rotation[start_idx:end_idx].detach()
                        opacity_block = original_opacity[start_idx:end_idx].detach()
                        f_dc = original_features_dc[start_idx:end_idx].detach()
                        f_rest = original_features_rest[start_idx:end_idx].detach()

                        block_tensor = torch.cat([
                            xyz_block,
                            scaling_block,
                            rotation_block,
                            opacity_block,
                            f_dc,
                            f_rest
                        ], dim=1)

                        updated_blocks_dict[block_id] = block_tensor

                    elif hasattr(gaussians, 'gpu_working_set_manager') and \
                         gaussians.gpu_working_set_manager.local_to_global_idx is not None:
                        # ================================================================
                        # SSD-backed path: read from GPU working set.
                        # ================================================================
                        global_indices_in_block = torch.arange(start_idx, end_idx, device='cuda')
                        manager = gaussians.gpu_working_set_manager
                        local_ids = manager.global_to_local(global_indices_in_block)
                        valid_mask = local_ids >= 0
                        
                        if valid_mask.any():
                            local_indices = local_ids[valid_mask]
                            
                            # Extract updated parameters from GPU (detach and move to CPU)
                            xyz_block = gaussians._xyz[local_indices].detach().cpu()
                            scaling_block = gaussians._scaling[local_indices].detach().cpu()
                            rotation_block = gaussians._rotation[local_indices].detach().cpu()
                            opacity_block = gaussians._opacity[local_indices].detach().cpu()
                            f_dc = gaussians._features_dc[local_indices].detach().cpu()
                            f_rest = gaussians._features_rest[local_indices].detach().cpu()
                            
                            # Assemble block tensor (59 dims total)
                            block_tensor = torch.cat([
                                xyz_block,       # 3
                                scaling_block,   # 3
                                rotation_block,  # 4
                                opacity_block,   # 1
                                f_dc,            # 3
                                f_rest           # 45
                            ], dim=1)  # Total = 59
                            
                            updated_blocks_dict[block_id] = block_tensor
                        else:
                            # This block wasn't in GPU working set, skip
                            log_file.write(f"[WARNING] Block {block_id} not in GPU working set, skipping writeback\n")
                else:
                    # ================================================================
                    # Archived fallback path; should rarely happen.
                    # ================================================================
                    log_file.write(f"[WARNING] Using RAM fallback for block {block_id}\n")
                    
                    xyz_block = original_xyz[start_idx:end_idx].detach()
                    scaling_block = original_scaling[start_idx:end_idx].detach()
                    rotation_block = original_rotation[start_idx:end_idx].detach()
                    opacity_block = original_opacity[start_idx:end_idx].detach()
                    
                    if hasattr(gaussians, 'parameters_buffer'):
                        shs_ram = gaussians.parameters_buffer[start_idx:end_idx]
                        f_dc = shs_ram[:, :3]
                        f_rest = shs_ram[:, 3:48]
                    else:
                        f_dc = original_features_dc[start_idx:end_idx].detach()
                        f_rest = original_features_rest[start_idx:end_idx].detach()
                    
                    block_tensor = torch.cat([
                        xyz_block, scaling_block, rotation_block, opacity_block,
                        f_dc, f_rest
                    ], dim=1)
                    
                    updated_blocks_dict[block_id] = block_tensor

            if not is_paper_ssd_mode:
                storage_adapter.async_sync_updated_blocks(updated_blocks_dict)

                log_file = utils.get_log_file()
                writeback_message = (
                    f"[SSD WRITEBACK][fast_ram/gpu] Iter {iteration}: Persisted {len(updated_block_ids)} "
                    f"updated blocks to SSD/cache\n"
                )
                log_file.write(writeback_message)

            torch.cuda.nvtx.range_pop()

    if defer_gpu_working_set_clear and hasattr(gaussians, 'gpu_working_set_manager'):
        gaussians.gpu_working_set_manager.clear()
        if is_paper_ssd_mode:
            _log_paper_working_set_cleared(log_file=log_file)

    _ts('stage5_writeback_done')
    # ------------------------------------------------------------------------
    # 5.3: Final synchronization and return
    # ------------------------------------------------------------------------
    torch.cuda.synchronize()
    _ts('stage5_sync_done')
    
    # ============================================================================
    # [MEMORY CLEANUP] Periodic cleanup to prevent memory accumulation
    # ============================================================================
    # Clear intermediate variables to help garbage collection
    if iteration % 100 == 0:
        # Force Python garbage collection periodically
        gc.collect()
        
        # Clear CUDA cache if GPU memory pressure is high
        if torch.cuda.is_available():
            gpu_mem_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
            if gpu_mem_percent > 0.9:
                torch.cuda.empty_cache()
    
    # ============================================================================
    # [STATS] Log double buffer and prefetch statistics periodically
    # ============================================================================
    if iteration % 1000 == 0:
        global _double_buffer_gpu
        _log_double_buffer_stats(
            iteration=iteration,
            double_buffer=_double_buffer_gpu,
            log_file=log_file,
        )
    
    # [PERF PROFILE] Print stage timings
    _ts('iter_end')
    if _perf_log:
        if is_paper_ssd_mode:
            _log_paper_perf_profile(
                iteration=iteration,
                perf_times=_perf_t,
                log_file=log_file,
                storage_adapter=storage_adapter,
            )
        else:
            def _dt(a, b):
                return (_perf_t.get(b, _perf_t['iter_start']) - _perf_t.get(a, _perf_t['iter_start'])) * 1000
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
        log_file.flush()

    return losses, ordered_cams, sparsity


def clm_offload_eval_one_cam(camera, gaussians, background, scene):
    # Prepare parameters.
    xyz_gpu = gaussians.get_xyz
    opacity_gpu_origin = gaussians.get_opacity
    scaling_gpu_origin = gaussians.get_scaling
    rotation_gpu_origin = gaussians.get_rotation

    filters, _, _ = calculate_filters(
        [camera], xyz_gpu, opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
    )

    del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
    this_filter = filters[0]

    filtered_xyz_gpu = torch.gather(
        xyz_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3)
    )
    filtered_opacity_gpu = torch.gather(
        gaussians._opacity, 0, this_filter.reshape(-1, 1)
    )
    filtered_scaling_gpu = torch.gather(
        gaussians._scaling, 0, this_filter.reshape(-1, 1).expand(-1, 3)
    )
    filtered_rotation_gpu = torch.gather(
        gaussians._rotation, 0, this_filter.reshape(-1, 1).expand(-1, 4)
    )

    filtered_opacity_gpu = gaussians.opacity_activation(filtered_opacity_gpu)
    filtered_scaling_gpu = gaussians.scaling_activation(filtered_scaling_gpu)
    filtered_rotation_gpu = gaussians.rotation_activation(filtered_rotation_gpu)

    this_filter_cpu = this_filter.to("cpu")
    filtered_shs_gpu = torch.gather(
        gaussians._parameters, 0, this_filter_cpu.reshape(-1, 1).expand(-1, 48)
    ).to("cuda")

    # Do rendering.
    rendered_image, _, _ = pipeline_forward_one_step(
        filtered_opacity_gpu=filtered_opacity_gpu,
        filtered_scaling_gpu=filtered_scaling_gpu,
        filtered_rotation_gpu=filtered_rotation_gpu,
        filtered_xyz_gpu=filtered_xyz_gpu,
        filtered_shs=filtered_shs_gpu,
        camera=camera,
        scene=scene,
        gaussians=gaussians,
        background=background,
        pipe_args=None,
        eval=True,
    )

    return rendered_image
