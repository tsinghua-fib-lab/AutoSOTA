import os
import sys
import json
import gc
import psutil
from pathlib import Path

# import faulthandler
# faulthandler_log = open("fault.log", "w")
# faulthandler.enable(file=faulthandler_log, all_threads=True)

import torch
import torch.multiprocessing
from torch.cuda import nvtx
from tqdm import tqdm

from utils.mem_monitor import MemMonitor
from argparse import ArgumentParser
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)

from scene import Scene, OffloadSceneDataset
from strategies.tide_engine.gaussian_model import TideGaussianModel
from strategies.tide_engine.runtime import (
    train_tide_batch,
    validate_tide_runtime_args,
)

from utils.general_utils import safe_state, prepare_output_and_logger
import utils.general_utils as utils
from utils.timer import Timer, End2endTimer

from storage.tide_storage_adapter import TideStorageAdapter
from storage.schedule_utils import get_camera_batch_schedule
from storage.pure_ssd_checkpoint import (
    is_pure_ssd_checkpoint,
    load_pure_ssd_checkpoint_manifest,
    write_pure_ssd_incremental_checkpoint,
    write_pure_ssd_snapshot_checkpoint,
)


def _load_pure_ssd_prebuilt_manifest(args):
    manifest_path = getattr(args, "pure_ssd_prebuilt_manifest", "")
    if not manifest_path:
        return None

    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_dir = manifest_path.parent
    base_file = Path(getattr(args, "pure_ssd_prebuilt_base_file", "") or manifest["base_file"])
    block_bounds = Path(getattr(args, "pure_ssd_prebuilt_block_bounds", "") or manifest["block_bounds"])
    if not base_file.is_absolute():
        base_file = manifest_dir / base_file
    if not block_bounds.is_absolute():
        block_bounds = manifest_dir / block_bounds
    base_file = base_file.resolve()
    block_bounds = block_bounds.resolve()
    if not base_file.is_file():
        raise FileNotFoundError(f"Pure SSD prebuilt base_file not found: {base_file}")
    if not block_bounds.is_file():
        raise FileNotFoundError(f"Pure SSD prebuilt block_bounds not found: {block_bounds}")

    total_points = int(manifest["total_points"])
    param_dim = int(manifest.get("param_dim", 59))
    expected_size = total_points * param_dim * 4
    actual_size = base_file.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Pure SSD prebuilt base_file size mismatch: got {actual_size}, expected {expected_size}"
        )

    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["base_file"] = str(base_file)
    manifest["block_bounds"] = str(block_bounds)
    manifest.setdefault("param_dim", param_dim)
    manifest.setdefault("next_iteration", 1)
    manifest["_prebuilt_base_reuse"] = True
    return manifest


def _validate_pure_ssd_runtime(args, gaussians, storage_adapter, resolved_backend, log_file):
    """Fail early if TideGS silently falls back to a RAM-resident state."""
    if not getattr(args, "pure_ssd_offload", False):
        return

    if storage_adapter is None:
        raise RuntimeError("[PURE SSD CHECK] storage_adapter is required")
    if getattr(storage_adapter, "execution_mode", "fast_ram") != "paper":
        raise RuntimeError("[PURE SSD CHECK] storage_adapter.execution_mode must be 'paper'")
    if resolved_backend != "tiered_cache":
        raise RuntimeError(
            f"[PURE SSD CHECK] BlockReader backend must be tiered_cache, got {resolved_backend!r}"
        )
    block_reader = getattr(gaussians, "_block_reader", None)
    if block_reader is None or block_reader.__class__.__name__ != "TieredCacheBlockReader":
        raise RuntimeError(
            "[PURE SSD CHECK] gaussians._block_reader must be a TieredCacheBlockReader"
        )
    if getattr(gaussians, "_unified_params", None) is not None:
        raise RuntimeError("[PURE SSD CHECK] _unified_params must be released")
    if not getattr(gaussians, "_paper_unified_params_freed", False):
        raise RuntimeError("[PURE SSD CHECK] _paper_unified_params_freed marker is missing")
    if getattr(args, "paper_optimizer_backend", "cpu") != "gpu_resident":
        raise RuntimeError("[PURE SSD CHECK] optimizer backend must be gpu_resident")
    if getattr(args, "paper_optimizer_state_mode", "full_cpu") != "resident_blocks":
        raise RuntimeError("[PURE SSD CHECK] optimizer state mode must be resident_blocks")

    total_gaussians = getattr(gaussians, "_paper_unified_params_num_total", None)
    total_desc = f"{int(total_gaussians):,}" if total_gaussians is not None else "unknown"
    num_blocks = getattr(storage_adapter, "num_blocks", None)
    cache = getattr(storage_adapter, "cache", None)
    cache_limit_gb = getattr(cache, "max_ram_bytes", 0) / (1024 ** 3) if cache is not None else 0.0
    init_state = (
        "_unified_params never allocated"
        if getattr(gaussians, "_paper_unified_params_never_allocated", False)
        else "_unified_params released"
    )
    message = (
        "[PURE SSD CHECK] Runtime path verified: "
        f"gaussians={total_desc} blocks={num_blocks} "
        f"block_reader=TieredCacheBlockReader optimizer=GPUResidentAdam "
        f"state=resident_blocks init={init_state} ram_cache_limit={cache_limit_gb:.2f}GB\n"
    )
    log_file.write(message)
    utils.print_rank_0(message.strip())


def training(dataset_args, opt_args, pipe_args, args, log_file):
    """Main training loop for the pure SSD/Tide release path."""

    # ============================================================================
    # STAGE 1: INITIALIZATION
    # ============================================================================

    assert args.dataset_cache_and_stream_mode in [
        "load_from_disk_on_demand"
    ], f"Only load_from_disk_on_demand is supported for now, but got {args.dataset_cache_and_stream_mode}"
    try:
        validate_tide_runtime_args(args)
    except RuntimeError as exc:
        raise ValueError(
            "train_tidegs.py is the pure SSD/Tide release entry. "
            "Use scripts/train_matrixcity_1b.sh or enable the TideGS SSD flags."
        ) from exc
    # ------------------------------------------------------------------------
    # 1.1: Setup auxiliary tools and GPU configuration
    # ------------------------------------------------------------------------
    gc.set_threshold(700, 10, 500)  # gen0, gen1, gen2

    torch.cuda.set_device(args.gpu)
    timers = Timer(args)
    utils.set_timers(timers)
    prepare_output_and_logger(dataset_args)
    utils.log_cpu_memory_usage("at the beginning of training")
    start_from_this_iteration = 1
    pure_ssd_resume_manifest = None
    pure_ssd_prebuilt_manifest = _load_pure_ssd_prebuilt_manifest(args)
    if pure_ssd_prebuilt_manifest is not None and args.start_checkpoint != "":
        raise ValueError(
            "--pure_ssd_prebuilt_manifest is for fresh runs from an existing SSD base; "
            "use --start_checkpoint alone for checkpoint resume."
        )
    if pure_ssd_prebuilt_manifest is not None:
        args._pure_ssd_prebuilt_manifest = pure_ssd_prebuilt_manifest
        args.gaussian_block_size = int(pure_ssd_prebuilt_manifest["block_size"])
        prebuilt_msg = (
            "[PURE SSD PREBUILT] loaded manifest: "
            f"{args.pure_ssd_prebuilt_manifest} "
            f"points={int(pure_ssd_prebuilt_manifest['total_points']):,} "
            f"base={pure_ssd_prebuilt_manifest.get('base_file')}"
        )
        utils.print_rank_0(prebuilt_msg)
        log_file.write(prebuilt_msg + "\n")
    if args.start_checkpoint != "":
        if not is_pure_ssd_checkpoint(args.start_checkpoint):
            raise ValueError(
                "train_tidegs.py only resumes pure SSD checkpoints."
            )
        pure_ssd_resume_manifest = load_pure_ssd_checkpoint_manifest(args.start_checkpoint)
        args._pure_ssd_resume_manifest = pure_ssd_resume_manifest
        start_from_this_iteration = int(pure_ssd_resume_manifest["next_iteration"])
        args.gaussian_block_size = int(pure_ssd_resume_manifest["block_size"])
        resume_msg = (
            "[PURE SSD RESUME] loaded checkpoint: "
            f"{args.start_checkpoint} next_iteration={start_from_this_iteration} "
            f"base={pure_ssd_resume_manifest.get('base_file')}"
        )
        utils.print_rank_0(resume_msg)
        log_file.write(resume_msg + "\n")

    # Configure multiprocessing sharing strategy if needed
    if args.sharing_strategy != "default":
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)

    # ------------------------------------------------------------------------
    # 1.2: Initialize pure SSD Gaussian shell
    # ------------------------------------------------------------------------
    gaussians = TideGaussianModel(sh_degree=dataset_args.sh_degree)
    utils.print_rank_0("Using TideGaussianModel for TideGS/SSD")
    log_file.write("Using TideGaussianModel for TideGS/SSD\n")

    storage_adapter = None
    ssd_training_schedule = None

    with torch.no_grad():
        scene = Scene(args, gaussians)
        utils.print_rank_0("[SSD] Initializing Tide storage engine...")
        storage_adapter = TideStorageAdapter(
            gaussians=gaussians,
            cameras=scene.getTrainCamerasInfo(),
            storage_dir=args.ssd_cache_dir,
            block_size=args.gaussian_block_size,
            max_ram_gb=args.max_ram_gb,
            num_clusters=args.num_clusters,
            use_6plane=args.use_6plane,
            execution_mode=args.ssd_execution_mode,
        )

        log_file.write(f"[SSD] Execution mode: {args.ssd_execution_mode}\n")

        ssd_schedule_ordering = getattr(args, "ssd_schedule_ordering", "trajectory")
        ssd_schedule_shuffle = ssd_schedule_ordering == "shuffle"
        ssd_training_schedule = storage_adapter.get_training_schedule(shuffle=ssd_schedule_shuffle)
        utils.print_rank_0(f"[SSD] Schedule ordering: {ssd_schedule_ordering}")
        log_file.write(f"[SSD] Schedule ordering: {ssd_schedule_ordering}\n")

        gaussians.offload_params_to_ssd_storage()
        utils.print_rank_0("[PURE SSD] Training will use the SSD → RAM → GPU pipeline")


        gaussians.training_setup(opt_args)

        from storage.block_reader import TieredCacheBlockReader, resolve_block_reader_backend

        requested_backend = args.paper_block_reader_backend
        resolved_backend = resolve_block_reader_backend(
            requested_backend,
            getattr(storage_adapter, 'execution_mode', 'paper'),
        )
        if resolved_backend != 'tiered_cache':
            raise RuntimeError(
                "train_tidegs.py requires paper_block_reader_backend=tiered_cache "
                f"(resolved {resolved_backend!r})"
            )

        total_gaussians = getattr(gaussians, '_paper_unified_params_num_total', None)
        if total_gaussians is None:
            total_gaussians = getattr(storage_adapter, 'num_points', None)
        gaussians._block_reader = TieredCacheBlockReader(
            cache_manager=storage_adapter.cache,
            total_gaussians=int(total_gaussians),
            block_size=int(args.gaussian_block_size),
        )

        utils.print_rank_0(
            f"[SSD] BlockReader backend = {resolved_backend} "
            f"(requested={requested_backend}, ssd_execution_mode={storage_adapter.execution_mode})"
        )
        log_file.write(
            f"[SSD] BlockReader backend = {resolved_backend} "
            f"(requested={requested_backend}, ssd_execution_mode={storage_adapter.execution_mode})\n"
        )

        gaussians.free_unified_params()
        utils.print_rank_0(
            "[SSD] ✓ Paper mode: _unified_params released; "
            "reads served by TieredCacheBlockReader, writeback via direct GPU→cache"
        )

        _validate_pure_ssd_runtime(args, gaussians, storage_adapter, resolved_backend, log_file)

        if pure_ssd_resume_manifest is not None:
            msg = "[PURE SSD RESUME] GPUResidentAdam cold-started; Adam moments are not restored"
            utils.print_rank_0(msg)
            log_file.write(msg + "\n")

        scene.log_scene_info_to_file(log_file, "Scene Info Before Training")
    utils.check_initial_gpu_memory_usage("after init and before training loop")

    # ------------------------------------------------------------------------
    # 1.3: Initialize data loader
    # ------------------------------------------------------------------------
    train_dataset = OffloadSceneDataset(scene.getTrainCamerasInfo())

    # ------------------------------------------------------------------------
    # 1.4: Initialize background and CUDA streams
    # ------------------------------------------------------------------------
    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else None

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Dedicated stream for CPU↔GPU communication (overlapped with compute)
    comm_stream = torch.cuda.Stream(device=args.gpu, priority=args.comm_stream_priority)

    # ------------------------------------------------------------------------
    # 1.5: Initialize training loop state
    # ------------------------------------------------------------------------
    end2end_timers = End2endTimer(args)
    end2end_timers.start()
    progress_bar = tqdm(
        range(1, opt_args.iterations + 1),
        desc="Training progress",
    )
    progress_bar.update(start_from_this_iteration - 1)
    num_trained_batches = 0

    mem_mon = MemMonitor(log_dir=args.model_path, warn_avail_gb=15.0)

    # Random number generator for camera ordering in retention-based offloading
    perm_generator = torch.Generator(device="cuda")
    perm_generator.manual_seed(1)

    # Training state variables
    ema_loss_for_log = 0
    last_iteration = None

    optimizer_churn_tsv = None
    optimizer_churn_state = {
        'current_epoch': None,
        'last_optimizer_rows_total': 0,
        'last_cold_rows_total': 0,
        'last_cold_restarts_total': 0,
        'last_state_evictions_total': 0,
    }

    def _snapshot_optimizer_churn_counters():
        gpu_resident_optimizer = getattr(gaussians, '_paper_gpu_resident_optimizer', None)
        if gpu_resident_optimizer is None:
            return None
        stats = gpu_resident_optimizer.get_stats()
        if stats.get('state_mode', 'full_cpu') != 'resident_blocks':
            return None
        return {
            'optimizer_rows_touched_total': int(stats.get('optimizer_rows_touched_total', 0)),
            'cold_restarted_rows_touched_total': int(stats.get('cold_restarted_rows_touched_total', 0)),
            'cold_restarts_total': int(stats.get('cold_restarts', 0)),
            'state_evictions_total': int(stats.get('state_evictions', 0)),
            'mean_resident_streak': float(stats.get('mean_resident_streak', 0.0)),
        }

    def _write_optimizer_churn_epoch(epoch_zero_based: int, iteration_end: int):
        if optimizer_churn_tsv is None:
            return
        stats = _snapshot_optimizer_churn_counters()
        if stats is None:
            return

        epoch_optimizer_rows = max(0, stats['optimizer_rows_touched_total'] - optimizer_churn_state['last_optimizer_rows_total'])
        epoch_cold_rows = max(0, stats['cold_restarted_rows_touched_total'] - optimizer_churn_state['last_cold_rows_total'])
        epoch_cold_restarts = max(0, stats['cold_restarts_total'] - optimizer_churn_state['last_cold_restarts_total'])
        epoch_state_evictions = max(0, stats['state_evictions_total'] - optimizer_churn_state['last_state_evictions_total'])
        epoch_cold_ratio_pct = 100.0 * epoch_cold_rows / max(1, epoch_optimizer_rows)

        optimizer_churn_tsv.write(
            f"{epoch_zero_based + 1}\t{iteration_end}\t{epoch_optimizer_rows}\t{epoch_cold_rows}\t"
            f"{epoch_cold_ratio_pct:.6f}\t{epoch_cold_restarts}\t{epoch_state_evictions}\t"
            f"{stats['mean_resident_streak']:.6f}\n"
        )
        log_file.write(
            f"[OPTIMIZER CHURN] Epoch {epoch_zero_based + 1}: rows={epoch_optimizer_rows}, "
            f"cold_rows={epoch_cold_rows}, cold_ratio={epoch_cold_ratio_pct:.4f}%, "
            f"cold_restarts={epoch_cold_restarts}, state_evictions={epoch_state_evictions}, "
            f"mean_streak={stats['mean_resident_streak']:.4f}\n"
        )

        optimizer_churn_state['last_optimizer_rows_total'] = stats['optimizer_rows_touched_total']
        optimizer_churn_state['last_cold_rows_total'] = stats['cold_restarted_rows_touched_total']
        optimizer_churn_state['last_cold_restarts_total'] = stats['cold_restarts_total']
        optimizer_churn_state['last_state_evictions_total'] = stats['state_evictions_total']

    def _enable_optimizer_churn_logging(reason: str):
        nonlocal optimizer_churn_tsv
        if optimizer_churn_tsv is not None:
            return
        churn_tsv_path = os.path.join(args.model_path, 'optimizer_state_churn_by_epoch.tsv')
        optimizer_churn_tsv = open(churn_tsv_path, 'w', buffering=1)
        optimizer_churn_tsv.write(
            'epoch\titeration_end\toptimizer_rows_updated\tcold_restarted_rows_updated\t'
            'cold_restarted_row_ratio_pct\tcold_restarts\tstate_evictions\tmean_resident_streak\n'
        )
        log_file.write(f"[OPTIMIZER CHURN] Per-epoch churn logging enabled: {churn_tsv_path} ({reason})\n")
        log_file.flush()

    gaussians._paper_optimizer_state_mode = str(getattr(args, 'paper_optimizer_state_mode', 'resident_blocks')).lower()
    gaussians._paper_optimizer_block_size = int(getattr(args, 'gaussian_block_size', 4096))
    gaussians._paper_optimizer_backend = 'gpu_resident'
    _enable_optimizer_churn_logging("pure_ssd_gpu_resident")
    # ============================================================================
    # STAGE 2: MAIN TRAINING LOOP
    # ============================================================================

    for iteration in range(
        start_from_this_iteration, opt_args.iterations + 1, args.bsz
    ):
        # # rewrite the checking iterations
        # ------------------------------------------------------------------------
        # 2.1: Iteration setup and profiling
        # ------------------------------------------------------------------------
        # Optional: trace CUDA memory usage for debugging
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (
                iteration % args.densification_interval
            ) == 0:
                torch.cuda.memory._record_memory_history()
                log_file.write(
                    "[ITER {}] Tracing cuda memory usage.\n".format(iteration)
                )

        # Update progress bar and iteration state
        if iteration // args.bsz % 30 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
        progress_bar.update(args.bsz)
        utils.set_cur_iter(iteration)
        gaussians.update_learning_rate(iteration)  # Learning rate scheduling
        num_trained_batches += 1
        last_iteration = iteration

        # Optional: reset memory tracking for per-iteration profiling
        if args.reset_each_iter:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()

        # Start timing this iteration
        timers.clear()
        timers.start("[iteration end2end]")

        # Optional: NSight Systems profiling
        if args.nsys_profile:
            if iteration == args.nsys_profile_start_iter:
                torch.cuda.cudart().cudaProfilerStart()
            if (
                iteration == args.nsys_profile_end_iter
                or iteration == opt_args.iterations
            ):
                torch.cuda.cudart().cudaProfilerStop()
            if (
                iteration >= args.nsys_profile_start_iter
                and iteration < args.nsys_profile_end_iter
            ):
                nvtx.range_push(f"iteration[{iteration},{iteration+args.bsz})")

        # Gradually increase spherical harmonics degree (every 1000 iterations)
        if utils.check_update_at_this_iter(iteration, args.bsz, 1000, 0):
            gaussians.oneupSHdegree()

        # ------------------------------------------------------------------------
        # 2.2: Load training data (camera images)
        # ------------------------------------------------------------------------
        timers.start("dataloader: load the next image from disk and decode")

        if (
            optimizer_churn_tsv is None
            and getattr(args, 'pure_ssd_offload', False)
            and str(getattr(args, 'paper_optimizer_backend', '')).lower() == 'gpu_resident'
            and str(getattr(args, 'paper_optimizer_state_mode', '')).lower() == 'resident_blocks'
        ):
            _enable_optimizer_churn_logging("pure_ssd_gpu_resident")

        schedule_info = get_camera_batch_schedule(
            training_schedule=ssd_training_schedule,
            iteration=iteration,
            batch_size=args.bsz,
            schedule_ordering=getattr(args, "ssd_schedule_ordering", "trajectory"),
        )
        batch_indices = schedule_info.batch_indices
        epoch = schedule_info.epoch
        within_epoch_idx = schedule_info.within_epoch_idx
        n_batches = schedule_info.num_batches
        epoch_camera_offset = schedule_info.epoch_camera_offset

        if optimizer_churn_tsv is not None:
            if optimizer_churn_state['current_epoch'] is None:
                optimizer_churn_state['current_epoch'] = epoch
            elif epoch != optimizer_churn_state['current_epoch']:
                _write_optimizer_churn_epoch(
                    epoch_zero_based=optimizer_churn_state['current_epoch'],
                    iteration_end=max(start_from_this_iteration, iteration - args.bsz),
                )
                optimizer_churn_state['current_epoch'] = epoch

        batched_cameras = [train_dataset[idx] for idx in batch_indices]

        if iteration % 100 == 0:
            log_file.write(
                f"[SSD Schedule] Iter {iteration}: epoch={epoch}, "
                f"within_epoch_batch={within_epoch_idx}/{n_batches}, "
                f"camera_offset={epoch_camera_offset}, "
                f"Camera indices: {batch_indices[:3]}...{batch_indices[-1]}\n"
            )

        timers.stop("dataloader: load the next image from disk and decode")

        for uid, (c, global_idx) in enumerate(zip(batched_cameras, batch_indices)):
            c.uid = uid
            c.global_idx = global_idx

        # ------------------------------------------------------------------------
        # 2.3: Transfer camera matrices to GPU
        # ------------------------------------------------------------------------
        timers.start("send cam matrices to gpu")
        # Transfer world-view and projection transforms
        for camera in batched_cameras:
            camera.world_view_transform = camera.world_view_transform.cuda()
            camera.full_proj_transform = camera.full_proj_transform.cuda()

        # Create camera intrinsics (K matrix) and compute camera-to-world transforms
        batched_world_view_transform = []
        for camera in batched_cameras:
            camera.K = camera.create_k_on_gpu()
            batched_world_view_transform.append(
                camera.world_view_transform.transpose(0, 1)
            )

        # Batch process: compute inverse transforms for all cameras
        batched_world_view_transform = torch.stack(batched_world_view_transform)
        batched_world_view_transform_inverse = torch.inverse( # torch.tensor, (4, 4, 4)
            batched_world_view_transform
        )
        batched_world_view_transform_inverse = torch.unbind( # tuple
            batched_world_view_transform_inverse, dim=0
        )

        # Store camera-to-world transforms (for view direction computation)
        for camera, wvt in zip(batched_cameras, batched_world_view_transform_inverse):
            camera.camtoworlds = wvt.unsqueeze(0)
        # TODO: maybe we can save them on GPU during initialization. After all, they do not take up lots of memory.
        timers.stop("send cam matrices to gpu")

        # ------------------------------------------------------------------------
        # 2.4: Load ground-truth images to GPU
        # ------------------------------------------------------------------------
        with torch.no_grad():
            timers.start("load_cameras")
            for camera in batched_cameras:
                camera.original_image = camera.original_image_backup.cuda()
            timers.stop("load_cameras")
        assert args.bsz > 1, "Pipelined offload requires batch size > 1"
        losses, ordered_cams, sparsity = train_tide_batch(
            gaussians=gaussians,
            scene=scene,
            batched_cameras=batched_cameras,
            parameters_grad_buffer=gaussians.parameters_grad_buffer,
            background=background,
            pipe_args=pipe_args,
            comm_stream=comm_stream,
            perm_generator=perm_generator,
            storage_adapter=storage_adapter,
            training_schedule=ssd_training_schedule,
        )

        mem_mon.tick(iteration)

        if len(losses) == 0:
            log_file.write(
                f"[WARNING] Iteration {iteration}: All {len(batched_cameras)} cameras see no Gaussians; "
                "skipping optimizer step.\n"
            )
            continue

        batched_cameras = [batched_cameras[i] for i in ordered_cams]

        timers.start("sync_loss_and_log")
        batched_losses = torch.stack(losses)
        batched_loss_cpu = batched_losses.cpu().numpy()

        ema_loss_for_log = (
            batched_loss_cpu.mean()
            if ema_loss_for_log is None
            else 0.6 * ema_loss_for_log + 0.4 * batched_loss_cpu.mean()
        )

        train_dataset.update_losses(batched_loss_cpu)

        batched_loss_cpu = [round(loss, 6) for loss in batched_loss_cpu]
        log_file.write(
            "iteration[{},{}), loss: {} sparsity: {} image: {}\n".format(
                iteration,
                iteration + args.bsz,
                batched_loss_cpu,
                sparsity,
                [viewpoint_cam.image_name for viewpoint_cam in batched_cameras],
            )
        )

        with torch.no_grad():
            if any(
                [
                    iteration <= save_iteration < iteration + args.bsz
                    for save_iteration in args.save_iterations
                ]
            ):
                utils.print_rank_0("\n[ITER {}] Saving End2end".format(iteration))
                end2end_timers.stop()
                end2end_timers.print_time(log_file, iteration + args.bsz)

                skip_msg = (
                    f"[ITER {iteration}] SKIP model save: pure SSD release "
                    "keeps the full parameter table out-of-core. Use "
                    "--checkpoint_iterations for resumable state, or "
                    "tools/export_pure_ssd_checkpoint_to_ply.py for PLY preview/export.\n"
                )
                utils.print_rank_0(skip_msg.rstrip())
                log_file.write(skip_msg)

                end2end_timers.start()

            # ------------------------------------------------------------------------
            # 2.10: Save training checkpoint (for resuming)
            # ------------------------------------------------------------------------
            if any(
                [
                    iteration <= checkpoint_iteration < iteration + args.bsz
                    for checkpoint_iteration in args.checkpoint_iterations
                ]
            ):
                end2end_timers.stop()
                matched_checkpoint_iterations = [
                    checkpoint_iteration
                    for checkpoint_iteration in args.checkpoint_iterations
                    if iteration <= checkpoint_iteration < iteration + args.bsz
                ]
                checkpoint_iteration = matched_checkpoint_iterations[-1]
                save_folder = os.path.join(
                    scene.model_path,
                    "checkpoints",
                    str(checkpoint_iteration),
                )
                utils.print_rank_0(
                    f"\n[ITER {iteration}] Saving Pure SSD Checkpoint {checkpoint_iteration}"
                )
                log_file.write(
                    f"[ITER {iteration}] Saving Pure SSD Checkpoint {checkpoint_iteration}\n"
                )
                pure_ssd_checkpoint_mode = str(
                    getattr(args, "pure_ssd_checkpoint_mode", "incremental")
                ).lower()
                if pure_ssd_checkpoint_mode == "snapshot":
                    write_pure_ssd_snapshot_checkpoint(
                        storage_adapter=storage_adapter,
                        gaussians=gaussians,
                        checkpoint_dir=save_folder,
                        iteration=checkpoint_iteration,
                        next_iteration=iteration + args.bsz,
                        args=args,
                        chunk_blocks=getattr(args, "pure_ssd_checkpoint_chunk_blocks", 256),
                        log_file=log_file,
                    )
                else:
                    write_pure_ssd_incremental_checkpoint(
                        storage_adapter=storage_adapter,
                        gaussians=gaussians,
                        checkpoint_dir=save_folder,
                        iteration=checkpoint_iteration,
                        next_iteration=iteration + args.bsz,
                        args=args,
                        log_file=log_file,
                    )
                end2end_timers.start()

        # ------------------------------------------------------------------------
        # 2.12: Iteration cleanup
        # ------------------------------------------------------------------------
        torch.cuda.synchronize()  # Ensure all GPU operations are complete

        # Release camera image memory
        for viewpoint_cam in batched_cameras:
            viewpoint_cam.original_image = None

        # End profiling range if active
        if args.nsys_profile:
            if (
                iteration >= args.nsys_profile_start_iter
                and iteration < args.nsys_profile_end_iter
            ):
                nvtx.range_pop()

        # Print timing statistics
        if utils.check_enable_python_timer():
            timers.stop("[iteration end2end]")
            timers.printTimers(iteration, mode="sum")

        # Dump CUDA memory trace if enabled
        if args.trace_cuda_mem:
            if (iteration % args.log_interval) == 1 or (
                iteration % args.densification_interval
            ) == 0:
                dump_name = args.model_path + f"/trace_dump/iter={iteration}"
                torch.cuda.memory._dump_snapshot(filename=dump_name)
                torch.cuda.memory._record_memory_history(enabled=None)

        utils.memory_report("at the end of the iteration")
        log_file.flush()

    if optimizer_churn_tsv is not None and optimizer_churn_state['current_epoch'] is not None:
        _write_optimizer_churn_epoch(
            epoch_zero_based=optimizer_churn_state['current_epoch'],
            iteration_end=last_iteration if last_iteration is not None else opt_args.iterations,
        )
        optimizer_churn_tsv.close()
        optimizer_churn_tsv = None

    # ============================================================================
    # STAGE 3: POST-TRAINING CLEANUP AND REPORTING
    # ============================================================================

    # Clean up CUDA resources
    del comm_stream

    # Print final timing statistics
    if opt_args.iterations not in args.save_iterations:
        end2end_timers.print_time(log_file, opt_args.iterations)

    # Log peak memory usage
    log_file.write(
        "Max Memory usage: {} GB.\n".format(
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        )
    )

    # Close progress bar and clean up scene
    progress_bar.close()
    mem_mon.close()

    if storage_adapter is not None:
        storage_adapter.shutdown()

    scene.clean_up()

    # Stop profiler if active
    if args.nsys_profile:
        torch.cuda.cudart().cudaProfilerStop()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    args = parser.parse_args(sys.argv[1:])

    init_args(args)

    args = utils.get_args()

    # create log folder
    os.makedirs(args.log_folder, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    with open(args.log_folder + "/args.json", "w") as f:
        json.dump(vars(args), f)

    # create cuda trace dump folder
    if args.trace_cuda_mem:
        os.makedirs(os.path.join(args.model_path, "trace_dump"))

    # Initialize log file and print all args
    log_file = open(
        args.log_folder + "/python.log",
        "a" if args.auto_start_checkpoint else "w",
    )
    utils.set_log_file(log_file)

    # Initialize system state (RNG). In quiet mode regular stdout is mirrored to
    # python.log, while tqdm progress bars still use stderr for terminal progress.
    safe_state(args.quiet, log_file=log_file)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print_all_args(args, log_file)

    p = psutil.Process()
    log_file.write(
        f"Initial pinned memory: {p.memory_info().shared / 1024 / 1024 / 1024} GB\n"
    )

    training(lp.extract(args), op.extract(args), pp.extract(args), args, log_file)

    # All done
    utils.print_rank_0("\nTraining complete.")
    log_file.flush()
    log_file.close()
