#
# Copyright (C) 2023, Inria GRAPHDECO research group
# Copyright (C) 2025, New York University
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE file.
#
# Original 3D Gaussian Splatting code from:
# https://github.com/graphdeco-inria/gaussian-splatting
#
# CLM-GS modifications by NYU Systems Group
# https://github.com/nyu-systems/CLM-GS
#

from argparse import ArgumentParser, Namespace
import sys
import os
import utils.general_utils as utils


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    type_to_use = int
                    if len(value) > 0:
                        type_to_use = type(value[0])
                    group.add_argument(
                        "--" + key, default=value, nargs="+", type=type_to_use
                    )
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class AuxiliaryParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        # ====================================================================
        # OFFLOADING CONFIGURATION
        # ====================================================================
        # TideGS release runs use pure SSD out-of-core training.  The older
        # no/naive flags remain in the parser for compatibility with existing
        # config files.
        # ====================================================================

        # --- No Offload (GPU Baseline) ---
        self.no_offload = False  # Enable no offload mode

        # --- NAIVE Offload ---
        self.naive_offload = False  # Enable naive offload mode

        # --- TideGS offload strategy flag ---
        self.clm_offload = (
            False  # Internal strategy switch used by the TideGS release path
        )
        self.prealloc_capacity = -1  # Pre-allocated capacity for parameters
        # --- Offload advanced flags (usually left unchanged) ---
        self.comm_stream_priority = (
            -1
        )  # by default, use -1 as the priority of the stream
        self.grid_size_H = (
            32  # Grid size for height dimension (used in filtering/spatial hashing)
        )
        self.grid_size_D = 128  # Grid size for depth dimension
        self.reorder_by_min_sparsity_at_end = (
            True  # Reorder Gaussians by minimum sparsity at end of training
        )

        # --- Shared Offload Optimization Flags (used by both modes) ---
        self.sparse_adam = (
            False  # Use sparse Adam optimizer (only update visible Gaussians)
        )

        # ====================================================================
        # MEMORY MANAGEMENT
        # ====================================================================

        # ====================================================================
        # DATA LOADING & DATASET
        # ====================================================================
        self.dataset_cache_and_stream_mode = (
            "load_from_disk_on_demand"  # or "load_from_cpuram_on_demand"
        )
        self.decode_dataset_path = ""  # Path for decoded dataset storage
        self.multiprocesses_decode_dataset_to_disk = True
        self.num_workers = 0  # Number of worker threads for data loading
        self.sharing_strategy = "default"  # PyTorch multiprocessing sharing strategy: "default" ("file_descriptor"), or "file_system"
        self.llffhold = 8  # LLFF dataset hold-out value

        self.initial_point_cloud_downsampled_ratio = 1.0
        # ====================================================================
        # MODEL I/O
        # ====================================================================
        self.load_ply_path = ""  # Path to load pre-trained PLY file
        self.load_ply_max = 1_000_000  # Maximum number of points to load from PLY
        self.load_pt_path = ""  # Path to load PyTorch checkpoint
        self.dense_ply_file = ""  # Path to dense PLY file
        self.start_checkpoint = ""  # Checkpoint to resume training from
        self.auto_start_checkpoint = (
            False  # Automatically find and load latest checkpoint
        )
        self.pure_ssd_checkpoint_chunk_blocks = 256  # Blocks per chunk when writing pure SSD snapshot checkpoints
        self.pure_ssd_checkpoint_mode = "incremental"  # incremental by default; snapshot is debug/fallback
        self.pure_ssd_prebuilt_manifest = ""  # Reuse an existing streaming_init_manifest.json and skip PLY streaming init
        self.pure_ssd_prebuilt_base_file = ""  # Optional override for the manifest base_file path
        self.pure_ssd_prebuilt_block_bounds = ""  # Optional override for the manifest block_bounds path

        # ====================================================================
        # LOGGING & MONITORING
        # ====================================================================
        self.log_folder = "/tmp/gaussian_splatting"  # Folder for logging outputs
        self.log_interval = 250  # Iteration interval for logging
        self.quiet = False  # Suppress verbose output

        # ====================================================================
        # TRAINING CONTROL & EVALUATION
        # ====================================================================
        self.test_iterations = [
            7_000,
            30_000,
        ]  # Iterations at which to run test evaluation
        self.save_iterations = []  # Iterations at which to save model
        self.checkpoint_iterations = []  # Iterations at which to save checkpoints

        # ====================================================================
        # DEBUGGING & PROFILING
        # ====================================================================
        self.debug_from = -1  # Start debugging from this iteration
        self.detect_anomaly = False  # Enable PyTorch anomaly detection

        # ====================================================================
        # HARDWARE & DEVICE
        # ====================================================================
        self.gpu = 0  # GPU device ID to use

        # ====================================================================
        # DATASET-SPECIFIC FLAGS
        # ====================================================================
        self.matrixcity_ocean_mask = (
            False  # Enable ocean masking for MatrixCity dataset
        )

        # ====================================================================
        # EXPERIMENTAL / ADVANCED FLAGS
        # ====================================================================
        self.packed = False  # Use packed representation

        self.use_block_scheduler = False
        self.camera_block_size = 50
        self.visualize_clustering = False  # Generate K-Means clustering visualization

        self.num_save_images_during_eval = 0

        self.use_ssd_offload = False
        self.pure_ssd_offload = False  # Strict paper path: SSD is the full-scene source of truth
        self.pure_ssd_init_backend = "auto"  # {auto, streaming, inmemory}; pure SSD PLY initialization backend
        self.ssd_cache_dir = "./output/ssd_cache"
        self.gaussian_block_size = 4096
        self.max_ram_gb = 32.0
        self.num_clusters = 64
        self.visualize_ssd_schedule = False  # Generate TSP schedule visualization
        self.pure_ssd_schedule_cache_dir = ""  # Optional persistent cache for pure SSD camera TSP schedules
        self.pure_ssd_disable_schedule_cache = False  # Disable pure SSD camera schedule cache
        self.enable_hotspot_retention = True  # Enable GPU hotspot retention to reduce RAM→GPU bandwidth
        self.ssd_execution_mode = "fast_ram"  # {fast_ram, paper}; TideGS routes reads through the SSD→RAM cache path
        self.paper_optimizer_deferred_mode = "off"  # {off, same_iter, cross_iter}; optimizer/writeback defer mode
        self.paper_resident_selection_policy = "topc_balanced"  # {passthrough_active_set, topc_strict, topc_balanced}; resident-set selection policy
        self.paper_resident_lambda = 0.3  # Eq.(5) mixing weight for next-step usefulness vs. recency
        self.paper_resident_recency_decay = 0.95  # multiplicative aging factor for resident recency
        self.paper_balanced_seed_fraction = 0.25  # topc_balanced camera-seed capacity fraction
        self.paper_resident_capacity_blocks = 2048  # resident capacity in blocks
        self.paper_optimizer_state_mode = "full_cpu"  # {full_cpu, resident_blocks}; optimizer-state placement
        self.paper_optimizer_backend = "cpu"  # {cpu, gpu_resident}; optimizer update backend
        self.paper_block_reader_backend = "auto"  # {auto, unified_params, tiered_cache}; source for per-iteration block reads
        self.paper_free_unified_params = False  # release _unified_params after init to unlock >100GB scenes; requires paper_block_reader_backend != unified_params and paper_optimizer_backend=gpu_resident
        self.paper_debug_logging = False  # Enable verbose TideGS diagnostics for development runs
        # Public TideGS aliases. These map onto the internal paper_* names for
        # checkpoint and args.json compatibility.
        self.tide_optimizer_deferred_mode = ""
        self.tide_resident_selection_policy = ""
        self.tide_resident_lambda = ""
        self.tide_resident_recency_decay = ""
        self.tide_balanced_seed_fraction = ""
        self.tide_resident_capacity_blocks = ""
        self.tide_optimizer_state_mode = ""
        self.tide_optimizer_backend = ""
        self.tide_block_reader_backend = ""
        self.tide_free_unified_params = False
        self.tide_debug_logging = False
        self.pure_ssd_max_inmemory_init_points = 50_000_000  # Refuse accidental full PLY init above this many points in pure SSD mode (-1 disables)
        self.pure_ssd_bucket_bits = 10  # Initial Morton bucket bits for streaming PLY init; increase for billion-scale skewed PLYs
        self.pure_ssd_sort_memory_mb = 512.0  # Max RAM per streaming PLY Morton bucket sort before recursive split

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        return g


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.radius_clip = 0.0
        self._source_path = ""
        self._model_path = "/tmp/gaussian_splatting"
        self._images = "images"
        self._white_background = False
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.lr_scale_loss = 1.0
        self.lr_scale_pos_and_scale = 1.0
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.disable_auto_densification = False
        self.random_background = False
        self.min_opacity = 0.005
        self.lr_scale_mode = "sqrt"  # can be "linear", "sqrt", or "accumu"

        # Dataset and Model save
        self.bsz = 1  # batch size.
        self.multiprocesses_image_loading = False  # Disable multiprocess image loading by default to avoid out of shared memory
        self.num_train_cameras = -1
        self.num_test_cameras = -1

        self.exact_filter = True

        self.max_num_images_to_evaluate = int(
            1e9
        )  # maximum number of images to evaluate during testing.
        super().__init__(parser, "Optimization Parameters")


class BenchmarkParams(ParamGroup):
    def __init__(self, parser):
        self.enable_timer = False  # Log running time from python side.
        self.end2end_time = True  # Log end2end training time.
        self.check_gpu_memory = False  # check gpu memory usage.
        self.check_cpu_memory = False  # check cpu memory usage.
        self.log_memory_summary = False
        self.trace_cuda_mem = False
        self.log_block_visibility = False  # Log block-level visibility statistics (SSD offload mode)
        self.debug_frustum = False  # Enable frustum culling visualization for debugging camera orientation
        self.analyze_block_bounds = False  # Analyze block bounding box size distribution after Morton sorting
        self.use_hierarchical_morton = False  # Use hierarchical Morton code for better spatial locality in sparse scenes
        self.use_6plane = True  # Use CPU 6-plane block-wise frustum culling.
        self.projection_max_cameras_per_chunk = -1  # -1 = auto; positive value forces camera chunking inside fully_fused_projection

        super().__init__(parser, "Benchmark Parameters")


class DebugParams(ParamGroup):
    def __init__(self, parser):
        self.stop_update_param = (
            False  # stop updating parameters. No optimizer.step() will be called.
        )
        self.time_image_loading = False  # Log image loading time.

        self.nsys_profile = False  # profile with nsys.
        self.nsys_profile_start_iter = 1  # profile with nsys start iteration.
        self.nsys_profile_end_iter = 1000000  # profile with nsys end iteration.
        self.drop_initial_3dgs_p = 0.0  # profile with nsys.
        self.drop_duplicate_gaussians_coeff = 1.0
        self.do_not_save = False  # Do not save model
        self.reset_each_iter = False  # Reset max memory for  each iteration
        self.save_tensors = False  # Save model parameters as .pt file
        self.debug_max_train_cameras = -1  # Limit train cameras for fast smoke/debug runs
        self.debug_max_test_cameras = -1  # Limit test cameras for fast smoke/debug runs
        self.debug_camera_sample_mode = "linspace"  # {linspace, contiguous, window}; debug camera subset policy
        self.debug_camera_sample_start = 0  # Start index for debug_camera_sample_mode=window
        self.debug_fast_init_scales = False  # Use approximate scale init to speed up billion-scale smoke runs

        self.reinit_ply = False

        super().__init__(parser, "Debug Parameters")


def get_combined_args(parser: ArgumentParser, auto_find_cfg_args_path=False):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        if auto_find_cfg_args_path:
            if hasattr(args_cmdline, "load_ply_path"):
                path = args_cmdline.load_ply_path
                while not os.path.exists(
                    os.path.join(path, "cfg_args")
                ) and os.path.exists(path):
                    path = os.path.join(path, "..")
                cfgfilepath = os.path.join(path, "cfg_args")
        else:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def print_all_args(args, log_file):
    # print all arguments in a readable format, each argument in a line.
    log_file.write("arguments:\n")
    log_file.write("-" * 30 + "\n")
    for arg in vars(args):
        log_file.write("{}: {}\n".format(arg, getattr(args, arg)))
    log_file.write("-" * 30 + "\n\n")
    log_file.write("bsz: " + str(args.bsz) + "\n")


def find_latest_checkpoint(log_folder):
    checkpoint_folder = os.path.join(log_folder, "checkpoints")
    if os.path.exists(checkpoint_folder):
        all_sub_folders = os.listdir(checkpoint_folder)
        valid_checkpoints = []
        for folder_name in all_sub_folders:
            try:
                iteration = int(folder_name)
            except ValueError:
                continue
            folder_path = os.path.join(checkpoint_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue
            has_pure_ssd_manifest = os.path.exists(
                os.path.join(folder_path, "pure_ssd_checkpoint.json")
            )
            has_legacy_pth = any(
                file_name.endswith(".pth") for file_name in os.listdir(folder_path)
            )
            if has_pure_ssd_manifest or has_legacy_pth:
                valid_checkpoints.append((iteration, folder_path))
        if valid_checkpoints:
            valid_checkpoints.sort(key=lambda x: x[0], reverse=True)
            return valid_checkpoints[0][1]
    return ""


def _apply_pure_ssd_release_defaults(args):
    """Make --pure_ssd_offload select the TideGS release path by default."""
    if not getattr(args, "pure_ssd_offload", False):
        return

    args.use_ssd_offload = True
    if not any(
        getattr(args, name, False)
        for name in ("clm_offload", "naive_offload", "no_offload")
    ):
        args.clm_offload = True

    if str(getattr(args, "pure_ssd_init_backend", "auto")).lower() == "auto":
        args.pure_ssd_init_backend = "streaming"
    if str(getattr(args, "ssd_execution_mode", "fast_ram")).lower() == "fast_ram":
        args.ssd_execution_mode = "paper"
    if str(getattr(args, "paper_block_reader_backend", "auto")).lower() == "auto":
        args.paper_block_reader_backend = "tiered_cache"
    if str(getattr(args, "paper_optimizer_backend", "cpu")).lower() == "cpu":
        args.paper_optimizer_backend = "gpu_resident"
    if str(getattr(args, "paper_optimizer_state_mode", "full_cpu")).lower() == "full_cpu":
        args.paper_optimizer_state_mode = "resident_blocks"
    args.paper_free_unified_params = True
    args.disable_auto_densification = True
    args.sparse_adam = True


def _apply_tide_aliases(args):
    """Map public TideGS option aliases to internal compatibility fields."""
    alias_pairs = (
        ("tide_optimizer_deferred_mode", "paper_optimizer_deferred_mode"),
        ("tide_resident_selection_policy", "paper_resident_selection_policy"),
        ("tide_resident_lambda", "paper_resident_lambda"),
        ("tide_resident_recency_decay", "paper_resident_recency_decay"),
        ("tide_balanced_seed_fraction", "paper_balanced_seed_fraction"),
        ("tide_resident_capacity_blocks", "paper_resident_capacity_blocks"),
        ("tide_optimizer_state_mode", "paper_optimizer_state_mode"),
        ("tide_optimizer_backend", "paper_optimizer_backend"),
        ("tide_block_reader_backend", "paper_block_reader_backend"),
    )
    for alias_name, internal_name in alias_pairs:
        value = getattr(args, alias_name, "")
        if value != "":
            setattr(args, internal_name, value)
    if getattr(args, "tide_free_unified_params", False):
        args.paper_free_unified_params = True
    if getattr(args, "tide_debug_logging", False):
        args.paper_debug_logging = True


def init_args(args):
    _apply_tide_aliases(args)
    _apply_pure_ssd_release_defaults(args)

    assert (
        sum([args.clm_offload, args.naive_offload, args.no_offload]) == 1
    ), "Exactly one of clm_offload, naive_offload, or no_offload must be True"

    # Logging are saved with where model is saved.
    args.log_folder = args.model_path

    if hasattr(args, "ssd_execution_mode"):
        args.ssd_execution_mode = str(args.ssd_execution_mode).lower()
        assert args.ssd_execution_mode in {"fast_ram", "paper"}, (
            f"Invalid ssd_execution_mode={args.ssd_execution_mode!r}; expected one of fast_ram, paper"
        )
    if hasattr(args, "paper_optimizer_deferred_mode"):
        args.paper_optimizer_deferred_mode = str(args.paper_optimizer_deferred_mode).lower()
        assert args.paper_optimizer_deferred_mode in {"off", "same_iter", "cross_iter"}, (
            "Invalid paper_optimizer_deferred_mode="
            f"{args.paper_optimizer_deferred_mode!r}; expected one of off, same_iter, cross_iter"
        )
    if hasattr(args, "paper_resident_selection_policy"):
        args.paper_resident_selection_policy = str(args.paper_resident_selection_policy).lower()
        if args.paper_resident_selection_policy == "topc":
            args.paper_resident_selection_policy = "topc_strict"
        assert args.paper_resident_selection_policy in {"passthrough_active_set", "topc_strict", "topc_balanced"}, (
            "Invalid paper_resident_selection_policy="
            f"{args.paper_resident_selection_policy!r}; expected one of passthrough_active_set, topc_strict, topc_balanced"
        )
    if hasattr(args, "paper_resident_lambda"):
        args.paper_resident_lambda = float(args.paper_resident_lambda)
        assert 0.0 <= args.paper_resident_lambda <= 1.0, (
            f"Invalid paper_resident_lambda={args.paper_resident_lambda!r}; expected a value in [0, 1]"
        )
    if hasattr(args, "paper_resident_recency_decay"):
        args.paper_resident_recency_decay = float(args.paper_resident_recency_decay)
        assert 0.0 <= args.paper_resident_recency_decay <= 1.0, (
            "Invalid paper_resident_recency_decay="
            f"{args.paper_resident_recency_decay!r}; expected a value in [0, 1]"
        )
    if hasattr(args, "paper_balanced_seed_fraction"):
        args.paper_balanced_seed_fraction = float(args.paper_balanced_seed_fraction)
        assert 0.0 <= args.paper_balanced_seed_fraction <= 1.0, (
            "Invalid paper_balanced_seed_fraction="
            f"{args.paper_balanced_seed_fraction!r}; expected a value in [0, 1]"
        )
    if hasattr(args, "paper_resident_capacity_blocks"):
        args.paper_resident_capacity_blocks = int(args.paper_resident_capacity_blocks)
        assert args.paper_resident_capacity_blocks == -1 or args.paper_resident_capacity_blocks > 0, (
            "Invalid paper_resident_capacity_blocks="
            f"{args.paper_resident_capacity_blocks!r}; expected -1 (auto) or a positive integer"
        )
    if hasattr(args, "debug_camera_sample_mode"):
        args.debug_camera_sample_mode = str(args.debug_camera_sample_mode).lower()
        assert args.debug_camera_sample_mode in {"linspace", "contiguous", "window"}, (
            "Invalid debug_camera_sample_mode="
            f"{args.debug_camera_sample_mode!r}; expected one of linspace, contiguous, window"
        )
    if hasattr(args, "debug_camera_sample_start"):
        args.debug_camera_sample_start = int(args.debug_camera_sample_start)
        assert args.debug_camera_sample_start >= 0, (
            f"Invalid debug_camera_sample_start={args.debug_camera_sample_start!r}; expected >= 0"
        )

    if hasattr(args, "paper_optimizer_state_mode"):
        args.paper_optimizer_state_mode = str(args.paper_optimizer_state_mode).lower()
        assert args.paper_optimizer_state_mode in {"full_cpu", "resident_blocks"}, (
            "Invalid paper_optimizer_state_mode="
            f"{args.paper_optimizer_state_mode!r}; expected one of full_cpu, resident_blocks"
        )
        if args.paper_optimizer_state_mode == "resident_blocks":
            assert getattr(args, "sparse_adam", False), (
                "paper_optimizer_state_mode=resident_blocks currently requires --sparse_adam"
            )
            assert getattr(args, "disable_auto_densification", False), (
                "paper_optimizer_state_mode=resident_blocks currently requires --disable_auto_densification"
            )
            assert getattr(args, "paper_optimizer_deferred_mode", "off") == "off", (
                "paper_optimizer_state_mode=resident_blocks currently requires --paper_optimizer_deferred_mode off"
            )

    if hasattr(args, "paper_block_reader_backend"):
        args.paper_block_reader_backend = str(args.paper_block_reader_backend).lower()
        assert args.paper_block_reader_backend in {"auto", "unified_params", "tiered_cache"}, (
            "Invalid paper_block_reader_backend="
            f"{args.paper_block_reader_backend!r}; expected one of auto, unified_params, tiered_cache"
        )

    if hasattr(args, "paper_free_unified_params"):
        args.paper_free_unified_params = bool(args.paper_free_unified_params)
        if args.paper_free_unified_params:
            assert getattr(args, "ssd_execution_mode", "fast_ram") == "paper", (
                "paper_free_unified_params=True requires --ssd_execution_mode paper"
            )
            assert getattr(args, "paper_block_reader_backend", "auto") != "unified_params", (
                "paper_free_unified_params=True is incompatible with "
                "--paper_block_reader_backend unified_params (freeing the tensor "
                "would break the reader); use 'auto' or 'tiered_cache'"
            )
            assert getattr(args, "paper_optimizer_backend", "cpu") == "gpu_resident", (
                "paper_free_unified_params=True currently requires "
                "--paper_optimizer_backend gpu_resident (the CPU-Adam backend "
                "still writes to _unified_params.grad directly)"
            )
            assert getattr(args, "disable_auto_densification", False), (
                "paper_free_unified_params=True currently requires "
                "--disable_auto_densification (densification expects the full "
                "CPU parameter table)"
            )

    if hasattr(args, "paper_optimizer_backend"):
        args.paper_optimizer_backend = str(args.paper_optimizer_backend).lower()
        assert args.paper_optimizer_backend in {"cpu", "gpu_resident"}, (
            "Invalid paper_optimizer_backend="
            f"{args.paper_optimizer_backend!r}; expected one of cpu, gpu_resident"
        )
        if args.paper_optimizer_backend == "gpu_resident":
            assert getattr(args, "paper_optimizer_state_mode", "full_cpu") == "resident_blocks", (
                "paper_optimizer_backend=gpu_resident currently requires --paper_optimizer_state_mode resident_blocks"
            )
            assert getattr(args, "sparse_adam", False), (
                "paper_optimizer_backend=gpu_resident currently requires --sparse_adam"
            )
            assert getattr(args, "disable_auto_densification", False), (
                "paper_optimizer_backend=gpu_resident currently requires --disable_auto_densification"
            )
            assert getattr(args, "paper_optimizer_deferred_mode", "off") == "off", (
                "paper_optimizer_backend=gpu_resident currently requires --paper_optimizer_deferred_mode off"
            )

    if hasattr(args, "pure_ssd_init_backend"):
        args.pure_ssd_init_backend = str(args.pure_ssd_init_backend).lower()
        assert args.pure_ssd_init_backend in {"auto", "streaming", "inmemory"}, (
            "Invalid pure_ssd_init_backend="
            f"{args.pure_ssd_init_backend!r}; expected one of auto, streaming, inmemory"
        )

    if getattr(args, "pure_ssd_offload", False):
        args.use_ssd_offload = True
        if getattr(args, "pure_ssd_init_backend", "auto") == "auto":
            args.pure_ssd_init_backend = "streaming"

    if getattr(args, "use_ssd_offload", False):
        if getattr(args, "paper_block_reader_backend", "auto") == "auto":
            args.paper_block_reader_backend = "tiered_cache"

    if hasattr(args, "pure_ssd_max_inmemory_init_points"):
        args.pure_ssd_max_inmemory_init_points = int(args.pure_ssd_max_inmemory_init_points)
        assert args.pure_ssd_max_inmemory_init_points == -1 or args.pure_ssd_max_inmemory_init_points > 0, (
            "pure_ssd_max_inmemory_init_points must be -1 or a positive integer"
        )

    if hasattr(args, "pure_ssd_checkpoint_chunk_blocks"):
        args.pure_ssd_checkpoint_chunk_blocks = int(args.pure_ssd_checkpoint_chunk_blocks)
        assert args.pure_ssd_checkpoint_chunk_blocks > 0, (
            "pure_ssd_checkpoint_chunk_blocks must be a positive integer"
        )
    if hasattr(args, "pure_ssd_checkpoint_mode"):
        args.pure_ssd_checkpoint_mode = str(args.pure_ssd_checkpoint_mode).lower()
        assert args.pure_ssd_checkpoint_mode in {"incremental", "snapshot"}, (
            "pure_ssd_checkpoint_mode must be incremental or snapshot"
        )

    if hasattr(args, "pure_ssd_sort_memory_mb"):
        args.pure_ssd_sort_memory_mb = float(args.pure_ssd_sort_memory_mb)
        assert args.pure_ssd_sort_memory_mb > 0, (
            "pure_ssd_sort_memory_mb must be positive"
        )

    if hasattr(args, "pure_ssd_bucket_bits"):
        args.pure_ssd_bucket_bits = int(args.pure_ssd_bucket_bits)
        assert 1 <= args.pure_ssd_bucket_bits <= 14, (
            "pure_ssd_bucket_bits must be in [1, 14]"
        )

    if getattr(args, "pure_ssd_prebuilt_manifest", ""):
        args.pure_ssd_prebuilt_manifest = os.path.abspath(args.pure_ssd_prebuilt_manifest)
        assert os.path.isfile(args.pure_ssd_prebuilt_manifest), (
            f"pure_ssd_prebuilt_manifest not found: {args.pure_ssd_prebuilt_manifest}"
        )
    if getattr(args, "pure_ssd_prebuilt_base_file", ""):
        args.pure_ssd_prebuilt_base_file = os.path.abspath(args.pure_ssd_prebuilt_base_file)
        assert os.path.isfile(args.pure_ssd_prebuilt_base_file), (
            f"pure_ssd_prebuilt_base_file not found: {args.pure_ssd_prebuilt_base_file}"
        )
    if getattr(args, "pure_ssd_prebuilt_block_bounds", ""):
        args.pure_ssd_prebuilt_block_bounds = os.path.abspath(args.pure_ssd_prebuilt_block_bounds)
        assert os.path.isfile(args.pure_ssd_prebuilt_block_bounds), (
            f"pure_ssd_prebuilt_block_bounds not found: {args.pure_ssd_prebuilt_block_bounds}"
        )

    if getattr(args, "pure_ssd_offload", False):
        assert getattr(args, "clm_offload", False), (
            "pure SSD offload requires --clm_offload as the training strategy"
        )
        assert not getattr(args, "naive_offload", False) and not getattr(args, "no_offload", False), (
            "pure SSD offload is incompatible with --naive_offload and --no_offload"
        )
        assert getattr(args, "ssd_execution_mode", "fast_ram") == "paper", (
            "pure SSD offload requires --ssd_execution_mode paper"
        )
        assert getattr(args, "paper_block_reader_backend", "auto") == "tiered_cache", (
            "pure SSD offload requires --paper_block_reader_backend tiered_cache"
        )
        assert getattr(args, "paper_optimizer_backend", "cpu") == "gpu_resident", (
            "pure SSD offload requires --paper_optimizer_backend gpu_resident"
        )
        assert getattr(args, "paper_optimizer_state_mode", "full_cpu") == "resident_blocks", (
            "pure SSD offload requires --paper_optimizer_state_mode resident_blocks"
        )
        assert getattr(args, "paper_free_unified_params", False), (
            "pure SSD offload requires --paper_free_unified_params so the full CPU parameter table is released"
        )
        assert getattr(args, "disable_auto_densification", False), (
            "pure SSD offload requires --disable_auto_densification; current densification uses full-scene tensors"
        )
        assert getattr(args, "sparse_adam", False), (
            "pure SSD offload requires --sparse_adam for resident-block optimizer updates"
        )
        assert not getattr(args, "use_block_scheduler", False), (
            "pure SSD offload is incompatible with --use_block_scheduler"
        )
        assert getattr(args, "paper_optimizer_deferred_mode", "off") == "off", (
            "pure SSD offload requires --paper_optimizer_deferred_mode off"
        )
        assert getattr(args, "paper_resident_selection_policy", "passthrough_active_set") in {"topc_strict", "topc_balanced"}, (
            "pure SSD offload requires --paper_resident_selection_policy topc_strict or topc_balanced"
        )
        assert getattr(args, "paper_resident_capacity_blocks", -1) > 0, (
            "pure SSD offload requires a positive --paper_resident_capacity_blocks so VRAM residency is explicitly bounded"
        )
        assert getattr(args, "pure_ssd_init_backend", "streaming") in {"streaming", "inmemory"}, (
            "pure SSD offload requires --pure_ssd_init_backend streaming or inmemory"
        )
        pure_ssd_checkpoint_resume = bool(
            getattr(args, "start_checkpoint", "")
            and os.path.isfile(
                os.path.join(args.start_checkpoint, "pure_ssd_checkpoint.json")
            )
        )
        if (
            getattr(args, "pure_ssd_init_backend", "streaming") == "streaming"
            and not getattr(args, "pure_ssd_prebuilt_manifest", "")
            and not pure_ssd_checkpoint_resume
        ):
            assert getattr(args, "debug_fast_init_scales", False), (
                "pure SSD streaming init requires --debug_fast_init_scales; full distCUDA2 init is not out-of-core"
            )

    if args.auto_start_checkpoint:
        args.start_checkpoint = find_latest_checkpoint(args.log_folder)

    # sort test_iterations
    args.test_iterations.sort()
    args.save_iterations.sort()
    if len(args.save_iterations) > 0 and args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.sort()

    # Set up global args
    utils.set_args(args)
