import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
import utils.general_utils as utils
from optimizer import ResidentAdamContext
import numba.cuda

from strategies.base_gaussian_model import BaseGaussianModel
from strategies.tide_engine.gpu_working_set import GPUWorkingSet


class TideGaussianModel(BaseGaussianModel):
    """Gaussian model shell shared by the release path and archived baselines."""

    def _get_device(self):
        return "cuda"

    def create_from_pcd(
        self,
        pcd: BasicPointCloud,
        spatial_lr_scale: float,
        subsample_ratio: float = 1.0,
    ):
        log_file = utils.get_log_file()
        self.spatial_lr_scale = spatial_lr_scale

        self.use_gpu_features = getattr(self.args, 'use_ssd_offload', False)

        if not self.use_gpu_features:
            parameters_buffer_array = numba.cuda.pinned_array( 
            (self.args.prealloc_capacity, 48), dtype=np.float32
        )
            self.parameters_buffer = torch.from_numpy(parameters_buffer_array)
            
            if not self.only_for_rendering:
                parameters_grad_buffer_array = numba.cuda.pinned_array( # TODO: 这里一样，在cpu pinned memory上allocate了sh的grad空间，有必要吗？
                    (self.args.prealloc_capacity, 48), dtype=np.float32
                )
                self.parameters_grad_buffer = torch.from_numpy(parameters_grad_buffer_array)
        else:
            self.parameters_buffer = None
            self.parameters_grad_buffer = None

        # ========================================================================
        # [HOTSPOT CACHING] Initialize inter-batch cache state
        # ========================================================================
        # This dictionary stores GPU tensors from the last batch to enable
        # hotspot reuse across batch boundaries (Inter-Batch Retention)
        self.block_cache_state = {
            "last_shs": None,           # SH features retained from last micro-batch of previous batch
            "last_filter": None,         # Visibility indices from last micro-batch of previous batch
            "last_retention_vec": None,  # Retention mapping vector from previous batch
        }
        
        if self.use_gpu_features:
            self.gpu_working_set = None

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()  # CPU tensor
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        N = fused_point_cloud.shape[0]
        print("Number of points before initialization : ", N)

        if subsample_ratio != 1.0:
            assert subsample_ratio > 0 and subsample_ratio < 1
            sub_N = int(N * subsample_ratio)
            print("Subsample ratio: ", subsample_ratio)
            print("Number of points after subsampling : ", sub_N)

            perm_generator = torch.Generator()
            perm_generator.manual_seed(1)
            subsampled_indices, _ = torch.randperm(N, generator=perm_generator)[:sub_N].sort()
            fused_point_cloud = fused_point_cloud[subsampled_indices]
            features = features[subsampled_indices]
            N = sub_N

        def _log_init_progress(message: str):
            print(message)
            if log_file is not None:
                log_file.write(message + "\n")
                log_file.flush()

        debug_fast_init_scales = getattr(self.args, "debug_fast_init_scales", False)

        # Distance-based scale initialization. For billion-scale smoke/debug runs we
        # optionally use a strided sample to estimate a global initial scale, which
        # avoids a full-N distCUDA2 pass while keeping the formal training path
        # unchanged unless the debug flag is explicitly enabled.
        with torch.no_grad():
            if debug_fast_init_scales:
                sample_size = min(N, 1_000_000)
                stride = max(1, N // sample_size)
                sample_indices = torch.arange(0, N, stride, dtype=torch.long)[:sample_size]
                _log_init_progress(
                    f"[INIT SCALES] Debug fast-init enabled: estimating scales from {sample_size:,}/{N:,} strided points (stride={stride})"
                )
                points_gpu = fused_point_cloud[sample_indices].cuda()
                dist2_gpu = torch.clamp_min(distCUDA2(points_gpu), 0.0000001)
                sampled_log_scales = torch.log(torch.sqrt(dist2_gpu))
                global_scale = float(sampled_log_scales.median().item())
                scales = torch.full((N, 3), global_scale, dtype=torch.float32, device="cpu")
                del points_gpu, dist2_gpu, sampled_log_scales, sample_indices
                torch.cuda.empty_cache()
                _log_init_progress(
                    f"[INIT SCALES] Fast debug estimate complete: median log-scale={global_scale:.6f}; broadcasting to all {N:,} points"
                )
            else:
                _log_init_progress(
                    f"[INIT SCALES] Running full distCUDA2 for {N:,} points; this can take a while on billion-scale scenes"
                )
                points_gpu = fused_point_cloud.cuda()
                dist2_gpu = torch.clamp_min(distCUDA2(points_gpu), 0.0000001)
                scales = torch.log(torch.sqrt(dist2_gpu))[..., None].repeat(1, 3).cpu()  # 立即移回CPU
                del points_gpu, dist2_gpu
                torch.cuda.empty_cache()
                _log_init_progress("[INIT SCALES] Full distCUDA2 scale initialization complete")

        print(f"[PURE SSD INIT] Initialization tensors on CPU, N={N:,}")

        # Large paper-mode runs (for example billion-scale measurements) can
        # exceed CUDA host-allocation limits if we pin the full geometry tensors at
        # startup. When the geometry state becomes very large, fall back to ordinary
        # CPU tensors and rely on the block-wise streaming path during training.
        geometry_state_bytes = N * (3 + 3 + 4 + 1) * 4
        disable_auto_densification = getattr(self.args, "disable_auto_densification", False)
        paper_execution = str(getattr(self.args, "ssd_execution_mode", "fast_ram")).lower() == "paper"
        paper_free_unified = bool(getattr(self.args, "paper_free_unified_params", False))
        use_pinned_geometry = not (
            self.use_gpu_features
            and (
                paper_execution
                or paper_free_unified
                or geometry_state_bytes > 8 * (1024**3)
            )
        )
        if not use_pinned_geometry:
            geometry_state_gb = geometry_state_bytes / (1024**3)
            reason = "paper out-of-core mode" if paper_execution or paper_free_unified else "large geometry state"
            _log_init_progress(
                f"[OUT-OF-CORE INIT] {reason}: geometry state is {geometry_state_gb:.2f} GB; "
                "using ordinary CPU tensors instead of pinned host tensors during initialization."
            )

        rots = torch.zeros((N, 4), device="cpu") # 1560 MiB 
        rots[:, 0] = 1

        opacities = inverse_sigmoid( # 780 MiB
            0.1 * torch.ones((N, 1), dtype=torch.float, device="cpu")
        )

        if use_pinned_geometry:
            try:
                xyz_storage = torch.empty((N, 3), dtype=torch.float32, pin_memory=True)
                xyz_storage.copy_(fused_point_cloud)

                scaling_storage = torch.empty((N, 3), dtype=torch.float32, pin_memory=True)
                scaling_storage.copy_(scales)

                rotation_storage = torch.empty((N, 4), dtype=torch.float32, pin_memory=True)
                rotation_storage.copy_(rots)

                opacity_storage = torch.empty((N, 1), dtype=torch.float32, pin_memory=True)
                opacity_storage.copy_(opacities)
            except RuntimeError as exc:
                _log_init_progress(
                    f"[OUT-OF-CORE INIT] Pinned geometry allocation failed ({exc}); "
                    "falling back to ordinary CPU tensors."
                )
                use_pinned_geometry = False

        if not use_pinned_geometry:
            xyz_storage = fused_point_cloud.contiguous()
            scaling_storage = scales.contiguous()
            rotation_storage = rots.contiguous()
            opacity_storage = opacities.contiguous()

        self._xyz = nn.Parameter(xyz_storage, requires_grad=True)
        self._scaling = nn.Parameter(scaling_storage, requires_grad=True)
        self._rotation = nn.Parameter(rotation_storage, requires_grad=True)
        self._opacity = nn.Parameter(opacity_storage, requires_grad=True)

        # Extract DC and Rest features from the input (still on CPU at this point)
        features_dc = (
            features[:, :, 0:1].transpose(1, 2).contiguous().view(N, -1)
        )  # (N, 1, 3) -> (N, 3)
        features_rest = (
            features[:, :, 1:].transpose(1, 2).contiguous().view(N, -1)
        )  # (N, 15, 3) -> (N, 45)
        dims = [features_dc.shape[1], features_rest.shape[1]]

        
        if self.use_gpu_features:
            # In pure SSD / paper mode we stream SH block-wise into the GPU working set.
            # Keeping the full SH tensor in ordinary CPU memory is sufficient here and avoids
            # large pinned-memory allocations during startup (especially features_rest).
            features_dc_cpu = features_dc.contiguous()
            self._features_dc = nn.Parameter(features_dc_cpu, requires_grad=True)

            features_rest_cpu = features_rest.contiguous()
            self._features_rest = nn.Parameter(features_rest_cpu, requires_grad=True)
            self._parameters = None
            
        else:
            # Historical split-feature mode stores SH features in CPU pinned memory.
            torch.cat((features_dc, features_rest), dim=1, out=self.parameters_buffer[:N])
            self._parameters = nn.Parameter(self.parameters_buffer[:N].requires_grad_(True))
            self._features_dc, self._features_rest = torch.split(
                self._parameters, dims, dim=1
            )
    
        if self.use_gpu_features:
            # Initialize GPU working set manager for block-wise loading
            block_size = getattr(self.args, 'gaussian_block_size', 4096)
            self.gpu_working_set_manager = GPUWorkingSet(
                num_total_gaussians=N,
                block_size=block_size,
                device='cuda',
                verbose=bool(getattr(self.args, "paper_debug_logging", False)),
            )
        if disable_auto_densification:
            log_file.write(
                "[PAPER MODE] Auto densification disabled; skipping full-size GPU densification buffers.\n"
            )
            self.max_radii2D = torch.empty((0,), device="cuda")
            self.sum_visible_count_in_one_batch = torch.empty((0,), device="cuda")
        else:
            self.max_radii2D = torch.zeros((N), device="cuda")
            self.sum_visible_count_in_one_batch = torch.zeros((N), device="cuda")

        self.param_dims = torch.tensor(dims, dtype=torch.int, device="cuda")

    def _refresh_gpu_working_set_topology(self):
        if self.use_gpu_features and hasattr(self, 'gpu_working_set_manager') and self.gpu_working_set_manager is not None:
            self.gpu_working_set_manager.refresh_topology(int(self._xyz.shape[0]))

    def prepare_streaming_ply_init(self, ply_path: str, spatial_lr_scale: float):
        """Prepare a lightweight model shell for pure SSD streaming init.

        Full Gaussian parameters are not allocated here.  The storage adapter
        will stream the PLY directly into SSD and then call
        initialize_from_streaming_ssd_manifest() with the resulting metadata.
        """
        self.spatial_lr_scale = spatial_lr_scale
        self.use_gpu_features = True
        self.parameters_buffer = None
        self.parameters_grad_buffer = None
        self._parameters = None
        self._streaming_ply_init_pending = True
        self._streaming_init_ply_path = ply_path
        self._streaming_ssd_initialized = False
        self._pure_ssd_resume_pending = False
        self._pure_ssd_resume_manifest = None
        self._unified_params = None
        self._xyz = None
        self._opacity = None
        self._scaling = None
        self._rotation = None
        self._features_dc = None
        self._features_rest = None
        self.block_cache_state = {
            "last_shs": None,
            "last_filter": None,
            "last_retention_vec": None,
        }
        log_file = utils.get_log_file()
        if log_file is not None:
            log_file.write(
                f"[STREAMING PLY INIT] Prepared lightweight Gaussian shell for {ply_path}\n"
            )

    def prepare_pure_ssd_checkpoint_resume(self, manifest: dict, spatial_lr_scale: float):
        """Prepare a lightweight shell that will bind to an SSD checkpoint snapshot."""
        self.spatial_lr_scale = spatial_lr_scale
        self.use_gpu_features = True
        self.parameters_buffer = None
        self.parameters_grad_buffer = None
        self._parameters = None
        self._streaming_ply_init_pending = False
        self._streaming_init_ply_path = manifest.get("source_ply", "")
        self._streaming_ssd_initialized = False
        self._pure_ssd_resume_pending = True
        self._pure_ssd_resume_manifest = dict(manifest)
        self._unified_params = None
        self._xyz = None
        self._opacity = None
        self._scaling = None
        self._rotation = None
        self._features_dc = None
        self._features_rest = None
        self.block_cache_state = {
            "last_shs": None,
            "last_filter": None,
            "last_retention_vec": None,
        }
        log_file = utils.get_log_file()
        message = (
            "[PURE SSD RESUME] loaded checkpoint shell: "
            f"next_iteration={manifest.get('next_iteration')} "
            f"base={manifest.get('base_file')}\n"
        )
        if log_file is not None:
            log_file.write(message)
        print(message.strip())

    def prepare_pure_ssd_prebuilt_manifest(self, manifest: dict, spatial_lr_scale: float):
        """Prepare a lightweight shell that will bind to an existing SSD base."""
        self.prepare_pure_ssd_checkpoint_resume(manifest, spatial_lr_scale)
        self._pure_ssd_prebuilt_pending = True
        log_file = utils.get_log_file()
        message = (
            "[PURE SSD PREBUILT] loaded prebuilt shell: "
            f"base={manifest.get('base_file')}\n"
        )
        if log_file is not None:
            log_file.write(message)
        print(message.strip())

    def initialize_from_streaming_ssd_manifest(self, manifest: dict):
        """Bind model metadata after streaming PLY-to-SSD base creation."""
        total_points = int(manifest["total_points"])
        block_size = int(manifest["block_size"])
        self.use_gpu_features = True
        self.parameters_buffer = None
        self.parameters_grad_buffer = None
        self._parameters = None
        self._streaming_ply_init_pending = False
        self._streaming_ssd_initialized = True
        self._pure_ssd_resume_pending = False
        self._streaming_init_manifest = dict(manifest)
        self._paper_unified_params_freed = True
        self._paper_unified_params_never_allocated = True
        self._paper_unified_params_num_total = total_points
        self._paper_unified_params_param_width = 59

        # Zero-row placeholders satisfy existing optimizer setup without
        # allocating an N x 59 CPU table.  Training replaces these with the
        # current GPU working set before forward/backward.
        self._xyz = nn.Parameter(torch.empty((0, 3), dtype=torch.float32), requires_grad=True)
        self._opacity = nn.Parameter(torch.empty((0, 1), dtype=torch.float32), requires_grad=True)
        self._scaling = nn.Parameter(torch.empty((0, 3), dtype=torch.float32), requires_grad=True)
        self._rotation = nn.Parameter(torch.empty((0, 4), dtype=torch.float32), requires_grad=True)
        self._features_dc = nn.Parameter(torch.empty((0, 3), dtype=torch.float32), requires_grad=True)
        self._features_rest = nn.Parameter(torch.empty((0, 45), dtype=torch.float32), requires_grad=True)
        self._unified_params = None

        self.gpu_working_set = None
        self.gpu_working_set_manager = GPUWorkingSet(
            num_total_gaussians=total_points,
            block_size=block_size,
            device='cuda',
            verbose=bool(getattr(self.args, "paper_debug_logging", False)),
        )
        self.max_radii2D = torch.empty((0,), device="cuda")
        self.sum_visible_count_in_one_batch = torch.empty((0,), device="cuda")
        self.param_dims = torch.tensor([3, 45], dtype=torch.int, device="cuda")

        log_file = utils.get_log_file()
        message = (
            f"[PAPER MODE] _unified_params never allocated; streaming SSD base has "
            f"{total_points:,} Gaussians across {manifest.get('num_blocks')} blocks\n"
        )
        if log_file is not None:
            log_file.write(message)
        print(message.strip())

    def initialize_from_pure_ssd_checkpoint_manifest(self, manifest: dict):
        """Bind model metadata after selecting a pure SSD checkpoint snapshot."""
        self.initialize_from_streaming_ssd_manifest(manifest)
        self._pure_ssd_resume_pending = False
        self._pure_ssd_resume_initialized = True
        self._pure_ssd_resume_manifest = dict(manifest)
        if "active_sh_degree" in manifest:
            self.active_sh_degree = int(manifest["active_sh_degree"])

        log_file = utils.get_log_file()
        message = (
            "[PURE SSD RESUME] using SSD snapshot base: "
            f"{manifest.get('base_file')} "
            f"next_iteration={manifest.get('next_iteration')}\n"
        )
        if log_file is not None:
            log_file.write(message)
        print(message.strip())
    
    @property
    def get_xyz(self):
        """
        Get XYZ positions.
        - During training (GPU working set active): Returns GPU tensor subset
        - During densification/pruning: Returns full CPU tensor
        - Standard mode: Returns parameter tensor directly
        """
        if self.use_gpu_features and hasattr(self, 'gpu_working_set_manager'):
            if self.gpu_working_set_manager.gpu_xyz is not None:
                # Training mode: return GPU working set
                return self.gpu_working_set_manager.gpu_xyz
        # Densification/initialization: return full CPU tensor
        return self._xyz
    
    @property
    def get_scaling(self):
        """Get scaling parameters with activation."""
        if self.use_gpu_features and hasattr(self, 'gpu_working_set_manager'):
            if self.gpu_working_set_manager.gpu_scaling is not None:
                return self.scaling_activation(self.gpu_working_set_manager.gpu_scaling)
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        """Get rotation parameters with activation."""
        if self.use_gpu_features and hasattr(self, 'gpu_working_set_manager'):
            if self.gpu_working_set_manager.gpu_rotation is not None:
                return self.rotation_activation(self.gpu_working_set_manager.gpu_rotation)
        return self.rotation_activation(self._rotation)
    
    @property
    def get_opacity(self):
        """Get opacity with activation."""
        if self.use_gpu_features and hasattr(self, 'gpu_working_set_manager'):
            if self.gpu_working_set_manager.gpu_opacity is not None:
                return self.opacity_activation(self.gpu_working_set_manager.gpu_opacity)
        return self.opacity_activation(self._opacity)
    
    @property
    def get_features(self):
        """Get SH features (DC + Rest)."""
        if self.use_gpu_features and hasattr(self, 'gpu_working_set_manager'):
            if self.gpu_working_set_manager.gpu_features_dc is not None:
                return torch.cat((
                    self.gpu_working_set_manager.gpu_features_dc,
                    self.gpu_working_set_manager.gpu_features_rest
                ), dim=1)
        # Fallback to CPU tensors
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    # ========================================================================
    # End of property overrides
    # ========================================================================
    
    def offload_params_to_ssd_storage(self):
        """
        Mark the model as SSD-backed after the storage adapter is initialized.

        The pure SSD/Tide release path keeps the full model in the log-structured
        SSD store and materializes only resident blocks through the tiered cache.
        This method only sets the bookkeeping flag; ``free_unified_params()``
        performs the actual release of the temporary unified CPU tensor.
        """
        if not self.use_gpu_features:
            print("[WARNING] offload_params_to_ssd_storage() called in non-SSD mode, ignoring")
            return
        
        if not hasattr(self, '_params_offloaded_to_ssd'):
            self._params_offloaded_to_ssd = False
        
        if self._params_offloaded_to_ssd:
            print("[PURE SSD] Parameters already marked as SSD-backed")
            return
        
        # Mark as offloaded
        self._params_offloaded_to_ssd = True
        
        log_file = utils.get_log_file()
        log_file.write("\n" + "="*70 + "\n")
        log_file.write("[PURE SSD] Parameters are backed by SSD storage\n")
        log_file.write("[PURE SSD] Training will materialize resident blocks via SSD -> RAM -> GPU\n")
        log_file.write("="*70 + "\n\n")
        
        print("[PURE SSD] Parameters marked as SSD-backed")
        print("[PURE SSD] Training will use the SSD -> RAM -> GPU pipeline")

    def free_unified_params(self):
        """Release ``_unified_params`` and its column views for paper mode.

        After this call, the gaussian model no longer holds an N × 59 pinned
        CPU tensor.  The full parameter table lives exclusively in the tiered
        SSD + RAM cache; per-iteration working sets are materialised through
        the ``BlockReader`` and optimizer state is kept on GPU (``gpu_resident``
        backend).

        Preconditions (validated at CLI in arguments/__init__.py):
            * ``ssd_execution_mode == "paper"``
            * ``paper_block_reader_backend != "unified_params"``
            * ``paper_optimizer_backend == "gpu_resident"``
            * ``disable_auto_densification`` (densification still expects the
              full CPU parameter table)

        After the call:
            * ``self._unified_params = None``
            * ``self._xyz, self._opacity, ..., self._features_rest = None``
              (these were views into ``_unified_params``; callers must re-bind
              to the per-iter GPU working set)
            * ``self.optimizer.param_groups[0]['params']`` is replaced with a
              dummy zero-width leaf tensor so metadata/checkpoint code cannot
              retain a stale reference to the released storage.
        """
        if not getattr(self, 'use_gpu_features', False):
            print("[WARNING] free_unified_params() called in non-SSD mode, ignoring")
            return
        if not hasattr(self, '_unified_params') or self._unified_params is None:
            if getattr(self, '_paper_unified_params_never_allocated', False):
                return
            return

        old_tensor = self._unified_params
        num_total = int(old_tensor.shape[0])
        param_width = int(old_tensor.shape[1])
        if getattr(self, '_paper_unified_params_never_allocated', False):
            num_total = int(getattr(self, '_paper_unified_params_num_total', num_total))
            param_width = int(getattr(self, '_paper_unified_params_param_width', param_width))
        total_gb = old_tensor.numel() * old_tensor.element_size() / 1e9

        # Break optimizer references before releasing the tensor so no stale
        # pointer remains inside optimizer param-group bookkeeping.
        optimizer = getattr(self, 'optimizer', None)
        replacement_leaf = torch.nn.Parameter(
            torch.empty(0, param_width, dtype=old_tensor.dtype),
            requires_grad=False,
        )
        if optimizer is not None:
            for group in getattr(optimizer, 'param_groups', []):
                params_list = group.get('params', [])
                for i, p in enumerate(params_list):
                    if p is old_tensor:
                        params_list[i] = replacement_leaf
            # Drop any cached Adam state that was keyed on old_tensor.
            inner_state = getattr(optimizer, 'state', None)
            if isinstance(inner_state, dict) and old_tensor in inner_state:
                inner_state.pop(old_tensor, None)

        self._unified_params = None
        self._xyz = None
        self._opacity = None
        self._scaling = None
        self._rotation = None
        self._features_dc = None
        self._features_rest = None

        # Release the pinned-memory storage back to the allocator.
        del old_tensor
        import gc
        gc.collect()

        self._paper_unified_params_freed = True
        self._paper_unified_params_num_total = num_total
        self._paper_unified_params_param_width = param_width

        log_file = utils.get_log_file()
        log_file.write("\n" + "=" * 70 + "\n")
        if getattr(self, '_paper_unified_params_never_allocated', False):
            log_file.write(
                f"[PAPER MODE] _unified_params never allocated ({num_total:,} × {param_width}); streaming SSD base is source of truth\n"
            )
        else:
            log_file.write(
                f"[PAPER MODE] Released _unified_params ({num_total:,} × {param_width} ≈ {total_gb:.2f} GB) from CPU RAM\n"
            )
        log_file.write(
            "[PAPER MODE] Training will now read blocks exclusively via TieredCacheBlockReader\n"
        )
        log_file.write(
            "[PAPER MODE] Writeback will bypass _unified_params and go GPU → per-block CPU → cache directly\n"
        )
        log_file.write("=" * 70 + "\n\n")
        if getattr(self, '_paper_unified_params_never_allocated', False):
            print(
                f"[PAPER MODE] ✓ _unified_params never allocated — streaming paper out-of-core mode active"
            )
        else:
            print(
                f"[PAPER MODE] ✓ Released _unified_params (~{total_gb:.2f} GB) — paper out-of-core mode active"
            )

    def all_parameters(self):
        """Return all trainable parameters."""
        if self.use_gpu_features:
            # Paper mode: return the full CPU table before it is released;
            # after paper_free_unified_params, return only the
            # currently materialized GPU working-set tensors.
            unified_params = getattr(self, '_unified_params', None)
            if unified_params is not None:
                return [unified_params]

            return [
                param for param in (
                    self._xyz,
                    self._opacity,
                    self._scaling,
                    self._rotation,
                    self._features_dc,
                    self._features_rest,
                )
                if param is not None
            ]
        else:
            # Historical split-feature path returns geometry plus unified SH parameters.
            return [
                self._xyz,
                self._opacity,
                self._scaling,
                self._rotation,
                self._parameters,
            ]

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense # for densification，该分裂还是克隆的阈值
        if getattr(training_args, "disable_auto_densification", False):
            self.xyz_gradient_accum = torch.empty((0, 1), device=self.device)
            self.denom = torch.empty((0, 1), device=self.device)
        else:
            self.xyz_gradient_accum = torch.zeros(
                (self.get_xyz.shape[0], 1), device=self.device
            )
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        args = utils.get_args()
        log_file = utils.get_log_file()

        # ========================================================================
        # [DEBUG] Check parameter devices BEFORE building optimizer param list
        # ========================================================================
        log_file.write(f"\n[PRE-TRAINING-SETUP VERIFICATION]\n")
        log_file.write(f"  _xyz device: {self._xyz.device}, is_cuda: {self._xyz.is_cuda}\n")
        log_file.write(f"  _opacity device: {self._opacity.device}, is_cuda: {self._opacity.is_cuda}\n")
        log_file.write(f"  _scaling device: {self._scaling.device}, is_cuda: {self._scaling.is_cuda}\n")
        log_file.write(f"  _rotation device: {self._rotation.device}, is_cuda: {self._rotation.is_cuda}\n")
        
        if hasattr(self, '_features_dc') and isinstance(self._features_dc, nn.Parameter):
            log_file.write(f"  _features_dc device: {self._features_dc.device}, is_cuda: {self._features_dc.is_cuda}\n")
        if hasattr(self, '_features_rest') and isinstance(self._features_rest, nn.Parameter):
            log_file.write(f"  _features_rest device: {self._features_rest.device}, is_cuda: {self._features_rest.is_cuda}\n")
        if hasattr(self, '_parameters') and isinstance(self._parameters, nn.Parameter):
            log_file.write(f"  _parameters device: {self._parameters.device}, is_pinned: {self._parameters.is_pinned()}\n")
        log_file.write(f"\n")

        # ====================================================================
        # Build parameter list based on storage mode
        # ====================================================================
        if self.use_gpu_features:
            # ================================================================
            # Pure SSD/Paper mode: concatenate all params into one unified tensor
            # ================================================================
            # Keep one metadata parameter group with column-wise learning rates.
            # Layout: xyz(3) | opacity(1) | scaling(3) | rotation(4) | features_dc(3) | features_rest(45)
            # Total: 59 columns per Gaussian
            
            log_file.write("[PAPER MODE] Concatenating all parameters into unified tensor\n")
            
            # ================================================================
            # [CRITICAL FIX] Preserve pinned memory when concatenating
            # ================================================================
            # torch.cat() on pinned tensors returns a NON-pinned tensor!
            # Solution: Create pinned buffer first, then copy data
            
            N = self._xyz.shape[0]
            paper_execution_mode = getattr(self.args, 'ssd_execution_mode', 'fast_ram') == 'paper'
            
            if paper_execution_mode:
                unified_buffer = torch.empty((N, 59), dtype=torch.float32)
                log_file.write("  [PAPER MODE] Using ordinary CPU memory for unified_params (not pinned)\n")
            else:
                unified_buffer = torch.empty((N, 59), dtype=torch.float32).pin_memory()
                log_file.write("  [FAST_RAM MODE] Using pinned CPU memory for unified_params\n")
            
            # Copy each parameter into the buffer
            unified_buffer[:, 0:3] = self._xyz           # (N, 3)
            unified_buffer[:, 3:4] = self._opacity       # (N, 1)
            unified_buffer[:, 4:7] = self._scaling       # (N, 3)
            unified_buffer[:, 7:11] = self._rotation     # (N, 4)
            unified_buffer[:, 11:14] = self._features_dc # (N, 3)
            unified_buffer[:, 14:59] = self._features_rest  # (N, 45)
            
            log_file.write(f"  unified_buffer.is_pinned() BEFORE nn.Parameter = {unified_buffer.is_pinned()}\n")
            
            self._unified_params = nn.Parameter(unified_buffer, requires_grad=True)
            
            log_file.write(f"  Unified params shape: {self._unified_params.shape}\n")
            log_file.write(f"  Unified params device: {self._unified_params.device}\n")
            log_file.write(f"  Unified params pinned: {self._unified_params.is_pinned()}\n")
            log_file.write(f"  id(unified_buffer) = {id(unified_buffer)}\n")
            log_file.write(f"  id(self._unified_params.data) = {id(self._unified_params.data)}\n")
            log_file.write(f"  Same tensor? {id(unified_buffer) == id(self._unified_params.data)}\n")
            
            assert self._unified_params.device.type == 'cpu', \
                f"unified_params should be on CPU but is on {self._unified_params.device}"
            if paper_execution_mode:
                log_file.write("  ✅ Unified params verified on CPU (paper mode, non-pinned allowed)\n")
            else:
                assert self._unified_params.is_pinned(), \
                    f"unified_params should be pinned but is_pinned()={self._unified_params.is_pinned()}"
                log_file.write("  ✅ Unified params verified as pinned\n")
            
            # Single metadata parameter group for the SSD-backed unified table.
            l = [
                {
                    "params": [self._unified_params],
                    "lr": training_args.position_lr_init  # Base LR (will be overridden by columns_lr)
                    * self.spatial_lr_scale
                    * args.lr_scale_pos_and_scale,
                    "name": "unified_params",
                },
            ]
            
            # Column-wise learning rates for the unified parameter layout.
            # Each column gets its own LR based on parameter type
            column_sizes = [3, 1, 3, 4, 3, 45]  # xyz, opacity, scaling, rotation, features_dc, features_rest
            column_lrs = [
                training_args.position_lr_init * self.spatial_lr_scale * args.lr_scale_pos_and_scale,  # xyz
                training_args.opacity_lr,                                                               # opacity
                training_args.scaling_lr * args.lr_scale_pos_and_scale,                                # scaling
                training_args.rotation_lr,                                                              # rotation
                training_args.feature_lr,                                                               # features_dc
                training_args.feature_lr / 20.0,                                                        # features_rest
            ]
            
            log_file.write(f"  Column sizes: {column_sizes}\n")
            log_file.write(f"  Column LRs: {column_lrs}\n")
            
            # ================================================================
            # [CRITICAL] Update individual parameter references to views of unified tensor
            # ================================================================
            # After creating _unified_params, update _xyz, _opacity, etc. to be views
            # This ensures get_xyz, get_scaling, etc. still work correctly
            # Layout: xyz(3) | opacity(1) | scaling(3) | rotation(4) | features_dc(3) | features_rest(45)
            self._xyz = self._unified_params[:, 0:3]
            self._opacity = self._unified_params[:, 3:4]
            self._scaling = self._unified_params[:, 4:7]
            self._rotation = self._unified_params[:, 7:11]
            self._features_dc = self._unified_params[:, 11:14]
            self._features_rest = self._unified_params[:, 14:59]
            
            log_file.write("[PAPER MODE] Individual parameters updated to views of unified tensor\n")
        else:
            # Split-feature compatibility path: geometry on GPU, SH on CPU (5 parameters).
            l = [
                {
                    "params": [self._xyz],
                    "lr": training_args.position_lr_init
                    * self.spatial_lr_scale
                    * args.lr_scale_pos_and_scale,
                    "name": "xyz",
                },
                {
                    "params": [self._opacity],
                    "lr": training_args.opacity_lr,
                    "name": "opacity",
                },
                {
                    "params": [self._scaling],
                    "lr": training_args.scaling_lr * args.lr_scale_pos_and_scale,
                    "name": "scaling",
                },
                {
                    "params": [self._rotation],
                    "lr": training_args.rotation_lr,
                    "name": "rotation",
                },
                {
                    "params": [self._parameters],  # concatenated SH features on CPU
                    "lr": training_args.feature_lr,
                    "name": "parameters",
                },
            ]
            column_sizes = [3, 45]
            column_lrs = [training_args.feature_lr, training_args.feature_lr / 20.0]

        # ====================================================================
        # Verify parameter devices before passing to optimizer
        # ====================================================================
        log_file.write(f"\n{'='*60}\n")
        setup_mode = "SSD-backed working set" if self.use_gpu_features else "split-feature compatibility"
        log_file.write(f"[Optimizer Setup] Mode: {setup_mode}\n")
        log_file.write(f"{'='*60}\n")
        
        # for param_dict in l:
        #     # import pdb; pdb.set_trace()
        #     param = param_dict["params"][0]
        #     param_name = param_dict["name"]
        #     device_info = f"device={param.device}, is_cuda={param.is_cuda}, is_pinned={param.is_pinned()}"
        #     log_file.write(f"  {param_name:20s}: {device_info}, shape={param.shape}\n")
            
        #     # ====================================================================
        #     # Paper-mode conditional device verification
        #     # ====================================================================
        #     if self.use_gpu_features:
        #         # Pure SSD/Paper mode: all parameters should be CPU-backed here.
        #         # They will be loaded to GPU working set dynamically during training
        #         if param.is_cuda:
        #             error_msg = f"ERROR: '{param_name}' should be on CPU (pinned) but is on {param.device}"
        #             log_file.write(f"  ❌ {error_msg}\n")
        #             raise AssertionError(error_msg)
                
        #         # Verify it's on CPU and preferably pinned (but pinned check may be unreliable for nn.Parameter)
        #         if param.device.type != 'cpu':
        #             error_msg = f"ERROR: '{param_name}' should be on CPU but is on {param.device}"
        #             log_file.write(f"  ❌ {error_msg}\n")
        #             raise AssertionError(error_msg)
        #     else:
        #         # Split-feature compatibility path: geometry on GPU, SH features on CPU pinned.
        #         if param_name == "parameters":
        #             if not param.is_pinned():
        #                 error_msg = f"ERROR: 'parameters' should be pinned but is on {param.device}"
        #                 log_file.write(f"  ❌ {error_msg}\n")
        #                 raise AssertionError(error_msg)
        #         else:
        #             if not param.is_cuda:
        #                 error_msg = f"ERROR: '{param_name}' should be on CUDA but is on {param.device}"
        #                 log_file.write(f"  ❌ {error_msg}\n")
        #                 raise AssertionError(error_msg)
        
        if self.use_gpu_features:
            log_file.write(f"✅ [PAPER MODE] Unified parameter tensor created\n")
            log_file.write(f"   Shape: {l[0]['params'][0].shape}\n")
            log_file.write(f"   Device: {l[0]['params'][0].device}\n")
            log_file.write(f"   Pinned: {l[0]['params'][0].is_pinned()}\n")
            log_file.write(f"   Column sizes: {column_sizes}\n")
            log_file.write(f"   Column LRs: {column_lrs}\n")
            log_file.write(f"   Will be loaded to GPU working set dynamically during training\n")
        else:
            log_file.write("✅ [LEGACY SPLIT-FEATURE] 4 params on CUDA, 1 param on CPU (pinned)\n")
        log_file.write(f"{'='*60}\n\n")
        
        self.optimizer = ResidentAdamContext(
            l,
            column_sizes,
            column_lrs,
            lr=0.0,
            bias_correction=True,  # This True is required.
            betas=(0.9, 0.999),
            eps=1e-15,
            weight_decay=0,
            amsgrad=False,
            adamw_mode=False,
            fp32_optimizer_states=True,
            fused=True,
            sparse=self.args.sparse_adam,
        )

        # Scale learning rates according to bsz.
        bsz = args.bsz
        for param_group in self.optimizer.param_groups:
            if training_args.lr_scale_mode == "linear":
                lr_scale = bsz
                param_group["lr"] *= lr_scale
            elif training_args.lr_scale_mode == "sqrt":
                lr_scale = np.sqrt(bsz)
                param_group["lr"] *= lr_scale # TODO: 这里的param_group["lr"] 用到了吗？ 不是用的是 columns_lr 吗？
                if "eps" in param_group:  # Adam
                    param_group["eps"] /= lr_scale
                    param_group["betas"] = [beta**bsz for beta in param_group["betas"]]
                    log_file.write(
                        param_group["name"]
                        + " betas: "
                        + str(param_group["betas"])
                        + "\n"
                    )
            elif training_args.lr_scale_mode == "accumu":
                lr_scale = 1
            else:
                assert (
                    False
                ), f"lr_scale_mode {training_args.lr_scale_mode} not supported."

        # Scale the per-column learning rates consumed by GPUResidentAdam.
        # Scale column-wise learning rates when the optimizer exposes them.
        if hasattr(self.optimizer, 'columns_lr') and self.optimizer.columns_lr is not None:
            if training_args.lr_scale_mode == "linear":
                lr_scale_cols = bsz
                self.optimizer.columns_lr *= lr_scale_cols
            elif training_args.lr_scale_mode == "sqrt":
                lr_scale_cols = np.sqrt(bsz)
                self.optimizer.columns_lr *= lr_scale_cols

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init
            * self.spatial_lr_scale
            * lr_scale
            * args.lr_scale_pos_and_scale,
            lr_final=training_args.position_lr_final
            * self.spatial_lr_scale
            * lr_scale
            * args.lr_scale_pos_and_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        utils.check_initial_gpu_memory_usage("after training_setup")

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step.
        
        In the split-feature compatibility path, the param_group named 'xyz' is updated directly.
        In SSD-backed unified mode, there is only one param_group named 'unified_params'
        and the unified layout stores per-column LRs in self.optimizer.columns_lr,
        so columns_lr[0] (the xyz column) is updated as well.
        """
        lr = self.xyz_scheduler_args(iteration)
        
        for param_group in self.optimizer.param_groups:
            # Case A: split-feature compatibility path — param_group is named "xyz"
            if param_group["name"] == "xyz":
                param_group["lr"] = lr
                return lr
            
            # Case B: SSD Offload mode — single group named "unified_params"
            elif param_group["name"] == "unified_params":
                # Update the param_group lr (used by some logging code)
                param_group["lr"] = lr
                
                # columns_lr[0] corresponds to xyz in the unified layout.
                if hasattr(self.optimizer, 'columns_lr'):
                    cols_lr = self.optimizer.columns_lr
                    if isinstance(cols_lr, torch.Tensor):
                        cols_lr[0] = lr
                    elif isinstance(cols_lr, list):
                        cols_lr[0] = lr
                    # Also update the underlying cpu_adam's columns_lr if it exists
                    if hasattr(self.optimizer, 'cpu_adam') and self.optimizer.cpu_adam is not None:
                        cpu_cols_lr = self.optimizer.cpu_adam.columns_lr
                        if isinstance(cpu_cols_lr, torch.Tensor):
                            cpu_cols_lr[0] = lr
                        elif isinstance(cpu_cols_lr, list):
                            cpu_cols_lr[0] = lr
                return lr
        
        return lr  # fallback (should not reach here)

    def save_tensors(self, parent_path):
        """Compatibility in-memory tensor save path; pure SSD release uses block-wise checkpoints."""
        from utils.tensor_save_utils import save_tensor_compact

        mkdir_p(parent_path)

        save_tensor_compact(self._xyz, os.path.join(parent_path, "xyz.pt"))
        save_tensor_compact(self._opacity, os.path.join(parent_path, "opacity.pt"))
        save_tensor_compact(self._scaling, os.path.join(parent_path, "scaling.pt"))
        save_tensor_compact(self._rotation, os.path.join(parent_path, "rotation.pt"))

        # In paper mode _parameters may be None; save features from split views instead.
        if self._parameters is not None:
            save_tensor_compact(self._parameters, os.path.join(parent_path, "parameters.pt"))
        else:
            features = torch.cat(
                [self._features_dc.detach().contiguous().clone(),
                 self._features_rest.detach().contiguous().clone()],
                dim=1,
            )
            if features.is_cuda:
                features = features.cpu()
            from utils.tensor_save_utils import safe_torch_save
            safe_torch_save(features, os.path.join(parent_path, "parameters.pt"))

    def load_tensors(self, parent_path):
        """Compatibility in-memory tensor load path; pure SSD release resumes from SSD checkpoints."""
        _xyz = torch.load(os.path.join(parent_path, "xyz.pt"), map_location="cpu")
        _opacity = torch.load(
            os.path.join(parent_path, "opacity.pt"), map_location="cpu"
        )
        _scaling = torch.load(
            os.path.join(parent_path, "scaling.pt"), map_location="cpu"
        )
        _rotation = torch.load(
            os.path.join(parent_path, "rotation.pt"), map_location="cpu"
        )
        _features = torch.load(
            os.path.join(parent_path, "parameters.pt"), map_location="cpu"
        )

        N = _xyz.shape[0]
        print("Number of points before initialization : ", N)

        # ========================================================================
        # [LOAD TENSORS] Rendering-only eval prefers exact-size host SH storage
        # ========================================================================
        log_file = utils.get_log_file()
        requested_capacity = int(getattr(self.args, "prealloc_capacity", -1))
        if requested_capacity < N:
            requested_capacity = N

        if self.only_for_rendering:
            if log_file is not None:
                log_file.write(
                    f"[RENDER-ONLY LOAD] Using exact CPU SH storage for {N:,} Gaussians (requested prealloc={getattr(self.args, 'prealloc_capacity', -1)})\n"
                )
            print(f"[RENDER-ONLY LOAD] Using exact CPU SH storage for {N:,} Gaussians")
            self.use_gpu_features = False
            self.parameters_buffer = _features.contiguous()
            self.parameters_grad_buffer = None
        else:
            # ========================================================================
            # Initialize SSD-backed compatibility mode for compatibility tensor loading.
            # ========================================================================
            self.use_gpu_features = hasattr(self.args, 'use_ssd_offload') and self.args.use_ssd_offload

            if self.use_gpu_features:
                log_file.write("[LEGACY LOAD TENSORS] SH features will be stored on GPU\n")
                print("[LEGACY LOAD TENSORS] SH features will be stored on GPU")
                self.parameters_buffer = torch.zeros(
                    (requested_capacity, 48),
                    dtype=torch.float32,
                    device='cuda'
                )
                self.parameters_grad_buffer = torch.zeros(
                    (requested_capacity, 48),
                    dtype=torch.float32,
                    device='cuda'
                )
            else:
                log_file.write("[LEGACY LOAD TENSORS] SH features will be stored in CPU pinned memory\n")
                print("[LEGACY LOAD TENSORS] SH features will be stored in CPU pinned memory")
                parameters_buffer_array = numba.cuda.pinned_array(
                    (requested_capacity, 48), dtype=np.float32
                )
                self.parameters_buffer = torch.from_numpy(parameters_buffer_array)
                assert self.parameters_buffer.is_pinned()
                parameters_grad_buffer_array = numba.cuda.pinned_array(
                    (requested_capacity, 48), dtype=np.float32
                )
                self.parameters_grad_buffer = torch.from_numpy(parameters_grad_buffer_array)
                assert self.parameters_grad_buffer.is_pinned()

        # ========================================================================
        # Initialize parameters based on mode
        # ========================================================================
        if self.only_for_rendering:
            self._xyz = _xyz.to("cuda")
            self._opacity = _opacity.to("cuda")
            self._scaling = _scaling.to("cuda")
            self._rotation = _rotation.to("cuda")
            self._parameters = self.parameters_buffer[:N]
            self._features_dc, self._features_rest = torch.split(
                self._parameters, [3, 45], dim=1
            )
        else:
            # Always move geometry parameters to GPU
            self._xyz = nn.Parameter(_xyz.to("cuda").requires_grad_(True))
            self._opacity = nn.Parameter(_opacity.to("cuda").requires_grad_(True))
            self._scaling = nn.Parameter(_scaling.to("cuda").requires_grad_(True))
            self._rotation = nn.Parameter(_rotation.to("cuda").requires_grad_(True))

            # Handle SH features based on mode
            if self.use_gpu_features:
                # SSD-backed compatibility mode stores SH features on GPU as separate parameters.
                _features_gpu = _features.cuda()
                self.parameters_buffer[:N].copy_(_features_gpu)

                self._features_dc = nn.Parameter(
                    _features_gpu[:, :3].requires_grad_(True)
                )
                self._features_rest = nn.Parameter(
                    _features_gpu[:, 3:48].requires_grad_(True)
                )
                self._parameters = self.parameters_buffer[:N]  # View, not a parameter
            else:
                # Split-feature compatibility path stores SH features in CPU pinned memory.
                self.parameters_buffer[:N].copy_(_features)
                self._parameters = nn.Parameter(self.parameters_buffer[:N].requires_grad_(True))
                self._features_dc, self._features_rest = torch.split(
                    self._parameters, [3, 45], dim=1
                )

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def save_sub_plys(self, path, n_split, split_size):
        args = utils.get_args()
        _xyz = _features_dc = _features_rest = _opacity = _scaling = _rotation = None
        utils.log_cpu_memory_usage("start save_ply")
        _xyz = self._xyz
        _features_dc = self._features_dc
        _features_rest = self._features_rest
        _opacity = self._opacity
        _scaling = self._scaling
        _rotation = self._rotation

        for i in range(n_split):
            assert path.endswith(".ply")
            this_path = (
                path[:-4] + "_rk" + str(i) + "_ws" + str(n_split) + ".ply"
            )  # TODO: modify the file_name.
            mkdir_p(os.path.dirname(this_path))

            start = i * split_size
            end = min((i + 1) * split_size, _xyz.shape[0])

            xyz = _xyz.detach()[start:end].cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = (
                _features_dc.detach()[start:end]
                .contiguous()
                .view(-1, 1, 3)
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            f_rest = (
                _features_rest.detach()[start:end]
                .contiguous()
                .view(-1, 15, 3)
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            opacities = _opacity.detach()[start:end].cpu().numpy()
            scale = _scaling.detach()[start:end].cpu().numpy()
            rotation = _rotation.detach()[start:end].cpu().numpy()

            utils.log_cpu_memory_usage(
                f"[{i/n_split}] after change gpu tensor to cpu numpy"
            )

            dtype_full = [
                (attribute, "f4") for attribute in self.construct_list_of_attributes()
            ]

            elements = np.empty(end - start, dtype=dtype_full)
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
            )
            del xyz, normals, f_dc, f_rest, opacities, scale, rotation
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, "vertex")

            utils.log_cpu_memory_usage(
                f"[{i/n_split}] after change numpy to plyelement before writing ply file"
            )
            PlyData([el]).write(this_path)

        utils.log_cpu_memory_usage("finish write ply file")
        # remark: max_radii2D, xyz_gradient_accum and denom are not saved here; they are save elsewhere.

    def load_raw_ply(self, path):
        print("Loading ", path)
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        args = utils.get_args()

        if args.drop_initial_3dgs_p > 0.0:
            # drop each point with probability args.drop_initial_3dgs_p
            drop_mask = np.random.rand(xyz.shape[0]) > args.drop_initial_3dgs_p
            xyz = xyz[drop_mask]
            features_dc = features_dc[drop_mask]
            features_extra = features_extra[drop_mask]
            scales = scales[drop_mask]
            rots = rots[drop_mask]
            opacities = opacities[drop_mask]

        return xyz, features_dc, features_extra, opacities, scales, rots

    def one_file_load_ply(self, folder):
        path = os.path.join(folder, "point_cloud.ply")
        xyz, features_dc, features_extra, opacities, scales, rots = self.load_raw_ply(
            path
        )
        N = xyz.shape[0]

        _xyz = torch.from_numpy(xyz)
        _opacity = torch.from_numpy(opacities)
        _scaling = torch.from_numpy(scales)
        _rotation = torch.from_numpy(rots)
        _features_dc = torch.from_numpy(features_dc)
        _features_rest = torch.from_numpy(features_extra)

        if _features_dc.ndim == 3:
            _features_dc = _features_dc.permute(0, 2, 1).reshape(N, -1)
        if _features_rest.ndim == 3:
            _features_rest = _features_rest.permute(0, 2, 1).reshape(N, -1)

        # ========================================================================
        # Initialize SSD-backed compatibility mode for compatibility PLY loading.
        # ========================================================================
        self.use_gpu_features = hasattr(self.args, 'use_ssd_offload') and self.args.use_ssd_offload
        log_file = utils.get_log_file()
        
        if self.use_gpu_features:
            log_file.write("[LEGACY LOAD PLY] SH features will be stored on GPU\n")
            print("[LEGACY LOAD PLY] SH features will be stored on GPU")
        else:
            log_file.write("[LEGACY LOAD PLY] SH features will be stored in CPU pinned memory\n")
            print("[LEGACY LOAD PLY] SH features will be stored in CPU pinned memory")

        # ========================================================================
        # Allocate buffers based on mode
        # ========================================================================
        if self.use_gpu_features:
            # SSD-backed compatibility mode allocates SH buffers on GPU.
            self.parameters_buffer = torch.zeros(
                (self.args.prealloc_capacity, 48), 
                dtype=torch.float32, 
                device='cuda'
            )
            if not self.only_for_rendering:
                self.parameters_grad_buffer = torch.zeros(
                    (self.args.prealloc_capacity, 48), 
                    dtype=torch.float32, 
                    device='cuda'
                )
        else:
            # Split-feature compatibility path allocates SH buffers in CPU pinned memory.
            parameters_buffer_array = numba.cuda.pinned_array(
                (self.args.prealloc_capacity, 48), dtype=np.float32
            )
            self.parameters_buffer = torch.from_numpy(parameters_buffer_array)
            assert self.parameters_buffer.is_pinned()
            if not self.only_for_rendering:
                parameters_grad_buffer_array = numba.cuda.pinned_array(
                    (self.args.prealloc_capacity, 48), dtype=np.float32
                )
                self.parameters_grad_buffer = torch.from_numpy(parameters_grad_buffer_array)
                assert self.parameters_grad_buffer.is_pinned()

        # ========================================================================
        # Initialize parameters based on storage mode
        # ========================================================================
        if self.use_gpu_features:
            # Paper mode: parameters are CPU-backed before block-wise materialization.
            
            # Geometry parameters in pinned memory
            xyz_pinned = torch.empty(_xyz.shape, dtype=torch.float32).pin_memory()
            xyz_pinned.copy_(_xyz.to(torch.float))
            self._xyz = nn.Parameter(xyz_pinned.requires_grad_(True))
            
            scaling_pinned = torch.empty(_scaling.shape, dtype=torch.float32).pin_memory()
            scaling_pinned.copy_(_scaling.to(torch.float))
            self._scaling = nn.Parameter(scaling_pinned.requires_grad_(True))
            
            rotation_pinned = torch.empty(_rotation.shape, dtype=torch.float32).pin_memory()
            rotation_pinned.copy_(_rotation.to(torch.float))
            self._rotation = nn.Parameter(rotation_pinned.requires_grad_(True))
            
            opacity_pinned = torch.empty(_opacity.shape, dtype=torch.float32).pin_memory()
            opacity_pinned.copy_(_opacity.to(torch.float))
            self._opacity = nn.Parameter(opacity_pinned.requires_grad_(True))
            
            log_file.write("[PAPER MODE] Loaded geometry parameters to pinned memory\n")
        else:
            # Split-feature compatibility path keeps geometry on GPU and SH on CPU pinned memory.
            self._xyz = nn.Parameter(
                _xyz.to(torch.float).cuda().requires_grad_(True)
            )
            self._opacity = nn.Parameter(
                _opacity.to(torch.float).cuda().requires_grad_(True)
            )
            self._scaling = nn.Parameter(
                _scaling.to(torch.float).cuda().requires_grad_(True)
            )
            self._rotation = nn.Parameter(
                _rotation.to(torch.float).cuda().requires_grad_(True)
            )

        # Handle SH features based on mode
        dims = [_features_dc.shape[1], _features_rest.shape[1]]
        
        if self.use_gpu_features:
            # SSD-backed compatibility mode stores SH features on GPU as separate parameters.
            _features_dc_gpu = _features_dc.to(torch.float).cuda()
            _features_rest_gpu = _features_rest.to(torch.float).cuda()
            
            self._features_dc = nn.Parameter(_features_dc_gpu.requires_grad_(True))
            self._features_rest = nn.Parameter(_features_rest_gpu.requires_grad_(True))
            
            # Populate unified buffer (on GPU)
            torch.cat((_features_dc_gpu, _features_rest_gpu), dim=1, out=self.parameters_buffer[:N])
            self._parameters = self.parameters_buffer[:N]  # View, not a parameter
        else:
            # Split-feature compatibility path stores SH features in CPU pinned memory.
            torch.cat((_features_dc, _features_rest), dim=1, out=self.parameters_buffer[:N])
            self._parameters = nn.Parameter(self.parameters_buffer[:N].requires_grad_(True))
            self._features_dc, self._features_rest = torch.split(
                self._parameters, dims, dim=1
            )

        self.active_sh_degree = self.max_sh_degree
        
        # ========================================================================
        # [DEBUG] Verify all parameters are on correct devices after loading
        # ========================================================================
        log_file.write(f"\n[POST-LOAD VERIFICATION]\n")
        log_file.write(f"  _xyz device: {self._xyz.device}, is_cuda: {self._xyz.is_cuda}\n")
        log_file.write(f"  _opacity device: {self._opacity.device}, is_cuda: {self._opacity.is_cuda}\n")
        log_file.write(f"  _scaling device: {self._scaling.device}, is_cuda: {self._scaling.is_cuda}\n")
        log_file.write(f"  _rotation device: {self._rotation.device}, is_cuda: {self._rotation.is_cuda}\n")
        
        if self.use_gpu_features:
            log_file.write(f"  _features_dc device: {self._features_dc.device}, is_cuda: {self._features_dc.is_cuda}\n")
            log_file.write(f"  _features_rest device: {self._features_rest.device}, is_cuda: {self._features_rest.is_cuda}\n")
        else:
            log_file.write(f"  _parameters device: {self._parameters.device}, is_pinned: {self._parameters.is_pinned()}\n")
        
        # ====================================================================
        # Conditional device verification
        # ====================================================================
        if not self.use_gpu_features:
            # Split-feature compatibility path requires geometry parameters on GPU.
            assert self._xyz.is_cuda, f"_xyz should be on GPU but is on {self._xyz.device}"
            assert self._opacity.is_cuda, f"_opacity should be on GPU but is on {self._opacity.device}"
            assert self._scaling.is_cuda, f"_scaling should be on GPU but is on {self._scaling.device}"
            assert self._rotation.is_cuda, f"_rotation should be on GPU but is on {self._rotation.device}"
            log_file.write(f"✅ All geometry parameters verified on GPU after load\n\n")
        else:
            # Paper mode: parameters are CPU-backed after loading PLY.
            log_file.write(f"[PAPER MODE] Loaded PLY parameters on CPU\n\n")

    def load_ply(self, path):
        self.one_file_load_ply(path)

    def reset_opacity(self):
        utils.LOG_FILE.write("Resetting opacity to 0.01\n")
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_unified_adam(
            opacities_new, "opacity"
        )
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_unified_adam(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                assert group["params"][
                    0
                ].is_cuda, "Not implemented for parameters on cpu yet."
                stored_state = self.optimizer.gpu_adam.state.get(
                    group["params"][0], None
                )
                assert stored_state is not None, "Optimizer state is required for this path."
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.zeros_like(tensor)
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.gpu_adam.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.gpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_unified_adam(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["params"][0].is_cuda:
                stored_state = self.optimizer.gpu_adam.state.get(
                    group["params"][0], None
                )
                mask = mask.to(group["params"][0].device.type)
                assert stored_state is not None, "Optimizer state is required for this path."

                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = stored_state["momentum_buffer"][
                        mask
                    ]
                else:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.gpu_adam.state[group["params"][0]]

                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )

                self.optimizer.gpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]

            else:
                stored_state = self.optimizer.cpu_adam.state.get(
                    group["params"][0], None
                )
                mask = mask.to(group["params"][0].device.type)
                assert stored_state is not None, "Optimizer state is required for this path."

                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = stored_state["momentum_buffer"][
                        mask
                    ]
                else:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.cpu_adam.state[group["params"][0]]

                assert mask.dim() == 1
                self.parameters_buffer[: torch.sum(mask)] = group["params"][0][mask]
                group["params"][0] = nn.Parameter(
                    (self.parameters_buffer[: torch.sum(mask)].requires_grad_(True))
                )

                self.optimizer.cpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]

        # ====================================================================
        # Merge optimizer states conditionally
        # ====================================================================
        # In GPU mode, cpu_adam is None, so only use gpu_adam.state
        if self.optimizer.cpu_adam is not None:
            self.optimizer.state = (
                self.optimizer.gpu_adam.state | self.optimizer.cpu_adam.state
            )
        else:
            self.optimizer.state = self.optimizer.gpu_adam.state
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_unified_adam(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # ====================================================================
        # Handle SH features based on storage mode
        # ====================================================================
        if self.use_gpu_features:
            # SSD-backed compatibility mode updates _features_dc and _features_rest.
            self._features_dc = optimizable_tensors["features_dc"]
            self._features_rest = optimizable_tensors["features_rest"]
            
            # Resize parameters_buffer if needed to avoid deprecated resize warning
            num_gaussians = self._features_dc.shape[0]
            if self.parameters_buffer.shape[0] < num_gaussians:
                # Reallocate larger buffer (grow by 1.5x to reduce reallocation frequency)
                new_size = int(num_gaussians * 1.5)
                self.parameters_buffer = torch.empty((new_size, 48), 
                                                      dtype=self._features_dc.dtype,
                                                      device=self._features_dc.device)
            
            # Update parameters_buffer view for compatibility
            torch.cat([self._features_dc, self._features_rest], dim=1, 
                      out=self.parameters_buffer[:num_gaussians])
            self._parameters = self.parameters_buffer[:num_gaussians]
        else:
            # Split-feature compatibility path updates _parameters and split views.
            self._parameters = optimizable_tensors["parameters"]
            dims = [3, 45]
            self._features_dc, self._features_rest = torch.split(
                self._parameters, dims, dim=1
            )

        # ====================================================================
        # Conditional device verification after pruning
        # ====================================================================
        if not self.use_gpu_features:
            # Split-feature compatibility path: strict device requirements.
            assert self._xyz.is_cuda, "xyz should be on GPU in split-feature compatibility mode"
            assert self._opacity.is_cuda, "opacity should be on GPU"
            assert self._scaling.is_cuda, "scaling should be on GPU"
            assert self._rotation.is_cuda, "rotation should be on GPU"
            assert self._parameters.is_pinned(), "SH parameters should be pinned"
            assert self._features_dc.is_pinned()
            assert self._features_rest.is_pinned()
        else:
            # Paper mode: parameters can be on CPU.
            # They will be loaded to GPU working set during next training iteration
            pass

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.sum_visible_count_in_one_batch = self.sum_visible_count_in_one_batch[
            valid_points_mask
        ]
        self._refresh_gpu_working_set_topology()

    def cat_tensors_to_unified_adam(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if tensors_dict[group["name"]] is None:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]

            if group["params"][0].is_cuda:
                stored_state = self.optimizer.gpu_adam.state.get(
                    group["params"][0], None
                )

                assert stored_state is not None, "Optimizer state is required for this path."
                # Update optimizer states.
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.cat(
                        (
                            stored_state["momentum_buffer"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                else:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                del self.optimizer.gpu_adam.state[group["params"][0]]

                # Update parameters.
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.gpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]

            else:
                stored_state = self.optimizer.cpu_adam.state.get(
                    group["params"][0], None
                )

                assert stored_state is not None, "Optimizer state is required for this path."
                # Update optimizer states.
                if "exp_avg" not in stored_state:
                    stored_state["momentum_buffer"] = torch.cat(
                        (
                            stored_state["momentum_buffer"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                else:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                del self.optimizer.cpu_adam.state[group["params"][0]]

                # Update parameters.
                N = group["params"][0].shape[0]
                N_ext = extension_tensor.shape[0]
                self.parameters_buffer[N : (N + N_ext)] = extension_tensor
                group["params"][0] = nn.Parameter(
                    self.parameters_buffer[: (N + N_ext)].requires_grad_(True)
                )
                self.optimizer.cpu_adam.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]

        # ====================================================================
        # Merge optimizer states conditionally
        # ====================================================================
        # In GPU mode, cpu_adam is None, so only use gpu_adam.state
        if self.optimizer.cpu_adam is not None:
            self.optimizer.state = (
                self.optimizer.gpu_adam.state | self.optimizer.cpu_adam.state
            )
        else:
            self.optimizer.state = self.optimizer.gpu_adam.state
        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_parameters=None,
    ):
        # ========================================================================
        # [HOTSPOT CACHING] Invalidate cache when point cloud topology changes
        # ========================================================================
        # When gaussians are added (densify) or removed (prune), the indexing
        # changes, making the cached last_filter indices invalid. We must force
        # a cold start for the next batch.
        if hasattr(self, 'block_cache_state'):
            self.block_cache_state["last_shs"] = None
            self.block_cache_state["last_filter"] = None
            self.block_cache_state["last_retention_vec"] = None
            # Log cache invalidation for debugging
            log_file = utils.get_log_file()
            log_file.write("[HOTSPOT CACHE] Invalidated due to densification/pruning\n")

        # ====================================================================
        # Dictionary keys must match optimizer param_groups names
        # ====================================================================
        # SSD-backed compatibility mode: "features_dc", "features_rest" (no "parameters")
        # Split-feature compatibility path: "parameters" (no "features_dc", "features_rest")
        d = {
            "xyz": new_xyz,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "parameters": new_parameters,
        }
        optimizable_tensors = self.cat_tensors_to_unified_adam(d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # ====================================================================
        # Handle SH features based on storage mode
        # ====================================================================
        if self.use_gpu_features:
            # SSD-backed compatibility mode updates _features_dc and _features_rest.
            self._features_dc = optimizable_tensors["features_dc"]
            self._features_rest = optimizable_tensors["features_rest"]
            
            # Resize parameters_buffer if needed to avoid deprecated resize warning
            num_gaussians = self._features_dc.shape[0]
            if self.parameters_buffer.shape[0] < num_gaussians:
                # Reallocate larger buffer (grow by 1.5x to reduce reallocation frequency)
                new_size = int(num_gaussians * 1.5)
                self.parameters_buffer = torch.empty((new_size, 48), 
                                                      dtype=self._features_dc.dtype,
                                                      device=self._features_dc.device)
            
            # Update parameters_buffer view for compatibility
            torch.cat([self._features_dc, self._features_rest], dim=1, 
                      out=self.parameters_buffer[:num_gaussians])
            self._parameters = self.parameters_buffer[:num_gaussians]
        else:
            # Split-feature compatibility path updates _parameters and split views.
            self._parameters = optimizable_tensors["parameters"]
            dims = [3, 45]
            self._features_dc, self._features_rest = torch.split(
                self._parameters, dims, dim=1
            )

        # ====================================================================
        # Conditional device verification after densification
        # ====================================================================
        if not self.use_gpu_features:
            # Split-feature compatibility path: strict device requirements.
            assert self._xyz.is_cuda, "xyz should be on GPU in split-feature compatibility mode"
            assert self._opacity.is_cuda, "opacity should be on GPU"
            assert self._scaling.is_cuda, "scaling should be on GPU"
            assert self._rotation.is_cuda, "rotation should be on GPU"
            assert self._parameters.is_pinned(), "SH parameters should be pinned"
            assert self._features_dc.is_pinned()
            assert self._features_rest.is_pinned()
        else:
            # Paper mode: parameters can remain CPU-backed after densification.
            # They will be loaded to GPU working set during next training iteration
            log_file = utils.get_log_file()
            log_file.write(f"[Densification] Parameters on device: {self._xyz.device}\\n")

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device=self.device
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self.sum_visible_count_in_one_batch = torch.zeros(
            (self.get_xyz.shape[0]), device=self.device
        )
        self._refresh_gpu_working_set_topology()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        
        # ====================================================================
        # Ensure scaling is on same device as grads
        # ====================================================================
        scaling_vals = self.get_scaling
        if scaling_vals.device != selected_pts_mask.device:
            scaling_vals = scaling_vals.to(selected_pts_mask.device)
        
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scaling_vals, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = scaling_vals[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        utils.get_log_file().write(
            "Number of split gaussians: {}\n".format(selected_pts_mask.sum().item())
        )

        selected_pts_mask_cpu = selected_pts_mask.cpu()
        
        # ====================================================================
        # Use CPU mask for CPU tensors, GPU mask for GPU tensors
        # ====================================================================
        # Determine which mask to use based on tensor location
        if self._rotation.device.type == 'cpu':
            idx_mask = selected_pts_mask_cpu
        else:
            idx_mask = selected_pts_mask
        
        rots = build_rotation(self._rotation[idx_mask]).repeat(N, 1, 1)
        
        # get_xyz returns GPU tensor if working set is active, else CPU
        xyz_for_split = self.get_xyz
        if xyz_for_split.device != samples.device:
            xyz_for_split = xyz_for_split.to(samples.device)
        new_xyz = torch.bmm(rots.to(samples.device), samples.unsqueeze(-1)).squeeze(-1) + xyz_for_split[
            selected_pts_mask if xyz_for_split.is_cuda else selected_pts_mask_cpu
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            scaling_vals[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[idx_mask].repeat(N, 1)
        new_opacities = self._opacity[idx_mask].repeat(N, 1)
        
        # ====================================================================
        # Extract SH features based on storage mode
        # ====================================================================
        if self.use_gpu_features:
            # SSD-backed compatibility mode extracts from _features_dc and _features_rest.
            # Use idx_mask for CPU tensors
            new_features_dc = self._features_dc[idx_mask].repeat(N, 1)
            new_features_rest = self._features_rest[idx_mask].repeat(N, 1)
            new_parameters = None
        else:
            # Split-feature compatibility path extracts from _parameters.
            new_parameters = self._parameters[selected_pts_mask_cpu].repeat(N, 1)
            new_features_dc = None
            new_features_rest = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_parameters,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device=self.device, dtype=bool
                ),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        
        # ====================================================================
        # Ensure scaling is on same device as grads
        # ====================================================================
        scaling_vals = self.get_scaling
        if scaling_vals.device != selected_pts_mask.device:
            scaling_vals = scaling_vals.to(selected_pts_mask.device)
        
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scaling_vals, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        utils.get_log_file().write(
            "Number of cloned gaussians: {}\n".format(selected_pts_mask.sum().item())
        )

        selected_pts_mask_cpu = selected_pts_mask.cpu()
        
        # ====================================================================
        # Use CPU mask for CPU tensors
        # ====================================================================
        if self._xyz.device.type == 'cpu':
            idx_mask = selected_pts_mask_cpu
        else:
            idx_mask = selected_pts_mask
        
        new_xyz = self._xyz[idx_mask]
        new_opacities = self._opacity[idx_mask]
        new_scaling = self._scaling[idx_mask]
        new_rotation = self._rotation[idx_mask]
        
        # ====================================================================
        # Extract SH features based on storage mode
        # ====================================================================
        if self.use_gpu_features:
            # SSD-backed compatibility mode extracts from _features_dc and _features_rest.
            new_features_dc = self._features_dc[idx_mask]
            new_features_rest = self._features_rest[idx_mask]
            new_parameters = None
        else:
            # Split-feature compatibility path extracts from _parameters.
            new_parameters = self._parameters[selected_pts_mask_cpu]
            new_features_dc = None
            new_features_rest = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_parameters,
        )

    def gsplat_add_densification_stats_exact_filter(
        self,
        viewspace_point_tensor_grad,
        radii,
        send2gpu_final_filter_indices,
        width,
        height,
    ):
        self.max_radii2D[send2gpu_final_filter_indices] = torch.max(
            self.max_radii2D[send2gpu_final_filter_indices], radii
        )
        grad = viewspace_point_tensor_grad  # (N, 2)
        # Normalize the gradients to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[send2gpu_final_filter_indices] += torch.norm(
            grad, dim=-1, keepdim=True
        )
        self.denom[send2gpu_final_filter_indices] += 1
