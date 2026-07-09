import math
import logging
import time
import multiprocessing


import torch
import numpy as np
from scipy import linalg as spla
from numpy import linalg as la
from tqdm.auto import tqdm
from ..utils import get_layer_by_name, get_layer_name, find_matched_pattern

logger = logging.getLogger(__name__)

# CLAMP_MIN = 1e-6
NUM_MATRIX_SQRT_ITERATIONS = 200


class ScaleHookFactoryDiagonal:
    """
    scale = diag( sqrt( E[ x_1^2]), sqrt( E[ x_2^2]), ..., sqrt( E[ x_n^2] ) )
    """

    def __init__(self, torch_dtype) -> None:
        self.scales = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.torch_dtype = torch_dtype
        self.handles = []

    def get_scale_hook(self, name: str) -> callable:
        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.view(-1, x.shape[-1])
            num_samples, _ = x.shape
            x = x.pow(2).sum(0)

            self.n_samples[name] += num_samples
            if self.scales[name] is None:
                self.compute_devices[name] = x.device
                if self.torch_dtype is None:
                    self.torch_dtype = x.dtype
                if self.compute_devices[name].type == "cpu":
                    logger.warning("Using CPU for computing scale, this may be slow")
                scale = x.to(self.torch_dtype)
            else:
                scale = self.scales[name].to(self.compute_devices[name])
                scale = scale + x.to(self.torch_dtype)

            self.scales[name] = scale

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, progress_bar=False) -> dict[str, torch.Tensor]:
        scale_names_prog_bar = tqdm(
            self.scales, desc="Computing scale", disable=not progress_bar, total=len(self.scales)
        )

        for name in scale_names_prog_bar:
            # for name in self.scales:
            scale = self.scales[name].to(self.compute_devices[name])
            scale = torch.sqrt(scale) * (1 / math.sqrt(self.n_samples[name]))
            # scale = torch.clamp(scale, min=CLAMP_MIN)
            self.scales[name] = scale

        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def sqrtm_newton_schulz(A, numIters=200):
    """Newton-Schulz iterations method to get matrix square root.

    Code copied from https://github.com/pytorch/pytorch/issues/25481

    Page 231, Eq 2.6b
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.6.8799&rep=rep1&type=pdf

    Args:
        A: the symmetric PSD matrix whose matrix square root be computed
        numIters: Maximum number of iterations.

    Returns:
        A^0.5

    Tensorflow Source:
        https://github.com/tensorflow/tensorflow/blob/df3a3375941b9e920667acfe72fb4c33a8f45503/tensorflow/contrib/opt/python/training/matrix_functions.py#L26C1-L73C42
    Torch Source:
        https://github.com/msubhransu/matrix-sqrt/blob/cc2289a3ed7042b8dbacd53ce8a34da1f814ed2f/matrix_sqrt.py#L74
    """

    normA = torch.linalg.matrix_norm(A, keepdim=True)
    err = normA + 1.0
    I = torch.eye(*A.shape[-2:], dtype=A.dtype, device=A.device)
    Z = torch.eye(*A.shape[-2:], dtype=A.dtype, device=A.device).expand_as(A)
    Y = A / normA
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y_new = Y.bmm(T)
        Z_new = T.bmm(Z)

        # This method require that we check for divergence every step.
        # Compute the error in approximation.
        mat_a_approx = torch.bmm(Y_new, Y_new) * normA
        residual = A - mat_a_approx
        current_err = torch.linalg.matrix_norm(residual, keepdim=True) / normA
        if torch.all(current_err > err):
            break

        err = current_err
        Y = Y_new
        Z = Z_new

    sA = Y * torch.sqrt(normA)

    return sA


def sqrtm_scipy(A: np.ndarray):
    if not isinstance(A, np.ndarray):
        raise RuntimeError("input matrix must be a numpy array")
    A_sqrt = spla.sqrtm(A)
    return A_sqrt




class ScaleHookFactoryHess:
    """
    For row vector x,
    scale = E[ 2 x^T x ], where Rxx = E[ 2 x^T x ] is the auto-correlation matrix
    """
    
    def __init__(self, torch_dtype) -> None:
        self.scales = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.torch_dtype = torch_dtype
        self.handles = []

        self._force_cpu = False

    @torch.no_grad()
    def get_scale_hook(self, name: str) -> callable:
        """ """

        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.reshape(-1, x.shape[-1])
            n_samples, in_features = x.shape
            if self.scales[name] is None:
                if self.torch_dtype is None:
                    self.torch_dtype = x.dtype
                self.compute_devices[name] = x.device
                if self.compute_devices[name].type == "cpu":
                    logger.warning("Using CPU for computing Rxx, this may be slow")

                if self._force_cpu:
                    self.scales[name] = np.zeros((in_features, in_features), dtype=np.float64)
                else:
                    self.scales[name] = torch.zeros(
                        in_features, in_features, dtype=torch.float64
                    )  # *: hard-coded float64
                    # self.scales[name] = torch.zeros(in_features, in_features, dtype=self.torch_dtype)

            if self._force_cpu:
                x = x.cpu().numpy()
                # delta = np.einsum("bi,bj->ij", x, x, optimize="optimal").astype(np.float64)
                delta = np.tensordot(x, x, axes=([0], [0])).astype(np.float64)
                self.scales[name] += delta
                self.n_samples[name] += n_samples
            else:
                compute_device = self.compute_devices[name]
                scales = self.scales[name].to(compute_device)
                x = x.float()
                x = x.to(compute_device)
                # batched outer product
                # *: outer product in self.torch_dtype (float32 is preferred), then accumulate in float64
                delta = torch.einsum("bi,bj->ij", x, x).to(torch.float64)  # *: hard-coded float64
                # delta = torch.einsum("bi,bj->ij", x, x).to(self.torch_dtype)
                scales += 2 * delta
                self.scales[name] = scales.cpu()
                self.n_samples[name] += n_samples

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, progress_bar=False) -> dict[str, torch.Tensor]:

        for name in self.scales:
            scale = self.scales[name]
            n_samples = self.n_samples[name]
            scale = scale / n_samples
            self.scales[name] = scale

        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    

class ScaleHookFactoryRxx:
    """
    For row vector x,
    scale = E[ x^T x ] ^ 0.5, where Rxx = E[ x^T x ] is the auto-correlation matrix

    For numerical stability, we compute (x^T x) in torch_dtype (float32 is preferred) and accumulate in float64 (hard-coded).
    Then sqrt is computed in float64 (hard-coded).
    """

    def __init__(self, torch_dtype) -> None:
        self.scales = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.torch_dtype = torch_dtype
        self.handles = []

        self._force_cpu = False

    @torch.no_grad()
    def get_scale_hook(self, name: str) -> callable:
        """ """

        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.reshape(-1, x.shape[-1])
            n_samples, in_features = x.shape
            if self.scales[name] is None:
                if self.torch_dtype is None:
                    self.torch_dtype = x.dtype
                self.compute_devices[name] = x.device
                if self.compute_devices[name].type == "cpu":
                    logger.warning("Using CPU for computing Rxx, this may be slow")

                if self._force_cpu:
                    self.scales[name] = np.zeros((in_features, in_features), dtype=np.float64)
                else:
                    self.scales[name] = torch.zeros(
                        in_features, in_features, dtype=torch.float64
                    )  # *: hard-coded float64
                    # self.scales[name] = torch.zeros(in_features, in_features, dtype=self.torch_dtype)

            if self._force_cpu:
                x = x.cpu().numpy()
                # delta = np.einsum("bi,bj->ij", x, x, optimize="optimal").astype(np.float64)
                delta = np.tensordot(x, x, axes=([0], [0])).astype(np.float64)
                self.scales[name] += delta
                self.n_samples[name] += n_samples
            else:
                compute_device = self.compute_devices[name]
                scales = self.scales[name].to(compute_device)
                x = x.to(self.torch_dtype)
                x = x.to(compute_device)
                # batched outer product
                # *: outer product in self.torch_dtype (float32 is preferred), then accumulate in float64
                delta = torch.einsum("bi,bj->ij", x, x).to(torch.float64)  # *: hard-coded float64
                # delta = torch.einsum("bi,bj->ij", x, x).to(self.torch_dtype)
                scales += delta
                self.scales[name] = scales.cpu()
                self.n_samples[name] += n_samples

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(
        self, progress_bar=False, sqrtm_implementation: str = "blocked", sqrtm_num_iters: int = 200
    ) -> dict[str, torch.Tensor]:

        # if sqrtm_implementation == "iterative":
        #     ...
        #     scale_names_prog_bar = tqdm(
        #         self.scales, desc="Computing scale", disable=not progress_bar, total=len(self.scales)
        #     )
        #     for name in scale_names_prog_bar:
        #         compute_device = self.compute_devices[name]
        #         scale = self.scales[name].to(compute_device)
        #         scale = scale.unsqueeze(0)
        #         scale = sqrtm_newton_schulz(scale, numIters=sqrtm_num_iters)
        #         scale = scale.squeeze(0)
        #         scale = scale * (1 / math.sqrt(self.n_samples[name]))
        #         scale = scale.cpu()
        #         self.scales[name] = scale
        # elif sqrtm_implementation == "blocked":
        if sqrtm_implementation == "blocked":
            # convert to numpy
            for name in self.scales:
                if not self._force_cpu:
                    self.scales[name] = self.scales[name].numpy()

            # *: compute the square root sequentially

            # scale_names_prog_bar = tqdm.tqdm(
            #     self.scales, desc="Computing scale", total=len(self.scales), disable=not progress_bar
            # )
            # for name in scale_names_prog_bar:
            #     scale = self.scales[name]
            #     scale = sqrtm_scipy(scale)
            #     self.scales[name] = scale

            # *: compute the square root in parallel, no progress bar

            # num_cores = multiprocessing.cpu_count()
            # num_processes = max(1, num_cores // 64)

            # with multiprocessing.Pool(num_processes) as pool:
            #     self.scales = dict(
            #         zip(
            #             self.scales.keys(),
            #             pool.map(sqrtm_scipy, self.scales.values()),
            #         )
            #     )

            # *: compute the square root in parallel, with progress bar
            num_cores = multiprocessing.cpu_count()
            # num_processes = max(1, num_cores // 64)
            num_processes = max(1, num_cores // 2)

            with multiprocessing.Pool(num_processes) as pool:
                with tqdm(total=len(self.scales), desc="Computing scale", disable=not progress_bar) as pbar:
                    for name, scale in zip(self.scales.keys(), pool.imap(sqrtm_scipy, self.scales.values())):
                        self.scales[name] = scale
                        pbar.update()

            # convert to torch tensor
            for name in self.scales:
                scale = self.scales[name]
                n_samples = self.n_samples[name]
                scale = torch.from_numpy(scale).to(self.torch_dtype).to(self.compute_devices[name])
                scale = scale * (1 / math.sqrt(n_samples))
                self.scales[name] = scale
        else:
            raise ValueError(f"Unknown sqrtm_implementation: {sqrtm_implementation}")

        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class ScaleHookFactoryMeanAbs:
    """
    The idea of this scale drived from https://arxiv.org/abs/2402.02446
    it calculates the mean of x.abs() among the last dimension.
    """

    def __init__(self, torch_dtype, scale_clamp_min: float = 1e-4) -> None:
        self.scales = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.torch_dtype = torch_dtype
        self.handles = []
        self.scale_clamp_min = scale_clamp_min

    @torch.no_grad()
    def get_scale_hook(self, name: str) -> callable:
        """ """

        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.reshape(-1, x.shape[-1])
            n_samples, in_features = x.shape
            if self.scales[name] is None:
                if self.torch_dtype is None:
                    self.torch_dtype = x.dtype
                self.compute_devices[name] = x.device
                if self.compute_devices[name].type == "cpu":
                    logger.warning("Using CPU for computing LQER, this may be slow")
                self.scales[name] = torch.zeros(in_features, dtype=torch.float64)  # *: hard-coded float64

            compute_device = self.compute_devices[name]
            scale = self.scales[name].to(compute_device)
            x = x.to(self.torch_dtype)
            x_abs = x.abs().view(-1, x.shape[-1]).mean(0)
            scale = torch.maximum(scale, x_abs)
            self.scales[name] = scale.cpu()
            self.n_samples[name] += n_samples

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, progress_bar=False) -> dict[str, torch.Tensor]:
        scale_names_prog_bar = tqdm(
            self.scales, desc="Computing scale", disable=not progress_bar, total=len(self.scales)
        )

        for name in scale_names_prog_bar:
            # for name in self.scales:
            scale = self.scales[name].to(self.compute_devices[name])
            scale = scale.clamp(min=self.scale_clamp_min)
            scale = scale / torch.sqrt(scale.min() * scale.max())
            self.scales[name] = scale

        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class ScaleHookFactoryIdentity:
    """
    identity scale, which is torch.ones, which is identity matrix after expansion
    """

    def __init__(self, torch_dtype) -> None:
        self.scales = {}
        self.in_features = {}
        self.handles = []
        self.torch_dtype = torch_dtype

    def get_scale_hook(self, name: str) -> callable:
        self.scales[name] = None
        self.in_features[name] = None

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ):
            if self.in_features[name] is None:
                if self.torch_dtype is None:
                    self.torch_dtype = input[0].dtype
                x = input[0]
                x = x.view(-1, x.shape[-1])
                self.in_features[name] = x.shape[-1]

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        for name in self.scales:
            self.scales[name] = torch.ones(self.in_features[name], dtype=self.torch_dtype)
        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class ScaleHookFactoryMixed:
    def __init__(self, torch_dtype):
        self.scale_hook_factory_diag = ScaleHookFactoryDiagonal(torch_dtype)
        self.scale_hook_factory_rxx = ScaleHookFactoryRxx(torch_dtype)
        self.scale_hook_factory_identity = ScaleHookFactoryIdentity(torch_dtype)
        self.handles = []

    def get_scale_hook_diag(self, name: str) -> callable:
        logger.debug(f"Getting diag scale hook for {name}")
        return self.scale_hook_factory_diag.get_scale_hook(name)

    def get_scale_hook_rxx(self, name: str) -> callable:
        logger.debug(f"Getting rxx scale hook for {name}")
        return self.scale_hook_factory_rxx.get_scale_hook(name)

    def get_scale_hook_identity(self, name: str) -> callable:
        logger.debug(f"Getting identity scale hook for {name}")
        return self.scale_hook_factory_identity.get_scale_hook(name)

    def get_scale_dict(
        self, progress_bar=False, sqrtm_implementation: str = "blocked", sqrtm_num_iters: int = 200
    ) -> dict[str, torch.Tensor]:
        scales_diag = self.scale_hook_factory_diag.get_scale_dict(progress_bar)
        scales_rxx = self.scale_hook_factory_rxx.get_scale_dict(progress_bar, sqrtm_implementation, sqrtm_num_iters)
        scales_identity = self.scale_hook_factory_identity.get_scale_dict()
        scales = {**scales_diag, **scales_rxx, **scales_identity}
        return scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def register_scale_hooks(
    model: torch.nn.Module,
    layers_to_register_and_share: list[str],
    mode: str = "diagonal",
    torch_dtype: torch.dtype = None,
    mode_map: dict[str, str] = None,
):
    if mode in ["diagonal", "diag", "rxx", "identity", "lqer", 'hess']:
        if mode in ["diagonal", "diag"]:
            hook_factory = ScaleHookFactoryDiagonal(torch_dtype)
        elif mode == "rxx":
            hook_factory = ScaleHookFactoryRxx(torch_dtype)
        elif mode == "identity":
            hook_factory = ScaleHookFactoryIdentity(torch_dtype)
        elif mode == "lqer":
            hook_factory = ScaleHookFactoryMeanAbs(torch_dtype)
        elif mode == 'hess':
            hook_factory = ScaleHookFactoryHess(torch_dtype)
        else:
            raise ValueError(f"mode {mode} is not supported")

        for target_and_share in layers_to_register_and_share:
            target_layer_name = target_and_share["target_layer"]
            target_layer = get_layer_by_name(model, target_layer_name)
            handle = target_layer.register_forward_hook(hook_factory.get_scale_hook(target_layer_name))
            hook_factory.handles.append(handle)
        
    elif mode == "mixed":
        assert isinstance(mode_map, dict)
        hook_factory = ScaleHookFactoryMixed(torch_dtype)

        for target_and_share in layers_to_register_and_share:
            target_layer_name = target_and_share["target_layer"]
            target_layer = get_layer_by_name(model, target_layer_name)
            matched_pattern = find_matched_pattern(target_layer_name, list(mode_map.keys()))
            assert matched_pattern is not None, f"Cannot find matched pattern for {target_layer_name}"
            matched_mode = mode_map[matched_pattern]
            if matched_mode == "diag":
                handle = target_layer.register_forward_hook(hook_factory.get_scale_hook_diag(target_layer_name))
            elif matched_mode == "rxx":
                handle = target_layer.register_forward_hook(hook_factory.get_scale_hook_rxx(target_layer_name))
            elif matched_mode == "identity":
                handle = target_layer.register_forward_hook(hook_factory.get_scale_hook_identity(target_layer_name))
            else:
                raise ValueError(f"Unknown matched pattern: {matched_mode}")
            hook_factory.handles.append(handle)
    else:
        raise ValueError(f"mode {mode} is not supported")

    return hook_factory


def share_scales(
    scale_dict: dict[str, torch.Tensor],
    layers_to_register_and_share: list[str],
):
    """
    Share scales among layers (share the same scale tensor rather than duplicate scales)

    Some layers in the model may share the same input, and thus the same scale.

    For example, the k_proj, q_proj, and v_proj in the self-attention layer in the transformer model share the same scale.
    """
    for target_and_share in layers_to_register_and_share:
        target_layer_name = target_and_share["target_layer"]
        layers_sharing_scale = target_and_share["layers_sharing_scale"]
        for layer_sharing_scale in layers_sharing_scale:
            scale_dict[layer_sharing_scale] = scale_dict[target_layer_name]
