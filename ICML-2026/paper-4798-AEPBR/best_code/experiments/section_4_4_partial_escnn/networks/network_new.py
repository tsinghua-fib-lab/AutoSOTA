from typing import Tuple
import time
import hashlib
from pathlib import Path

from escnn import group, nn  # IMPORTANT: do NOT import escnn.gspaces
from escnn.nn import EquivariantModule, GeometricTensor, FieldType

import os
import sys

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from escnn2.r3convolution import R3Conv
from escnn2 import gspaces  # IMPORTANT: use escnn2.gspaces everywhere for 3D

from .networks_cnn import EqResBlock, ResBlock


sys.path.append("..")
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)


_EQUIVARIANT_Q_CACHE = {}
_EQUIVARIANT_Q_CACHE_DIR = Path(
    os.environ.get("PARTIAL_ESCNN_Q_CACHE_DIR", "/tmp/partial_escnn_q_cache")
)


def _repr_signature(repr_) -> tuple:
    return (
        getattr(repr_, "name", None),
        getattr(repr_, "size", None),
        getattr(repr_, "id", None),
    )


def _field_type_signature(field_type: FieldType) -> tuple:
    return (
        getattr(field_type.gspace, "name", None),
        tuple(_repr_signature(repr_) for repr_ in field_type.representations),
    )


def _kernel_signature(kernel_size) -> tuple:
    if isinstance(kernel_size, tuple):
        return tuple(kernel_size)
    return (kernel_size, kernel_size, kernel_size)


def _equivariant_q_cache_key(in_type: FieldType, out_type: FieldType, kernel_size) -> tuple:
    return (
        _field_type_signature(in_type),
        _field_type_signature(out_type),
        _kernel_signature(kernel_size),
    )


def _equivariant_q_cache_path(cache_key: tuple) -> Path:
    digest = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()
    return _EQUIVARIANT_Q_CACHE_DIR / f"{digest}.pt"


def _persist_equivariant_q_cache(cache_path: Path, tensor: torch.Tensor) -> None:
    if os.environ.get("PARTIAL_ESCNN_DISABLE_Q_CACHE_WRITE", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        print("[network_new] Disk cache write disabled by PARTIAL_ESCNN_DISABLE_Q_CACHE_WRITE.")
        return

    _EQUIVARIANT_Q_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp.{os.getpid()}")
    try:
        # The legacy serializer is more robust here than the zip writer for these large cache tensors.
        torch.save(tensor, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, cache_path)
        print(f"[network_new] Saved complement basis cache to {cache_path}.")
    except Exception as exc:
        print(
            f"[network_new] Failed to persist complement basis cache to {cache_path}: {exc}. "
            "Continuing without a disk cache entry."
        )
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def _basis_dim(basisexpansion) -> int:
    if hasattr(basisexpansion, "dimension") and callable(basisexpansion.dimension):
        return int(basisexpansion.dimension())
    if hasattr(basisexpansion, "dim"):
        return int(basisexpansion.dim)
    if hasattr(basisexpansion, "D"):
        return int(basisexpansion.D)
    raise RuntimeError("Cannot read basisexpansion dimension (tried dimension(), dim, D).")


def _flatten_basis_filter(fi: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if fi.ndim == 3:
        return fi
    if fi.ndim == 5:
        Cout, Cin, k1, k2, k3 = fi.shape
        if not (k1 == k2 == k3 == kernel_size):
            raise RuntimeError(f"Unexpected kernel in basisexpansion output: {fi.shape}, expected K={kernel_size}")
        return fi.reshape(Cout, Cin, k1 * k2 * k3)
    raise RuntimeError(f"Unexpected basisexpansion output shape: {tuple(fi.shape)}")


@torch.no_grad()
def build_equivariant_Q_from_escnn2_r3conv(
    ref_r3conv: R3Conv,
    kernel_size: int,
    *,
    dtype=torch.float32,
) -> torch.Tensor:
    cache_key = _equivariant_q_cache_key(ref_r3conv.in_type, ref_r3conv.out_type, kernel_size)
    cached_q = _EQUIVARIANT_Q_CACHE.get(cache_key)
    if cached_q is not None:
        return cached_q.clone()

    cache_path = _equivariant_q_cache_path(cache_key)
    if cache_path.exists():
        try:
            cached_q = torch.load(cache_path, map_location="cpu")
            _EQUIVARIANT_Q_CACHE[cache_key] = cached_q
            return cached_q.clone()
        except (RuntimeError, OSError, EOFError, ValueError) as exc:
            print(
                f"[network_new] Failed to load cached complement basis from "
                f"{cache_path}: {exc}. Rebuilding cache entry."
            )
            try:
                cache_path.unlink()
            except FileNotFoundError:
                pass

    bm = ref_r3conv.basisexpansion
    D = _basis_dim(bm)

    w0 = torch.zeros(D, dtype=torch.float32)
    f0 = _flatten_basis_filter(bm(w0), kernel_size)
    Cout, Cin, K3 = f0.shape
    N = Cout * Cin * K3

    B = torch.empty(N, D, dtype=dtype)

    for i in range(D):
        wi = torch.zeros(D, dtype=torch.float32)
        wi[i] = 1.0
        fi = _flatten_basis_filter(bm(wi), kernel_size).to(dtype=dtype)
        B[:, i] = fi.reshape(-1)

    Q, _ = torch.linalg.qr(B, mode="reduced")
    Q = Q.to(dtype=torch.float32)
    _EQUIVARIANT_Q_CACHE[cache_key] = Q.cpu()
    _persist_equivariant_q_cache(cache_path, _EQUIVARIANT_Q_CACHE[cache_key])
    return Q.clone()


class _OrthogonalComplementParam(torch.nn.Module):
    def __init__(self, Q: torch.Tensor):
        super().__init__()
        if Q.ndim != 2:
            raise ValueError(f"Q must be 2D [N, D], got {tuple(Q.shape)}")
        self.register_buffer("Q", Q)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        w = W.reshape(-1)
        Q = self.Q
        w = w - Q @ (Q.T @ w)
        return w.view_as(W)


def _load_or_build_equivariant_q(
    in_type: FieldType,
    out_type: FieldType,
    kernel_size,
    *,
    stride=1,
    padding=0,
    dilation=1,
    padding_mode="zeros",
    groups=1,
    **r3conv_kwargs,
) -> torch.Tensor:
    cache_key = _equivariant_q_cache_key(in_type, out_type, kernel_size)
    cached_q = _EQUIVARIANT_Q_CACHE.get(cache_key)
    if cached_q is not None:
        return cached_q.clone()

    ref = R3Conv(
        in_type=in_type,
        out_type=out_type,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        padding_mode=padding_mode,
        groups=groups,
        bias=False,
        **r3conv_kwargs,
    )
    return build_equivariant_Q_from_escnn2_r3conv(ref, kernel_size=kernel_size)


class ComplementR3Conv(torch.nn.Module):
    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        padding_mode="zeros",
        groups=1,
        bias=False,
        **r3conv_kwargs,
    ):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("groups!=1 not implemented.")

        self.in_type = in_type
        self.out_type = out_type

        cache_key = _equivariant_q_cache_key(in_type, out_type, kernel_size)

        self.conv = torch.nn.Conv3d(
            in_channels=in_type.size,
            out_channels=out_type.size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        if cache_key in _EQUIVARIANT_Q_CACHE:
            Q = _EQUIVARIANT_Q_CACHE[cache_key].clone()
            print(
                f"[network_new] Reusing cached equivariant complement basis "
                f"(Cin={in_type.size}, Cout={out_type.size}, K={_kernel_signature(kernel_size)})."
            )
        else:
            print(
                f"[network_new] Building equivariant complement basis "
                f"(Cin={in_type.size}, Cout={out_type.size}, K={_kernel_signature(kernel_size)}) ..."
            )
            start_time = time.time()
            Q = _load_or_build_equivariant_q(
                in_type,
                out_type,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode=padding_mode,
                groups=groups,
                **r3conv_kwargs,
            )
            _EQUIVARIANT_Q_CACHE[cache_key] = Q.detach().cpu()
            print(
                f"[network_new] Finished complement basis build in "
                f"{time.time() - start_time:.2f}s."
            )
        parametrize.register_parametrization(self.conv, "weight", _OrthogonalComplementParam(Q))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PenalizedDenseR3Conv(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        padding_mode="zeros",
        groups=1,
        bias=False,
        **r3conv_kwargs,
    ):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("groups!=1 not implemented.")

        self.in_type = in_type
        self.out_type = out_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        cache_key = _equivariant_q_cache_key(in_type, out_type, kernel_size)
        if cache_key in _EQUIVARIANT_Q_CACHE:
            Q = _EQUIVARIANT_Q_CACHE[cache_key].clone()
            print(
                f"[network_new] Reusing cached equivariant basis "
                f"(Cin={in_type.size}, Cout={out_type.size}, K={_kernel_signature(kernel_size)})."
            )
        else:
            print(
                f"[network_new] Building equivariant basis "
                f"(Cin={in_type.size}, Cout={out_type.size}, K={_kernel_signature(kernel_size)}) ..."
            )
            start_time = time.time()
            Q = _load_or_build_equivariant_q(
                in_type,
                out_type,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode=padding_mode,
                groups=groups,
                **r3conv_kwargs,
            )
            _EQUIVARIANT_Q_CACHE[cache_key] = Q.detach().cpu()
            print(
                f"[network_new] Finished equivariant basis build in "
                f"{time.time() - start_time:.2f}s."
            )

        # Store Q on CPU (NOT a buffer) to avoid GPU memory blowup
        self._q_cpu = Q
        self.conv = torch.nn.Conv3d(
            in_channels=in_type.size,
            out_channels=out_type.size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    @property
    def Q(self) -> torch.Tensor:
        return self._q_cpu

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        y = self.conv(x.tensor)
        return GeometricTensor(y, self.out_type)

    def evaluate_output_shape(self, input_shape):
        B, _, Din, Hin, Win = input_shape
        k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        s = self.stride if isinstance(self.stride, int) else self.stride[0]
        p = self.padding if isinstance(self.padding, int) else self.padding[0]
        d = self.dilation if isinstance(self.dilation, int) else self.dilation[0]

        def out_dim(L):
            return (L + 2 * p - d * (k - 1) - 1) // s + 1

        return (B, self.out_type.size, out_dim(Din), out_dim(Hin), out_dim(Win))

    def _flatten_weight(self) -> torch.Tensor:
        return self.conv.weight.reshape(-1)

    def _qTw_chunked(self, w, chunk_mb=512):
        """Compute Q^T @ w in chunks to avoid GPU OOM."""
        q_cpu = self._q_cpu
        N, D = q_cpu.shape
        bytes_per_col = N * q_cpu.element_size()
        chunk_cols = max(1, int(chunk_mb * 1024 * 1024 / bytes_per_col))
        result = torch.zeros(D, device=w.device, dtype=w.dtype)
        for start in range(0, D, chunk_cols):
            end = min(start + chunk_cols, D)
            q_chunk = q_cpu[:, start:end].to(device=w.device, dtype=w.dtype, non_blocking=True)
            result[start:end] = q_chunk.T @ w
            del q_chunk
        return result

    def equivariant_projection_weight(self) -> torch.Tensor:
        w = self._flatten_weight()
        coeffs = self._qTw_chunked(w)
        q_cpu = self._q_cpu
        N, D = q_cpu.shape
        chunk_mb = 512
        bytes_per_col = N * q_cpu.element_size()
        chunk_cols = max(1, int(chunk_mb * 1024 * 1024 / bytes_per_col))
        result = torch.zeros(N, device=w.device, dtype=w.dtype)
        for start in range(0, D, chunk_cols):
            end = min(start + chunk_cols, D)
            q_chunk = q_cpu[:, start:end].to(device=w.device, dtype=w.dtype, non_blocking=True)
            result += q_chunk @ coeffs[start:end]
            del q_chunk
        return result.view_as(self.conv.weight)

    def _projection_norm_sq(self):
        """||Q @ Q^T @ w||^2 = ||Q^T @ w||^2 (orthonormal columns)."""
        w = self._flatten_weight()
        coeffs = self._qTw_chunked(w)
        return coeffs.pow(2).sum()

    def nonequivariant_projection_weight(self) -> torch.Tensor:
        return self.conv.weight - self.equivariant_projection_weight()

    def projection_penalty(self, conv_wd: float = 0.0, basic_wd: float = 0.0) -> torch.Tensor:
        w = self._flatten_weight()
        w_norm_sq = w.pow(2).sum()
        w_eq_norm_sq = self._projection_norm_sq()
        w_non_eq_norm_sq = w_norm_sq - w_eq_norm_sq
        return float(conv_wd) * w_eq_norm_sq + float(basic_wd) * w_non_eq_norm_sq

    def penalty_terms(self) -> dict[str, torch.Tensor]:
        w = self._flatten_weight()
        w_norm_sq = w.pow(2).sum()
        w_eq_norm_sq = self._projection_norm_sq()
        return {
            "equivariant_l2": w_eq_norm_sq,
            "nonequivariant_l2": w_norm_sq - w_eq_norm_sq,
        }

    def project_equivariant_inplace(self) -> None:
        with torch.no_grad():
            self.conv.weight.copy_(self.equivariant_projection_weight())

    def project_nonequivariant_inplace(self) -> None:
        with torch.no_grad():
            self.conv.weight.copy_(self.nonequivariant_projection_weight())


class TensorDropoutField(EquivariantModule):
    def __init__(self, field_type: FieldType, p: float):
        super().__init__()
        self.in_type = field_type
        self.out_type = field_type
        self.dropout = torch.nn.Dropout3d(p)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return GeometricTensor(self.dropout(x.tensor), self.out_type)

    def evaluate_output_shape(self, input_shape):
        return input_shape


class TensorBatchNorm3dField(EquivariantModule):
    def __init__(self, field_type: FieldType):
        super().__init__()
        self.in_type = field_type
        self.out_type = field_type
        self.bn = torch.nn.BatchNorm3d(field_type.size)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return GeometricTensor(self.bn(x.tensor), self.out_type)

    def evaluate_output_shape(self, input_shape):
        return input_shape


class TensorELUField(EquivariantModule):
    def __init__(self, field_type: FieldType):
        super().__init__()
        self.in_type = field_type
        self.out_type = field_type
        self.elu = torch.nn.ELU()

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return GeometricTensor(self.elu(x.tensor), self.out_type)

    def evaluate_output_shape(self, input_shape):
        return input_shape


class TensorAvgPool3dField(EquivariantModule):
    def __init__(self, field_type: FieldType, kernel_size, stride=None, padding=0):
        super().__init__()
        self.in_type = field_type
        self.out_type = field_type
        self.pool = torch.nn.AvgPool3d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        return GeometricTensor(self.pool(x.tensor), self.out_type)

    def evaluate_output_shape(self, input_shape):
        batch, channels, depth, height, width = input_shape

        def _triple(value):
            if isinstance(value, tuple):
                return value
            return (value, value, value)

        kernel = _triple(self.pool.kernel_size)
        stride = _triple(self.pool.stride)
        padding = _triple(self.pool.padding)

        def out_dim(size, k, s, p):
            return (size + 2 * p - k) // s + 1

        return (
            batch,
            channels,
            out_dim(depth, kernel[0], stride[0], padding[0]),
            out_dim(height, kernel[1], stride[1], padding[1]),
            out_dim(width, kernel[2], stride[2], padding[2]),
        )


class TensorResidualBlock(EquivariantModule):
    def __init__(self, block: EquivariantModule, skip: EquivariantModule):
        super().__init__()
        self.in_type = block.in_type
        self.out_type = block.out_type
        self.block = block
        self.skip = skip

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        block_out = self.block(x)
        skip_out = self.skip(x)
        return GeometricTensor(block_out.tensor + skip_out.tensor, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self.block.evaluate_output_shape(input_shape)


class RPPBlock(nn.EquivariantModule):
    def __init__(self, conveq, conv, layer_id):
        super(RPPBlock, self).__init__()
        self.conveq = conveq
        self.conv = conv
        self.out_type = conveq.out_type
        self.in_type = conveq.in_type
        self.layer_id = layer_id

    def forward(self, x):
        conv_out = self.conv(x.tensor)
        eq_out = self.conveq(x)
        return eq_out + self.out_type(conv_out)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return 0


class _PenalizedSteerable3DResnetBase(torch.nn.Module):
    def __init__(
        self,
        activation=nn.GatedNonLinearityUniform,
        last_tensor=False,
        n_classes=10,
        n_channels=1,
        mnist_type="single",
        f=True,
        N=-1,
        restrict=False,
        one_eq=False,
        channels=6,
        iteration=0,
        dropout=0,
        L_in=2,
        L_out=None,
        invariant=True,
        learn_eq=None
    ):
        super().__init__()

        self._L_in, self._L_out = L_in, L_out
        self._id_offset = 100 * iteration

        self._c_fac = [channels]
        splits = 1

        assert not (last_tensor and restrict), "last tensor and restrict cannot both be on"

        # IMPORTANT: build gspace using escnn2.gspaces
        if N == -1:
            if f:
                self.L = 2
                self.act_r2 = gspaces.flipRot3dOnR3(maximum_frequency=2 * self.L)
                self.restrict_id = (None, 1)
                self._group_name = "O3"
            else:
                self.L = 3
                self.act_r2 = gspaces.rot3dOnR3(maximum_frequency=2 * self.L)
                self.restrict_id = 1
                self._group_name = "SO3"

        elif N > 0:
            raise "not supported yet"

        else:
            self.L = 0
            if f:
                raise "not supported yet"
            else:
                self.act_r2 = gspaces.trivialOnR3()
                self.restrict_id = 1
                self._group_name = "C1"

        self._f = f
        self._N = N
        self.activation_fn = activation
        self._activation_name = activation.__name__
        self.last_tensor = last_tensor
        self.n_classes = n_classes
        self._n_channels = n_channels
        self.mnist_type = mnist_type
        self.restrict = restrict
        self._one_eq = one_eq
        self._splits = splits
        self.dropout = dropout
        self.invariant = invariant

        self._init_layers(splits)

    def _init_layers(self, splits):
        if self.mnist_type == "double":
            w = h = d = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif self.mnist_type == "single":
            w = h = d = 29
            padding_3 = (1, 1, 1)
            padding_4 = (2, 2, 2)

        # FieldType now uses escnn2 gspace (self.act_r2) -> satisfies escnn2 R3Conv assertion
        self.in_type = nn.FieldType(
            self.act_r2, [self.act_r2.trivial_repr for _ in range(self._n_channels)]
        )
        self.upsample = nn.R3Upsampling(self.in_type, size=(h, w, d))
        self.mask = nn.MaskModule(self.in_type, h, margin=1)

        self.layers_eq = torch.nn.ModuleList()
        dropout = self.dropout

        for channel in range(splits):
            # Block 1
            activation_1, out_type_1_pre = self._activation_and_out_type(self._c_fac[channel], min(2, self.L))
            block_1 = nn.SequentialModule(
                R3Conv(self.in_type, out_type_1_pre, kernel_size=7, padding=2),
                nn.FieldDropout(out_type_1_pre, 0.05 * dropout),
                nn.IIDBatchNorm3d(out_type_1_pre),
                activation_1,
            )
            out_type_1_post = block_1.out_type

            block_1_conv = torch.nn.Sequential(
                ComplementR3Conv(self.in_type, out_type_1_post, kernel_size=7, padding=2),
                torch.nn.Dropout(0.05 * dropout),
                torch.nn.BatchNorm3d(out_type_1_post.size),
                torch.nn.ELU(),
            )

            # Block 2
            activation_2, out_type_2_pre = self._activation_and_out_type(self._c_fac[channel] * 2, min(3, self.L))
            block_2 = nn.SequentialModule(
                R3Conv(block_1.out_type, out_type_2_pre, kernel_size=5, padding=2),
                nn.FieldDropout(out_type_2_pre, 0.05 * dropout),
                nn.IIDBatchNorm3d(out_type_2_pre),
                activation_2,
            )
            out_type_2_post = block_2.out_type

            block_2_conv = torch.nn.Sequential(
                ComplementR3Conv(out_type_1_post, out_type_2_post, kernel_size=5, padding=2),
                torch.nn.Dropout(0.05 * dropout),
                torch.nn.BatchNorm3d(out_type_2_post.size),
                torch.nn.ELU(),
            )

            skip_1 = nn.SequentialModule(
                R3Conv(block_1.in_type, out_type_2_pre, kernel_size=7, padding=2),
                activation_2,
            )
            skip_1_post = skip_1.out_type

            skip_1_conv = torch.nn.Sequential(
                ComplementR3Conv(block_1.in_type, skip_1_post, kernel_size=7, padding=2),
                torch.nn.BatchNorm3d(skip_1_post.size),
                torch.nn.ELU(),
            )

            resblock_eq_1 = EqResBlock(nn.SequentialModule(block_1, block_2), skip_1)
            resblock_conv_1 = ResBlock(torch.nn.Sequential(block_1_conv, block_2_conv), skip_1_conv)
            resblock_1 = RPPBlock(resblock_eq_1, resblock_conv_1, layer_id=0)

            pool_1 = nn.PointwiseAvgPoolAntialiased3D(block_2.out_type, sigma=0.66, stride=2, padding=1)

            # Block 3
            activation_3, out_type_3_pre = self._activation_and_out_type(self._c_fac[channel] * 4)
            block_3 = nn.SequentialModule(
                R3Conv(block_2.out_type, out_type_3_pre, kernel_size=3, stride=2, padding=padding_3),
                nn.FieldDropout(out_type_3_pre, 0.1 * dropout),
                nn.IIDBatchNorm3d(out_type_3_pre),
                activation_3,
            )
            out_type_3_post = block_3.out_type

            block_3_conv = torch.nn.Sequential(
                ComplementR3Conv(out_type_2_post, out_type_3_post, kernel_size=3, stride=2, padding=padding_3),
                torch.nn.Dropout(0.1 * dropout),
                torch.nn.BatchNorm3d(out_type_3_post.size),
                torch.nn.ELU(),
            )

            pool_2 = nn.PointwiseAvgPoolAntialiased3D(block_3.out_type, sigma=0.66, stride=2, padding=1)
            pool_2_conv = torch.nn.AvgPool3d(5, stride=2, padding=1)

            # Block 4
            activation_4, out_type_4_pre = self._activation_and_out_type(self._c_fac[channel] * 6)
            conv_4 = R3Conv(block_3.out_type, out_type_4_pre, kernel_size=3, stride=2, padding=padding_4)
            block_4 = nn.SequentialModule(
                conv_4,
                nn.IIDBatchNorm3d(activation_4.in_type),
                nn.FieldDropout(out_type_4_pre, 0.1 * dropout),
                activation_4,
            )
            out_type_4_post = block_4.out_type

            block_4_conv = torch.nn.Sequential(
                ComplementR3Conv(out_type_3_post, out_type_4_post, kernel_size=3, stride=2, padding=padding_4),
                torch.nn.Dropout(0.1 * dropout),
                torch.nn.BatchNorm3d(out_type_4_post.size),
                torch.nn.ELU(),
            )

            skip_2 = nn.SequentialModule(
                nn.PointwiseAvgPoolAntialiased3D(pool_1.out_type, sigma=0.66, stride=2, padding=0),
                R3Conv(block_3.in_type, out_type_4_pre, kernel_size=3, padding=0),
                activation_4,
            )
            skip_2_post = skip_2.out_type

            skip_2_conv = torch.nn.Sequential(
                torch.nn.AvgPool3d(5, stride=2, padding=0),
                ComplementR3Conv(block_3.in_type, skip_2_post, kernel_size=3, padding=0),
                torch.nn.BatchNorm3d(skip_2_post.size),
                torch.nn.ELU(),
            )

            resblock_eq_2 = EqResBlock(nn.SequentialModule(block_3, pool_2, block_4), skip_2)
            resblock_conv_2 = ResBlock(torch.nn.Sequential(block_3_conv, pool_2_conv, block_4_conv), skip_2_conv)
            resblock_2 = RPPBlock(resblock_eq_2, resblock_conv_2, 1)

            if self.restrict:
                restriction_4, block_5_in = self._restrict_layer(block_4.out_type, self.restrict_id)
            else:
                restriction_4 = lambda x: x
                block_5_in = block_4.out_type

            # Block 5
            activation_5, out_type_5_pre = self._activation_and_out_type(self._c_fac[channel] * 6, min(2, self.L))
            conv_5 = R3Conv(block_5_in, out_type_5_pre, kernel_size=3, stride=1, padding=1)
            block_5 = nn.SequentialModule(
                conv_5,
                nn.IIDBatchNorm3d(out_type_5_pre),
                nn.FieldDropout(out_type_5_pre, 0.1 * dropout),
                activation_5,
            )
            out_type_5_post = block_5.out_type

            block_5_conv = torch.nn.Sequential(
                ComplementR3Conv(block_5_in, out_type_5_post, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm3d(out_type_5_post.size),
                torch.nn.Dropout(0.1 * dropout),
                torch.nn.ELU(),
            )

            pool_3 = nn.PointwiseAvgPoolAntialiased3D(block_5.out_type, sigma=0.66, stride=1, padding=1)
            pool_3_conv = torch.nn.AvgPool3d(3, stride=1, padding=0)

            if self.last_tensor:
                out_type_6 = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.irreps[3] + self.act_r2.irreps[2]] * (self._c_fac[channel] * 8),
                )
                tensor_out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * (self._c_fac[channel] * 8),
                )
                tensor_act = nn.TensorProductModule(out_type_6, tensor_out_type)
            elif not self.invariant:
                irreps = [self.act_r2.fibergroup.irrep(*irr) for irr in self.act_r2.fibergroup.bl_irreps(self.L)]
                out_type_6 = nn.FieldType(self.act_r2, irreps * int(self._c_fac[channel] * 2))
            else:
                out_type_6 = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * int(self._c_fac[channel] * 8),
                )

            block_6 = nn.SequentialModule(
                R3Conv(block_5.out_type, out_type_6, kernel_size=1, bias=False),
            )
            block_6_conv = torch.nn.Sequential(
                ComplementR3Conv(out_type_5_post, out_type_6, kernel_size=1, bias=False),
            )

            skip_3 = nn.SequentialModule(
                R3Conv(block_5_in, out_type_6, kernel_size=3, padding=0),
            )
            skip_3_conv = torch.nn.Sequential(
                ComplementR3Conv(block_5_in, out_type_6, kernel_size=3, padding=0),
            )

            resblock_eq_3 = EqResBlock(nn.SequentialModule(block_5, pool_3, block_6), skip_3)
            resblock_conv_3 = ResBlock(torch.nn.Sequential(block_5_conv, pool_3_conv, block_6_conv), skip_3_conv)
            resblock_3 = RPPBlock(resblock_eq_3, resblock_conv_3, 2)

            layers = [
                self.upsample,
                resblock_1,
                pool_1,
                resblock_2,
                restriction_4,
                resblock_3,
            ]
            if not self.restrict:
                del layers[-2]
            if self.last_tensor:
                layers.append(tensor_act)

            self.layers_eq.append(nn.SequentialModule(*layers))

        nr_features = int(splits * out_type_6.size)
        self.full_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(nr_features),
            torch.nn.ELU(),
            torch.nn.Linear(nr_features, self.n_classes),
        )

    def _activation_and_out_type(self, channels, L=None):
        channels = int(channels)
        if L is None:
            L = self.L
        irreps = self.act_r2.fibergroup.bl_irreps(L)

        if self.activation_fn in {nn.FourierELU, nn.FourierPointwise}:
            L = 2
            irreps = self.act_r2.fibergroup.bl_irreps(L)
            try:
                N = self.act_r2.fibergroup.bl_regular_representation(L).size
            except AttributeError:
                N = self.act_r2.fibergroup.regular_representation.size
            if self._f:
                N //= 2
            activation = self.activation_fn(self.act_r2, irreps=irreps, N=N, channels=channels, type="thomson")
            out_type = activation.in_type

        elif self.activation_fn in {nn.QuotientFourierELU, nn.QuotientFourierPointwise}:
            irreps = self.act_r2.fibergroup.bl_irreps(L)
            try:
                N = self.act_r2.fibergroup.bl_regular_representation(L).size
            except AttributeError:
                N = self.act_r2.fibergroup.regular_representation.size
            if self._f:
                N //= 2
            activation = self.activation_fn(
                self.act_r2,
                subgroup_id=(False, -1) if not self._f else (False, True, -1),
                irreps=irreps,
                N=N,
                channels=channels,
                type="thomson",
            )
            out_type = activation.in_type

        elif self.activation_fn in {nn.NormNonLinearity}:
            c = 2
            irreps = c * channels * [group.directsum([self.act_r2.irrep(*id) for id in irreps])]
            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(out_type)

        elif self.activation_fn == nn.GatedNonLinearityUniform:
            irreps = channels * [
                group.directsum(
                    [self.act_r2.trivial_repr for _ in range((len(irreps)))]
                    + [self.act_r2.irrep(*id) for id in irreps]
                )
            ]
            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(out_type)

        elif self.activation_fn == nn.GatedNonLinearity1:
            irreps = channels * [self.act_r2.trivial_repr] + channels * [
                group.directsum([self.act_r2.irrep(*id) for id in irreps])
            ]
            out_type = nn.FieldType(self.act_r2, irreps)
            activation = self.activation_fn(out_type, gates=channels * ["gate"] + channels * ["gated"])

        elif self.activation_fn == nn.TensorProductModule:
            c = 2
            out_type = nn.FieldType(
                self.act_r2,
                c * channels * [group.directsum([self.act_r2.irrep(*id) for id in irreps])],
            )
            tensor_out_type = nn.FieldType(
                self.act_r2,
                c * channels * [group.directsum([self.act_r2.irrep(*id) for id in irreps])],
            )
            activation = self.activation_fn(out_type, tensor_out_type)

        return activation, out_type

    def forward(self, x):
        x = self.in_type(x)
        outs = [layers(x).tensor for layers in self.layers_eq]
        x = torch.cat(outs, axis=1)
        x = self.full_net(x.reshape(x.shape[0], -1))
        return x

    @classmethod
    def from_group(
        cls,
        group,
        activation=nn.GatedNonLinearityUniform,
        last_tensor=False,
        n_classes=10,
        n_channels=1,
        mnist_type="single",
        restrict=False,
        one_eq=True,
        channels=6,
        iteration=0,
        L_in=2,
        L_out=4,
        invariant=True,
        learn_eq=None,
        normalise_basis=None,
    ):
        try:
            if group == "SO3":
                N = -1
                f = False
            elif group == "O3":
                N = -1
                f = True
            elif group == "trivial":
                f = False
                N = 0
            else:
                raise AssertionError("invalid group")
        except Exception as e:
            raise AssertionError(f"invalid group, found exception: {e}")
        return cls(
            f=f,
            N=N,
            activation=activation,
            last_tensor=last_tensor,
            n_classes=n_classes,
            n_channels=n_channels,
            mnist_type=mnist_type,
            restrict=restrict,
            one_eq=one_eq,
            channels=channels,
            iteration=iteration,
            L_in=L_in,
            L_out=L_out,
            invariant=invariant,
        )

    @staticmethod
    def supported_activations():
        return {
            nn.NormNonLinearity,
            nn.GatedNonLinearity1,
            nn.GatedNonLinearityUniform,
            nn.FourierPointwise,
            nn.FourierELU,
            nn.TensorProductModule,
        }

    @property
    def network_name(self):
        return f"{self._group_name}{self.__class__.__name__}"


class PenalizedSteerableApprox3DResnet(_PenalizedSteerable3DResnetBase):
    """
    Single-branch dense-conv approximate equivariant 3D ResNet.

    Each convolution is a regular Conv3d wrapped as an EquivariantModule so the
    architecture keeps the same field-type flow, but the model exposes a
    projection penalty based on the cached equivariant subspace basis Q.

    By convention, ``conv_wd`` penalizes the equivariant projection norm and
    ``basic_wd`` penalizes the non-equivariant complement norm, matching the
    historical weighting names used elsewhere in the repo.
    """

    def __init__(self, *args, conv_wd: float = 0.0, basic_wd: float = 0.0, **kwargs):
        self._conv_wd = float(conv_wd)
        self._basic_wd = float(basic_wd)
        super().__init__(*args, **kwargs)

    def _restrict_layer(self, in_type, id):
        layers = []
        layers.append(nn.RestrictionModule(in_type, id))
        layers.append(nn.DisentangleModule(layers[-1].out_type))
        self.act_r2 = layers[-1].out_type.gspace
        self.L = 0
        restrict_layer = nn.SequentialModule(*layers)
        return restrict_layer, layers[-1].out_type

    def _init_layers(self, splits):
        if self.mnist_type == "double":
            w = h = d = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif self.mnist_type == "single":
            w = h = d = 29
            padding_3 = (1, 1, 1)
            padding_4 = (2, 2, 2)
        else:
            raise ValueError(f"Unsupported mnist_type: {self.mnist_type}")

        self.in_type = nn.FieldType(
            self.act_r2, [self.act_r2.trivial_repr for _ in range(self._n_channels)]
        )
        self.upsample = nn.R3Upsampling(self.in_type, size=(h, w, d))
        self.layers_eq = torch.nn.ModuleList()
        self._penalized_convs = torch.nn.ModuleList()
        dropout = self.dropout

        for channel in range(splits):
            activation_1, _ = self._activation_and_out_type(self._c_fac[channel], min(2, self.L))
            out_type_1 = activation_1.out_type
            conv_1 = PenalizedDenseR3Conv(self.in_type, out_type_1, kernel_size=7, padding=2)
            self._penalized_convs.append(conv_1)
            block_1 = nn.SequentialModule(
                conv_1,
                TensorDropoutField(out_type_1, 0.05 * dropout),
                TensorBatchNorm3dField(out_type_1),
                TensorELUField(out_type_1),
            )

            activation_2, _ = self._activation_and_out_type(self._c_fac[channel] * 2, min(3, self.L))
            out_type_2 = activation_2.out_type
            conv_2 = PenalizedDenseR3Conv(block_1.out_type, out_type_2, kernel_size=5, padding=2)
            self._penalized_convs.append(conv_2)
            block_2 = nn.SequentialModule(
                conv_2,
                TensorDropoutField(out_type_2, 0.05 * dropout),
                TensorBatchNorm3dField(out_type_2),
                TensorELUField(out_type_2),
            )

            skip_1_conv = PenalizedDenseR3Conv(block_1.in_type, out_type_2, kernel_size=7, padding=2)
            self._penalized_convs.append(skip_1_conv)
            skip_1 = nn.SequentialModule(
                skip_1_conv,
                TensorBatchNorm3dField(out_type_2),
                TensorELUField(out_type_2),
            )
            resblock_1 = TensorResidualBlock(nn.SequentialModule(block_1, block_2), skip_1)

            pool_1 = nn.PointwiseAvgPoolAntialiased3D(block_2.out_type, sigma=0.66, stride=2, padding=1)

            activation_3, _ = self._activation_and_out_type(self._c_fac[channel] * 4)
            out_type_3 = activation_3.out_type
            conv_3 = PenalizedDenseR3Conv(
                block_2.out_type,
                out_type_3,
                kernel_size=3,
                stride=2,
                padding=padding_3,
            )
            self._penalized_convs.append(conv_3)
            block_3 = nn.SequentialModule(
                conv_3,
                TensorDropoutField(out_type_3, 0.1 * dropout),
                TensorBatchNorm3dField(out_type_3),
                TensorELUField(out_type_3),
            )

            pool_2 = TensorAvgPool3dField(block_3.out_type, kernel_size=5, stride=2, padding=1)

            activation_4, _ = self._activation_and_out_type(self._c_fac[channel] * 6)
            out_type_4 = activation_4.out_type
            conv_4 = PenalizedDenseR3Conv(
                block_3.out_type,
                out_type_4,
                kernel_size=3,
                stride=2,
                padding=padding_4,
            )
            self._penalized_convs.append(conv_4)
            block_4 = nn.SequentialModule(
                conv_4,
                TensorDropoutField(out_type_4, 0.1 * dropout),
                TensorBatchNorm3dField(out_type_4),
                TensorELUField(out_type_4),
            )

            skip_2_conv = PenalizedDenseR3Conv(block_3.in_type, out_type_4, kernel_size=3, padding=0)
            self._penalized_convs.append(skip_2_conv)
            skip_2 = nn.SequentialModule(
                TensorAvgPool3dField(pool_1.out_type, kernel_size=5, stride=2, padding=0),
                skip_2_conv,
                TensorBatchNorm3dField(out_type_4),
                TensorELUField(out_type_4),
            )
            resblock_2 = TensorResidualBlock(nn.SequentialModule(block_3, pool_2, block_4), skip_2)

            if self.restrict:
                restriction_4, block_5_in = self._restrict_layer(block_4.out_type, self.restrict_id)
            else:
                restriction_4 = lambda x: x
                block_5_in = block_4.out_type

            activation_5, _ = self._activation_and_out_type(self._c_fac[channel] * 6, min(2, self.L))
            out_type_5 = activation_5.out_type
            conv_5 = PenalizedDenseR3Conv(block_5_in, out_type_5, kernel_size=3, stride=1, padding=1)
            self._penalized_convs.append(conv_5)
            block_5 = nn.SequentialModule(
                conv_5,
                TensorBatchNorm3dField(out_type_5),
                TensorDropoutField(out_type_5, 0.1 * dropout),
                TensorELUField(out_type_5),
            )

            pool_3 = TensorAvgPool3dField(block_5.out_type, kernel_size=3, stride=1, padding=0)

            if self.last_tensor:
                out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.irreps[3] + self.act_r2.irreps[2]] * (self._c_fac[channel] * 8),
                )
                tensor_out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * (self._c_fac[channel] * 8),
                )
                tensor_act = nn.TensorProductModule(out_type, tensor_out_type)
            elif not self.invariant:
                irreps = [self.act_r2.fibergroup.irrep(*irr) for irr in self.act_r2.fibergroup.bl_irreps(self.L)]
                out_type = nn.FieldType(self.act_r2, irreps * int(self._c_fac[channel] * 2))
            else:
                out_type = nn.FieldType(
                    self.act_r2,
                    [self.act_r2.trivial_repr] * int(self._c_fac[channel] * 8),
                )

            block_6_conv = PenalizedDenseR3Conv(block_5.out_type, out_type, kernel_size=1, bias=False)
            self._penalized_convs.append(block_6_conv)
            block_6 = nn.SequentialModule(block_6_conv)

            skip_3_conv = PenalizedDenseR3Conv(block_5_in, out_type, kernel_size=3, padding=0)
            self._penalized_convs.append(skip_3_conv)
            skip_3 = nn.SequentialModule(skip_3_conv)

            resblock_3 = TensorResidualBlock(nn.SequentialModule(block_5, pool_3, block_6), skip_3)

            layers = [
                self.upsample,
                resblock_1,
                pool_1,
                resblock_2,
                restriction_4,
                resblock_3,
            ]
            if not self.restrict:
                del layers[-2]
            if self.last_tensor:
                layers.append(tensor_act)

            self.layers_eq.append(nn.SequentialModule(*layers))

        nr_features = int(splits * out_type.size)
        self.full_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(nr_features),
            torch.nn.ELU(),
            torch.nn.Linear(nr_features, self.n_classes),
        )

    def projection_penalty(self) -> torch.Tensor:
        total = None
        for conv in self._penalized_convs:
            penalty = conv.projection_penalty(conv_wd=self._conv_wd, basic_wd=self._basic_wd)
            total = penalty if total is None else total + penalty
        if total is None:
            return next(self.parameters()).new_zeros(())
        return total

    def projection_penalty_terms(self) -> dict[str, torch.Tensor]:
        eq_total = None
        non_eq_total = None
        for conv in self._penalized_convs:
            terms = conv.penalty_terms()
            eq_total = terms["equivariant_l2"] if eq_total is None else eq_total + terms["equivariant_l2"]
            non_eq_total = (
                terms["nonequivariant_l2"]
                if non_eq_total is None
                else non_eq_total + terms["nonequivariant_l2"]
            )
        if eq_total is None:
            zero = next(self.parameters()).new_zeros(())
            eq_total = zero
            non_eq_total = zero
        return {
            "equivariant_l2": eq_total,
            "nonequivariant_l2": non_eq_total,
        }

    def project_equivariant_inplace(self) -> None:
        for conv in self._penalized_convs:
            conv.project_equivariant_inplace()

    def project_nonequivariant_inplace(self) -> None:
        for conv in self._penalized_convs:
            conv.project_nonequivariant_inplace()

    @classmethod
    def from_group(
        cls,
        group,
        activation=nn.GatedNonLinearityUniform,
        last_tensor=False,
        n_classes=10,
        n_channels=1,
        mnist_type="single",
        restrict=False,
        one_eq=True,
        channels=6,
        iteration=0,
        L_in=2,
        L_out=4,
        invariant=True,
        learn_eq=None,
        normalise_basis=None,
        conv_wd: float = 0.0,
        basic_wd: float = 0.0,
    ):
        try:
            if group == "SO3":
                N = -1
                f = False
            elif group == "O3":
                N = -1
                f = True
            elif group == "trivial":
                f = False
                N = 0
            else:
                raise AssertionError("invalid group")
        except Exception as e:
            raise AssertionError(f"invalid group, found exception: {e}")
        return cls(
            f=f,
            N=N,
            activation=activation,
            last_tensor=last_tensor,
            n_classes=n_classes,
            n_channels=n_channels,
            mnist_type=mnist_type,
            restrict=restrict,
            one_eq=one_eq,
            channels=channels,
            iteration=iteration,
            L_in=L_in,
            L_out=L_out,
            invariant=invariant,
            conv_wd=conv_wd,
            basic_wd=basic_wd,
        )
