from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from softjax.projections_permutahedron import (
    _proj_permutahedron,
    _proj_permutahedron_smooth_sort,
)
from softjax.projections_simplex import _proj_simplex
from softjax.projections_transport_polytope import _proj_transport_polytope
from softjax.sorting_network import (
    _argsort_via_sorting_network,
    _sort_via_sorting_network,
)
from softjax.utils import (
    _canonicalize_axis,
    _ensure_float,
    _map_in_chunks,
    _quantile_interpolation_params,
    _reduce_in_chunks,
    _standardize_and_squash,
    _unsquash_and_destandardize,
    _validate_softness,
)


SoftBool = Float[Array, "..."]  # probability in [0, 1]
SoftIndex = Float[Array, "..."]  # probabilities, or log probabilities, along the last axis


def _validate_log_prob_request(
    return_log_probs: bool,
    log_prob_eps: float | Array | None,
) -> None:
    if log_prob_eps is None:
        return
    if not return_log_probs:
        raise ValueError("log_prob_eps can only be set when return_log_probs=True")
    if isinstance(log_prob_eps, jax.core.Tracer):
        return

    eps = jnp.asarray(log_prob_eps)
    if eps.shape != ():
        raise ValueError(f"log_prob_eps must be scalar, got shape {eps.shape}")
    eps_value = float(eps)
    if eps_value <= 0 or eps_value > 1:
        raise ValueError(f"log_prob_eps must be in (0, 1], got {log_prob_eps}")


def _log_soft_index_from_probs(
    soft_index: SoftIndex,
    log_prob_eps: float | Array | None,
) -> SoftIndex:
    if log_prob_eps is not None:
        eps = jnp.asarray(log_prob_eps, dtype=soft_index.dtype)
        soft_index = jnp.clip(soft_index, eps, 1.0)
        soft_index = soft_index / jnp.sum(soft_index, axis=-1, keepdims=True)
    return jnp.log(soft_index)


def _renormalize_log_soft_index(
    soft_index: SoftIndex,
    log_prob_eps: float | Array | None,
) -> SoftIndex:
    if log_prob_eps is None:
        return soft_index
    log_eps = jnp.log(jnp.asarray(log_prob_eps, dtype=soft_index.dtype))
    soft_index = jnp.maximum(soft_index, log_eps)
    return soft_index - jax.nn.logsumexp(soft_index, axis=-1, keepdims=True)


def _maybe_log_soft_index(
    soft_index: SoftIndex | None,
    return_log_probs: bool,
    already_log: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex | None:
    if soft_index is None:
        return None
    if return_log_probs:
        if already_log:
            return _renormalize_log_soft_index(soft_index, log_prob_eps)
        return _log_soft_index_from_probs(soft_index, log_prob_eps)
    return soft_index


def _soft_index_probs(soft_index: SoftIndex, is_log: bool) -> SoftIndex:
    if is_log:
        return jnp.exp(soft_index)
    return soft_index


def _logaddexp_weighted(
    x_log: SoftIndex,
    y_log: SoftIndex,
    y_weight: Array,
) -> SoftIndex:
    return jnp.logaddexp(
        jnp.log1p(-y_weight) + x_log,
        jnp.log(y_weight) + y_log,
    )


def _neuralsort_A_sum(x_last, mode, softness):
    """A_sum[..., j] = sum_i soft_abs(x[..., i] - x[..., j])."""
    n = x_last.shape[-1]
    x_flat = x_last.reshape(-1, n)

    def _single(x_row):
        def _chunk_fn(x_chunk_j):
            return abs(
                x_row[:, None] - x_chunk_j[None, :], mode=mode, softness=softness
            ).sum(axis=0)

        return _map_in_chunks(f=_chunk_fn, xs=x_row, chunk_size=128)

    return jax.vmap(_single)(x_flat).reshape(x_last.shape)



def _softsort_fused_sort(x_last, batch_dims, softness, mode, descending, standardize, gated_grad):
    """Sorted values via SoftSort, O(n) memory."""
    n = x_last.shape[-1]
    if standardize:
        x_std = _standardize_and_squash(x_last, axis=-1)
    else:
        x_std = x_last

    x_orig_flat = x_last.reshape(-1, n)
    x_std_flat = x_std.reshape(-1, n)

    def _single(x_orig_row, x_std_row):
        anchors_row = jnp.sort(x_std_row, descending=descending)

        def _chunk_fn(anchors_chunk):
            diff = jnp.abs(anchors_chunk[:, None] - x_std_row[None, :])
            P_chunk = _proj_simplex(-diff, axis=-1, softness=softness, mode=mode)
            if not gated_grad:
                P_chunk = jax.lax.stop_gradient(P_chunk)
            return jnp.einsum("cn,n->c", P_chunk, x_orig_row)

        return _map_in_chunks(f=_chunk_fn, xs=anchors_row, chunk_size=128)

    result = jax.vmap(_single)(x_orig_flat, x_std_flat)  # (B, n)
    return result.reshape(*batch_dims, n)


def _neuralsort_fused_sort(x_last, batch_dims, softness, mode, descending, standardize, gated_grad):
    """Sorted values via NeuralSort, O(n) memory."""
    n = x_last.shape[-1]
    if standardize:
        x_std = _standardize_and_squash(x_last, axis=-1)
    else:
        x_std = x_last

    A_sum = _neuralsort_A_sum(x_last=x_std, mode=mode, softness=softness)

    i = jnp.arange(1, n + 1)
    if descending:
        i = i[::-1]
    coef = n + 1 - 2 * i
    coef = jnp.broadcast_to(coef, (*batch_dims, n))

    x_orig_flat = x_last.reshape(-1, n)
    x_std_flat = x_std.reshape(-1, n)
    A_sum_flat = A_sum.reshape(-1, n)
    coef_flat = coef.reshape(-1, n)

    def _single(x_orig_row, x_std_row, a_sum_row, coef_row):
        def _chunk_fn(coef_chunk):
            z_chunk = -(coef_chunk[:, None] * x_std_row[None, :] + a_sum_row[None, :])
            P_chunk = _proj_simplex(z_chunk, axis=-1, softness=softness, mode=mode)
            if not gated_grad:
                P_chunk = jax.lax.stop_gradient(P_chunk)
            return jnp.einsum("cn,n->c", P_chunk, x_orig_row)

        return _map_in_chunks(f=_chunk_fn, xs=coef_row, chunk_size=128)

    result = jax.vmap(_single)(x_orig_flat, x_std_flat, A_sum_flat, coef_flat)  # (B, n)
    return result.reshape(*batch_dims, n)


def _softsort_fused_rank(x_last, batch_dims, softness, mode, descending):
    """Ranks via SoftSort, O(n) memory. x_last should already be standardized."""
    n = x_last.shape[-1]
    nums = jnp.arange(1, n + 1, dtype=x_last.dtype)
    x_flat = x_last.reshape(-1, n)

    def _single(x_row):
        anchors_row = jnp.sort(x_row, descending=descending)

        def _chunk_fn(x_chunk):
            diff = jnp.abs(x_chunk[:, None] - anchors_row[None, :])
            P_chunk = _proj_simplex(-diff, axis=-1, softness=softness, mode=mode)
            return jnp.einsum("cn,n->c", P_chunk, nums)

        return _map_in_chunks(f=_chunk_fn, xs=x_row, chunk_size=128)

    result = jax.vmap(_single)(x_flat)  # (B, n)
    return result.reshape(*batch_dims, n)


def _neuralsort_fused_rank(x_last, batch_dims, softness, mode, descending):
    """Ranks via NeuralSort, O(n) memory. x_last should already be standardized."""
    n = x_last.shape[-1]
    nums = jnp.arange(1, n + 1, dtype=x_last.dtype)

    row_sums = _neuralsort_A_sum(x_last=x_last, mode=mode, softness=softness)

    i = jnp.arange(1, n + 1)
    if descending:
        i = i[::-1]
    coef = n + 1 - 2 * i
    coef = jnp.broadcast_to(
        coef.reshape(*(1,) * len(batch_dims), n), (*batch_dims, n)
    )

    x_flat = x_last.reshape(-1, n)
    row_sums_flat = row_sums.reshape(-1, n)
    coef_flat = coef.reshape(-1, n)

    def _single(x_row, row_sums_row, coef_row):
        # P_t[j, i] where j=sorted position, i=element.
        # rank[i] = sum_j nums[j] * P_t[j, i] / sum_k P_t[k, i]
        coef_and_nums = jnp.stack([coef_row, nums], axis=-1)  # (n, 2)

        def _chunk_fn(data_chunk):
            coef_chunk = data_chunk[:, 0]
            nums_chunk = data_chunk[:, 1]
            z_chunk = -(coef_chunk[:, None] * x_row[None, :] + row_sums_row[None, :])
            P_chunk = _proj_simplex(z_chunk, axis=-1, softness=softness, mode=mode)
            col_sum_contrib = P_chunk.sum(axis=0)  # (n,)
            weighted_contrib = (nums_chunk[:, None] * P_chunk).sum(axis=0)  # (n,)
            return jnp.stack([col_sum_contrib, weighted_contrib])  # (2, n)

        result = _reduce_in_chunks(f=_chunk_fn, xs=coef_and_nums, chunk_size=128)
        col_sums = result[0]  # (n,)
        weighted_sums = result[1]  # (n,)
        return weighted_sums / jnp.clip(col_sums, min=1e-10)

    result = jax.vmap(_single)(x_flat, row_sums_flat, coef_flat)  # (B, n)
    return result.reshape(*batch_dims, n)


# Selection operators


def where(condition: SoftBool, x: Array, y: Array) -> Array:
    """Performs a soft version of [jax.numpy.where](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html) as `x * condition + y * (1.0 - condition)`.

    **Arguments:**
    - `condition`: SoftBool condition Array, same shape as `x` and `y`.
    - `x`: First input Array, same shape as `condition`.
    - `y`: Second input Array, same shape as `condition`.

    **Returns:**

    Array of the same shape as `x` and `y`, interpolating between `x` and `y` according to `condition` in [0, 1].
    """
    return x * condition + y * (1.0 - condition)


def take_along_axis(
    x: Array,  # (..., n, ...)
    soft_index: SoftIndex,  # (..., k, ..., [n])
    axis: int | None = -1,
) -> Array:  # (..., k, ...)
    """Performs a soft version of [jax.numpy.take_along_axis](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take_along_axis.html) via a weighted dot product.

    !!! example "Relation to `jnp.take_along_axis`"

        ```python
        x = jnp.array([[1, 2, 3], [4, 5, 6]])

        indices = jnp.array([[0, 2], [1, 0]])
        print(jnp.take_along_axis(x, indices, axis=1))

        indices_onehot = jax.nn.one_hot(indices, x.shape[1])
        print(sj.take_along_axis(x, indices_onehot, axis=1))
        ```

        ```
        [[1. 3.]
         [5. 4.]]
        [[1. 3.]
         [5. 4.]]
        ```

    ??? example "Interaction with [`softjax.argmax`][]"

        ```python
        x = jnp.array([[5, 3, 4], [2, 7, 6]])

        indices = jnp.argmin(x, axis=1, keepdims=True)
        print("argmin_jnp:", jnp.take_along_axis(x, indices, axis=1))

        indices_onehot = sj.argmin(x, axis=1, mode="hard", keepdims=True)
        print("argmin_val_onehot:", sj.take_along_axis(x, indices_onehot, axis=1))

        indices_soft = sj.argmin(x, axis=1, mode="smooth", softness=1.0,
            keepdims=True)
        print("argmin_val_soft:", sj.take_along_axis(x, indices_soft, axis=1))
        ```

        ```
        argmin_jnp: [[3]
                     [2]]
        argmin_val_onehot: [[3.]
                            [2.]]
        argmin_val_soft: [[3.42478962]
                          [2.10433824]]
        ```

    ??? example "Interaction with [`softjax.argsort`][]"

        ```python
        x = jnp.array([[5, 3, 4], [2, 7, 6]])

        indices = jnp.argsort(x, axis=1)
        print("sorted_jnp:", jnp.take_along_axis(x, indices, axis=1))

        indices_onehot = sj.argsort(x, axis=1, mode="hard")
        print("sorted_sj_hard:", sj.take_along_axis(x, indices_onehot, axis=1))

        indices_soft = sj.argsort(x, axis=1, mode="smooth", softness=1.0)
        print("sorted_sj_soft:", sj.take_along_axis(x, indices_soft, axis=1))
        ```

        ```
        sorted_jnp: [[3 4 5]
                     [2 6 7]]
        sorted_sj_hard: [[3. 4. 5.]
                         [2. 6. 7.]]
        sorted_sj_soft: [[3.2918137  4.         4.7081863 ]
                         [2.00000045 6.26894107 6.73105858]]
        ```

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `soft_index`: A SoftIndex of shape (..., k, ..., [n]) (positive Array which sums to 1 over the last dimension). If axis is None, must be two-dimensional. If axis is not None, must have x.ndim + 1 == soft_index.ndim, and x must be broadcast-compatible with soft_index along dimensions other than axis.
    - `axis`: Axis along which to apply the soft index. If None, the array will be flattened before indexing is applied.

    **Returns:**

    Array of shape (..., k, ...), representing the result after soft selection along the specified axis.
    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)
    if x.ndim + 1 != soft_index.ndim:
        raise ValueError(
            f"Input x and soft_index must have compatible dimensions, "
            f"but got x.ndim={x.ndim} and soft_index.ndim={soft_index.ndim}. "
            f"Should be x.ndim + 1 == soft_index.ndim."
        )
    x = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
    soft_index = jnp.moveaxis(soft_index, axis, -2)  # (..., ..., k, [n])
    dotprod = jnp.einsum("...n,...kn->...k", x, soft_index)  # (..., ..., k)
    dotprod = jnp.moveaxis(dotprod, -1, axis)  # (..., k, ...)
    return dotprod


def take(
    x: Array,  # (..., n, ...)
    soft_index: SoftIndex,  # (k, [n])
    axis: int | None = None,
) -> Array:  # (..., k, ...)
    """Performs a soft version of [jax.numpy.take](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html) via a weighted dot product.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `soft_index`: A SoftIndex of shape (k, [n]) (positive Array which sums to 1 over the last dimension).
    - `axis`: Axis along which to apply the soft index. If None, the input is flattened.

    **Returns:**

    Array of shape (..., k, ...) after soft selection.
    """
    if soft_index.ndim != 2:
        raise ValueError(
            f"soft_index must be of shape (k, [n]), but got shape {soft_index.shape}."
        )
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)
        x = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
    soft_index = jnp.reshape(
        soft_index, (1,) * (x.ndim - 1) + soft_index.shape
    )  # (1..., 1..., k, [n])
    x = jnp.expand_dims(x, axis)  # (..., 1, ..., n)
    soft_index = jnp.moveaxis(soft_index, -2, axis)  # (1..., k, 1..., [n])
    y = jnp.sum(x * soft_index, axis=-1)  # (..., k, ...)
    return y


def choose(
    soft_index: SoftIndex,  # (..., [n])
    choices: Array,  # (n, ...)
) -> Array:  # (...,)
    """Performs a soft version of [jax.numpy.choose](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.choose.html)
    via a weighted dot product.

    **Arguments:**

    - `soft_index`: A SoftIndex of shape (..., [n]) (positive Array which sums to 1 over the last dimension). Represents the weights for each choice.
    - `choices`: Array of shape (n, ...) supplying the values to mix.

    **Returns:**

    Array of shape (..., ...) after softly selecting among `choices`.
    """
    if soft_index.ndim != choices.ndim or soft_index.shape[-1] != choices.shape[0]:
        raise ValueError(
            f"soft_index and choices must have compatible dimensions, but got "
            f"soft_index.shape={soft_index.shape} and choices.shape={choices.shape}. "
            f"Should be soft_index.shape=(..., [n]) and choices.shape=(n, ...)."
        )
    tgt_shape = jnp.broadcast_shapes(choices.shape[1:], soft_index.shape[:-1])
    choices_bcast = jnp.broadcast_to(choices, (choices.shape[0], *tgt_shape))
    choices_bcast = jnp.moveaxis(choices_bcast, 0, -1)  # (..., C)
    result = jnp.sum(choices_bcast * soft_index, axis=-1)  # (...)
    return result


def dynamic_index_in_dim(
    x: Array,  # (..., n, ...)
    soft_index: SoftIndex,  # ([n],)
    axis: int = 0,
    keepdims: bool = True,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.lax.dynamic_index_in_dim](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_index_in_dim.html) via a weighted dot product.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `soft_index`: A SoftIndex of shape ([n],) (positive Array which sums to 1 over the last dimension).
    - `axis`: Axis along which to apply the soft index.
    - `keepdims`: If True, keeps the reduced dimension as a singleton {1}.

    **Returns:**

    Array after soft indexing, shape (..., {1}, ...).
    """
    axis = _canonicalize_axis(axis, x.ndim)
    if x.shape[axis] != soft_index.shape[0]:
        raise ValueError(
            f"Dimension mismatch between x and soft_index along axis {axis}: "
            f"x.shape[{axis}]={x.shape[axis]} vs soft_index.shape[0]={soft_index.shape[0]}"
        )
    x = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
    x_reshaped = jnp.reshape(x, (-1, x.shape[-1]))  # (B, n)
    dotprod = jnp.sum(x_reshaped * soft_index[None, :], axis=-1)  # (B,)
    y = jnp.reshape(dotprod, x.shape[:-1])  # (..., ...)
    if keepdims:
        y = jnp.expand_dims(y, axis=axis)  # (..., {1}, ...)
    return y


def dynamic_slice_in_dim(
    x: Array,  # (..., n, ...)
    soft_start_index: SoftIndex,  # ([n],)
    slice_size: int,
    axis: int = 0,
) -> Array:  # (..., slice_size, ...)
    """Performs a soft version of [jax.lax.dynamic_slice_in_dim](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice_in_dim.html) via a weighted dot product.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `soft_start_index`: A SoftIndex of shape ([n],) (positive Array which sums to 1 over the last dimension).
    - `slice_size`: Length of the slice to extract.
    - `axis`: Axis along which to apply the soft slice.

    **Returns:**

    Array of shape (..., slice_size, ...) after soft slicing.
    """
    axis = _canonicalize_axis(axis, x.ndim)
    if not (x.shape[axis] >= slice_size > 0):
        raise ValueError(
            f"slice_size must satisfy 0 < slice_size <= x.shape[axis], "
            f"got slice_size={slice_size}, x.shape[axis]={x.shape[axis]}"
        )

    x_last = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
    t_idx = jnp.arange(slice_size)

    def one_step(t: Array) -> Array:
        rolled = jnp.roll(x_last, shift=-t, axis=-1)  # (..., n)
        return jnp.einsum("...n,n->...", rolled, soft_start_index)  # (...)

    y_stack = jax.vmap(one_step)(t_idx)  # (slice_size, ...)
    y_last = jnp.moveaxis(y_stack, 0, -1)  # (..., slice_size)
    y = jnp.moveaxis(y_last, -1, axis)  # (..., slice_size, ...)
    return y


def dynamic_slice(
    x: Array,  # (n_1, n_2, ..., n_k)
    soft_start_indices: Sequence[SoftIndex],  # [([n_1],), ([n_2],), ..., ([n_k],)]
    slice_sizes: Sequence[int],  # [l_1, l_2, ..., l_k]
) -> Array:  # (l_1, l_2, ..., l_k)
    """Performs a soft version of [jax.lax.dynamic_slice](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html) via a weighted dot product.

    **Arguments:**

    - `x`: Input Array of shape (n_1, n_2, ..., n_k).
    - `soft_start_indices`: A list of SoftIndices of shape ([n_i],) (positive Array which sums to 1).
    Sequence of SoftIndex distributions of shapes ([n_1],), ([n_2],), ..., ([n_k]) each summing to 1.
    - `slice_sizes`: Sequence of slice lengths for each dimension.

    **Returns:**

    Array of shape (l_1, l_2, ..., l_k) after soft slicing.
    """
    if not (len(soft_start_indices) == len(slice_sizes) == x.ndim):
        raise ValueError(
            f"len(soft_start_indices) == len(slice_sizes) == x.ndim required, "
            f"got {len(soft_start_indices)}, {len(slice_sizes)}, {x.ndim}"
        )
    y = x
    for axis, (soft_start_index, slice_size) in enumerate(
        zip(soft_start_indices, slice_sizes)
    ):
        y = dynamic_slice_in_dim(
            y,
            soft_start_index=soft_start_index,
            slice_size=slice_size,
            axis=axis,
        )
    return y


# Array-valued operators


def argmax(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Performs a soft version of [jax.numpy.argmax](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmax.html) of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: The axis along which to compute the argmax. If None, the input Array is flattened before computing the argmax.
    - `keepdims`: If True, keeps the reduced dimension as a singleton {1}.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns the result of jnp.argmax with a one-hot encoding of the indices.
        - `smooth`: C∞ smooth (based on logistic/softmax/entropic regularizer), computed in closed-form via a softmax operation.
        - `c0`: C0 continuous (based on euclidean/L2 regularizer), computed via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
        - `c1`: C1 differentiable (p=3/2 p-norm), computed in closed form via quadratic formula.
        - `c2`: C2 twice differentiable (p=4/3 p-norm), computed in closed form via Cardano's method.
    - `method`: Method to compute the soft argmax. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Computes the max element via optimal transport projection onto a 2-point support.
        - `softsort`: Computes the max element of the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038). Reduces to projecting `x` onto the unit simplex.
        - `neuralsort`: Computes the max element of the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
    - `standardize`: If True, standardizes and squashes the input `x` along the specified axis before applying the softargmax operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.
    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.
    - `return_log_probs`: If True, returns log probabilities instead of probabilities. Exact zero probabilities are represented as `-inf`.
    - `log_prob_eps`: Optional probability floor used only with `return_log_probs=True`. When set, probabilities are floored before taking `log` and renormalized along the soft-index axis.

    **Returns:**

    A SoftIndex of shape (..., {1}, ..., [n]) representing the distribution over indices corresponding to the argmax along the specified axis.
    By default this contains probabilities that sum to 1 over the last dimension; if `return_log_probs=True`, it contains log probabilities instead.

    !!! warning "Log probabilities and sparse modes"
        For differentiable objectives over exact log probabilities, prefer `mode="smooth"`. Sparse modes (`"c0"`, `"c1"`, `"c2"`) can produce exact zero probabilities, whose exact log is `-inf`; replacing `-inf` after the log does not generally make gradients finite. For bounded sparse-mode logs, set `log_prob_eps` with `return_log_probs=True` to floor before `log` and renormalize along the soft-index axis.

    !!! tip "Usage"
        This function can be used as a differentiable relaxation to [jax.numpy.argmax](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmax.html), enabling backpropagation through index selection steps in neural networks or optimization routines. However, note that the output is not a discrete index but a `SoftIndex`, which is a distribution over indices. Therefore, functions which operate on indices have to be adjusted accordingly to accept a SoftIndex, see e.g. [`softjax.max`][] for an example of using [`softjax.take_along_axis`][] to retrieve the soft maximum value via the `SoftIndex`.

    !!! caveat "Difference to jax.nn.softmax"
        Note that [`softjax.argmax`][] in `smooth` mode is not fully equivalent to [jax.nn.softmax](https://docs.jax.dev/en/latest/_autosummary/jax.nn.softmax.html) because it moves the probability dimension into the last axis (this is a convention in the `SoftIndex` data type).
    """
    _validate_log_prob_request(return_log_probs, log_prob_eps)
    project_return_log = return_log_probs and log_prob_eps is None

    if mode == "hard" or mode == "_hard":
        indices = jnp.argmax(x, axis=axis, keepdims=keepdims)
        num_classes = jnp.size(x, axis=axis)
        soft_index = jax.nn.one_hot(indices, num_classes=num_classes, axis=-1)
        soft_index_is_log = False
    else:
        x = _ensure_float(x)
        if axis is None:
            num_dims = x.ndim
            x = jnp.ravel(x)
            _axis = 0
        else:
            _axis = _canonicalize_axis(axis, x.ndim)

        if standardize:
            x = _standardize_and_squash(x, axis=_axis)

        x_last = jnp.moveaxis(x, _axis, -1)  # (..., ..., n)
        *batch_dims, n = x_last.shape
        if method == "softsort":
            soft_index = _proj_simplex(
                x_last,
                axis=-1,
                softness=softness,
                mode=mode,
                return_log_probs=project_return_log,
            )  # (..., ..., [n])
            soft_index_is_log = project_return_log
        elif method == "neuralsort":
            A_sum = _neuralsort_A_sum(x_last=x_last, mode=mode, softness=softness)  # (..., n)
            z = (n - 1) * x_last - A_sum  # (..., ..., n)
            soft_index = _proj_simplex(
                z,
                axis=-1,
                softness=softness,
                mode=mode,
                return_log_probs=project_return_log,
            )  # (..., ..., [n])
            soft_index_is_log = project_return_log
        elif method == "ot":
            anchors = jnp.array([0.0, 1.0], dtype=x.dtype)  # (2,)
            anchors = jnp.broadcast_to(anchors, (*batch_dims, 2))  # (..., ..., 2)

            cost = (
                x_last[..., :, None] - anchors[..., None, :]
            ) ** 2  # (..., ..., n, 2)

            mu = jnp.ones((n,), dtype=x.dtype) / n  # ([n],)
            nu = jnp.array([(n - 1) / n, 1 / n], dtype=x.dtype)  # ([2],)

            if ot_kwargs is None:
                ot_kwargs = {}
            out = _proj_transport_polytope(
                cost=cost,
                mu=mu,
                nu=nu,
                softness=softness,
                mode=mode,
                return_log_probs=project_return_log,
                **ot_kwargs,
            )  # (..., ..., [n], 2)
            soft_index = out[..., :, 1]  # (..., ..., [n])
            soft_index_is_log = project_return_log
        elif method == "sorting_network":
            P = _argsort_via_sorting_network(
                x_last, softness, mode, descending=True, standardized=standardize
            )  # (..., ..., n, [n])
            soft_index = P[..., 0, :]  # (..., ..., [n])
            soft_index_is_log = False
        else:
            raise ValueError(f"Invalid method: {method}")

        if keepdims:
            if axis is None:
                soft_index = soft_index.reshape(
                    *(1,) * num_dims, n
                )  # (1..., 1, 1..., [n])
            else:
                soft_index = jnp.expand_dims(
                    soft_index, axis=_axis
                )  # (..., 1, ..., [n])
    return _maybe_log_soft_index(
        soft_index,
        return_log_probs,
        already_log=soft_index_is_log,
        log_prob_eps=log_prob_eps,
    )


def max(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.numpy.max](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.max.html) of `x` along the specified axis.

    For methods other than `fast_soft_sort` and `sorting_network`, implemented as [`softjax.argmax`][] followed by [`softjax.take_along_axis`][], see respective documentations for details.
    For `fast_soft_sort` and `sorting_network`, uses [`softjax.sort`][] to compute soft sorted values and retrieves the maximum as the first element. See [`softjax.sort`][] for method details.

    **Extra Arguments:**

    - `gated_grad`: If `False`, stops the gradient flow through the soft index. True gives gated 'SiLU-style' gradients, while False gives integrated 'Softplus-style' gradients.

    **Returns:**

    Array of shape (..., {1}, ...) representing the soft maximum of `x` along the specified axis.
    """
    if mode == "hard":
        max_val = jnp.max(x, axis=axis, keepdims=keepdims)
    else:
        if axis is None:
            num_dims = x.ndim
            x = jnp.ravel(x)
            _axis = 0
        else:
            _axis = _canonicalize_axis(axis, x.ndim)
        if method in ("fast_soft_sort", "smooth_sort", "sorting_network"):
            soft_sorted = sort(
                x,
                axis=_axis,
                descending=True,
                softness=softness,
                standardize=standardize,
                mode=mode,
                method=method,
            )  # (..., n, ...)
            max_val = jnp.take(soft_sorted, indices=0, axis=_axis)  # (..., ...)
            if axis is None:
                if keepdims:
                    max_val = max_val.reshape(*(1,) * num_dims)  # (1..., 1, 1...)
            elif keepdims:
                max_val = jnp.expand_dims(max_val, axis=_axis)  # (..., 1, ...)
        else:
            soft_index = argmax(
                x,
                axis=_axis,
                keepdims=True,
                softness=softness,
                mode=mode,
                method=method,
                standardize=standardize,
                ot_kwargs=ot_kwargs,
            )  # (..., 1, ..., [n])
            if not gated_grad:
                soft_index = jax.lax.stop_gradient(soft_index)
            max_val = take_along_axis(x, soft_index, axis=_axis)  # (..., 1, ...)
            if axis is None:
                max_val = max_val.reshape(*(1,) * num_dims)  # (1..., 1, 1...)
            if not keepdims:
                max_val = jnp.squeeze(max_val, axis=axis)  # (..., ...)
    return max_val


def argmin(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Performs a soft version of [jax.numpy.argmin](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmin.html) of `x` along the specified axis.
    Implemented as [`softjax.argmax`][] on `-x`, see respective documentation for details.
    """
    return argmax(
        -x,
        axis=axis,
        mode=mode,
        method=method,
        softness=softness,
        keepdims=keepdims,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        return_log_probs=return_log_probs,
        log_prob_eps=log_prob_eps,
    )


def min(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "softsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.numpy.min](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.min.html) of `x` along the specified axis.
    Implemented as -[`softjax.max`][] on `-x`, see respective documentation for details.

    **Returns:**

    Array of shape (..., {1}, ...) representing the soft minimum of `x` along the specified axis.
    """
    return -max(
        -x,
        axis=axis,
        softness=softness,
        mode=mode,
        method=method,
        keepdims=keepdims,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        gated_grad=gated_grad,
    )


def argsort(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    descending: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex:  # (..., n, ..., [n])
    """Performs a soft version of [jax.numpy.argsort](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argsort.html) of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: The axis along which to compute the argsort operation. If None, the input Array is flattened before computing the argsort.
    - `descending`: If True, sorts in descending order.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns the result of jnp.argsort with a one-hot encoding of the indices.
        - `smooth`: C∞ smooth (based on logistic/softmax/entropic regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: C0 continuous (based on euclidean/L2 regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft argsort. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses the approach in [Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885).
            Intuition: The sorted elements are selected by specifying n "anchors" and then transporting the ith-largest value to the ith-largest anchor.
            Note: Inaccurate for small `sinkhorn_max_iter` (can be passed as keyword argument), but can be very slow for large `sinkhorn_max_iter`.
        - `softsort`: Computes the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Computes the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
    - `standardize`: If True, standardizes and squashes the input `x` along the specified axis before applying the softargsort operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.
    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.
    - `return_log_probs`: If True, returns log probabilities instead of probabilities. Exact zero probabilities are represented as `-inf`.
    - `log_prob_eps`: Optional probability floor used only with `return_log_probs=True`. When set, probabilities are floored before taking `log` and renormalized along the soft-index axis.

    **Returns:**

    A SoftIndex of shape (..., n, ..., [n]).
    By default this contains probabilities that sum to 1 over the last dimension; if `return_log_probs=True`, it contains log probabilities instead.
    The elements in (..., i, ..., [n]) represent a distribution over values in x for the ith smallest element along the specified axis.

    !!! warning "Log probabilities and sparse modes"
        For differentiable objectives over exact log probabilities, prefer `mode="smooth"`. Sparse modes (`"c0"`, `"c1"`, `"c2"`) can produce exact zero probabilities, whose exact log is `-inf`; replacing `-inf` after the log does not generally make gradients finite. For bounded sparse-mode logs, set `log_prob_eps` with `return_log_probs=True` to floor before `log` and renormalize along the soft-index axis.

    !!! tip "Computing the expectation"
        Computing the soft sorted values means taking the expectation of `x` under the SoftIndex distribution. Similar to how with normal indices you would do
            ```python
            sorted_x = jnp.take_along_axis(x, indices, axis=axis)
            ```
        we offer the equivalent soft version via
            ```python
            soft_sorted_x = sj.take_along_axis(x, soft_index, axis=axis)
            ```
        This is what is done in [`softjax.sort`][].
    """
    _validate_log_prob_request(return_log_probs, log_prob_eps)
    project_return_log = return_log_probs and log_prob_eps is None

    if mode == "hard" or mode == "_hard":
        indices = jnp.argsort(x, axis=axis, descending=descending)  # (..., n, ...)
        num_classes = jnp.size(x, axis=axis)
        soft_index = jax.nn.one_hot(
            indices, num_classes=num_classes, axis=-1
        )  # (..., n, ..., [n])
        soft_index_is_log = False
    else:
        x = _ensure_float(x)
        if axis is None:
            x = jnp.ravel(x)
            axis = 0
        else:
            axis = _canonicalize_axis(axis, x.ndim)

        if standardize:
            x = _standardize_and_squash(x, axis=axis)

        x_last = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
        *batch_dims, n = x_last.shape

        if method == "softsort":
            anchors = jnp.sort(x_last, axis=-1, descending=descending)  # (..., ..., n)
            diff = jnp.abs(anchors[..., :, None] - x_last[..., None, :])  # (..., n, n)
            soft_index = _proj_simplex(
                -diff,
                axis=-1,
                softness=softness,
                mode=mode,
                return_log_probs=project_return_log,
            )  # (..., n, [n])
            soft_index_is_log = project_return_log
        elif method == "neuralsort":
            A_sum = _neuralsort_A_sum(x_last=x_last, mode=mode, softness=softness)  # (..., n)
            i = jnp.arange(1, n + 1)  # (n,)
            if descending:
                i = i[::-1]  # (n,)
            coef = n + 1 - 2 * i  # (n,)
            coef = jnp.broadcast_to(coef, (*batch_dims, n))  # (..., n)
            z = -(coef[..., :, None] * x_last[..., None, :] + A_sum[..., None, :])  # (..., n, n)
            soft_index = _proj_simplex(
                z,
                axis=-1,
                softness=softness,
                mode=mode,
                return_log_probs=project_return_log,
            )  # (..., n, [n])
            soft_index_is_log = project_return_log
        elif method == "ot":
            anchors = jnp.linspace(0, n, n, dtype=x.dtype) / n  # (n,)
            if descending:
                anchors = anchors[::-1]  # (n,)
            anchors = jnp.broadcast_to(anchors, (*batch_dims, n))  # (..., ..., n)

            cost = (
                x_last[..., :, None] - anchors[..., None, :]
            ) ** 2  # (..., ..., n, n)

            mu = jnp.ones((n,), dtype=x.dtype) / n  # ([n],)
            nu = jnp.ones((n,), dtype=x.dtype) / n  # ([n],)

            if ot_kwargs is None:
                ot_kwargs = {}
            out = _proj_transport_polytope(
                cost=cost,
                mu=mu,
                nu=nu,
                softness=softness,
                mode=mode,
                return_log_probs=project_return_log,
                **ot_kwargs,
            )  # (..., ..., [n], n)
            soft_index = jnp.swapaxes(out, -2, -1)  # (..., ..., n, [n])
            soft_index_is_log = project_return_log
        elif method == "sorting_network":
            soft_index = _argsort_via_sorting_network(
                x_last, softness, mode, descending, standardized=standardize
            )  # (..., ..., n, [n])
            soft_index_is_log = False
        else:
            raise ValueError(f"Invalid method: {method}")

        soft_index = jnp.moveaxis(soft_index, -2, axis)  # (..., n, ..., [n])
    return _maybe_log_soft_index(
        soft_index,
        return_log_probs,
        already_log=soft_index_is_log,
        log_prob_eps=log_prob_eps,
    )


def sort(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    descending: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> Array:  # (..., n, ...)
    """Performs a soft version of [jax.numpy.sort](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort.html) of `x` along the specified axis.

    Most methods go through [`softjax.argsort`][] + [`softjax.take_along_axis`][] to produce soft sorted values.
    The exceptions (`fast_soft_sort`, `smooth_sort`, `sorting_network`) bypass soft indices and compute values directly:

    - `fast_soft_sort`: permutahedron projection via PAV isotonic regression ([Blondel et al., 2020](https://arxiv.org/abs/2002.08871)). In `smooth` mode, uses an entropic (log-KL) variant that is piecewise smooth but not C∞ (discontinuities at argsort chamber boundaries).
    - `smooth_sort` (SoftJAX only, `smooth` mode only): permutahedron projection via ESP smooth majorization bounds + LBFGS dual, giving a truly C∞ relaxation.
    - `sorting_network`: soft bitonic sorting network ([Petersen et al., 2021](https://arxiv.org/abs/2105.04019)).

    **Extra Arguments:**

    - `gated_grad`: If `False`, stops the gradient flow through the soft index. True gives gated 'SiLU-style' gradients, while False gives integrated 'Softplus-style' gradients.

    **Returns:**

    Array of shape (..., n, ...) representing the soft sorted values of `x` along the specified axis.
    """
    if mode == "hard" or (
        mode == "_hard" and method in ("fast_soft_sort", "smooth_sort", "sorting_network")
    ):
        soft_values = jnp.sort(x, axis=axis, descending=descending)
    else:
        x = _ensure_float(x)
        if axis is None:
            x = jnp.ravel(x)
            axis = 0
        else:
            axis = _canonicalize_axis(axis, x.ndim)
        if method == "sorting_network":
            if standardize:
                x, mean, std = _standardize_and_squash(
                    x, axis=axis, return_mean_std=True
                )
            x_last = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
            soft_values = _sort_via_sorting_network(
                x_last,
                softness=softness,
                mode=mode,
                descending=descending,
                standardized=standardize,
            )
            soft_values = jnp.moveaxis(soft_values, -1, axis)  # (..., n, ...)
            if standardize:
                soft_values = _unsquash_and_destandardize(
                    y=soft_values, mean=mean, std=std
                )
        elif method == "fast_soft_sort":
            if standardize:
                x, mean, std = _standardize_and_squash(
                    x, axis=axis, return_mean_std=True
                )
            x_last = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
            *batch_dims, n = x_last.shape
            w = x_last
            anchors = jnp.arange(n, dtype=x.dtype) / jnp.maximum((n - 1), 1)  # (n,)
            anchors = jnp.broadcast_to(anchors, (*batch_dims, n))  # (..., ..., n)
            soft_values = _proj_permutahedron(anchors, w, softness=softness, mode=mode)
            soft_values = jnp.moveaxis(soft_values, -1, axis)  # (..., n, ...)
            if descending:
                soft_values = jnp.flip(soft_values, axis=axis)
            if standardize:
                soft_values = _unsquash_and_destandardize(
                    y=soft_values, mean=mean, std=std
                )
        elif method == "smooth_sort":
            if mode not in ("smooth",):
                raise ValueError(
                    f"smooth_sort only supports mode='smooth', got mode='{mode}'"
                )
            # smooth_sort skips standardize: the sigmoid squash / logit unsquash
            # round-trip amplifies small out-of-[0,1] deviations from the ESP
            # solver into extreme values.
            x_last = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
            *batch_dims, n = x_last.shape
            w = x_last
            anchors = jnp.arange(n, dtype=x.dtype) / jnp.maximum((n - 1), 1)  # (n,)
            anchors = jnp.broadcast_to(anchors, (*batch_dims, n))  # (..., ..., n)
            soft_values = _proj_permutahedron_smooth_sort(
                anchors, w, softness=softness
            )
            soft_values = jnp.moveaxis(soft_values, -1, axis)  # (..., n, ...)
            if descending:
                soft_values = jnp.flip(soft_values, axis=axis)
        elif method == "softsort" and mode != "_hard":
            x_last = jnp.moveaxis(x, axis, -1)
            *batch_dims, n = x_last.shape
            soft_values = _softsort_fused_sort(
                x_last=x_last, batch_dims=batch_dims, softness=softness, mode=mode,
                descending=descending, standardize=standardize, gated_grad=gated_grad,
            )
            soft_values = jnp.moveaxis(soft_values, -1, axis)
        elif method == "neuralsort" and mode != "_hard":
            x_last = jnp.moveaxis(x, axis, -1)
            *batch_dims, n = x_last.shape
            soft_values = _neuralsort_fused_sort(
                x_last=x_last, batch_dims=batch_dims, softness=softness, mode=mode,
                descending=descending, standardize=standardize, gated_grad=gated_grad,
            )
            soft_values = jnp.moveaxis(soft_values, -1, axis)
        else:
            soft_index = argsort(
                x=x,
                axis=axis,
                descending=descending,
                softness=softness,
                mode=mode,
                method=method,
                standardize=standardize,
                ot_kwargs=ot_kwargs,
            )  # (..., n, ..., [n])
            if not gated_grad:
                soft_index = jax.lax.stop_gradient(soft_index)
            soft_values = take_along_axis(x, soft_index, axis=axis)
    return soft_values  # (..., n, ...)


def argquantile(
    x: Array,  # (..., n, ...)
    q: Array,  # scalar or (k,)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "neuralsort",
    quantile_method: Literal[
        "linear", "lower", "higher", "nearest", "midpoint"
    ] = "linear",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Performs a soft version of [jax.numpy.quantile](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.quantile.html) of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `q`: Scalar quantile or 1-D Array of quantiles in [0, 1]. When a 1-D array of length k is passed, the q dimension is prepended to the output shape.
    - `axis`: The axis along which to compute the argquantile. If None, the input Array is flattened before computing the argquantile.
    - `keepdims`: If True, keeps the reduced dimension as a singleton {1}.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns a one/two-hot encoding of the indices corresponding to the jax.numpy.quantile definitions.
        - `smooth`: C∞ smooth (based on logistic/softmax/entropic regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: C0 continuous (based on euclidean/L2 regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft argquantile. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses a variation of the soft quantile approach in [Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885), which is adapted to converge to the jax quantile definitions for small softness. Depending on the quantile_method, either a lower and upper quantile are computed and combined, or just a single quantile is computed.
            Intuition: The sorted elements are selected by specifying 4 or 3 "anchors" and then transporting the upper/lower quantile values to the appropriate anchors.
            Note: Inaccurate for small `sinkhorn_max_iter` (can be passed as keyword argument), but can be very slow for large `sinkhorn_max_iter`.
        - `softsort`: Computes the upper and lower quantiles via the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Computes the upper and lower quantiles via the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
    - `quantile_method`: Method to compute the quantile, following the options in [jax.numpy.quantile](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.quantile.html).
    - `standardize`: If True, standardizes and squashes the input `x` along the specified axis before applying the softargquantile operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.
    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.
    - `return_log_probs`: If True, returns log probabilities instead of probabilities. Exact zero probabilities are represented as `-inf`.
    - `log_prob_eps`: Optional probability floor used only with `return_log_probs=True`. When set, probabilities are floored before taking `log` and renormalized along the soft-index axis.

    **Returns:**

    A SoftIndex of shape (..., {1}, ..., [n]) for scalar q, or (k, ..., {1}, ..., [n]) for vector q of length k (q dimension prepended). It represents a distribution over values in x being the q-quantile along the specified axis.
    By default this contains probabilities that sum to 1 over the last dimension; if `return_log_probs=True`, it contains log probabilities instead.

    !!! warning "Log probabilities and sparse modes"
        For differentiable objectives over exact log probabilities, prefer `mode="smooth"`. Sparse modes (`"c0"`, `"c1"`, `"c2"`) can produce exact zero probabilities, whose exact log is `-inf`; replacing `-inf` after the log does not generally make gradients finite. For bounded sparse-mode logs, set `log_prob_eps` with `return_log_probs=True` to floor before `log` and renormalize along the soft-index axis.
    """
    _validate_log_prob_request(return_log_probs, log_prob_eps)
    project_return_log = return_log_probs and log_prob_eps is None

    q_arr = jnp.asarray(q)
    if q_arr.ndim > 1:
        raise ValueError(
            f"q must be scalar or 1-D, got q with shape {q_arr.shape}"
        )
    if q_arr.ndim == 1:

        def _single(qi):
            return argquantile(
                x,
                q=qi,
                axis=axis,
                keepdims=keepdims,
                softness=softness,
                mode=mode,
                method=method,
                quantile_method=quantile_method,
                standardize=standardize,
                ot_kwargs=ot_kwargs,
                return_log_probs=return_log_probs,
                log_prob_eps=log_prob_eps,
            )

        return jax.vmap(_single)(q_arr)

    orig_axis_is_none = axis is None
    if axis is None:
        num_dims = x.ndim
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)

    if mode not in ("hard", "_hard"):
        x = _ensure_float(x)
    if standardize and mode not in ("hard", "_hard"):
        x = _standardize_and_squash(x, axis=axis)

    x_last = jnp.moveaxis(x, axis, -1)
    *batch_dims, n = x_last.shape

    q = jnp.clip(q, 0.0, 1.0)
    k, a, take_next = _quantile_interpolation_params(q, n, quantile_method)
    a_b = jnp.expand_dims(a, axis=-1)  # (..., ..., 1)
    kp1 = jnp.minimum(k + 1, n - 1)
    soft_index_is_log = False

    if mode == "hard" or mode == "_hard":
        indices = jnp.argsort(x_last, axis=-1, descending=False)  # (..., ..., n)
        if take_next:
            indices = jnp.stack(
                [indices[..., k], indices[..., kp1]], axis=-1
            )  # (..., ..., 2)
            soft_index = jax.nn.one_hot(
                indices, num_classes=n, axis=-1
            )  # (..., ..., 2, [n])
            soft_index = (1.0 - a_b) * soft_index[..., 0, :] + a_b * soft_index[
                ..., 1, :
            ]  # (..., ..., [n])
        else:
            indices = indices[..., k]  # (..., ...)
            soft_index = jax.nn.one_hot(
                indices, num_classes=n, axis=-1
            )  # (..., ..., [n])
    else:
        if method == "softsort":
            x_sorted = jnp.sort(x_last, axis=-1, descending=False)  # (..., ..., n)
            if take_next:
                anchors = jnp.stack(
                    [x_sorted[..., k], x_sorted[..., kp1]], axis=-1
                )  # (..., ..., 2)
                abs_diff = jnp.abs(
                    anchors[..., :, None] - x_last[..., None, :]
                )  # (..., ..., 2, n)
                proj = _proj_simplex(
                    -abs_diff,
                    axis=-1,
                    softness=softness,
                    mode=mode,
                    return_log_probs=project_return_log,
                )  # (..., ..., 2, [n])
                idx_k = proj[..., 0, :]  # (..., ..., [n])
                idx_kp1 = proj[..., 1, :]  # (..., ..., [n])
                if project_return_log:
                    soft_index = _logaddexp_weighted(idx_k, idx_kp1, a_b)
                    soft_index_is_log = True
                else:
                    soft_index = (1.0 - a_b) * idx_k + a_b * idx_kp1  # (..., ..., [n])
            else:
                anchors = x_sorted[..., k, None]  # (..., ..., 1)
                abs_diff = jnp.abs(
                    anchors[..., :, None] - x_last[..., None, :]
                )  # (..., ..., 1, n)
                soft_index = _proj_simplex(
                    -abs_diff,
                    axis=-1,
                    softness=softness,
                    mode=mode,
                    return_log_probs=project_return_log,
                )[..., 0, :]  # (..., ..., [n])
                soft_index_is_log = project_return_log
        elif method == "neuralsort":
            A_sum = _neuralsort_A_sum(x_last=x_last, mode=mode, softness=softness)  # (..., n)
            if take_next:
                i = jnp.array([k + 1, k + 2])  # (2,)
                coef = n + 1 - 2 * i  # (2,)
                coef = jnp.broadcast_to(coef, (*batch_dims, 2))  # (..., 2)
                z = -(
                    coef[..., :, None] * x_last[..., None, :] + A_sum[..., None, :]
                )  # (..., 2, n)
                proj = _proj_simplex(
                    z,
                    axis=-1,
                    softness=softness,
                    mode=mode,
                    return_log_probs=project_return_log,
                )  # (..., ..., 2, [n])
                idx_k = proj[..., 0, :]  # (..., ..., [n])
                idx_k1 = proj[..., 1, :]  # (..., ..., [n])
                if project_return_log:
                    soft_index = _logaddexp_weighted(idx_k, idx_k1, a_b)
                    soft_index_is_log = True
                else:
                    soft_index = (1.0 - a_b) * idx_k + a_b * idx_k1  # (..., ..., [n])
            else:
                coef = jnp.array([n + 1 - 2 * (k + 1)])  # (1,)
                coef = jnp.broadcast_to(coef, (*batch_dims, 1))  # (..., 1)
                z = -(
                    coef[..., :, None] * x_last[..., None, :] + A_sum[..., None, :]
                )  # (..., 1, n)
                soft_index = _proj_simplex(
                    z,
                    axis=-1,
                    softness=softness,
                    mode=mode,
                    return_log_probs=project_return_log,
                )[..., 0, :]  # (..., ..., [n])
                soft_index_is_log = project_return_log
        elif method == "ot":
            if take_next:
                mu = jnp.ones((n,), dtype=x.dtype) / n  # ([n],)
                nu = jnp.array(
                    [k / n, 1 / n, (kp1 - k) / n, (n - kp1 - 1) / n], dtype=x.dtype
                )  # (4,)

                anchors = jnp.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=x.dtype)
                anchors = jnp.broadcast_to(anchors, (*batch_dims, 4))  # (..., ..., 4)

                cost = (
                    x_last[..., :, None] - anchors[..., None, :]
                ) ** 2  # (..., ..., n, 4)

                if ot_kwargs is None:
                    ot_kwargs = {}
                out = _proj_transport_polytope(
                    cost=cost,
                    mu=mu,
                    nu=nu,
                    softness=softness,
                    mode=mode,
                    return_log_probs=project_return_log,
                    **ot_kwargs,
                )  # (..., ..., [n], 4)

                soft_index = jnp.swapaxes(out, -2, -1)  # (..., ..., 4, [n])
                idx_k = soft_index[..., 1, :]  # (...,  ..., [n])
                idx_k1 = soft_index[..., 2, :]  # (..., ..., [n])
                if project_return_log:
                    soft_index = _logaddexp_weighted(idx_k, idx_k1, a_b)
                    soft_index_is_log = True
                else:
                    soft_index = (1.0 - a_b) * idx_k + a_b * idx_k1  # (..., ..., [n])
            else:
                mu = jnp.ones((n,), dtype=x.dtype) / n  # ([n],)
                nu = jnp.array([k / n, 1 / n, (n - k - 1) / n], dtype=x.dtype)  # (3,)

                anchors = jnp.array([0.0, 0.5, 1.0], dtype=x.dtype)
                anchors = jnp.broadcast_to(anchors, (*batch_dims, 3))  # (..., ..., 3)

                cost = (
                    x_last[..., :, None] - anchors[..., None, :]
                ) ** 2  # (..., ..., n, 3)

                if ot_kwargs is None:
                    ot_kwargs = {}
                out = _proj_transport_polytope(
                    cost=cost,
                    mu=mu,
                    nu=nu,
                    softness=softness,
                    mode=mode,
                    return_log_probs=project_return_log,
                    **ot_kwargs,
                )  # (..., ..., [n], 3)

                soft_index = jnp.swapaxes(out, -2, -1)  # (..., ..., 3, [n])
                idx_k = soft_index[..., 1, :]  # (...,  ..., [n])
                soft_index = idx_k  # (..., ..., [n]))
                soft_index_is_log = project_return_log
        elif method == "sorting_network":
            P = _argsort_via_sorting_network(
                x_last, softness, mode, descending=False, standardized=standardize
            )  # (..., ..., n, [n])
            if take_next:
                soft_index = (1.0 - a_b) * P[..., k, :] + a_b * P[..., kp1, :]
            else:
                soft_index = P[..., k, :]  # (..., ..., [n])
        else:
            raise ValueError(f"Invalid method: {method}")

    if keepdims:
        if orig_axis_is_none:
            soft_index = soft_index.reshape(*(1,) * num_dims, n)
        else:
            soft_index = jnp.expand_dims(soft_index, axis=axis)  # (..., {1}, ..., [n])

    return _maybe_log_soft_index(
        soft_index,
        return_log_probs,
        already_log=soft_index_is_log,
        log_prob_eps=log_prob_eps,
    )


def quantile(
    x: Array,  # (..., n, ...)
    q: Array,  # quantile in [0, 1]
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "neuralsort",
    quantile_method: Literal[
        "linear", "lower", "higher", "nearest", "midpoint"
    ] = "linear",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.numpy.quantile](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.quantile.html) of `x` along the specified axis.

    For methods other than `fast_soft_sort` and `sorting_network`, implemented as [`softjax.argquantile`][] followed by [`softjax.take_along_axis`][], see respective documentations for details.
    For `fast_soft_sort` and `sorting_network`, uses [`softjax.sort`][] to compute soft sorted values, then retrieves the quantile as a combination of the appropriate elements depending on the quantile method. See [`softjax.sort`][] for method details.

    **Extra Arguments:**

    - `gated_grad`: If `False`, stops the gradient flow through the soft index. True gives gated 'SiLU-style' gradients, while False gives integrated 'Softplus-style' gradients.

    **Returns:**

    Array of shape (..., {1}, ...) for scalar q, or (k, ..., {1}, ...) for vector q of length k (q dimension prepended). Represents the soft q-quantile of `x` along the specified axis.
    """
    if mode == "hard":
        quantile_val = jnp.quantile(
            x, q=q, axis=axis, keepdims=keepdims, method=quantile_method
        )  # (..., {1}, ...)
    else:
        q_arr = jnp.asarray(q)
        if q_arr.ndim > 1:
            raise ValueError(
                f"q must be scalar or 1-D, got q with shape {q_arr.shape}"
            )
        if q_arr.ndim == 1:

            def _single(qi):
                return quantile(
                    x,
                    q=qi,
                    axis=axis,
                    keepdims=keepdims,
                    softness=softness,
                    mode=mode,
                    method=method,
                    quantile_method=quantile_method,
                    standardize=standardize,
                    ot_kwargs=ot_kwargs,
                    gated_grad=gated_grad,
                )

            return jax.vmap(_single)(q_arr)

        x = _ensure_float(x)
        if axis is None:
            num_dims = x.ndim
            x = jnp.ravel(x)
            _axis = 0
        else:
            _axis = _canonicalize_axis(axis, x.ndim)
        if method in ("fast_soft_sort", "smooth_sort", "sorting_network"):
            soft_sorted = sort(
                x,
                axis=_axis,
                descending=False,
                softness=softness,
                standardize=standardize,
                mode=mode,
                method=method,
            )  # (..., n, ...)
            k, a, take_next = _quantile_interpolation_params(
                q, soft_sorted.shape[_axis], quantile_method
            )
            quantile_val = jnp.take(soft_sorted, indices=k, axis=_axis)  # (..., ...)
            if take_next:
                kp1 = jnp.minimum(k + 1, soft_sorted.shape[_axis] - 1)
                quantile_val_next = jnp.take(
                    soft_sorted, indices=kp1, axis=_axis
                )  # (..., ...)
                quantile_val = (
                    1.0 - a
                ) * quantile_val + a * quantile_val_next  # (..., ...)
            quantile_val = jnp.expand_dims(quantile_val, axis=_axis)  # (..., 1, ...)
        else:
            soft_index = argquantile(
                x,
                q=q,
                axis=_axis,
                keepdims=True,
                softness=softness,
                mode=mode,
                method=method,
                quantile_method=quantile_method,
                standardize=standardize,
                ot_kwargs=ot_kwargs,
            )  # (..., 1, ..., [n])
            if not gated_grad:
                soft_index = jax.lax.stop_gradient(soft_index)
            quantile_val = take_along_axis(x, soft_index, axis=_axis)  # (..., 1, ...)

        if axis is None:
            quantile_val = quantile_val.reshape(*(1,) * num_dims)  # (1..., 1, 1...)
        if not keepdims:
            quantile_val = jnp.squeeze(quantile_val, axis=axis)  # (..., ...)
    return quantile_val


def argmedian(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Computes the soft argmedian of `x` along the specified axis.
    Implemented as [`softjax.argquantile`][] with q=0.5, see respective documentation for details.
    """
    return argquantile(
        x,
        q=0.5,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        quantile_method="midpoint",  # same as jnp.median
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        return_log_probs=return_log_probs,
        log_prob_eps=log_prob_eps,
    )


def median(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.numpy.median](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.median.html) of `x` along the specified axis.
    Implemented as [`softjax.quantile`][] with q=0.5, see respective documentation for details.
    """
    return quantile(
        x,
        q=0.5,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        quantile_method="midpoint",  # same as jnp.median
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        gated_grad=gated_grad,
    )


def argpercentile(
    x: Array,  # (..., n, ...)
    p: Array,  # percentile in [0, 100]
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "neuralsort",
    quantile_method: Literal[
        "linear", "lower", "higher", "nearest", "midpoint"
    ] = "linear",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Computes the soft p-argpercentile of `x` along the specified axis.
    Implemented as [`softjax.argquantile`][] with q=p/100, see respective documentation for details.
    """
    q = p / 100.0
    return argquantile(
        x,
        q=q,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        quantile_method=quantile_method,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        return_log_probs=return_log_probs,
        log_prob_eps=log_prob_eps,
    )


def percentile(
    x: Array,  # (..., n, ...)
    p: Array,  # percentile in [0, 100]
    axis: int | None = None,
    keepdims: bool = False,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "neuralsort",
    quantile_method: Literal[
        "linear", "lower", "higher", "nearest", "midpoint"
    ] = "linear",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.numpy.percentile](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.percentile.html) of `x` along the specified axis.
    Implemented as [`softjax.quantile`][] with q=p/100, see respective documentation for details.
    """
    q = p / 100.0
    return quantile(
        x,
        q=q,
        axis=axis,
        keepdims=keepdims,
        softness=softness,
        mode=mode,
        method=method,
        quantile_method=quantile_method,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        gated_grad=gated_grad,
    )


def _argtop_k(
    x: Array,  # (..., n, ...)
    k: int,
    axis: int = -1,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex:  # (..., k, ..., [n])
    """Performs a soft version of argtop_k of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `k`: The number of top elements to select.
    - `axis`: The axis along which to compute the top_k operation.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns the result of jax.lax.top_k with a one-hot encoding of the indices.
        - `smooth`: C∞ smooth (based on logistic/softmax/entropic regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: C0 continuous (based on euclidean/L2 regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft argsort. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses the approach in [Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf).
            Intuition: The top-k elements are selected by specifying k+1 "anchors" and then transporting the top_k values to the top k anchors, and the remaining (n-k) values to the last anchor.
            Note: Inaccurate for small `sinkhorn_max_iter` (can be passed as keyword argument), but can be very slow for large `sinkhorn_max_iter`.
        - `softsort`: Computes the top-k elements of the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Computes the top-k elements of the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
    - `standardize`: If True, standardizes and squashes the input `x` along the specified axis before applying the softtop_k operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.
    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.
    - `return_log_probs`: If True, returns log probabilities instead of probabilities. Exact zero probabilities are represented as `-inf`.
    - `log_prob_eps`: Optional probability floor used only with `return_log_probs=True`. When set, probabilities are floored before taking `log` and renormalized along the soft-index axis.

    **Returns:**

    SoftIndex of shape (..., k, ..., [n]) representing the soft indices of the top-k values.
    By default this contains probabilities that sum to 1 over the last dimension; if `return_log_probs=True`, it contains log probabilities instead.

    !!! warning "Log probabilities and sparse modes"
        For differentiable objectives over exact log probabilities, prefer `mode="smooth"`. Sparse modes (`"c0"`, `"c1"`, `"c2"`) can produce exact zero probabilities, whose exact log is `-inf`; replacing `-inf` after the log does not generally make gradients finite. For bounded sparse-mode logs, set `log_prob_eps` with `return_log_probs=True` to floor before `log` and renormalize along the soft-index axis.
    """
    _validate_log_prob_request(return_log_probs, log_prob_eps)
    project_return_log = return_log_probs and log_prob_eps is None

    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")
    n = x.shape[_canonicalize_axis(axis, x.ndim)]
    if k > n:
        raise ValueError(f"k={k} exceeds dimension size {n} along axis={axis}")
    if mode == "hard" or mode == "_hard":
        axis = _canonicalize_axis(axis, x.ndim)
        x_last = jnp.moveaxis(x, axis, -1)
        _, indices = jax.lax.top_k(x_last, k=k)  # (..., ..., k), (..., ..., k)
        indices = jnp.moveaxis(indices, -1, axis)
        num_classes = x.shape[axis]
        soft_index = jax.nn.one_hot(
            indices, num_classes=num_classes, axis=-1
        )  # (..., ..., k, [n])
        soft_index_is_log = False
    else:
        x = _ensure_float(x)
        axis = _canonicalize_axis(axis, x.ndim)
        x_last = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
        if standardize:
            x_last = _standardize_and_squash(x_last, axis=-1)
        *batch_dims, n = x_last.shape
        if method == "softsort":
            anchors, _ = jax.lax.top_k(x_last, k=k)  # (..., ..., k)
            diff = jnp.abs(anchors[..., :, None] - x_last[..., None, :])  # (..., k, n)
            soft_index = _proj_simplex(
                -diff,
                axis=-1,
                softness=softness,
                mode=mode,
                return_log_probs=project_return_log,
            )  # (..., k, [n])
            soft_index_is_log = project_return_log
        elif method == "neuralsort":
            A_sum = _neuralsort_A_sum(x_last=x_last, mode=mode, softness=softness)  # (..., n)

            i = jnp.arange(n - k + 1, n + 1)  # (k,)
            i = i[::-1]  # (k,)

            coef = n + 1 - 2 * i  # (k,)
            coef = jnp.broadcast_to(coef, (*batch_dims, k))  # (..., ..., k)

            z = -(coef[..., :, None] * x_last[..., None, :] + A_sum[..., None, :])  # (..., k, n)
            soft_index = _proj_simplex(
                z,
                axis=-1,
                softness=softness,
                mode=mode,
                return_log_probs=project_return_log,
            )  # (..., k, [n])
            soft_index_is_log = project_return_log
        elif method == "ot":
            if k == n:
                soft_index = argsort(
                    x=x_last,
                    axis=-1,
                    descending=True,
                    softness=softness,
                    mode=mode,
                    method=method,
                    standardize=False,
                    ot_kwargs=ot_kwargs,
                    return_log_probs=project_return_log,
                )  # (..., ..., k, [n])
                soft_index_is_log = project_return_log
            else:
                anchors = jnp.linspace(0, k, k + 1, dtype=x.dtype) / k  # (k+1,)
                anchors = anchors[::-1]  # (k+1,)
                anchors = jnp.broadcast_to(
                    anchors, (*batch_dims, k + 1)
                )  # (..., ..., k+1)

                cost = (
                    x_last[..., :, None] - anchors[..., None, :]
                ) ** 2  # (..., ..., k+1, n)

                mu = jnp.ones((n,), dtype=x.dtype) / n  # ([n],)
                nu = jnp.concatenate(
                    [jnp.ones(k, dtype=x.dtype) / n, jnp.array((n - k) / n)[None]]
                )  # ([k+1],)

                if ot_kwargs is None:
                    ot_kwargs = {}
                out = _proj_transport_polytope(
                    cost=cost,
                    mu=mu,
                    nu=nu,
                    softness=softness,
                    mode=mode,
                    return_log_probs=project_return_log,
                    **ot_kwargs,
                )  # (..., ..., [n], k+1)
                soft_index = jnp.swapaxes(out, -2, -1)  # (..., ..., k+1, [n])
                soft_index = soft_index[..., :k, :]  # (..., ..., k, [n])
                soft_index_is_log = project_return_log
        elif method == "sorting_network":
            P = _argsort_via_sorting_network(
                x_last, softness, mode, descending=True, standardized=standardize
            )  # (..., ..., n, [n])
            soft_index = P[..., :k, :]  # (..., ..., k, [n])
            soft_index_is_log = False
        else:
            raise ValueError(f"Invalid method: {method}")

        soft_index = jnp.moveaxis(soft_index, -2, axis)  # (..., k, ..., [n])
    return _maybe_log_soft_index(
        soft_index,
        return_log_probs,
        already_log=soft_index_is_log,
        log_prob_eps=log_prob_eps,
    )


def top_k(
    x: Array,  # (..., n, ...)
    k: int,
    axis: int = -1,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal[
        "ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"
    ] = "neuralsort",
    standardize: bool = True,
    ot_kwargs: dict | None = None,
    gated_grad: bool = True,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> tuple[Array, SoftIndex | None]:  # (..., k, ...), (..., k, ..., [n])
    """Performs a soft version of [jax.lax.top_k](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html) of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `k`: The number of top elements to select.
    - `axis`: The axis along which to compute the top_k operation.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns the result of jax.lax.top_k with a one-hot encoding of the indices.
        - `smooth`: C∞ smooth (based on logistic/softmax/entropic regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: C0 continuous (based on euclidean/L2 regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft argsort. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses the approach in [Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf).
            Intuition: The top-k elements are selected by specifying k+1 "anchors" and then transporting the top_k values to the top k anchors, and the remaining (n-k) values to the last anchor.
            Note: Inaccurate for small `sinkhorn_max_iter` (can be passed as keyword argument), but can be very slow for large `sinkhorn_max_iter`.
        - `softsort`: Computes the top-k elements of the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Computes the top-k elements of the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850).
        - `fast_soft_sort`: Uses the `FastSoftSort` operator from [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871) to directly compute the soft sorted values, via projection onto the permutahedron. The projection is solved via a PAV algorithm as proposed in [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871). The top-k values are then retrieved by taking the first k values from the soft sorted output.
            Note: This method does not return the soft indices, only the soft values.
        - `sorting_network`: Uses a soft bitonic sorting network as proposed in [Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision](https://arxiv.org/abs/2105.04019), replacing hard compare-and-swap with soft versions. The top-k values are retrieved by taking the first k values from the soft sorted output.
            Note: This method does not return the soft indices, only the soft values.
    - `standardize`: If True, standardizes and squashes the input `x` along the specified axis before applying the softtop_k operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.
    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.
    - `gated_grad`: If `False`, stops the gradient flow through the soft index. True gives gated 'SiLU-style' gradients, while False gives integrated 'Softplus-style' gradients.
    - `return_log_probs`: If True, returns `soft_index` as log probabilities instead of probabilities. `soft_values` are unchanged.
    - `log_prob_eps`: Optional probability floor used only with `return_log_probs=True`. When set, probabilities are floored before taking `log` and renormalized along the soft-index axis.

    **Returns:**

    - `soft_values`: Top-k values of `x`, shape (..., k, ...).
    - `soft_index`: SoftIndex of shape (..., k, ..., [n]) representing the soft indices of the top-k values. By default this contains probabilities that sum to 1 over the last dimension; if `return_log_probs=True`, it contains log probabilities instead.

    !!! warning "Log probabilities and sparse modes"
        For differentiable objectives over exact log probabilities, prefer `mode="smooth"`. Sparse modes (`"c0"`, `"c1"`, `"c2"`) can produce exact zero probabilities, whose exact log is `-inf`; replacing `-inf` after the log does not generally make gradients finite. For bounded sparse-mode logs, set `log_prob_eps` with `return_log_probs=True` to floor before `log` and renormalize along the soft-index axis.
    """
    _validate_log_prob_request(return_log_probs, log_prob_eps)

    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")
    _axis = _canonicalize_axis(axis if axis is not None else -1, x.ndim)
    n = x.shape[_axis] if axis is not None else x.size
    if k > n:
        raise ValueError(f"k={k} exceeds dimension size {n} along axis={axis}")
    if mode == "hard":
        if axis is None:
            x_flat = jnp.ravel(x)
        else:
            axis = _canonicalize_axis(axis, x.ndim)
            x_flat = jnp.moveaxis(x, axis, -1)
        values, indices = jax.lax.top_k(x_flat, k=k)  # (..., ..., k), (..., ..., k)
        if axis is not None:
            values = jnp.moveaxis(values, -1, axis)
            indices = jnp.moveaxis(indices, -1, axis)
        num_classes = x_flat.shape[-1]
        soft_index = jax.nn.one_hot(
            indices, num_classes=num_classes, axis=-1
        )  # (..., k, ..., [n])
        soft_index = _maybe_log_soft_index(
            soft_index, return_log_probs, log_prob_eps=log_prob_eps
        )
    else:
        if axis is None:
            x = jnp.ravel(x)
            axis = 0
        else:
            axis = _canonicalize_axis(axis, x.ndim)
        if method in ("fast_soft_sort", "smooth_sort", "sorting_network"):
            soft_sorted = sort(
                x,
                axis=axis,
                descending=True,
                softness=softness,
                standardize=standardize,
                mode=mode,
                method=method,
            )  # (..., n, ...)
            values = jnp.take(soft_sorted, jnp.arange(k), axis=axis)  # (..., k, ...)
            soft_index = None
        else:
            soft_index_is_log = (
                return_log_probs and log_prob_eps is None and mode == "smooth"
            )
            soft_index = _argtop_k(
                x=x,
                k=k,
                axis=axis,
                softness=softness,
                mode=mode,
                method=method,
                standardize=standardize,
                ot_kwargs=ot_kwargs,
                return_log_probs=soft_index_is_log,
            )  # (..., k, ..., [n])
            if not gated_grad:
                soft_index_tmp = jax.lax.stop_gradient(soft_index)
            else:
                soft_index_tmp = soft_index
            values = take_along_axis(
                x, _soft_index_probs(soft_index_tmp, soft_index_is_log), axis=axis
            )  # (..., k, ...)
            soft_index = _maybe_log_soft_index(
                soft_index,
                return_log_probs,
                already_log=soft_index_is_log,
                log_prob_eps=log_prob_eps,
            )
    return values, soft_index


def rank(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    softness: float | Array = 0.1,
    mode: Literal["hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"] = "neuralsort",
    descending: bool = True,
    standardize: bool = True,
    ot_kwargs: dict | None = None,
) -> Array:  # (..., n, ...)
    """Computes the soft ranks of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: The axis along which to compute the rank operation. If None, the input Array is flattened before computing the rank.
    - `descending`: If True, larger inputs receive smaller ranks (best rank is 0). If False, ranks increase with the input values.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Type of regularizer in the projection operators.
        - `hard`: Returns rank computed as two jnp.argsort calls.
        - `smooth`: C∞ smooth (based on logistic/softmax/entropic regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via a softmax operation.
            - For optimal transport (`ot` method), transport plan is computed via Sinkhorn iterations (see [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)).
        - `c0`: C0 continuous (based on euclidean/L2 regularizer).
            - For unit simplex projection (`softsort`/`neuralsort` methods), projection is computed in closed-form via the algorithm in [Projection onto the probability simplex: An eﬃcient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541).
            - For optimal transport (`ot` method), transport plan is computed via LBFGS (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
        - `c1`/`c2`: C1 differentiable / C2 twice differentiable. Similar to `c0`, but using p-norm regularizers with p=3/2 and p=4/3, respectively.
    - `method`: Method to compute the soft rank. All approaches were originally proposed for the smooth mode, we extend them to the c0,c1,c2 modes as well.
        - `ot`: Uses the approach in [Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885).
            Intuition: Run an OT procedure as in [`softjax.argsort`][], then transport the sorted ranks (0, 1, ..., n-1) back to the ranks of the original values by using the transpose of the transport plan.
            Note: Inaccurate for small `sinkhorn_max_iter` (can be passed as keyword argument), but can be very slow for large `sinkhorn_max_iter`.
        - `softsort`: Adapts the "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038) for rank by using a single **column**-wise projection instead of row-wise projection.
            Note: Can introduce gradient discontinuities when elements in `x` are not unique, but is much faster than OT-based method.
        - `neuralsort`: Adapts the "NeuralSort" operator from [Stochastic Optimization of Sorting Networks via Continuous Relaxations](https://arxiv.org/abs/1903.08850) for rank by renormalizing over columns after the row-wise projection.
        - `fast_soft_sort`: Uses the `FastSoftSort` operator from [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871) to directly compute the soft ranks, via projection onto the permutahedron. The projection is solved via a PAV algorithm as proposed in [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871).
    - `standardize`: If True, standardizes and squashes the input `x` along the specified axis before applying the softrank operation. This can improve numerical stability and performance, especially when the values in `x` vary widely in scale.
    - `ot_kwargs`: Additional optional keyword arguments to pass to the OT projection operator, e.g., to control the number of max iterations or tolerance.

    **Returns:**

    A positive Array of shape (..., n, ...) with values in [1, n].
    The elements in (..., i, ...) represent the soft rank of the ith element along the specified axis.
    """
    x = _ensure_float(x)
    if mode == "hard" or mode == "_hard":
        indices = jnp.argsort(x, axis=axis, descending=descending)  # (..., n, ...)
        indices = indices.astype(x.dtype)
        ranks = jnp.argsort(indices, axis=axis, descending=False) + 1  # (..., n, ...)
        ranks = ranks.astype(x.dtype)
    else:
        if axis is None:
            x = jnp.ravel(x)
            axis = 0
        else:
            axis = _canonicalize_axis(axis, x.ndim)
        x_last = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
        *batch_dims, n = x_last.shape

        if method == "smooth_sort":
            # smooth_sort skips standardize (see sort() for rationale)
            if mode not in ("smooth",):
                raise ValueError(
                    f"smooth_sort only supports mode='smooth', got mode='{mode}'"
                )
            w = -x_last if descending else x_last
            scale = jnp.maximum((n - 1), 1)
            anchors = jnp.arange(1, n + 1, dtype=x.dtype) / scale  # (n,)
            anchors = jnp.broadcast_to(anchors, (*batch_dims, n))  # (..., ..., n)
            ranks = _proj_permutahedron_smooth_sort(w, anchors, softness=softness)
            ranks = ranks * scale  # (..., n, ...)
            ranks = jnp.moveaxis(ranks, -1, axis)  # (..., n, ...)
        else:
            if standardize:
                x_last = _standardize_and_squash(x_last, axis=-1)
            if method == "fast_soft_sort":
                w = -x_last if descending else x_last
                scale = jnp.maximum((n - 1), 1)
                anchors = jnp.arange(1, n + 1, dtype=x.dtype) / scale  # (n,)
                anchors = jnp.broadcast_to(anchors, (*batch_dims, n))  # (..., ..., n)
                ranks = _proj_permutahedron(w, anchors, softness=softness, mode=mode)
                ranks = ranks * scale  # (..., n, ...)
                ranks = jnp.moveaxis(ranks, -1, axis)  # (..., n, ...)
            elif method == "softsort":
                ranks = _softsort_fused_rank(
                    x_last=x_last, batch_dims=batch_dims, softness=softness,
                    mode=mode, descending=descending,
                )
                ranks = jnp.moveaxis(ranks, -1, axis)
            elif method == "neuralsort":
                ranks = _neuralsort_fused_rank(
                    x_last=x_last, batch_dims=batch_dims, softness=softness,
                    mode=mode, descending=descending,
                )
                ranks = jnp.moveaxis(ranks, -1, axis)
            else:
                if method == "ot":
                    anchors = jnp.linspace(0, n, n, dtype=x.dtype) / n  # (n,)
                    if descending:
                        anchors = anchors[::-1]  # (n,)
                    anchors = jnp.broadcast_to(anchors, (*batch_dims, n))  # (..., ..., n)
                    cost = (
                        x_last[..., :, None] - anchors[..., None, :]
                    ) ** 2  # (..., ..., n, n)

                    mu = jnp.ones((n,), dtype=x.dtype) / n  # ([n],)
                    nu = jnp.ones((n,), dtype=x.dtype) / n  # ([n],)

                    if ot_kwargs is None:
                        ot_kwargs = {}
                    out = _proj_transport_polytope(
                        cost=cost, mu=mu, nu=nu, softness=softness, mode=mode, **ot_kwargs
                    )  # (..., ..., [n], n)
                    soft_index = out / jnp.clip(
                        jnp.sum(out, axis=-1, keepdims=True), min=1e-10
                    )  # (..., ..., n, [n])
                elif method == "sorting_network":
                    P = _argsort_via_sorting_network(
                        x_last, softness, mode, descending=descending, standardized=standardize
                    )  # (..., ..., n, [n])
                    # Transpose: P[sorted_pos, elem] → soft_index[elem, sorted_pos]
                    soft_index = jnp.swapaxes(P, -2, -1)  # (..., ..., n, [n])
                    soft_index = soft_index / jnp.clip(
                        jnp.sum(soft_index, axis=-1, keepdims=True), min=1e-10
                    )  # (..., ..., n, [n])
                else:
                    raise ValueError(f"Invalid method: {method}")

                nums = jnp.arange(1, n + 1, dtype=x.dtype)  # (n,)
                nums = jnp.broadcast_to(nums, (*batch_dims, n))  # (..., n)
                nums = jnp.moveaxis(nums, -1, axis)  # (..., n, ...)
                soft_rank_indices = jnp.moveaxis(soft_index, -2, axis)  # (..., n, ..., [n])
                ranks = take_along_axis(nums, soft_rank_indices, axis=axis)  # (..., n, ...)
    return ranks


# Autograd-safe operators


def sqrt(x: Array) -> Array:
    """Autograd-safe version of [jax.numpy.sqrt](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sqrt.html) via the double-where trick.

    Returns ``jnp.sqrt(x)`` for ``x > 0`` and ``0`` otherwise, without
    producing NaN gradients at ``x = 0`` (unlike ``jnp.sqrt``).

    **Arguments:**

    - `x`: Input Array.

    **Returns:**

    Elementwise square root of `x`, safe for autodiff.
    """
    safe_x = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, jnp.sqrt(safe_x), 0.0)


def arcsin(x: Array) -> Array:
    """Autograd-safe version of [jax.numpy.arcsin](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arcsin.html) via the double-where trick.

    Returns ``jnp.arcsin(x)`` for ``|x| < 1`` and ``±π/2`` at ``x = ±1``,
    without producing NaN gradients at the boundary (unlike ``jnp.arcsin``).

    **Arguments:**

    - `x`: Input Array.

    **Returns:**

    Elementwise arcsine of `x`, safe for autodiff.
    """
    safe_x = jnp.where(jnp.abs(x) < 1, x, 0.0)
    return jnp.where(jnp.abs(x) < 1, jnp.arcsin(safe_x), jnp.sign(x) * (jnp.pi / 2))


def arccos(x: Array) -> Array:
    """Autograd-safe version of [jax.numpy.arccos](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arccos.html) via the double-where trick.

    Returns ``jnp.arccos(x)`` for ``|x| < 1``, ``0`` at ``x >= 1``, and
    ``π`` at ``x <= -1``, without producing NaN gradients at the boundary
    (unlike ``jnp.arccos``).

    **Arguments:**

    - `x`: Input Array.

    **Returns:**

    Elementwise arccosine of `x`, safe for autodiff.
    """
    safe_x = jnp.where(jnp.abs(x) < 1, x, 0.0)
    return jnp.where(jnp.abs(x) < 1, jnp.arccos(safe_x), jnp.where(x >= 1, 0.0, jnp.pi))


def div(x: Array, y: Array) -> Array:
    """Autograd-safe division via the double-where trick.

    Returns ``x / y`` when ``y != 0`` and ``0`` otherwise, without
    producing NaN gradients at ``y = 0`` (unlike plain ``x / y``).

    **Arguments:**

    - `x`: Numerator Array.
    - `y`: Denominator Array.

    **Returns:**

    Elementwise ``x / y``, safe for autodiff.
    """
    safe_y = jnp.where(y != 0, y, 1.0)
    return jnp.where(y != 0, x / safe_y, 0.0)


def log(x: Array) -> Array:
    """Autograd-safe version of [jax.numpy.log](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.log.html) via the double-where trick.

    Returns ``jnp.log(x)`` for ``x > 0`` and ``0`` otherwise, without
    producing NaN gradients at ``x = 0`` (unlike ``jnp.log``).

    **Arguments:**

    - `x`: Input Array.

    **Returns:**

    Elementwise natural logarithm of `x`, safe for autodiff.
    """
    safe_x = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, jnp.log(safe_x), 0.0)


def norm(x: Array, axis=None, keepdims=False) -> Array:
    """Autograd-safe L2 norm via :func:`sqrt`.

    Computes ``sqrt(sum(x**2, ...))`` using the autograd-safe :func:`sqrt`,
    avoiding NaN gradients when the norm is zero (unlike ``jnp.linalg.norm``).

    **Arguments:**

    - `x`: Input Array.
    - `axis`: Axis or axes along which to compute the norm.
    - `keepdims`: If ``True``, retains reduced axes with size 1.

    **Returns:**

    L2 norm of `x` along the given axis, safe for autodiff.
    """
    return sqrt(jnp.sum(x * x, axis=axis, keepdims=keepdims))


# Elementwise operators


def sigmoidal(
    x: Array,
    softness: float | Array = 0.1,
    mode: Literal["smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"] = "smooth",
) -> SoftBool:
    """Sigmoidal functions defining a characteristic S-shaped curve.

    **Arguments:**

    - `x`: Input Array.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Choice of smoothing family for the surrogate step.
        - `smooth`: C∞ smooth (based on logistic/softmax/entropic regularizer). Smooth sigmoidal based on the logistic function.
        - `c0`: C0 continuous (based on euclidean/L2 regularizer). Continuous sigmoidal based on a piecewise quadratic polynomial.
        - `c1`: C1 differentiable (cubic Hermite). Differentiable sigmoidal based on a piecewise cubic polynomial.
        - `c2`: C2 twice differentiable (quintic Hermite). Twice differentiable sigmoidal based on a piecewise quintic polynomial.

    **Returns:**

    SoftBool of same shape as `x` (Array with values in [0, 1]).
    """
    _validate_softness(softness)
    x = x / softness
    if mode == "smooth":  # smooth
        # closed form of argmax([0,x], softness=softness, mode="smooth", standardize=False)[1]
        y = jax.nn.sigmoid(x)
    else:
        # Piecewise modes are defined on [-1, 1]. Scale by 1/5 so the effective
        # transition region [-5s, 5s] matches smooth's range (sigmoid(±5) ≈ 0/1).
        x = x / 5.0
        if mode == "c0":  # continuous
            # closed form of argmax([0,x], softness=5*softness, mode="c0", standardize=False)[1]
            y = jnp.polyval(jnp.array([0.5, 0.5], dtype=x.dtype), x)
            y = jnp.where(x < -1.0, 0.0, jnp.where(x < 1.0, y, 1.0))
        elif mode == "c1":  # differentiable
            # C1 Hermite smoothstep (f=0/1 and f'=0 at boundaries)
            y = jnp.polyval(jnp.array([-0.25, 0.0, 0.75, 0.5], dtype=x.dtype), x)
            y = jnp.where(x < -1.0, 0.0, jnp.where(x < 1.0, y, 1.0))
        elif mode == "_c1_pnorm":  # differentiable, p-norm based
            # closed form of argmax([0,x], softness=5*softness, mode="c1", standardize=False)[1]
            # y = (x + sqrt(2 - x^2))^2 / 4 on [-1, 1]
            y = jnp.square(x + sqrt(2.0 - jnp.square(x))) / 4.0
            y = jnp.where(x < -1.0, 0.0, jnp.where(x < 1.0, y, 1.0))
        elif mode == "c2":  # twice differentiable
            # C2 Hermite smoothstep (f=0/1, f'=0, and f''=0 at boundaries)
            y = jnp.polyval(jnp.array([0.1875, 0.0, -0.625, 0.0, 0.9375, 0.5], dtype=x.dtype), x)
            y = jnp.where(x < -1.0, 0.0, jnp.where(x < 1.0, y, 1.0))
        elif mode == "_c2_pnorm":  # twice differentiable, p-norm based
            # closed form of argmax([0,x], softness=5*softness, mode="c2", standardize=False)[1]
            # depressed cubic: t^3 + A*t + B = 0, where A = 3d^2/4, B = -1/2, d = x
            d = x
            p_coeff = -1.5 * d
            A = 0.75 * d**2
            B = jnp.full_like(x, -0.5)
            # hyperbolic Cardano with cbrt fallback for A~0 (avoids float32 underflow)
            A_safe = jnp.where(A > 1e-12, A, 1.0)
            sA3 = sqrt(A_safe / 3.0)
            arg = 3.0 * B / (2.0 * A_safe * sA3)
            t_hyp = -2.0 * sA3 * jnp.sinh(jnp.arcsinh(arg) / 3.0)
            t_cbrt = -jnp.sign(B) * jnp.abs(B) ** (1.0 / 3.0)
            t = jnp.where(A > 1e-12, t_hyp, t_cbrt)
            b = t - p_coeff / 3.0
            y = b**3
            y = jnp.where(x < -1.0, 0.0, jnp.where(x < 1.0, y, 1.0))
        else:
            raise ValueError(f"Invalid mode: {mode}")
    return y


def softrelu(
    x: Array,
    softness: float | Array = 0.1,
    mode: Literal["smooth", "c0", "c1", "_c1_pnorm", "c2", "_c2_pnorm"] = "smooth",
    gated: bool = False,
) -> Array:
    """Family of soft relaxations to ReLU.

    **Arguments:**

    - `x`: Input Array.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Choice of [`softjax.sigmoidal`][] smoothing mode, options are "smooth", "c0", "c1", "_c1_pnorm", "c2", or "_c2_pnorm".
    - `gated`: If True, uses the 'gated' version `x * sigmoidal(x)`. If False, uses the integral of the sigmoidal.

    **Returns:**

    Result of applying soft elementwise ReLU to `x`.
    """
    _validate_softness(softness)
    x = x / softness
    if mode == "smooth":
        if gated:
            # x * sigmoidal(x) = max([0,x], softness=softness, mode="smooth", standardize=False)
            y = x * sigmoidal(x, softness=1.0, mode="smooth")
        else:
            # closed form integral of sigmoidal(x, mode="smooth")
            y = jax.nn.softplus(x)
    else:
        # Piecewise modes: scale by 1/5 to match smooth's transition width.
        # Non-gated: softrelu = 5*G(u) where G is the antiderivative on [-1,1].
        # Gated: softrelu = x*sigmoidal(x), sigmoidal handles the 1/5 internally.
        u = x / 5.0
        if mode == "c0":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="c0")
            else:
                y = 5.0 * jnp.polyval(jnp.array([0.25, 0.5, 0.25], dtype=u.dtype), u)
                y = jnp.where(u < -1.0, 0.0, jnp.where(u < 1.0, y, x))
        elif mode == "c1":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="c1")
            else:
                y = 5.0 * jnp.polyval(
                    jnp.array([-0.0625, 0.0, 0.375, 0.5, 0.1875], dtype=u.dtype), u
                )
                y = jnp.where(u < -1.0, 0.0, jnp.where(u < 1.0, y, x))
        elif mode == "_c1_pnorm":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="_c1_pnorm")
            else:
                # F(u) = u/2 - (2 - u^2)^(3/2) / 6 + 2/3
                inside = 2.0 - u**2
                y = 5.0 * (u / 2.0 - inside * sqrt(inside) / 6.0 + 2.0 / 3.0)
                y = jnp.where(u < -1.0, 0.0, jnp.where(u < 1.0, y, x))
        elif mode == "c2":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="c2")
            else:
                y = 5.0 * jnp.polyval(
                    jnp.array(
                        [0.03125, 0.0, -0.15625, 0.0, 0.46875, 0.5, 0.15625],
                        dtype=u.dtype,
                    ),
                    u,
                )
                y = jnp.where(u < -1.0, 0.0, jnp.where(u < 1.0, y, x))
        elif mode == "_c2_pnorm":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="_c2_pnorm")
            else:
                # F(u) = 0.75 + u*g(u) - 0.75*((1-g(u))^(4/3) + g(u)^(4/3))
                y2 = sigmoidal(x, softness=1.0, mode="_c2_pnorm")
                safe_y2 = jnp.clip(y2, 0.0, 1.0)
                y = 5.0 * (
                    0.75
                    + u * safe_y2
                    - 0.75
                    * ((1.0 - safe_y2) ** (4.0 / 3.0) + safe_y2 ** (4.0 / 3.0))
                )
                y = jnp.where(u < -1.0, 0.0, jnp.where(u < 1.0, y, x))
        else:
            raise ValueError(f"Unknown mode '{mode}' for softrelu.")
    y = y * softness
    return y


def heaviside(
    x: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
) -> SoftBool:
    """Performs a soft version of [jax.numpy.heaviside](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.heaviside.html)(x,0.5).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact Heaviside step. Otherwise uses [`softjax.sigmoidal`][]-based "smooth", "c0", "c1", "c2" relaxations.

    **Returns:**

    SoftBool of same shape as `x` (Array with values in [0, 1]), relaxing the elementwise Heaviside step function.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return jnp.where(x < 0.0, 0.0, jnp.where(x > 0.0, 1.0, 0.5)).astype(x.dtype)
    else:
        return sigmoidal(x, softness=softness, mode=mode)


def round(
    x: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    neighbor_radius: int = 5,
) -> Array:
    """Performs a soft version of [jax.numpy.round](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.round.html).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", applies `jnp.round`. Otherwise uses a sigmoidal-based relaxation based on the algorithm described in [Smooth Approximations of the Rounding Function](https://arxiv.org/pdf/2504.19026v1).
        This function thereby inherits the different [`softjax.sigmoidal`][] modes "smooth", "c0", "c1", or "c2".
    - `neighbor_radius`: Number of neighbors on each side of the floor value to consider for the soft rounding.

    **Returns:**

    Result of applying soft elementwise rounding to `x`.
    """
    if mode == "hard":
        return jnp.round(x)
    else:
        x = _ensure_float(x)
        center = jax.lax.stop_gradient(jnp.floor(x))
        offsets = jnp.arange(
            -neighbor_radius, neighbor_radius + 1, dtype=x.dtype
        )  # (M,)
        n = center[..., None] + offsets  # (..., M)

        w_left = sigmoidal(x[..., None] - (n - 0.5), softness=softness, mode=mode)
        w_right = sigmoidal(x[..., None] - (n + 0.5), softness=softness, mode=mode)
        return jnp.sum(n * (w_left - w_right), axis=-1)


def sign(
    x: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
) -> Array:
    """Performs a soft version of [jax.numpy.sign](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sign.html).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns `jnp.sign`. Otherwise uses [`softjax.sigmoidal`][]-based "smooth", "c0", "c1", or "c2" relaxations.

    **Returns:**

    Result of applying soft elementwise sign to `x`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return jnp.sign(x).astype(x.dtype)
    else:
        return sigmoidal(x, mode=mode, softness=softness) * 2.0 - 1.0


def abs(
    x: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
) -> Array:
    """Performs a soft version of [jax.numpy.abs](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.abs.html).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: Projection mode. "hard" returns the exact absolute value. Otherwise uses [`softjax.sigmoidal`][]-based "smooth", "c0", "c1", or "c2" relaxations.


    **Returns:**

    Result of applying soft elementwise absolute value to `x`.
    """
    if mode == "hard":
        return jnp.abs(x)
    else:
        return x * sign(x, mode=mode, softness=softness)


def relu(
    x: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    gated: bool = False,
) -> Array:
    """Performs a soft version of [jax.nn.relu](https://docs.jax.dev/en/latest/_autosummary/jax.nn.relu.html).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", applies `jax.nn.relu`. Otherwise uses [`softjax.softrelu`][] with "smooth", "c0", "c1", or "c2" relaxations.
    - `gated`: See [`softjax.softrelu`][] documentation.

    **Returns:**

    Result of applying soft elementwise ReLU to `x`.
    """
    if mode == "hard":
        return jax.nn.relu(x)
    else:
        return softrelu(x, mode=mode, softness=softness, gated=gated)


def clip(
    x: Array,
    a: Array,
    b: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    gated: bool = False,
) -> Array:
    """Performs a soft version of [jax.numpy.clip](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.clip.html).
    Implemented via two [`softjax.softrelu`][] calls.

    **Arguments:**

    - `x`: Input Array of any shape.
    - `a`: Lower bound scalar.
    - `b`: Upper bound scalar.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", applies `jnp.clip`. Otherwise uses [`softjax.softrelu`][]-based "smooth", "c0", "c1", or "c2" relaxations.
    - `gated`: See [`softjax.softrelu`][] documentation.

    **Returns:**

    Result of applying soft elementwise clipping to `x`.
    """
    if mode == "hard":
        return jnp.clip(x, a, b)
    else:
        tmp1 = softrelu(x - a, mode=mode, softness=softness, gated=gated)
        tmp2 = softrelu(x - b, mode=mode, softness=softness, gated=gated)
        return a + tmp1 - tmp2


# Comparison operators


def greater(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x > y`.
    Uses a Heaviside relaxation so the output approaches 0 at equality.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, greater returns 0 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise `x > y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return jnp.greater(x, y).astype(x.dtype)
    else:
        return sigmoidal(x - y - epsilon, softness=softness, mode=mode)


def greater_equal(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x >= y`.
    Uses a Heaviside relaxation so the output approaches 1 at equality.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, greater_equal returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise `x >= y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return jnp.greater_equal(x, y).astype(x.dtype)
    else:
        return sigmoidal(x - y + epsilon, softness=softness, mode=mode)


def less(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x < y`.
    Uses a Heaviside relaxation so the output approaches 0 at equality.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, less returns 0 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise `x < y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return jnp.less(x, y).astype(x.dtype)
    else:
        return logical_not(
            greater_equal(x, y, softness=softness, mode=mode, epsilon=epsilon)
        )


def less_equal(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x <= y`.
    Uses a Heaviside relaxation so the output approaches 1 at equality.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, less_equal returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise `x <= y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return jnp.less_equal(x, y).astype(x.dtype)
    else:
        return logical_not(greater(x, y, softness=softness, mode=mode, epsilon=epsilon))


def equal(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x == y`.
    Implemented as a soft `abs(x - y) <= 0` comparison.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, equal returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise `x == y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return jnp.equal(x, y).astype(x.dtype)
    else:
        diff = abs(x - y, softness=softness, mode=mode)
        # diff >= 0, so less_equal(diff, 0) (a sigmoid on a non-positive input)
        # is in [0, 0.5], and the 2x scaling keeps the result in [0, 1].
        return 2.0 * less_equal(
            diff, jnp.zeros_like(diff), mode=mode, softness=softness, epsilon=epsilon
        )


def not_equal(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x != y`.
    Implemented as a soft `abs(x - y) > 0` comparison.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, not_equal returns 0 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise `x != y`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return jnp.not_equal(x, y).astype(x.dtype)
    else:
        diff = abs(x - y, softness=softness, mode=mode)
        # diff >= 0, so greater(diff, 0) (a sigmoid on a non-negative input)
        # is in [0.5, 1], and the 2x-1 scaling keeps the result in [0, 1].
        tmp = greater(
            diff,
            jnp.zeros_like(diff),
            mode=mode,
            softness=softness,
            epsilon=epsilon,
        )
        return 2.0 * tmp - 1.0


def isclose(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    mode: Literal[
        "hard", "smooth", "c0", "c1", "c2"
    ] = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to `jnp.isclose` for elementwise comparison.
    Implemented as a soft `abs(x - y) <= atol + rtol * abs(y)` comparison.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero.
    - `rtol`: Relative tolerance.
    - `atol`: Absolute tolerance.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft Heaviside.
    - `epsilon`: Small offset so that as softness->0, isclose returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise `isclose(x, y)`.
    """
    x = _ensure_float(x)
    if mode == "hard":
        return jnp.isclose(x, y, atol=atol, rtol=rtol).astype(x.dtype)
    else:
        diff = abs(x - y, softness=softness, mode=mode)
        y_abs = abs(y, softness=softness, mode=mode)
        # diff >= 0 and atol + rtol * y_abs >= 0, so less_equal(diff, tol)
        # is in [0, 0.5] when diff > tol, and the 2x scaling keeps the result in [0, 1].
        return 2.0 * less_equal(
            diff,
            atol + rtol * y_abs,
            mode=mode,
            softness=softness,
            epsilon=epsilon,
        )


# Logical operators


def logical_not(x: SoftBool) -> SoftBool:
    """Computes soft elementwise logical NOT of a SoftBool Array.
    Fuzzy logic implemented as `1.0 - x`.

    **Arguments:**
    - `x`: SoftBool input Array.

    **Returns:**

    SoftBool of same shape as `x` (Array with values in [0, 1]), relaxing the elementwise logical NOT.
    """
    return 1.0 - x


def all(
    x: SoftBool,
    axis: int = -1,
    epsilon: float = 1e-10,
    use_geometric_mean: bool = False,
) -> SoftBool:
    """Computes soft elementwise logical AND across a specified axis.
    Fuzzy logic implemented as the geometric mean along the axis.

    **Arguments:**
    - `x`: SoftBool input Array.
    - `axis`: Axis along which to compute the logical AND. Default is -1 (last axis).
    - `epsilon`: Minimum value for numerical stability inside the log.
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft AND. Otherwise, the product is used.

    **Returns:**

    SoftBool (Array with values in [0, 1]) with the specified axis reduced, relaxing the logical ALL along that axis.
    """
    if use_geometric_mean:
        return jnp.exp(jnp.mean(jnp.log(jnp.clip(x, min=epsilon)), axis=axis))
    else:
        return jnp.prod(x, axis=axis)


def any(x: SoftBool, axis: int = -1, use_geometric_mean: bool = False) -> SoftBool:
    """Computes soft elementwise logical OR across a specified axis.
    Fuzzy logic implemented as `1.0 - all(logical_not(x), axis=axis)`.

    **Arguments:**
    - `x`: SoftBool input Array.
    - `axis`: Axis along which to compute the logical OR. Default is -1 (last axis).
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft ALL inside the logical NOT. Otherwise, the product is used.

    **Returns:**

    SoftBool (Array with values in [0, 1]) with the specified axis reduced, relaxing the logical ANY along that axis.
    """
    return logical_not(
        all(logical_not(x), axis=axis, use_geometric_mean=use_geometric_mean)
    )


def logical_and(x: SoftBool, y: SoftBool, use_geometric_mean: bool = False) -> SoftBool:
    """Computes soft elementwise logical AND between two SoftBool Arrays.
    Fuzzy logic implemented as `all(stack([x, y], axis=-1), axis=-1)`.

    **Arguments:**

    - `x`: First SoftBool input Array.
    - `y`: Second SoftBool input Array.
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft AND. Otherwise, the product is used.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise logical AND.
    """
    return all(
        jnp.stack([x, y], axis=-1), axis=-1, use_geometric_mean=use_geometric_mean
    )


def logical_or(x: SoftBool, y: SoftBool, use_geometric_mean: bool = False) -> SoftBool:
    """Computes soft elementwise logical OR between two SoftBool Arrays.
    Fuzzy logic implemented as `any(stack([x, y], axis=-1), axis=-1)`.

    **Arguments:**
    - `x`: First SoftBool input Array.
    - `y`: Second SoftBool input Array.
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft ALL inside the logical NOT. Otherwise, the product is used.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise logical OR.
    """
    return any(
        jnp.stack([x, y], axis=-1), axis=-1, use_geometric_mean=use_geometric_mean
    )


def logical_xor(x: SoftBool, y: SoftBool, use_geometric_mean: bool = False) -> SoftBool:
    """Computes soft elementwise logical XOR between two SoftBool Arrays.

    **Arguments:**
    - `x`: First SoftBool input Array.
    - `y`: Second SoftBool input Array.
    - `use_geometric_mean`: If True, uses the geometric mean to compute the soft AND and OR operations. Otherwise, the product is used.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the elementwise logical XOR.
    """
    tmp1 = logical_and(x, logical_not(y), use_geometric_mean=use_geometric_mean)
    tmp2 = logical_and(logical_not(x), y, use_geometric_mean=use_geometric_mean)
    return logical_or(tmp1, tmp2, use_geometric_mean=use_geometric_mean)
