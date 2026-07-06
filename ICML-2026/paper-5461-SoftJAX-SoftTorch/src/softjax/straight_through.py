import functools
import inspect
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

import softjax as sj


def _replace_value_keep_grad(forward, backward):
    if forward is None or backward is None:
        return forward
    y = jax.lax.stop_gradient(forward - backward) + backward
    if hasattr(y, "dtype") and jnp.issubdtype(y.dtype, jnp.inexact):
        same_nonfinite = (forward == backward) & ~jnp.isfinite(forward)
        y = jnp.where(jnp.isnan(y) & same_nonfinite, jax.lax.stop_gradient(forward), y)
    return y


def grad_replace(fn: Callable) -> Callable:
    """This decorator calls the decorated function twice: once with `forward=True` and once with `forward=False`.
    It returns the output from the forward pass, but uses the output from the backward pass to compute gradients.

    **Arguments:**

    - `fn`: The function to be wrapped. It should accept a `forward` argument that specifies which computation to perform depending on forward or backward pass.

    **Returns:**
        A wrapped function that behaves like the `forward=True` version during the forward pass, but computes gradients using the `forward=False` version during the backward pass.
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        fw_y = fn(*args, **kwargs, forward=True)
        bw_y = fn(*args, **kwargs, forward=False)
        fw_leaves, fw_treedef = jtu.tree_flatten(fw_y, is_leaf=lambda x: x is None)
        bw_leaves, bw_treedef = jtu.tree_flatten(bw_y, is_leaf=lambda x: x is None)
        out_leaves = [_replace_value_keep_grad(f, b) for f, b in zip(fw_leaves, bw_leaves)]
        return jtu.tree_unflatten(fw_treedef, out_leaves)

    return wrapped


def st(fn: Callable) -> Callable:
    """This decorator calls the decorated function twice: once with `mode="hard"` and once with the specified `mode`.
    It returns the output from the hard forward pass, but uses the output from the soft backward pass to compute gradients.

    **Arguments:**

    - `fn`: The function to be wrapped. It may accept a `mode` argument. If `fn` has no `mode` parameter, it defaults to `"smooth"` and `mode` is passed through via `**kwargs`.

    **Returns:**
        A wrapped function that behaves like the `mode="hard"` version during the forward pass, but computes gradients using the specified `mode` and `softness` during the backward pass.
    """
    sig = inspect.signature(fn)
    mode_param = sig.parameters.get("mode")
    if mode_param is not None:
        mode_default = mode_param.default
        mode_idx = list(sig.parameters.keys()).index("mode")
    else:
        mode_default = "smooth"
        mode_idx = None

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if mode_idx is not None and len(args) > mode_idx:
            # mode was passed positionally — extract it
            mode = args[mode_idx]
            args = args[:mode_idx] + args[mode_idx + 1:]
        else:
            mode = kwargs.pop("mode", mode_default)
        fw_y = fn(*args, **kwargs, mode="hard")
        bw_y = fn(*args, **kwargs, mode=mode)
        fw_leaves, fw_treedef = jtu.tree_flatten(fw_y, is_leaf=lambda x: x is None)
        bw_leaves, bw_treedef = jtu.tree_flatten(bw_y, is_leaf=lambda x: x is None)
        out_leaves = [_replace_value_keep_grad(f, b) for f, b in zip(fw_leaves, bw_leaves)]
        return jtu.tree_unflatten(fw_treedef, out_leaves)

    return wrapped


_st_cache = {}


def _cached_st(fn):
    """Return a cached st() wrapper, avoiding repeated inspect.signature() calls."""
    if fn not in _st_cache:
        _st_cache[fn] = st(fn)
    return _st_cache[fn]


def abs_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.abs`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.abs)`.

    This function returns the hard `abs` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.abs)(*args, **kwargs)


def argmax_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.argmax`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.argmax)`.

    This function returns the hard `argmax` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.argmax)(*args, **kwargs)


def argmedian_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.argmedian`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.argmedian)`.

    This function returns the hard `argmedian` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.argmedian)(*args, **kwargs)


def argmin_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.argmin`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.argmin)`.

    This function returns the hard `argmin` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.argmin)(*args, **kwargs)


def argpercentile_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.argpercentile`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.argpercentile)`.

    This function returns the hard `argpercentile` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.argpercentile)(*args, **kwargs)


def argquantile_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.argquantile`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.argquantile)`.

    This function returns the hard `argquantile` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.argquantile)(*args, **kwargs)


def argsort_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.argsort`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.argsort)`.

    This function returns the hard `argsort` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.argsort)(*args, **kwargs)


def clip_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.clip`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.clip)`.

    This function returns the hard `clip` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.clip)(*args, **kwargs)


def equal_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.equal`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.equal)`.

    This function returns the hard `equal` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.equal)(*args, **kwargs)


def greater_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.greater`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.greater)`.

    This function returns the hard `greater` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.greater)(*args, **kwargs)


def greater_equal_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.greater_equal`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.greater_equal)`.

    This function returns the hard `greater_equal` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.greater_equal)(*args, **kwargs)


def heaviside_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.heaviside`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.heaviside)`.

    This function returns the hard `heaviside` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.heaviside)(*args, **kwargs)


def isclose_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.isclose`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.isclose)`.

    This function returns the hard `isclose` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.isclose)(*args, **kwargs)


def less_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.less`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.less)`.

    This function returns the hard `less` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.less)(*args, **kwargs)


def less_equal_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.less_equal`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.less_equal)`.

    This function returns the hard `less_equal` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.less_equal)(*args, **kwargs)


def max_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.max`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.max)`.

    This function returns the hard `max` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.max)(*args, **kwargs)


def median_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.median`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.median)`.

    This function returns the hard `median` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.median)(*args, **kwargs)


def min_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.min`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.min)`.

    This function returns the hard `min` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.min)(*args, **kwargs)


def not_equal_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.not_equal`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.not_equal)`.

    This function returns the hard `not_equal` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.not_equal)(*args, **kwargs)


def percentile_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.percentile`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.percentile)`.

    This function returns the hard `percentile` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.percentile)(*args, **kwargs)


def quantile_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.quantile`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.quantile)`.

    This function returns the hard `quantile` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.quantile)(*args, **kwargs)


def rank_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.rank`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.rank)`.

    This function returns the hard `rank` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.rank)(*args, **kwargs)


def relu_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.relu`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.relu)`.

    This function returns the hard `relu` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.relu)(*args, **kwargs)


def round_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.round`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.round)`.

    This function returns the hard `round` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.round)(*args, **kwargs)


def sign_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.sign`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.sign)`.

    This function returns the hard `sign` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.sign)(*args, **kwargs)


def sort_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.sort`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.sort)`.

    This function returns the hard `sort` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.sort)(*args, **kwargs)


def top_k_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.top_k`][].
    Implemented using the [`softjax.st`][] decorator as `st(softjax.top_k)`.

    This function returns the hard `top_k` during the forward pass, but uses a soft relaxation (controlled by the `mode` argument) for the backward pass (i.e., gradients are computed through the soft version).
    """
    return _cached_st(sj.top_k)(*args, **kwargs)
