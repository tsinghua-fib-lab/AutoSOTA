# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Feature communication used by communicating deployed models.

The Python implementation is an identity operation. During export the two
primitives lower to FFI calls which are implemented by chemtrain-deploy and
LAMMPS. Keeping the eager implementation as an identity is useful for model
initialisation and single-domain testing.
"""

from __future__ import annotations

import math

import jax
from jax import numpy as jnp


FORWARD_TARGET = "chemtrain_deploy.gather_forward"
REVERSE_TARGET = "chemtrain_deploy.gather_reverse"
CUSTOM_CALL_TARGETS = (FORWARD_TARGET, REVERSE_TARGET)


class ExportCommunication:
    """Communication interface used while tracing a deployment variant.

    ``widths`` records static packed layouts during a metadata trace. During
    the real export, the same sequence is validated and a fixed float32 token
    chains calls through the public FFI. The token value is irrelevant; only
    the data dependency between successive calls matters.

    Width recording is a Python tracing-time side effect, not part of the
    compiled computation. The exporter therefore creates a new instance for
    every trace; retracing replaces the complete record instead of appending
    to metadata from an earlier trace. This is safe for functionally pure
    models whose communication sites and packed widths are structurally
    fixed. The validation trace rejects a different site sequence or width.
    """

    def __init__(
        self,
        enabled: bool = False,
        expected_widths: tuple[int, ...] | None = None,
    ):
        self.enabled = enabled
        self.widths: list[int] = []
        self.expected_widths = expected_widths
        self.token = jnp.zeros((1,), dtype=jnp.float32)

    def gather(self, tree):
        """Record one communication site and optionally emit its FFI call."""
        width = packed_width(tree)
        site = len(self.widths)
        if self.expected_widths is not None:
            if (site >= len(self.expected_widths) or
                    width != self.expected_widths[site]):
                expected = (
                    self.expected_widths[site]
                    if site < len(self.expected_widths) else "no site"
                )
                raise ValueError(
                    "Communication structure changed while exporting: "
                    f"site {site} has width {width}, expected "
                    f"{expected}"
                )
        # This list belongs only to the current trace. It is consumed by the
        # exporter after tracing and is never observed by compiled model code.
        self.widths.append(width)
        if not self.enabled:
            return tree
        communicated, self.token = _gather_with_token(tree, self.token)
        return communicated

    def validate(self) -> None:
        """Check that a validation trace visited every expected site."""
        if (self.expected_widths is not None and
                tuple(self.widths) != self.expected_widths):
            raise ValueError(
                "Communication structure changed while exporting: "
                f"observed widths {tuple(self.widths)}, expected "
                f"{self.expected_widths}"
            )


def packed_width(tree) -> int:
    """Return the scalar width of a packed atom-leading floating pytree."""
    arrays, _ = _validated_arrays(tree)
    return sum(math.prod(array.shape[1:]) for array in arrays)


def _validated_arrays(tree):
    """Flatten and validate the common layout required by communication."""
    leaves, treedef = jax.tree.flatten(tree)
    if not leaves:
        raise ValueError("comm.gather requires a non-empty pytree")

    arrays = [jnp.asarray(leaf) for leaf in leaves]
    first = arrays[0]
    if first.ndim < 1:
        raise ValueError("comm.gather leaves must have an atom-leading axis")
    if not jnp.issubdtype(first.dtype, jnp.floating):
        raise TypeError("comm.gather supports floating-point arrays only")

    for array in arrays[1:]:
        if array.ndim < 1:
            raise ValueError(
                "comm.gather leaves must have an atom-leading axis"
            )
        if array.dtype != first.dtype:
            raise TypeError("comm.gather leaves must have the same dtype")
        if array.shape[0] != first.shape[0]:
            raise ValueError(
                "comm.gather leaves must have the same atom-leading size"
            )
    return arrays, treedef


def _exchange(target, buffer, token):
    """Invoke the public typed FFI with an explicit array dependency token."""
    result_shapes = (
        jax.ShapeDtypeStruct(buffer.shape, buffer.dtype),
        jax.ShapeDtypeStruct(token.shape, token.dtype),
    )
    return jax.ffi.ffi_call(
        target,
        result_shapes,
        has_side_effect=True,
        vmap_method="sequential",
    )(buffer, token)


@jax.custom_vjp
def _communicate(buffer, token):
    return _exchange(FORWARD_TARGET, buffer, token)


def _communicate_fwd(buffer, token):
    output = _exchange(FORWARD_TARGET, buffer, token)
    return output, None


def _communicate_bwd(_, cotangents):
    buffer_cotangent, token_cotangent = cotangents
    return _exchange(REVERSE_TARGET, buffer_cotangent, token_cotangent)


_communicate.defvjp(_communicate_fwd, _communicate_bwd)


def _gather_with_token(tree, token):
    """Gather atom-leading floating-point arrays with one FFI call.

    Leaves are packed into a single ``[n_atoms, packed_width]`` matrix. The
    reverse-mode transpose is a reverse communication call at the matching
    point in backpropagation.
    """

    arrays, treedef = _validated_arrays(tree)
    first = arrays[0]

    widths = [math.prod(array.shape[1:]) for array in arrays]

    packed = jnp.concatenate(
        [
            array.reshape((array.shape[0], width))
            for array, width in zip(arrays, widths)
        ],
        axis=1,
    )

    communicated, token = _communicate(packed, token)

    if communicated.shape != packed.shape:
        raise ValueError(
            "comm.gather changed shape from "
            f"{packed.shape} to {communicated.shape}"
        )

    unpacked = []
    start = 0
    for array, width in zip(arrays, widths):
        unpacked.append(
            communicated[:, start:start + width].reshape(array.shape)
        )
        start += width

    return jax.tree.unflatten(treedef, unpacked), token


def gather(tree):
    """Identity communication used outside an exporter-managed chain.

    Deployment models receive :class:`ExportCommunication` explicitly. The
    module-level function stays executable without a registered native FFI,
    which keeps model initialization and single-domain reference tests simple.
    """
    arrays, treedef = _validated_arrays(tree)
    return jax.tree.unflatten(treedef, arrays)
