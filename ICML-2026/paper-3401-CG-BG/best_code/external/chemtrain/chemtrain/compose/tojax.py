"""Loads MACE torch checkpoints through the toJax patching path."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Callable, Dict, Tuple

import numpy as np

import jax
import jax.core as jcore
import jax.numpy as jnp

from jax_md import partition, space
from jax_md_mod import custom_partition

from mace_jax.cli import mace_jax_from_torch
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig

from tojax import tojax
from tojax.patches import patch_module
from tojax.wrapper import TensorWrapper, jax_dtype, unwrap, wrap

from .mace_jax import AtomicNumberMapping, SpeciesMapping

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import e3nn.nn._extract as e3nn_extract
import mace.modules.models as mace_models
import mace.modules.utils as mace_utils
import torch


if not getattr(TensorWrapper, "_chemtrain_mace_compat", False):
    def _new_zeros(self, *size, dtype=None, **_):
        size = unwrap(size[0] if isinstance(size[0], tuple) else size)
        return TensorWrapper(
            jnp.zeros(
                size,
                dtype=self.data.dtype if dtype is None else jax_dtype(dtype),
            )
        )

    def _new_ones(self, *size, dtype=None, **_):
        size = unwrap(size[0] if isinstance(size[0], tuple) else size)
        return TensorWrapper(
            jnp.ones(
                size,
                dtype=self.data.dtype if dtype is None else jax_dtype(dtype),
            )
        )

    def _new_full(self, size, fill_value, *, dtype=None, **_):
        size = unwrap(size[0] if isinstance(size[0], tuple) else size)
        return TensorWrapper(
            jnp.full(
                size,
                fill_value,
                dtype=self.data.dtype if dtype is None else jax_dtype(dtype),
            )
        )

    TensorWrapper.new_zeros = _new_zeros
    TensorWrapper.new_ones = _new_ones
    TensorWrapper.new_full = _new_full
    TensorWrapper._chemtrain_mace_compat = True


if not getattr(e3nn_extract.Extract, "_chemtrain_mace_compat", False):

    def _extract_forward(self, x):
        is_wrapped = isinstance(x, TensorWrapper)
        data = unwrap(x) if is_wrapped else x

        outs = []
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            pieces = []
            for s_out, i_in in zip(irreps_out.slices(), ins):
                del s_out
                i_start = self.irreps_in[:i_in].dim
                i_len = self.irreps_in[i_in].dim
                pieces.append(data[..., i_start : i_start + i_len])

            if is_wrapped:
                outs.append(jnp.concatenate(pieces, axis=-1))
            else:
                outs.append(torch.cat(pieces, dim=-1))

        if len(outs) == 1:
            return wrap(outs[0]) if is_wrapped else outs[0]
        return tuple(wrap(out) for out in outs) if is_wrapped else tuple(outs)

    e3nn_extract.Extract.forward = _extract_forward
    e3nn_extract.Extract._chemtrain_mace_compat = True



import mace.modules.irreps_tools as irreps_tools
if getattr(irreps_tools, "_chemtrain_mace_compat_mask", True):
    def new_mask_head(x: torch.Tensor, head: torch.Tensor, num_heads: int) -> torch.Tensor:
        mask = x.new_zeros((x.shape[0], x.shape[1] // num_heads, num_heads))
        idx = torch.arange(mask.shape[0])
        mask[idx, :, head] = 1
        mask = mask.permute(0, 2, 1).reshape(*x.shape)
        return x * mask
    irreps_tools.mask_head = new_mask_head
    import mace.modules.blocks as blocks
    blocks.mask_head = new_mask_head
    irreps_tools._chemtrain_mace_compat_mask = False

def load_foundational_model(family: str = "mp", version: str = "medium-0b3"):
    """Load a foundation MACE torch model and extract its conversion config."""

    torch_model = mace_jax_from_torch._load_torch_model_from_foundations(
        family, version
    )
    torch_model.eval()

    config = mace_jax_from_torch.extract_config_mace_model(torch_model)
    if "error" in config:
        raise RuntimeError(config["error"])

    return torch_model, config


def tojax_vectors_from_torch(
    config: Dict[str, Any],
    torch_model: Any,
    *,
    per_particle: bool = False,
    scale_pot: float = 96.485,
    species_mapping: SpeciesMapping = SpeciesMapping(),
    cueq_config: CuEquivarianceConfig = None,
    use_custom_batch_fn: bool = False,
    head: str | None = None,
) -> Tuple[Any, Callable]:
    """Wrap a torch MACE model as a vector-first JAX callable via toJax."""

    del cueq_config, use_custom_batch_fn

    torch_model = patch_module(deepcopy(torch_model))
    torch_model.eval()

    heads = tuple(str(h) for h in (config.get("heads") or ("Default",)))
    head_name = heads[0] if head is None else str(head)
    if head_name not in heads:
        raise ValueError(
            f"Requested head '{head_name}' not present in model heads {heads}."
        )
    head_index = heads.index(head_name)
    num_species = len(torch_model.atomic_numbers)

    def _predict(vectors, senders, receivers, species, mask):
        num_atoms = species.shape[0]

        node_attrs = torch.nn.functional.one_hot(species, num_species).to(vectors.dtype)
        node_attrs = node_attrs * mask[:, None]

        data = {
            # "vectors": vectors,
            "node_attrs": node_attrs,
            "node_attrs_index": species,
            "species": species,
            "edge_index": torch.stack((senders, receivers), dim=0),
            "batch": torch.zeros((num_atoms,), dtype=torch.int32),
            "natoms": num_atoms,
            "ptr": torch.tensor([0, num_atoms], dtype=torch.int32),
            "positions": torch.zeros((num_atoms, 3), dtype=vectors.dtype),
            "unit_shifts": torch.zeros_like(vectors),
            "shifts": vectors, # torch.zeros_like(vectors),
            "cell": torch.zeros((1, 3, 3), dtype=vectors.dtype),
            "head": torch.tensor([head_index], dtype=torch.int32),
            "lammps_class": None,
        }

        out = torch_model(
            data,
            compute_force=False,
            compute_stress=False,
            compute_displacement=False,
            lammps_mliap=False,
        )
        return out["node_energy"] * mask

    _apply_fn = jax.jit(tojax(_predict))

    def apply_fn(
        params: Any,
        vectors: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        species: jax.Array,
        mask: jax.Array | None = None,
    ):
        del params

        if isinstance(species_mapping, AtomicNumberMapping) and not isinstance(
            species, jcore.Tracer
        ):
            atomic_numbers = np.asarray(config["atomic_numbers"], dtype=np.int32)
            species_np = np.asarray(species, dtype=np.int32)
            valid = np.isin(species_np, atomic_numbers)
            if not np.all(valid):
                invalid = sorted({int(value) for value in species_np[~valid]})
                raise ValueError(
                    "Species contains atomic numbers not supported by the MACE model: "
                    f"{invalid}. Supported atomic numbers are "
                    f"{atomic_numbers.tolist()}."
                )

        if mask is None:
            mask = jnp.ones(species.shape[0], dtype=jnp.bool_)

        mapped_species = species_mapping(species, config)
        per_atom_energies = _apply_fn(
            vectors,
            senders,
            receivers,
            mapped_species,
            mask,
        )
        per_atom_energies *= scale_pot

        if per_particle:
            return per_atom_energies
        return jnp.sum(per_atom_energies)

    return None, apply_fn


def tojax_neighborlist_from_torch(
    config: Dict[str, Any],
    torch_model: Any,
    displacement: space.DisplacementFn,
    max_edge_multiplier: float = 1.25,
    per_particle: bool = False,
    scale_pos: float = 0.1,
    scale_pot: float = 96.485,
    species_mapping: SpeciesMapping = SpeciesMapping(),
    cueq_config: CuEquivarianceConfig = None,
    use_custom_batch_fn: bool = False,
    head: str | None = None,
) -> Tuple[Any, Callable]:
    """Compose a torch MACE checkpoint with a chemtrain neighborlist frontend."""

    variables, apply_fn = tojax_vectors_from_torch(
        config,
        torch_model,
        per_particle=per_particle,
        scale_pot=scale_pot,
        species_mapping=species_mapping,
        cueq_config=cueq_config,
        use_custom_batch_fn=use_custom_batch_fn,
        head=head,
    )

    r_cutoff = jnp.asarray(config["r_max"], dtype=jnp.float32) * scale_pos
    edges_per_particle = float(config["avg_num_neighbors"]) * float(max_edge_multiplier)

    def apply_neighbor_fn(
        params: Any,
        position: jax.Array,
        neighbor: partition.NeighborList,
        species: jax.Array = None,
        mask: jax.Array = None,
        **dynamic_kwargs,
    ):
        assert species is not None, "Species must be provided."
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        vectors, senders, receivers = custom_partition.readout_vectors(
            displacement,
            r_cutoff,
            position,
            neighbor,
            species,
            mask,
            edges_per_particle=edges_per_particle,
            sort=True,
            **dynamic_kwargs,
        )

        vectors /= scale_pos
        return apply_fn(
            params,
            vectors,
            senders,
            receivers,
            species,
            mask=mask,
        )

    return variables, apply_neighbor_fn


mace_jax_neighborlist_from_torch = tojax_neighborlist_from_torch


__all__ = [
    "AtomicNumberMapping",
    "SpeciesMapping",
    "load_foundational_model",
    "mace_jax_neighborlist_from_torch",
    "tojax_neighborlist_from_torch",
    "tojax_vectors_from_torch",
]
