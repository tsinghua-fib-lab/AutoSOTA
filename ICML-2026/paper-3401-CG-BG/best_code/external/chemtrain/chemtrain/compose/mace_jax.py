# MIT License
#
# Copyright (c) 2022 mace-jax
# Copyright (c) 2026 Multiscale Modeling of Fluid Materials, TU Munich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Loads a MACE model from PyTorch via MACE-JAX."""

import functools

from typing import Dict, Any, Tuple, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from e3nn_jax import Irreps

from jax_md_mod import custom_partition
from jax_md import space, partition

from mace_jax.modules.wrapper_ops import (
    CuEquivarianceConfig,
    EquivarianceConfig,
    resolve_equivariance_config,
)
from mace_jax.modules import models as mace_jax_models
from mace_jax.nnx_config import ConfigDict, ConfigVar

from mace_jax.cli import mace_jax_from_torch
import torch

from . import utils

from mace_jax.tools.model_builder import (
    _as_irreps,
    _build_jax_model,
    _prepare_template_data,
)


def _variable_value(value):
    """Read an NNX variable value across Flax versions."""
    get_value = getattr(value, "get_value", None)
    if callable(get_value):
        return get_value()
    return value.value


def _extract_nnx_value(value):
    """Convert NNX variables to JAX pytrees while preserving config semantics."""
    if isinstance(value, ConfigVar):
        raw = _variable_value(value)
        # Dict-valued ConfigVars are model configuration, not nested state.
        # ConfigDict keeps them as a single registered pytree object.
        if isinstance(raw, dict) and not isinstance(raw, ConfigDict):
            return ConfigDict(raw)
        return raw
    if isinstance(value, nnx.Variable):
        return _variable_value(value)
    return value


def _state_to_legacy_variables(state):
    """Map MACE-JAX's NNX State back to Chemtrain's old variables dict.

    Chemtrain training code expects trainable weights under variables["params"].
    Newer MACE-JAX returns one flat NNX State, so split out Param leaves and
    keep all other state/config entries visible at the top level as before.
    The Param subtree itself is also reshaped to the historical MACE-JAX
    grouping used by saved Chemtrain pickles, e.g. interactions_0 instead of
    interactions[0].
    """
    if not isinstance(state, nnx.State):
        return state

    params_state, nonparam_state = nnx.split_state(state, nnx.Param, ...)
    params = _nnx_params_to_legacy_params(
        nnx.to_pure_dict(params_state, extract_fn=_extract_nnx_value)
    )
    nonparams = nnx.to_pure_dict(nonparam_state, extract_fn=_extract_nnx_value)

    if "params" in nonparams:
        raise ValueError("MACE-JAX non-param state contains reserved key 'params'.")

    return {"params": params, **nonparams}


def _nnx_params_to_legacy_params(params):
    """Expose NNX module-list params like old Chemtrain MACE-JAX params."""
    if not isinstance(params, dict):
        return params

    legacy = {
        key: value
        for key, value in params.items()
        if key not in ("interactions", "products", "readouts")
    }

    interactions = params.get("interactions")
    if isinstance(interactions, dict):
        interaction_indices = sorted(idx for idx in interactions if isinstance(idx, int))
        legacy["interactions"] = {
            str(idx): {"conv_tp_weights": {}}
            for idx in interaction_indices
        }
        for idx in interaction_indices:
            value = interactions[idx]
            legacy[f"interactions_{idx}"] = _legacy_layer_names(value)

    for group_name in ("products", "readouts"):
        group = params.get(group_name)
        if isinstance(group, dict):
            for idx in sorted(idx for idx in group if isinstance(idx, int)):
                value = group[idx]
                legacy[f"{group_name}_{idx}"] = value

    return legacy


def _legacy_params_to_nnx_params(params):
    """Accept old saved MACE-JAX params and rebuild the NNX apply tree."""
    if not isinstance(params, dict):
        return params

    nnx_params = {
        key: value
        for key, value in params.items()
        if not (
            key == "interactions"
            or key.startswith("interactions_")
            or key.startswith("products_")
            or key.startswith("readouts_")
        )
    }

    interactions = _collect_legacy_group(params, "interactions", _nnx_layer_names)
    if interactions:
        nnx_params["interactions"] = interactions

    for group_name in ("products", "readouts"):
        group = _collect_legacy_group(params, group_name)
        if group:
            nnx_params[group_name] = group

    return nnx_params


def _collect_legacy_group(params, prefix, value_fn=lambda value: value):
    group = {}
    marker = f"{prefix}_"
    for key, value in params.items():
        if not isinstance(key, str) or not key.startswith(marker):
            continue
        index = key[len(marker):]
        if index.isdigit():
            group[int(index)] = value_fn(value)
    return group


def _legacy_layer_names(tree):
    """Convert NNX MLP layer indices to old layer0/layer1 names."""
    return _rename_layer_container(tree, from_key="layers", to_key="layer")


def _nnx_layer_names(tree):
    """Convert old layer0/layer1 names back to NNX's layers[index] form."""
    return _rename_layer_container(tree, from_key="layer", to_key="layers")


def _rename_layer_container(tree, *, from_key, to_key):
    if not isinstance(tree, dict):
        return tree
    converted = dict(tree)
    conv_weights = converted.get("conv_tp_weights")
    if not isinstance(conv_weights, dict):
        return converted

    conv_weights = dict(conv_weights)
    if from_key == "layers":
        layers = conv_weights.pop("layers", None)
        if isinstance(layers, dict):
            for idx, value in layers.items():
                conv_weights[f"{to_key}{idx}"] = value
    else:
        layers = {}
        for key in list(conv_weights):
            if isinstance(key, str) and key.startswith(from_key):
                index = key[len(from_key):]
                if index.isdigit():
                    layers[int(index)] = conv_weights.pop(key)
        if layers:
            conv_weights[to_key] = layers

    converted["conv_tp_weights"] = conv_weights
    return converted



def load_foundational_model(family: str = "mp", version: str = "medium-0b3"):
    """Loads a foundational MACE model from PyTorch.

    Args:
        family: Model family to load, e.g., "mp" or "off".
        version: Model version to load, e.g., "medium-0b3".

    Returns:
        A tuple of the loaded PyTorch model and its configuration dictionary.

    """

    torch_model = mace_jax_from_torch._load_torch_model_from_foundations(
        family, version
    )
    torch_model.eval()

    config = mace_jax_from_torch.extract_config_mace_model(torch_model)
    if "error" in config:
        raise RuntimeError(config["error"])

    return torch_model, config


def load_torch_model(model_file: str):
    """Loads a foundational MACE model from PyTorch.

    Args:
        model_file: Filename / path of PyTorch model to load.

    Returns:
        A tuple of the loaded PyTorch model and its configuration dictionary.

    """

    torch_model = torch.load(model_file)
    torch_model.eval()

    config = mace_jax_from_torch.extract_config_mace_model(torch_model)
    if "error" in config:
        raise RuntimeError(config["error"])

    return torch_model, config


class SpeciesMapping:
    """Identity mapping for species."""

    def __call__(self, species: jnp.ndarray, config: Dict) -> jnp.ndarray:
        del config  # Unused

        return species


class AtomicNumberMapping(SpeciesMapping):
    """Maps atomic numbers to MACE-JAX species."""

    def __init__(self, max_number: int = 100):
        self.max_number = max_number

    def __call__(self, species: jnp.ndarray, config: Dict) -> jnp.ndarray:

        atomic_numbers = jnp.asarray(config["atomic_numbers"], dtype=jnp.int32)

        # Create lookup table, mapping from atomic number to index
        lookup_table = jnp.argmax(
            jnp.arange(self.max_number)[:, None] + 1 == atomic_numbers[None, :], axis=-1
        )

        return lookup_table[species - 1]


def mace_jax_neighborlist_from_torch(
    config: Dict[str, Any],
    torch_model: Any,
    displacement: space.DisplacementFn,
    max_edge_multiplier: None | float = 1.25,
    per_particle: bool = False,
    scale_pos: float = 0.1,
    scale_pot: float = 96.485,
    species_mapping: SpeciesMapping = SpeciesMapping(),
    equivariance_config: EquivarianceConfig = None,
    cueq_config: CuEquivarianceConfig = None,
    use_custom_batch_fn: bool = False,
    comm: Any = None,
) -> Tuple[Any, Callable]:
    """MACE model for property prediction.

    Args:
        config: Configuration dictionary for the MACE model.
        torch_model: The PyTorch MACE model to convert.
        displacement: Jax_md displacement function
        max_edge_multiplier: Multiplier to limit the maximum number of
            edges per particle.
        per_particle: Return per-particle energies instead of total energy.
        scale_pos: Scaling factor for positions, i.e., to convert units.
        scale_pot: Scaling factor for potentials, i.e., to convert units.
        species_mapping: Mapping for species to model-compatible indices.
        equivariance_config: Backend-neutral equivariance configuration.
        cueq_config: Deprecated alias for a CuEquivariance-only configuration.
        use_custom_batch_fn: Whether to use custom batch function.
            Required when cueq_config is enabled, optional otherwise.
        comm: Optional deployment communication interface. It must provide a
            ``gather`` method used between message-passing blocks.

    Returns:
        Returns a tuple of parameters and an apply function.

    """

    equivariance_config = resolve_equivariance_config(
        equivariance_config, cueq_config=cueq_config
    )
    jax_model, state, template_data = mace_jax_from_torch.convert_model(
        torch_model, config, equivariance_config=equivariance_config
    )
    uses_nnx_state = isinstance(state, nnx.State)
    variables = _state_to_legacy_variables(state)

    cueq = (
        None if equivariance_config is None else equivariance_config.cueq_config
    )
    cueq_enabled = False if cueq is None else cueq.enabled

    del template_data  # Unused

    r_cutoff = jnp.array(config["r_max"], dtype=jnp.float32) * scale_pos
    edges_per_particle = (
        float(config["avg_num_neighbors"]) * float(max_edge_multiplier)
        if max_edge_multiplier is not None
        else None
    )

    default_comm = comm

    def _apply_fn(
        params, senders, receivers, edge_feats, node_feats, *, state=None, comm=None
    ):
        (vectors,) = edge_feats
        species, mask = node_feats

        data = {
            "edge_index": jnp.stack([senders, receivers], axis=0),
            "node_attrs": jax.nn.one_hot(
                species,
                num_classes=config["num_elements"],
                dtype=vectors.dtype,
            ) * mask[:, None],
            "node_attrs_index": species,
            "positions": jnp.zeros((species.shape[0], 3)),  # Unused, but some models require it
            "cell": jnp.eye(3)[None, :, :],  # Unused, but some models require it
            "shifts": vectors, # Hack to extract vectors by us
            "ptr": jnp.asarray((0, species.shape[0]), dtype=jnp.int32),  # Unused, but some models require it
            "num_species": config["num_elements"],
            "batch": jnp.zeros(species.shape, dtype=jnp.int32),  # Unused, but some models require it
            "unit_shifts": jnp.zeros((vectors.shape[0], 3)),  # Unused, but some models require it
        }

        if state is None:
            out, _ = jax_model.apply(params)(
                data, compute_force=False, compute_stress=False, comm=comm)
        else:
            out, _ = jax_model.apply(params, state)(
                data, compute_force=False, compute_stress=False, comm=comm)
        return out["node_energy"] * mask

    use_batched_apply = cueq_enabled or use_custom_batch_fn

    def apply_fn(
        params: Any,
        position: jax.Array,
        neighbor: partition.NeighborList,
        species: jax.Array = None,
        mask: jax.Array = None,
        comm: Any = default_comm,
        **dynamic_kwargs,
    ):
        assert species is not None, "Species must be provided."
        species = species_mapping(species, config)

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

        if uses_nnx_state and isinstance(params, dict) and "params" in params:
            # Keep Chemtrain's public variables["params"] contract, but pass
            # params and non-param config separately to Flax's public
            # GraphDef.apply(state, *states) merge path.
            model_params = _legacy_params_to_nnx_params(params["params"])
            model_state = {
                key: value for key, value in params.items() if key != "params"
            } or None
        else:
            model_params, model_state = params, None
        model_args = (model_params, senders, receivers, (vectors,), (species, mask))
        if use_batched_apply and comm is None:
            # Construct the batching transform only for the ordinary model;
            # ``comm`` is a static JIT property, so this branch is specialized
            # independently from the communicating deployment variant.
            batched_apply = utils.batch_apply_fn(
                functools.partial(_apply_fn, state=model_state, comm=comm)
            )
            per_atom_energies = batched_apply(*model_args)
        else:
            # Deployment passes one already flattened graph, so feature
            # communication does not need the custom batching transform. Its
            # backward rule intentionally recomputes the model from its inputs;
            # bypassing it lets JAX retain the communicated primal features and
            # avoids a second forward halo exchange before the reverse one.
            per_atom_energies = _apply_fn(*model_args, state=model_state, comm=comm)
        per_atom_energies *= scale_pot

        if per_particle:
            return per_atom_energies
        else:
            return jnp.sum(per_atom_energies)

    return variables, jax.jit(
        apply_fn, static_argnames=("comm",)
    )


def mace_jax_neighborlist(
    *,
    displacement: space.DisplacementFn,
    r_cutoff: float,
    n_species: int = 100,
    positions_test: jnp.ndarray = None,
    neighbor_test: partition.NeighborList = None,
    max_edge_multiplier: float = 1.25,
    edges_per_particle: float = None,
    avg_num_neighbors: float = None,
    mode: str = "energy",
    per_particle: bool = False,
    equivariance_config: EquivarianceConfig = None,
    cueq_config: CuEquivarianceConfig = None,
    use_custom_batch_fn: bool = False,
    mace_config: Dict[str, Any] = None,
) -> Tuple[Any, Callable, Dict[str, Any]]:
    """MACE model for property prediction.
    We use the same initialization as in our chemutils implementation and convert it to the
    MACE JAX format.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        mace_config: Kwargs to change the default structure of MACE.
            For definition of the kwargs, see MACE.

    Returns:
        Returns a tuple of parameters and an apply function.

    """
    equivariance_config = resolve_equivariance_config(
        equivariance_config, cueq_config=cueq_config
    )
    species_mapping = AtomicNumberMapping(n_species)

    # Keys based on MACE JAX
    default_mace_config = {
        "atomic_numbers": jnp.arange(n_species),
        "r_max": r_cutoff,
        "num_interactions": 2,
        "hidden_irreps": "32x0e + 32x1o",
        "max_ell": 3,
        "num_species": n_species,
        "atomic_energies": jnp.zeros(n_species),
        "avg_num_neighbors": avg_num_neighbors,
        "MLP_irreps": "16x0e",
        "correlation": 3,
        "interaction_cls": "RealAgnosticInteractionBlock",
        "interaction_cls_first": "RealAgnosticInteractionBlock",
        "radial_type": "bessel",
        "pair_repulsion": False,
        "use_so3": False,
        "num_polynomial_cutoff": 6,
        "num_bessel": 8,
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
    }
    config = (
        default_mace_config | mace_config
    )  # Overwrite defaults with any values present in mace_config

    try:
        jax_model = _build_jax_model(
            config,
            equivariance_config=equivariance_config,
            init_normalize2mom_consts=False,
        )
    except TypeError as exc:
        if "equivariance_config" in str(exc):
            jax_model = _build_jax_model(
                config,
                init_normalize2mom_consts=False,
            )
        else:
            raise

    # Prepare template data and initialize JAX model parameters
    template_data = _prepare_template_data(config)
    template_vars = jax_model.init(jax.random.PRNGKey(0), template_data)
    del template_data  # Unused

    cueq = (
        None if equivariance_config is None else equivariance_config.cueq_config
    )
    cueq_enabled = False if cueq is None else cueq.enabled

    # We need a different __call__ method
    jax_model.__class__ = JaxMACE

    edges_per_particle = float(config["avg_num_neighbors"]) * float(max_edge_multiplier)

    def _apply_fn(params, senders, receivers, edge_feats, node_feats):
        (vectors,) = edge_feats
        species, mask = node_feats

        return jax_model.apply(
            params,
            vectors,
            senders,
            receivers,
            species,
            mask,
            num_species=config["num_species"],
        )

    # Apply custom batch function if enabled via cueq or explicitly requested
    if cueq_enabled or use_custom_batch_fn:
        _apply_fn = utils.batch_apply_fn(_apply_fn)

    def apply_fn(
        params: Any,
        position: jax.Array,
        neighbor: partition.NeighborList,
        species: jax.Array = None,
        mask: jax.Array = None,
        **dynamic_kwargs,
    ):
        assert species is not None, "Species must be provided."
        species = species_mapping(species, config)

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

        per_atom_energies = _apply_fn(
            params, senders, receivers, (vectors,), (species, mask)
        )

        if per_particle:
            return per_atom_energies
        else:
            return jnp.sum(per_atom_energies)

    return jax.tree.map(jnp.asarray, template_vars), jax.jit(apply_fn), config
