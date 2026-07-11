import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, jit
from functools import partial
import operator
from ntkunlimited.empirical.statistics import StatsRecorder, ArrayShape
from math import ceil
from pathlib import Path
from math import sqrt
import random
import click
from tqdm import tqdm
from ntkunlimited.nn import create_network, get_layer_widths, ParameterizationSetup
from dataclasses import dataclass
from typing import Protocol, Callable

import ntkunlimited.utils as utils
import json

# jax.config.update("jax_enable_x64", True)

is_x64 = jax.config.read("jax_enable_x64")
prec = "x64_" if is_x64 else ""


@dataclass
class StabilityRunConfig:
    """Configuration for tensor stability estimation runs."""

    layers: list
    key: jax.Array
    x1: jax.Array
    input_ind: jax.Array
    parameterization_setup: ParameterizationSetup
    output_dir: Path
    run_meta: dict
    n_samples: int
    n_samples_stats: int
    batch_size: int


def setup_stability_run(
    n_layers: int,
    layer_width: int,
    nonlin: str,
    c_w: float,
    c_b: float,
    param_seed: int,
    data_seed: int,
    data_size: int,
    input_dim: int,
    input_file: str | None,
    input_rescale: float,
    n_samples: int,
    n_samples_stats: int,
    batch_size: int,
    output_dir: str,
    input_ind: list[list[int]],
) -> StabilityRunConfig:
    """Set up common configuration for stability estimation runs."""
    output_path = Path(output_dir) / "tensors"
    output_path.mkdir(parents=True, exist_ok=True)

    parameterization_setup = ParameterizationSetup("standard", c_w, c_b)
    layers = create_network(n_layers, layer_width, nonlin, parameterization_setup)
    key = jax.random.key(param_seed)

    if input_file is not None:
        x1 = load_input_from_file(input_file, input_rescale, data_size)
    else:
        x1 = input_rescale * jax.random.normal(
            jax.random.key(data_seed), (data_size, input_dim)
        )

    input_ind_arr = jnp.array(input_ind)

    run_meta = {
        "n_layers": n_layers,
        "layer_width": layer_width,
        "nonlin": nonlin,
        "data_size": data_size,
        "input_ind": [tuple(x) for x in input_ind],
        "c_w": c_w,
        "c_b": c_b,
        "data_seed": data_seed if input_file is None else None,
        "x1": x1.tolist(),
        "input_rescale": input_rescale,
        "input_hash": utils.hash_array(np.array(x1)),
        "n_samples": n_samples,
        "parameterization": parameterization_setup.parameterization,
    }

    return StabilityRunConfig(
        layers=layers,
        key=key,
        x1=x1,
        input_ind=input_ind_arr,
        parameterization_setup=parameterization_setup,
        output_dir=output_path,
        run_meta=run_meta,
        n_samples=n_samples,
        n_samples_stats=n_samples_stats,
        batch_size=batch_size,
    )


class TensorEstimator(Protocol):
    """Protocol for tensor stability estimators."""

    @property
    def name(self) -> str:
        """Name of the tensor being estimated."""
        ...

    def get_shapes(self, n_layers: int, n_input_inds: int) -> list[ArrayShape]:
        """Return the shapes of the estimated tensors for each layer."""
        ...

    def estimate(
        self,
        config: StabilityRunConfig,
        key: jax.Array,
    ) -> list:
        """Estimate the tensor values for all layers."""
        ...


def run_stability_estimation(
    config: StabilityRunConfig,
    estimator: TensorEstimator,
) -> StatsRecorder:
    """Run stability estimation for a single tensor estimator."""
    n_layers = len(config.layers)
    n_input_inds = len(config.input_ind)
    shapes = estimator.get_shapes(n_layers, n_input_inds)
    stats = StatsRecorder.clean_init(shapes, batch_dim=-1)

    key, *subkeys = jax.random.split(config.key, config.n_samples_stats + 1)
    for subkey in tqdm(subkeys, desc=f"{estimator.name} stats"):
        estimate = estimator.estimate(config, subkey)
        estimate = jax.tree.map(partial(jnp.expand_dims, axis=-1), estimate)
        stats.update(estimate)

    return stats


def save_tensor_stats(
    config: StabilityRunConfig,
    tensor_name: str,
    stats: StatsRecorder,
    suffix: str = "",
) -> Path:
    """Save tensor statistics to a file."""
    tensor_deps = [
        "nonlin",
        "n_layers",
        "layer_width",
        "c_w",
        "c_b",
        "input_hash",
        "parameterization",
    ]

    if suffix:
        suffix = f"_{suffix}"

    filename = (
        f"{tensor_name}_{prec}stats_"
        f"{utils.create_identifier_string(config.run_meta, tensor_deps)}{suffix}.json"
    )
    filepath = config.output_dir / filename
    stats.save_to_file(filepath, config.run_meta)
    return filepath


class V4Estimator:
    """Estimator for V4 (connected 4-point function) tensor."""

    def __init__(self):
        self.names = ["V4"]

    def get_shapes(
        self, n_layers: int, n_input_inds: int
    ) -> dict[str, list[ArrayShape]]:
        shapes = [ArrayShape((n_input_inds,)) for _ in range(n_layers)]
        return {"V4": shapes}

    def estimate(
        self,
        config: StabilityRunConfig,
        key: jax.Array,
    ) -> dict[str, list]:
        layers = jax.tree.map(jax.tree_util.Partial, config.layers)
        result = est_expec_v4_all_layers(
            n_samples=config.n_samples,
            layers=layers,
            key=key,
            batchsize=config.batch_size,
            parameterization_setup=config.parameterization_setup,
            x1=config.x1,
            input_ind=config.input_ind,
        )
        return {"V4": result}


class AlphabetTensorsEstimator:
    """Estimator for alphabet tensors (D, F, A, B) computed together."""

    def __init__(self, tensor_fns_map: list[dict]):
        """
        Args:
            tensor_fns_map: List of dicts with "name" and "func" keys,
                e.g., [{"name": "D", "func": calc_D_tensor}, ...]
        """
        self.tensor_fns_map = tensor_fns_map
        self.names = [t["name"] for t in tensor_fns_map]

    def get_shapes(
        self, n_layers: int, n_input_inds: int
    ) -> dict[str, list[ArrayShape]]:
        shapes = [ArrayShape((n_input_inds,)) for _ in range(n_layers)]
        return {name: shapes for name in self.names}

    def estimate(
        self,
        config: StabilityRunConfig,
        key: jax.Array,
    ) -> dict[str, list]:
        tensor_fns = [t["func"] for t in self.tensor_fns_map]
        results = est_expec_4d_tensor_all_layers(
            n_samples=config.n_samples,
            layers=config.layers,
            key=key,
            batchsize=config.batch_size,
            x1=config.x1,
            input_ind=config.input_ind,
            tensor_fns=tensor_fns,
            parameterization_setup=config.parameterization_setup,
        )
        return {t["name"]: r for t, r in zip(self.tensor_fns_map, results)}


def run_stability_estimation(
    config: StabilityRunConfig,
    estimator: V4Estimator | AlphabetTensorsEstimator,
) -> dict[str, StatsRecorder]:
    """Run stability estimation for an estimator (single or grouped)."""
    n_layers = len(config.layers)
    n_input_inds = len(config.input_ind)
    shapes_dict = estimator.get_shapes(n_layers, n_input_inds)

    stats_dict = {
        name: StatsRecorder.clean_init(shapes, batch_dim=-1)
        for name, shapes in shapes_dict.items()
    }

    key, *subkeys = jax.random.split(config.key, config.n_samples_stats + 1)
    desc = "+".join(estimator.names)
    for subkey in tqdm(subkeys, desc=f"{desc} stats"):
        estimates = estimator.estimate(config, subkey)
        for name, estimate in estimates.items():
            estimate = jax.tree.map(partial(jnp.expand_dims, axis=-1), estimate)
            stats_dict[name].update(estimate)

    return stats_dict


@partial(vmap, in_axes=(None, 0, None), out_axes=0)
def _calc_inter_layer_derivs(apply_fn, input, params):
    def apply_fn_fixed(input):
        return apply_fn(params, input[None, ...])[0]

    inter_layer_derivs = jax.jacfwd(apply_fn_fixed)(input)
    return inter_layer_derivs


@partial(vmap, in_axes=(None, 0, None), out_axes=0)
def _calc_intra_layer_derivs(apply_fn, input, params):
    if len(params) == 0:
        return ()

    def apply_fn_fixed(params):
        return apply_fn(params, input[None, ...])[0]

    intra_layer_derivs = jax.jacfwd(apply_fn_fixed)(params)
    return intra_layer_derivs


def _flatten_jacobian(jacobian, output_shape):
    def flatten_leaf(leaf):
        if leaf is None:
            return jnp.zeros((*output_shape, 1))
        return jnp.reshape(leaf, (*output_shape, -1))

    jacobian = jax.tree.map(flatten_leaf, jacobian, is_leaf=lambda x: x is None)
    jacobian = jax.tree.leaves(jacobian)
    return jacobian


def _outer_prod_jacobian(jacobian_l, jacobian_r, params):
    if params == ():
        return ()

    def get_param_axes(leaf):
        param_axes = list(range(-1, -1 - leaf.ndim, -1))
        return param_axes

    param_axes = jax.tree.map(get_param_axes, params)

    def outer_prod_leaf(leaf_l, leaf_r, param_axes):
        summed_jac = jnp.tensordot(leaf_l, leaf_r, axes=(param_axes, param_axes))
        return jnp.transpose(summed_jac, axes=(0, 2, 1, 3))

    outer_prod = jax.tree.map(outer_prod_leaf, jacobian_l, jacobian_r, param_axes)
    return jax.tree.reduce(operator.add, outer_prod)


def _fix_normalization_conventions(
    derivs, param_shape, parameterization_setup: ParameterizationSetup
):
    """Adjusts the normalization factors such that they agree witht the conventions in the book."""
    if not parameterization_setup.convert_to_book:
        return derivs

    if param_shape is None:
        return derivs

    parameterization = parameterization_setup.parameterization
    C_w = parameterization_setup.C_w
    C_b = parameterization_setup.C_b

    lambda_b = 1.0
    lambda_w = 1.0

    if parameterization == "standard":
        if len(param_shape) == 1:
            return lambda_b * derivs

        if len(param_shape) == 2:
            # Convention of the book
            return sqrt(lambda_w / param_shape[0]) * derivs

    else:
        if len(param_shape) == 1:
            return lambda_b / sqrt(C_b) * derivs

        if len(param_shape) == 2:
            # Convention of the book, need to undo the rescaling by the prefactor C_w in the
            # derivatives
            return sqrt(lambda_w / C_w) * derivs


@partial(jit, static_argnames=("parameterization_setup",))
def calc_emp_ntk_next_layer(
    apply_fn,
    params: jax.Array,
    key: jax.Array,
    input: jax.Array,
    prev_full_ntk: jax.Array,
    parameterization_setup: ParameterizationSetup,
):
    inter_layer_derivs = _calc_inter_layer_derivs(apply_fn, input, params)
    intra_layer_derivs = _calc_intra_layer_derivs(apply_fn, input, params)

    full_ntk = jnp.einsum(
        "ali,bkj,abij->ablk", inter_layer_derivs, inter_layer_derivs, prev_full_ntk
    )
    param_shapes = utils._get_pytree_array_shapes(params)
    apply_convention = partial(
        _fix_normalization_conventions, parameterization_setup=parameterization_setup
    )
    intra_layer_derivs = jax.tree.map(
        apply_convention, intra_layer_derivs, param_shapes
    )
    outer_prod = _outer_prod_jacobian(intra_layer_derivs, intra_layer_derivs, params)
    if not (isinstance(outer_prod, tuple) and len(outer_prod) == 0):
        full_ntk += outer_prod

    return full_ntk


@partial(jit, static_argnames=("trace_avg", "parameterization_setup"))
def calc_emp_ntks_all_layers(
    layers: list,
    key: jax.Array,
    x1: jax.Array,
    x2: jax.Array | None,
    neural_pair: jax.Array | None,
    trace_avg: bool,
    parameterization_setup: ParameterizationSetup,
):
    """
    Calculate the empirical NTK for all layers in a neural network.

    Each layer's NTK can be computed via the chain rule, using the previous layers full NTK. This
    is why we need to temporarily store the full NTK of the previous layer. After the new one
    is computed, the previous full NTK can be discarded. We only store the diagonal
    components of the NTK for each layer since the off-diagonal
    have an expected value of 0, and we optionally average over this diagonal.

    Args:
        layers (list): List of layers in the neural network.
        key (jax.Array): Random key for initialization.
        x1 (jax.Array): First batch of inputs.
        x2 (jax.Array | None): Second batch of inputs (optional), if None, then `x2=x1`.
    """
    all_params = []
    ntks = []

    if x2 is not None:
        raise NotImplementedError("Not yet implemented for second batch of inputs")

    n_layers = len(layers)
    key, *subkeys = jax.random.split(key, n_layers + 1)
    prev_output = x1
    full_ntk = jnp.zeros((x1.shape[0], x1.shape[0], x1.shape[1], x1.shape[1]))
    if trace_avg:
        ntks.append(jnp.zeros((x1.shape[0],) * 2))
    else:
        ntks.append(jnp.zeros((x1.shape[0],) * 2 + (x1.shape[1],)))

    for i, layer in enumerate(layers):
        output_shape, params = layer[0](subkeys[i], input_shape=prev_output.shape)
        all_params.append(params)
        apply_fn = jax.tree_util.Partial(layer[1])

        full_ntk = calc_emp_ntk_next_layer(
            apply_fn, params, subkeys[i], prev_output, full_ntk, parameterization_setup
        )
        ntk = jnp.diagonal(full_ntk, axis1=-2, axis2=-1)

        if trace_avg:
            ntk = jnp.mean(ntk, axis=-1)

        ntks.append(ntk)
        prev_output = layer[1](params, prev_output)

    return ntks, all_params


def sample_layer(init_fn, apply_fn, key, input):
    output_shape, params = init_fn(key, input_shape=input.shape)
    layer_output = apply_fn(params, input)
    return layer_output, params


@partial(jit, static_argnames=("trace_avg",))
def calc_emp_nngps_all_layers(
    layers: list,
    key: jax.Array,
    x1: jax.Array,
    x2: jax.Array | None,
    neural_pair: jax.Array | None,
    trace_avg: bool,
):
    """
    Calculate the empirical NNGP for all layers in a neural network.

    The NNGP is computed for each layer by taking the outer product of the layer's outputs. This function
    iteratively computes the NNGP for each layer, starting from the input layer and propagating through
    the network. Only the diagonal components of the neural indices of the NNGP are kept. The function
    also collects the parameters of all layers during the computation.

    Args:
        layers (list): List of layers in the neural network. Each layer is a tuple containing
                       (init_fn, apply_fn, other_fn).
        key (jax.Array): Random key for initialization.
        x1 (jax.Array): First batch of inputs.
        x2 (jax.Array | None): Second batch of inputs (optional). If None, then `x2=x1`.
        neural_pairs (jax.Array | None): Optional pairs of neurons to compute the NNGP for. Shape:
            (2,).

    Returns:
        tuple: A tuple containing:
            - nngps (list): List of NNGP matrices for each layer.
            - all_params (list): List of parameters for each layer.
    """
    all_params = []
    nngps = []

    if x2 is not None:
        raise NotImplementedError("Not yet implemented for second batch of inputs")

    n_layers = len(layers)
    key, *subkeys = jax.random.split(key, n_layers + 1)
    layer_output = x1

    if neural_pair is None:
        nngps.append(jnp.einsum("ai,bi->abi", layer_output, layer_output))
        if trace_avg:
            nngps[-1] = jnp.mean(nngps[-1], axis=-1)
    else:
        nngps.append(
            jnp.outer(layer_output[:, neural_pair[0]], layer_output[:, neural_pair[1]])
        )

    for i, layer in enumerate(layers):
        layer_output, params = sample_layer(
            layer[0], layer[1], subkeys[i], layer_output
        )
        all_params.append(params)
        if neural_pair is None:
            nngp = jnp.einsum("ai,bi->abi", layer_output, layer_output)
            if trace_avg:
                nngp = jnp.mean(nngp, axis=-1)
            nngps.append(nngp)
        else:
            nngps.append(
                jnp.outer(
                    layer_output[:, neural_pair[0]], layer_output[:, neural_pair[1]]
                )
            )

    return nngps, all_params


def est_expec_kernel_stats_all_layers(
    kernel: str,
    n_samples: int,
    layers: list,
    key: jax.Array,
    batchsize: int,
    parameterization_setup: ParameterizationSetup,
    x1: jax.Array,
    x2: jax.Array | None = None,
    neural_pair: jax.Array | None = None,
    trace_avg: bool = False,
    kn_stats: StatsRecorder | None = None,
) -> StatsRecorder:
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, n_samples)
    if kernel == "ntk":
        emp_kn_fn = partial(
            calc_emp_ntks_all_layers, parameterization_setup=parameterization_setup
        )
    else:
        emp_kn_fn = calc_emp_nngps_all_layers
    batched_calc_emp_kernel_all_layers = vmap(
        emp_kn_fn, in_axes=(None, 0, None, None, None, None), out_axes=-1
    )
    layers = jax.tree.map(jax.tree_util.Partial, layers)

    if kn_stats is None:
        dummy_ensemble = batched_calc_emp_kernel_all_layers(
            layers, jnp.expand_dims(subkeys[0], 0), x1, None, neural_pair, trace_avg
        )
        kn_shapes = utils._get_pytree_array_shapes(dummy_ensemble[0])
        kn_stats = StatsRecorder.clean_init(kn_shapes, batch_dim=-1)

    for i in tqdm(
        range(ceil(n_samples / batchsize)),
        desc="Batched kernel stats computation",
        leave=False,
    ):
        batch_lower = i * batchsize
        batch_upper = min((i + 1) * batchsize, n_samples)
        kn_ensembles, _ = batched_calc_emp_kernel_all_layers(
            layers, subkeys[batch_lower:batch_upper], x1, None, neural_pair, trace_avg
        )
        kn_stats.update(kn_ensembles)
    return kn_stats


def partition_into_pairs(values: list, neural_inds: list):
    if len(values) == 2:
        return [tuple(values)]
    if len(values) < 2:
        raise ValueError("Single value cannot be partitioned into pairs")
    pairs = []
    for i, v in enumerate(values[1:]):
        if neural_inds[0] != neural_inds[i + 1]:
            continue
        first_pair = (values[0], v)
        leftover_v = values[1:]
        leftover_v.pop(i)
        leftover_n = neural_inds[1:]
        leftover_n.pop(i)
        left_pairs = partition_into_pairs(leftover_v, leftover_n)

        for pair in left_pairs:
            if isinstance(pair, tuple):
                pairs.append([first_pair, pair])
            else:
                pairs.append([first_pair] + list(pair))

    return pairs


def wick_contract(input_ind: list, corr_2pts: jax.Array, neural_pair: jax.Array):
    neural_inds = jnp.repeat(neural_pair, 2)
    partition = partition_into_pairs(input_ind, neural_inds.tolist())
    contraction = 0.0
    for part in partition:
        term = 1.0
        for pair in part:
            term *= corr_2pts[*pair]
        contraction += term
    return contraction


def wick_contract_batched_all_layers(
    layers: list,
    key: jax.Array,
    x1: jax.Array,
    neural_pair: jax.Array,
    n_samples: int,
    batchsize: int,
    input_inds: list,
    parameterization_setup: ParameterizationSetup,
):
    n_inputs = x1.shape[0]
    if n_inputs % 2 != 0:
        raise ValueError("Number of inputs must be even for Wick contraction")

    # shape (input_dim, input_dim, selected_neurals, selected_neurals)
    corr_2pts = est_expec_kernel_stats_all_layers(
        "nngp",
        n_samples,
        layers,
        key,
        batchsize,
        parameterization_setup,
        x1,
        x2=None,
        neural_pair=None,  # Use the fact that the expectation along the diagonal is the same
        trace_avg=True,
        kn_stats=None,
    ).mean[1:]

    def batched_wc(corr_2pts_leaf):
        wcs = []
        for input_ind in input_inds:
            # here, neural_pair is non-trivial because we want to isolate V in the channel (i, i, j,
            # j)
            wcs.append(wick_contract(input_ind, corr_2pts_leaf, neural_pair))
        return jnp.stack(wcs, axis=0)

    wick_cons = jax.tree.map(batched_wc, corr_2pts)

    return wick_cons


@jit
@partial(vmap, in_axes=(None, 0, 0), out_axes=0)
def calc_4pt_conn(layer_output: jax.Array, input_ind: jax.Array, wick_cont: jax.Array):
    first_pair = layer_output[input_ind[0], :] * layer_output[input_ind[1], :]
    second_pair = layer_output[input_ind[2], :] * layer_output[input_ind[3], :]
    # Subtract the diagonal
    pt4_conn = jnp.outer(first_pair, second_pair) - wick_cont
    pt4_conn = jnp.fill_diagonal(pt4_conn, 0.0, inplace=False)
    pt4_conn = jnp.sum(pt4_conn)
    n_out = layer_output.shape[1]
    pt4_conn *= 1 / (n_out * (n_out - 1))
    return pt4_conn


def calc_emp_4pt_conn_all_layers(
    layers: list,
    key: jax.Array,
    x1: jax.Array,
    input_ind: jax.Array,
    wick_cont: jax.Array,
) -> list:
    # Store one diagonal (i,i,j,j) defined by neural_pair
    pt4s_conn = []
    input_shape = x1.shape
    if input_shape[0] < 4:
        raise ValueError("Data set size must be at least 4 to compute V4 statistics")

    n_layers = len(layers)
    key, *subkeys = jax.random.split(key, n_layers + 1)
    layer_output = x1

    for i, layer in enumerate(layers):
        layer_output, params = sample_layer(
            layer[0], layer[1], subkeys[i], layer_output
        )
        pt4s_conn.append(calc_4pt_conn(layer_output, input_ind, wick_cont[i]))

    return pt4s_conn


def est_expec_v4_all_layers(
    n_samples: int | None,
    layers: list,
    key: jax.Array,
    batchsize: int,
    parameterization_setup: ParameterizationSetup,
    x1: jax.Array,
    input_ind: jax.Array,
):
    key, subkey1, subkey2 = jax.random.split(key, 3)
    batched_calc_emp_4pt_conn_all_layers = jit(
        vmap(
            calc_emp_4pt_conn_all_layers,
            in_axes=(None, 0, None, None, None),
            out_axes=-1,
        )
    )

    v_channel = jnp.array([0, 1])
    wick_cont = wick_contract_batched_all_layers(
        layers,
        subkey2,
        x1,
        v_channel,
        n_samples,
        batchsize,
        input_ind.tolist(),
        parameterization_setup,
    )

    dummy_ensemble = batched_calc_emp_4pt_conn_all_layers(
        layers, jnp.expand_dims(subkey1, 0), x1, input_ind, wick_cont
    )
    v4_shapes = utils._get_pytree_array_shapes(dummy_ensemble)
    pt4_conn_stats = StatsRecorder.clean_init(v4_shapes, batch_dim=-1)

    if n_samples is None:
        return pt4_conn_stats.mean

    subkeys = jax.random.split(subkey1, n_samples)

    for i in tqdm(
        range(ceil(n_samples / batchsize)),
        desc="Batched V4 stats computation",
        leave=False,
    ):
        batch_lower = i * batchsize
        batch_upper = min((i + 1) * batchsize, n_samples)
        pt4_conn_ensembles = batched_calc_emp_4pt_conn_all_layers(
            layers, subkeys[batch_lower:batch_upper], x1, input_ind, wick_cont
        )
        pt4_conn_stats.update(pt4_conn_ensembles)

    layer_widths = get_layer_widths([layer[0] for layer in layers], x1.shape)
    v4 = jax.tree.map(lambda x, y: x * y, layer_widths[:-1], pt4_conn_stats.mean)
    return v4


def load_input_from_file(
    input_file: str, input_rescale: float, data_size: int
) -> jax.Array:
    with open(input_file, "r") as f:
        x1 = jnp.array(json.load(f)["input"])[:data_size, :data_size]
    return input_rescale * x1


@jit
@partial(vmap, in_axes=(None, None, 0, None), out_axes=0)
def calc_D_tensor(
    layer_output: jax.Array, ntk_dev: jax.Array, input_ind: jax.Array, n_prev: int
):
    D = jnp.einsum(
        "i,i,jj->",
        layer_output[input_ind[0], :],
        layer_output[input_ind[1], :],
        ntk_dev[input_ind[2], input_ind[3], :, :],
    )
    # D -= jnp.einsum('i,i,ii->', layer_output[input_ind[0], :], layer_output[input_ind[1], :],
    #                 ntk_dev[input_ind[2], input_ind[3], :, :])

    # n_out = layer_output.shape[1]
    # D *= n_prev / (n_out * (n_out - 1))
    D *= n_prev / layer_output.shape[1] ** 2
    return D


@jit
@partial(vmap, in_axes=(None, None, 0, None), out_axes=0)
def calc_F_tensor(
    layer_output: jax.Array, ntk_dev: jax.Array, input_ind: jax.Array, n_prev: int
):
    F = jnp.einsum(
        "i,j,ij->",
        layer_output[input_ind[0], :],
        layer_output[input_ind[2], :],
        ntk_dev[input_ind[1], input_ind[3], :, :],
    )
    F *= n_prev / layer_output.shape[1] ** 2
    return F


@jit
@partial(vmap, in_axes=(None, None, 0, None), out_axes=0)
def calc_A_tensor(
    layer_output: jax.Array, ntk_dev: jax.Array, input_ind: jax.Array, n_prev: int
):
    A = jnp.einsum(
        "ii,jj->",
        ntk_dev[input_ind[0], input_ind[1], :, :],
        ntk_dev[input_ind[2], input_ind[3], :, :],
    )
    n_out = layer_output.shape[1]
    A *= n_prev / n_out**2
    return A


@jit
@partial(vmap, in_axes=(None, None, 0, None), out_axes=0)
def calc_B_tensor(
    layer_output: jax.Array, ntk_dev: jax.Array, input_ind: jax.Array, n_prev: int
):
    B = jnp.einsum(
        "ij,ij->",
        ntk_dev[input_ind[0], input_ind[2], :, :],
        ntk_dev[input_ind[1], input_ind[3], :, :],
    )
    B *= n_prev / layer_output.shape[1] ** 2
    return B


@jit
def add_diagonal(x: jax.Array, diag: jax.Array):
    n = x.shape[-1]
    return x.at[..., jnp.arange(n), jnp.arange(n)].add(diag)


@partial(jit, static_argnames=("parameterization_setup"))
def calc_emp_4d_tensor_all_layers(
    layers: list,
    key: jax.Array,
    x1: jax.Array,
    input_ind: jax.Array,
    tensor_fns: list,
    exp_ntk: jax.Array,
    parameterization_setup: ParameterizationSetup,
):
    # Compute a single empirical tensor using neural index symmetry
    emp_4d_tensors = [[] for _ in tensor_fns]
    input_shape = x1.shape
    if input_shape[0] < 4:
        raise ValueError(
            "Data set size must be at least 4 to compute 4d tensor statistics"
        )

    n_layers = len(layers)
    key, *subkeys = jax.random.split(key, n_layers + 1)
    layer_output = x1

    full_ntk = jnp.zeros((x1.shape[0], x1.shape[0], x1.shape[1], x1.shape[1]))
    prev_output = x1

    for i, layer in enumerate(layers):
        n_prev = layer_output.shape[1]
        layer_output, params = sample_layer(
            layer[0], layer[1], subkeys[i], layer_output
        )

        apply_fn = jax.tree_util.Partial(layer[1])
        full_ntk = calc_emp_ntk_next_layer(
            apply_fn, params, subkeys[i], prev_output, full_ntk, parameterization_setup
        )

        ntk_dev = jnp.copy(full_ntk)
        ntk_dev = add_diagonal(ntk_dev, -jnp.expand_dims(exp_ntk[i + 1], axis=-1))
        # ntk_dev_diag_avg = jnp.einsum('...ii->...', ntk_dev) / (ntk_dev.shape[-1])
        # print(f"ntk_dev diag avg: {ntk_dev_diag_avg}")

        for tensor_data, tensor_fn in zip(emp_4d_tensors, tensor_fns):
            tensor_data.append(tensor_fn(layer_output, ntk_dev, input_ind, n_prev))

        prev_output = layer[1](params, prev_output)

    return emp_4d_tensors


def est_expec_4d_tensor_all_layers(
    n_samples: int | None,
    layers: list,
    key: jax.Array,
    batchsize: int,
    x1: jax.Array,
    input_ind: jax.Array,
    tensor_fns: list,
    parameterization_setup: ParameterizationSetup,
):
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    batched_calc_emp_4d_tensor_all_layers = vmap(
        calc_emp_4d_tensor_all_layers,
        in_axes=(None, 0, None, None, None, None, None),
        out_axes=-1,
    )

    layers = jax.tree.map(jax.tree_util.Partial, layers)
    tensor_fns = jax.tree.map(jax.tree_util.Partial, tensor_fns)
    ntks = est_expec_kernel_stats_all_layers(
        "ntk",
        n_samples if n_samples is not None else 1,
        layers,
        subkey1,
        batchsize,
        parameterization_setup,
        x1,
        None,
        None,
        trace_avg=True,
    ).mean

    dummy_ensemble = batched_calc_emp_4d_tensor_all_layers(
        layers,
        jnp.expand_dims(subkey1, 0),
        x1,
        input_ind,
        tensor_fns,
        ntks,
        parameterization_setup,
    )[0]
    tensor_4d_shapes = utils._get_pytree_array_shapes(dummy_ensemble)
    tensor_4d_stats = [
        StatsRecorder.clean_init(tensor_4d_shapes, batch_dim=-1) for _ in tensor_fns
    ]

    # shortcut to obtain dummy object
    if n_samples is None:
        return [t.mean for t in tensor_4d_stats]

    subkeys = jax.random.split(subkey2, n_samples)

    for i in tqdm(
        range(ceil(n_samples / batchsize)),
        desc="Batched 4d tensor stats computation",
        leave=False,
    ):
        batch_lower = i * batchsize
        batch_upper = min((i + 1) * batchsize, n_samples)
        tensord_4d_ensembles = batched_calc_emp_4d_tensor_all_layers(
            layers,
            subkeys[batch_lower:batch_upper],
            x1,
            input_ind,
            tensor_fns,
            ntks,
            parameterization_setup,
        )
        for ts, te in zip(tensor_4d_stats, tensord_4d_ensembles):
            ts.update(te)

    return [t.mean for t in tensor_4d_stats]


@click.group()
def cli():
    pass


network_options = [
    click.option(
        "--n_layers",
        default=2,
        type=int,
        help="Number of layers in the network",
        show_default=True,
    ),
    click.option(
        "--layer_width",
        default=10,
        type=int,
        help="Width of the hidden layers",
        show_default=True,
    ),
    click.option(
        "--nonlin",
        default="Gelu",
        type=str,
        help="Nonlinearity to use",
        show_default=True,
    ),
    click.option(
        "--C_w",
        default=2.0,
        type=float,
        help="Variance of weight initialization",
        show_default=True,
    ),
    click.option(
        "--C_b",
        default=None,
        type=float,
        help="Variance of bias initialization. If not specified, defaults to None which corresponds"
            " to no biases",
        show_default=True,
    ),
]

seed_options = [
    click.option(
        "--param_seed",
        default=random.randrange(2**32),
        type=int,
        help="Random seed for initialization",
        show_default="random integer",
    ),
    click.option(
        "--data_seed",
        default=3042,
        type=int,
        help="Random seed for data generation",
        show_default=True,
    ),
]

data_size_option = click.option(
    "--data_size", default=None, type=int, help="Size of the data set", show_default=True
)

n_samples_option = click.option(
    "--n_samples",
    default=100,
    type=int,
    help="Number of samples to use",
    show_default=True,
)
n_samples_stats_option = click.option(
    "--n_samples_stats",
    default=10,
    type=int,
    help="Number of samples to estimate theestimators statistics",
    show_default=True,
)
batch_size_option = click.option(
    "--batch_size",
    default=2,
    type=int,
    help="Batch size for parallelcomputation",
    show_default=True,
)
input_options = [
    click.option(
        "--input_file",
        default=None,
        type=str,
        help="File containing input vectors for the NN (one row per sample). If --data_size is"
            " smaller than the number of inputs in the file, only the first data_size samples are used.",
        show_default=True,
    ),
    click.option(
        "--input_rescale",
        default=1.0,
        type=float,
        help="Rescaling factor for the input. Multiplies every vector by this number before passing"
        " it to the network",
        show_default=True,
    ),
]

output_dir_option = click.option(
    "--output_dir",
    default=".",
    type=str,
    help="Directory to save output files",
    show_default=True,
)

suffix_option = click.option(
    "--suffix", default="", type=str, help="Suffix to add to output files", show_default=True
)


@cli.command()
@click.argument(
    "kernel",
    type=click.Choice(["nngp", "ntk"], case_sensitive=False),
)
@click.option(
    "--trace", is_flag=True, help="Use the trace average to improve statistics"
)
@utils.add_cmd_options(
    [
        *network_options,
        *seed_options,
        data_size_option,
        n_samples_option,
        batch_size_option,
        *input_options,
        output_dir_option,
    ]
)
def kernel_stability(
    kernel: str,
    n_layers: int,
    data_size: int | None,
    layer_width: int,
    nonlin: str,
    n_samples: int,
    c_w: float,
    c_b: float,
    param_seed: int,
    data_seed: int,
    batch_size: int,
    input_file: str,
    input_rescale: float,
    trace: bool,
    output_dir: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameterization_setup = ParameterizationSetup("standard", c_w, c_b)
    if data_size is None:
        data_size = 2
    layers = create_network(n_layers, layer_width, nonlin, parameterization_setup)
    key = jax.random.key(param_seed)

    if input_file is not None:
        x1 = load_input_from_file(input_file, input_rescale, data_size)
    else:
        input_dim = 2
        x1 = input_rescale * jax.random.normal(
            jax.random.key(data_seed), (data_size, input_dim)
        )

    run_meta = {
        "kernel": kernel,
        "n_layers": n_layers,
        "layer_width": layer_width,
        "nonlin": nonlin,
        "data_size": data_size,
        "c_w": c_w,
        "c_b": c_b,
        "data_seed": data_seed,
        "x1": x1.tolist(),
        "input_rescale": input_rescale,
        "input_hash": utils.hash_array(np.array(x1)),
    }

    kernel_deps = ["nonlin", "n_layers", "layer_width", "c_w", "c_b", "input_hash"]

    key, subkey = jax.random.split(key)

    kn_stats_filename = (
        f"{kernel}_{prec}stats_"
        f"{utils.create_identifier_string(run_meta, kernel_deps)}.json"
    )
    tensors_dir = output_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    kn_file = tensors_dir / kn_stats_filename
    if kn_file.exists():
        kernel_stats = StatsRecorder.load_from_file(kn_file, relevant_meta=run_meta)
    else:
        kernel_stats = None
    kernel_stats = est_expec_kernel_stats_all_layers(
        kernel,
        n_samples,
        layers,
        subkey,
        batch_size,
        parameterization_setup,
        x1,
        x2=None,
        neural_pair=None,
        trace_avg=trace,
        kn_stats=kernel_stats,
    )
    kernel_stats.save_to_file(kn_file, run_meta)
    print(f"Output written to {kn_file.parent.resolve()}")


# Default input indices for 4-point tensors
DEFAULT_INPUT_IND = [
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 2, 2],
    [0, 1, 0, 3],
    [0, 0, 2, 3],
    [0, 1, 2, 3],
    [0, 0, 1, 1],
]

# Map of alphabet tensor names to their computation functions
ALPHABET_TENSOR_FNS = {
    "D": calc_D_tensor,
    "F": calc_F_tensor,
    "A": calc_A_tensor,
    "B": calc_B_tensor,
}


@cli.command()
@click.argument(
    "tensors",
    nargs=-1,
    type=click.Choice(["V4", "D", "F", "A", "B"], case_sensitive=False),
)
@utils.add_cmd_options(
    [
        *network_options,
        *seed_options,
        data_size_option,
        n_samples_option,
        n_samples_stats_option,
        batch_size_option,
        *input_options,
        suffix_option,
        output_dir_option,
    ]
)
def tensor_stability(
    tensors: tuple[str],
    n_layers: int,
    layer_width: int,
    nonlin: str,
    n_samples: int,
    n_samples_stats: int,
    c_w: float,
    c_b: float,
    param_seed: int,
    data_seed: int,
    data_size: int,
    batch_size: int,
    input_file: str,
    input_rescale: float,
    suffix: str,
    output_dir: str,
):
    """Compute stability statistics for specified tensors (V4, D, F, A, B)."""
    if not tensors:
        raise click.UsageError("At least one tensor must be specified")

    # Normalize tensor names to uppercase
    requested = {t.upper() for t in tensors}

    if data_size is None:
        data_size = 4
    input_dim = 4
    config = setup_stability_run(
        n_layers=n_layers,
        layer_width=layer_width,
        nonlin=nonlin,
        c_w=c_w,
        c_b=c_b,
        param_seed=param_seed,
        data_seed=data_seed,
        data_size=data_size,
        input_dim=input_dim,
        input_file=input_file,
        input_rescale=input_rescale,
        n_samples=n_samples,
        n_samples_stats=n_samples_stats,
        batch_size=batch_size,
        output_dir=output_dir,
        input_ind=DEFAULT_INPUT_IND,
    )

    # Build estimators
    estimators = []

    if "V4" in requested:
        estimators.append(V4Estimator())

    alphabet_requested = requested & set(ALPHABET_TENSOR_FNS.keys())
    if alphabet_requested:
        tensor_fns_map = [
            {"name": name, "func": ALPHABET_TENSOR_FNS[name]}
            for name in alphabet_requested
        ]
        estimators.append(AlphabetTensorsEstimator(tensor_fns_map))

    # Run estimation for each estimator group
    all_stats = {}
    for estimator in estimators:
        stats_dict = run_stability_estimation(config, estimator)
        all_stats.update(stats_dict)

    # Save results
    for tensor_name, stats in all_stats.items():
        save_tensor_stats(config, tensor_name, stats, suffix)

    print(f"Output written to {config.output_dir.resolve()}")


if __name__ == "__main__":
    cli()
