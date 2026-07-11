import json
import logging
from functools import partial
from pathlib import Path
import shutil

import click
import jax
import neural_tangents as nt
import numpy as np
from ntkunlimited.recursions.recursion_symbols import Gelu
from ntkunlimited.recursions.numerical_integration import CubatureConf
from ntkunlimited.recursions.symbolic_to_numerics import GaussExpecNumeric
from numpy.typing import ArrayLike
import sympy
from ntkunlimited.recursions.symmetric_tensor import (
    FULL_2D_SYM,
    NO_2D_T_IN_4D_D_SYM,
    NO_4D_SYM,
    PAIRWISE_4D_SYM,
    SymmetricTensor,
)
from tqdm import tqdm
from ntkunlimited.recursions import (
    V4_expressions,
    G1_expressions,
    F_expressions,
    D_expressions,
    H1_expressions,
    A_expressions,
    B_expressions,
)

import ntkunlimited.utils
from ntkunlimited.nn import create_network, get_layer_widths, ParameterizationSetup
from ntkunlimited.utils import _dicts_agree, add_cmd_options

from ntkunlimited.recursions.config import cache_dir


jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

logging.basicConfig(format="%(message)s", level=logging.INFO)


expec_deps = ["nonlin", "C_w", "C_b", "rtol", "max_subdiv", "input_hash"]
ntk_deps = ["parameterization"]
V4_deps = expec_deps
G_deps = expec_deps
H_deps = expec_deps + ntk_deps
G1_deps = expec_deps
F_deps = expec_deps + ntk_deps
D_deps = expec_deps + ntk_deps
H1_deps = expec_deps + ntk_deps
A_deps = expec_deps + ntk_deps
B_deps = expec_deps + ntk_deps

recursion_deps = {
    "V4": [],
    "G1": ["V4"],
    "F": [],
    "D": ["V4"],
    "H1": ["V4", "G1", "F", "D"],
    "A": ["V4", "D"],
    "B": [],
}


def extract_from_dict(d: dict, keys: list):
    return {key: d[key] for key in keys if key in d}


def find_compute_order(target, deps_map):
    ordered_list = []
    visited = set()

    def visit(node):
        if node in visited:
            return

        for dep in deps_map.get(node, []):
            visit(dep)
        visited.add(node)
        ordered_list.append(node)

    visit(target)
    return ordered_list


def comp_next_V(
    V4_rec_numeric_fn,
    V_prev: SymmetricTensor,
    G: np.ndarray,
    layer_width: int,
    prev_layer_width: int,
    C_w: float,
):
    V = SymmetricTensor(PAIRWISE_4D_SYM, partial(np.zeros, dtype=float))

    for comp in PAIRWISE_4D_SYM:
        V[comp] = V4_rec_numeric_fn(
            V_prev, G, *comp, layer_width, prev_layer_width, C_w
        )
    return V


def comp_V4s(
    Gs: list[ArrayLike],
    act_fn: sympy.Function,
    hidden_layer_width: int,
    layer_widths: list[int],
    gaussexpec_numeric: GaussExpecNumeric,
    meta: dict,
    force_recompute: bool = False,
    output_dir: Path = Path("tensors"),
):
    filename = (
        output_dir
        / f"V4_analytic_{ntkunlimited.utils.create_identifier_string(meta, V4_deps)}.json"
    )

    # FIX: Doesn't check for how many layers it was previously computed.
    if Path(filename).exists() and not force_recompute:
        try:
            Vs, _ = load_symtensors_json(
                filename, PAIRWISE_4D_SYM, hidden_layer_width, meta
            )
            return Vs
        except (FileNotFoundError, KeyError, RuntimeError):
            pass

    C_w = meta["C_w"]

    V4_rec_numeric_fn = V4_expressions.create_recursion_numeric_fn(
        gaussexpec_numeric, act_fn
    )

    # Convention: Input is layer 0, Vs start at layer 1 where they are 0 due to Gaussianity
    Vs = [SymmetricTensor(PAIRWISE_4D_SYM, partial(np.zeros, dtype=float))]
    for i, G in enumerate(
        tqdm(Gs[1:-1], desc="Computing V4 for all layers", position=1, leave=False),
        start=2,
    ):
        prev_layer_width = layer_widths[i - 2]
        Vs.append(
            comp_next_V(
                V4_rec_numeric_fn,
                Vs[-1],
                G,
                layer_widths[i - 1],
                prev_layer_width,
                C_w,
            )
        )
    gaussexpec_numeric.save_cache()

    update_symtensors_json(
        Vs, PAIRWISE_4D_SYM, hidden_layer_width, meta, filename, layer_start=1
    )

    return Vs


def comp_next_G1(
    G1_rec_numeric_fn,
    G1_prev: SymmetricTensor,
    V_prev: SymmetricTensor,
    G: np.ndarray,
    layer_width: int,
    prev_layer_width: int,
    C_w: float,
):
    sym = G1_prev.sym_map

    G1 = SymmetricTensor(sym, partial(np.zeros, dtype=float))
    Ginv = np.linalg.inv(G)

    for comp in sym:
        G1[comp] = G1_rec_numeric_fn(
            G1_prev, V_prev, G, Ginv, *comp, layer_width, prev_layer_width, C_w
        )
    return G1


def comp_G1s(
    Gs: list[ArrayLike],
    Vs: list[SymmetricTensor],
    act_fn: sympy.Function,
    hidden_layer_width: int,
    layer_widths: list[int],
    gaussexpec_numeric: GaussExpecNumeric,
    meta: dict,
    force_recompute: bool = False,
    output_dir: Path = Path("tensors"),
):
    sym = FULL_2D_SYM

    filename = (
        output_dir
        / f"G1_analytic_{ntkunlimited.utils.create_identifier_string(meta, G1_deps)}.json"
    )

    if Path(filename).exists() and not force_recompute:
        try:
            G1s, _ = load_symtensors_json(filename, sym, hidden_layer_width, meta)
            return G1s
        except (FileNotFoundError, KeyError, RuntimeError):
            pass

    C_w = meta["C_w"]

    G1_rec_numeric_fn = G1_expressions.create_recursion_numeric_fn(
        gaussexpec_numeric, act_fn
    )

    G1s = [SymmetricTensor(sym, partial(np.zeros, dtype=float))]
    for i, G in enumerate(
        tqdm(Gs[1:-1], desc="Computing G1 for all layers", position=1, leave=False),
        start=2,
    ):
        # i is the layer that we want to compute G1 for
        # layer 1: first preactivations, Gaussian
        # prev_layer_width: n_l-1, since we want to compute G1^l+1
        # G is from layer i-1
        prev_layer_width = layer_widths[i - 2]
        G1s.append(
            comp_next_G1(
                G1_rec_numeric_fn,
                G1s[-1],
                Vs[i - 2],  # V starts at layer 1
                G,
                layer_widths[i - 1],
                prev_layer_width,
                C_w,
            )
        )
    gaussexpec_numeric.save_cache()

    update_symtensors_json(G1s, sym, hidden_layer_width, meta, filename, layer_start=1)

    return G1s


def comp_next_F(
    F_rec_numeric_fn,
    F_prev: SymmetricTensor,
    G: np.ndarray,
    H: np.ndarray,
    layer_width: int,
    prev_layer_width: int,
    C_w: float,
):
    sym = F_prev.sym_map

    F = SymmetricTensor(sym, partial(np.zeros, dtype=float))
    Ginv = np.linalg.inv(G)

    for comp in sym:
        F[comp] = F_rec_numeric_fn(
            F_prev, G, Ginv, H, *comp, layer_width, prev_layer_width, C_w
        )
    return F


def comp_Fs(
    Gs: list[ArrayLike],
    Hs: list[ArrayLike],
    act_fn: sympy.Function,
    hidden_layer_width: int,
    layer_widths: list[int],
    gaussexpec_numeric: GaussExpecNumeric,
    meta: dict,
    force_recompute: bool = False,
    output_dir: Path = Path("tensors"),
):
    sym = NO_4D_SYM

    filename = (
        output_dir
        / f"F_analytic_{ntkunlimited.utils.create_identifier_string(meta, F_deps)}.json"
    )

    if Path(filename).exists() and not force_recompute:
        try:
            Fs, _ = load_symtensors_json(filename, sym, hidden_layer_width, meta)
            return Fs
        except (FileNotFoundError, KeyError, RuntimeError):
            pass

    C_w = meta["C_w"]

    F_rec_numeric_fn = F_expressions.create_recursion_numeric_fn(
        gaussexpec_numeric, act_fn
    )

    Fs = [SymmetricTensor(sym, partial(np.zeros, dtype=float))]
    for i, G in enumerate(
        tqdm(Gs[1:-1], desc="Computing Fs for all layers", position=1, leave=False),
        start=2,
    ):
        # layer 1: first preactivations, Gaussian
        # prev_layer_width: n_l-1, since we want to compute F^l+1
        # G is from layer i-1
        prev_layer_width = layer_widths[i - 2]
        Fs.append(
            comp_next_F(
                F_rec_numeric_fn,
                Fs[-1],
                G,
                Hs[i - 2],
                layer_widths[i - 1],
                prev_layer_width,
                C_w,
            )
        )

    gaussexpec_numeric.save_cache()

    update_symtensors_json(Fs, sym, hidden_layer_width, meta, filename, layer_start=1)

    return Fs


def comp_next_D(
    D_rec_numeric_fn,
    D_prev: SymmetricTensor,
    V: SymmetricTensor,
    G: np.ndarray,
    H: np.ndarray,
    layer_width: int,
    prev_layer_width: int,
    C_w: float,
):
    sym = D_prev.sym_map

    D = SymmetricTensor(sym, partial(np.zeros, dtype=float))

    for comp in sym:
        D[comp] = D_rec_numeric_fn(
            D_prev, G, H, V, *comp, layer_width, prev_layer_width, C_w
        )
    return D


def comp_Ds(
    Vs: list[SymmetricTensor],
    Gs: list[ArrayLike],
    Hs: list[ArrayLike],
    act_fn: sympy.Function,
    hidden_layer_width: int,
    layer_widths: list[int],
    gaussexpec_numeric: GaussExpecNumeric,
    meta: dict,
    force_recompute: bool = False,
    output_dir: Path = Path("tensors"),
):
    sym = NO_4D_SYM

    filename = (
        output_dir
        / f"D_analytic_{ntkunlimited.utils.create_identifier_string(meta, D_deps)}.json"
    )

    if Path(filename).exists() and not force_recompute:
        try:
            Ds, _ = load_symtensors_json(filename, sym, hidden_layer_width, meta)
            return Ds
        except (FileNotFoundError, KeyError, RuntimeError):
            pass

    C_w = meta["C_w"]

    D_rec_numeric_fn = D_expressions.create_recursion_numeric_fn(
        gaussexpec_numeric, act_fn
    )

    Ds = [SymmetricTensor(sym, partial(np.zeros, dtype=float))]
    for i, G in enumerate(
        tqdm(Gs[1:-1], desc="Computing Ds for all layers", position=1, leave=False),
        start=2,
    ):
        # layer 1: first preactivations, Gaussian
        # prev_layer_width: n_l-1, since we want to compute F^l+1
        # G is from layer i-1
        prev_layer_width = layer_widths[i - 2]
        Ds.append(
            comp_next_D(
                D_rec_numeric_fn,
                Ds[-1],
                Vs[i - 2],
                G,
                Hs[i - 2],
                layer_widths[i - 1],
                prev_layer_width,
                C_w,
            )
        )

    gaussexpec_numeric.save_cache()

    update_symtensors_json(Ds, sym, hidden_layer_width, meta, filename, layer_start=1)

    return Ds


def comp_next_H1(
    H1_rec_numeric_fn,
    H1_prev: SymmetricTensor,
    G1: SymmetricTensor,
    V: SymmetricTensor,
    F: SymmetricTensor,
    D: SymmetricTensor,
    G: np.ndarray,
    H: np.ndarray,
    layer_width: int,
    prev_layer_width: int,
    C_w: float,
):
    sym = H1_prev.sym_map

    H1 = SymmetricTensor(sym, partial(np.zeros, dtype=float))

    for comp in sym:
        H1[comp] = H1_rec_numeric_fn(
            H1_prev, G, H, G1, V, D, F, *comp, layer_width, prev_layer_width, C_w
        )
    return H1


def comp_H1s(
    Gs: list[ArrayLike],
    Hs: list[ArrayLike],
    G1s: list[SymmetricTensor],
    Vs: list[SymmetricTensor],
    Fs: list[SymmetricTensor],
    Ds: list[SymmetricTensor],
    act_fn: sympy.Function,
    hidden_layer_width: int,
    layer_widths: list[int],
    gaussexpec_numeric: GaussExpecNumeric,
    meta: dict,
    force_recompute: bool = False,
    output_dir: Path = Path("tensors"),
):
    # I think it is actually symmetric but that's not obvious right now when looking at the D and F
    # terms
    sym = NO_2D_T_IN_4D_D_SYM

    filename = (
        output_dir
        / f"H1_analytic_{ntkunlimited.utils.create_identifier_string(meta, H1_deps)}.json"
    )

    if Path(filename).exists() and not force_recompute:
        try:
            H1s, _ = load_symtensors_json(filename, sym, hidden_layer_width, meta)
            return H1s
        except (FileNotFoundError, KeyError, RuntimeError):
            pass

    C_w = meta["C_w"]

    H1_rec_numeric_fn = H1_expressions.create_recursion_numeric_fn(
        gaussexpec_numeric, act_fn
    )

    H1s = [SymmetricTensor(sym, partial(np.zeros, dtype=float))]
    for i, G in enumerate(
        tqdm(Gs[1:-1], desc="Computing H1s for all layers", position=1, leave=False),
        start=2,
    ):
        # layer 1: first preactivations, Gaussian
        # prev_layer_width: n_l-1, since we want to compute F^l+1
        # G is from layer i-1
        prev_layer_width = layer_widths[i - 2]
        H1s.append(
            comp_next_H1(
                H1_rec_numeric_fn,
                H1s[-1],
                G1s[i - 2],
                Vs[i - 2],
                Fs[i - 2],
                Ds[i - 2],
                G,
                Hs[i - 2],
                layer_widths[i - 1],
                prev_layer_width,
                C_w,
            )
        )

    gaussexpec_numeric.save_cache()

    update_symtensors_json(H1s, sym, hidden_layer_width, meta, filename, layer_start=1)

    return H1s


def comp_next_A(
    A_rec_numeric_fn,
    A_prev: SymmetricTensor,
    G: np.ndarray,
    H: np.ndarray,
    V: SymmetricTensor,
    D: SymmetricTensor,
    layer_width: int,
    prev_layer_width: int,
    C_w: float,
):
    sym = A_prev.sym_map

    A = SymmetricTensor(sym, partial(np.zeros, dtype=float))

    for comp in sym:
        A[comp] = A_rec_numeric_fn(
            A_prev, G, H, V, D, *comp, layer_width, prev_layer_width, C_w
        )
    return A


def comp_As(
    Gs: list[ArrayLike],
    Hs: list[ArrayLike],
    Vs: list[SymmetricTensor],
    Ds: list[SymmetricTensor],
    act_fn: sympy.Function,
    hidden_layer_width: int,
    layer_widths: list[int],
    gaussexpec_numeric: GaussExpecNumeric,
    meta: dict,
    force_recompute: bool = False,
    output_dir: Path = Path("tensors"),
):
    sym = PAIRWISE_4D_SYM

    filename = (
        output_dir
        / f"A_analytic_{ntkunlimited.utils.create_identifier_string(meta, A_deps)}.json"
    )

    if Path(filename).exists() and not force_recompute:
        try:
            As, _ = load_symtensors_json(filename, sym, hidden_layer_width, meta)
            return As
        except (FileNotFoundError, KeyError, RuntimeError):
            pass

    C_w = meta["C_w"]

    A_rec_numeric_fn = A_expressions.create_recursion_numeric_fn(
        gaussexpec_numeric, act_fn
    )

    As = [SymmetricTensor(sym, partial(np.zeros, dtype=float))]
    for i, G in enumerate(
        tqdm(Gs[1:-1], desc="Computing As for all layers", position=1, leave=False),
        start=2,
    ):
        # layer 1: first preactivations, Gaussian
        # prev_layer_width: n_l-1, since we want to compute A^l+1
        # G is from layer i-1
        prev_layer_width = layer_widths[i - 2]
        As.append(
            comp_next_A(
                A_rec_numeric_fn,
                As[-1],
                G,
                Hs[i - 2],
                Vs[i - 2],
                Ds[i - 2],
                layer_widths[i - 1],
                prev_layer_width,
                C_w,
            )
        )

    gaussexpec_numeric.save_cache()

    update_symtensors_json(As, sym, hidden_layer_width, meta, filename, layer_start=1)

    return As


def comp_next_B(
    B_rec_numeric_fn,
    B_prev: SymmetricTensor,
    G: np.ndarray,
    H: np.ndarray,
    layer_width: int,
    prev_layer_width: int,
    C_w: float,
):
    sym = B_prev.sym_map

    B = SymmetricTensor(sym, partial(np.zeros, dtype=float))

    for comp in sym:
        B[comp] = B_rec_numeric_fn(
            B_prev, G, H, *comp, layer_width, prev_layer_width, C_w
        )
    return B


def comp_Bs(
    Gs: list[ArrayLike],
    Hs: list[ArrayLike],
    act_fn: sympy.Function,
    hidden_layer_width: int,
    layer_widths: list[int],
    gaussexpec_numeric: GaussExpecNumeric,
    meta: dict,
    force_recompute: bool = False,
    output_dir: Path = Path("tensors"),
):
    sym = NO_4D_SYM

    filename = (
        output_dir
        / f"B_analytic_{ntkunlimited.utils.create_identifier_string(meta, B_deps)}.json"
    )

    if Path(filename).exists() and not force_recompute:
        try:
            Bs, _ = load_symtensors_json(filename, sym, hidden_layer_width, meta)
            return Bs
        except (FileNotFoundError, KeyError, RuntimeError):
            pass

    C_w = meta["C_w"]

    B_rec_numeric_fn = B_expressions.create_recursion_numeric_fn(
        gaussexpec_numeric, act_fn
    )

    Bs = [SymmetricTensor(sym, partial(np.zeros, dtype=float))]
    for i, G in enumerate(
        tqdm(Gs[1:-1], desc="Computing Bs for all layers", position=1, leave=False),
        start=2,
    ):
        # layer 1: first preactivations, Gaussian
        # prev_layer_width: n_l-1, since we want to compute A^l+1
        # G is from layer i-1
        prev_layer_width = layer_widths[i - 2]
        Bs.append(
            comp_next_B(
                B_rec_numeric_fn,
                Bs[-1],
                G,
                Hs[i - 2],
                layer_widths[i - 1],
                prev_layer_width,
                C_w,
            )
        )

    gaussexpec_numeric.save_cache()

    update_symtensors_json(Bs, sym, hidden_layer_width, meta, filename, layer_start=1)

    return Bs


def regularize_cov(cov: np.ndarray, diag_reg: float):
    eigvecs = np.linalg.eig(cov)[1]
    reg_mat = eigvecs @ np.eye(cov.shape[0]) * diag_reg @ eigvecs.T
    cov = cov + reg_mat
    return cov


def comp_infinite_kernels(
    kernel_fns, input, layer_widths, parameterization, diag_reg=None
):
    Gs = []
    Hs = []
    last_kernel = nt.stax.Identity()[2](input, None)

    Gs.append(last_kernel.nngp)
    for width, kernel_fn in zip(layer_widths[:-1], kernel_fns):
        last_kernel = kernel_fn(last_kernel)

        Gs.append(last_kernel.nngp)
        Hs.append(last_kernel.ntk)
        if diag_reg is not None:
            Gs[-1] = regularize_cov(Gs[-1], diag_reg)
    return Gs, Hs


def SymTensor_to_dict(V, tensor_sym):
    tensor_dict = {}
    for comp in tensor_sym:
        tensor_dict["-".join(map(str, comp))] = V[comp]
    return tensor_dict


def dict_to_SymTensor(tensor_dict, tensor_sym):
    tensor = SymmetricTensor(tensor_sym, partial(np.zeros, dtype=float))
    for comp in tensor_sym:
        key = "-".join(map(str, comp))
        if key not in tensor_dict:
            raise KeyError(f"Component {key} not found in tensor_dict")
        tensor[comp] = tensor_dict[key]
    return tensor


def load_symtensors_json(filename: str, sym, layer_width: int, meta: dict):
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"No such file: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    loaded_meta = data["meta"]
    if not _dicts_agree(loaded_meta, extract_from_dict(meta, loaded_meta.keys())):
        raise RuntimeError(
            f"Meta information in the file {filepath} does not match "
            "the relevant meta information of this run.\n\tmeta from "
            f"file:\n{loaded_meta}\n\n\tmeta from this run:\n{meta}"
        )
    width_key = f"width-{layer_width}"

    if width_key not in data["data"]:
        raise KeyError(f"No data found for {width_key} in file {filepath}")

    layers = data["data"][width_key]
    # sort by layer number (layer-1, layer-2, ...)
    layer_items = sorted(layers.items(), key=lambda kv: int(kv[0].split("-")[1]))
    symtensors = [dict_to_SymTensor(layer_dict, sym) for _, layer_dict in layer_items]

    return symtensors, meta


def update_symtensors_json(
    symtensors: list[ArrayLike],
    sym,
    layer_width: int,
    meta: dict,
    filename: str,
    layer_start: int,
):
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        with open(filepath, "r") as f:
            data = json.load(f)
            if not _dicts_agree(data["meta"], meta):
                raise RuntimeError(
                    f"Meta information in the file {filepath} does not match "
                    "the relevant meta information of this run.\n\tmeta from "
                    f"file:\n{data['meta']}\n\n\tmeta from this run:\n{meta}"
                )
        if f"width-{layer_width}" not in data["data"]:
            data["data"][f"width-{layer_width}"] = {}
    else:
        data = {
            "meta": meta,
            "data": {f"width-{layer_width}": {}},
        }
    for layer, symtensor in enumerate(symtensors, start=layer_start):
        symtensor_dict = SymTensor_to_dict(symtensor, sym)
        data["data"][f"width-{layer_width}"][f"layer-{layer}"] = symtensor_dict

    with open(filepath, "w") as f:
        json.dump(data, f)


def save_sym_full_tensor_json(
    tensors: list[np.ndarray],
    sym,
    layer_width: int,
    meta: dict,
    filename: str,
    layer_start: int,
):
    symtensors = [full_tensor_to_sym_tensor(tensor, sym) for tensor in tensors]
    update_symtensors_json(symtensors, sym, layer_width, meta, filename, layer_start)


def full_tensor_to_sym_tensor(tensor: np.ndarray, sym: list[tuple[int, ...]]):
    sym_tensor = SymmetricTensor(sym, partial(np.zeros, dtype=float))
    for comp in sym:
        sym_tensor[comp] = tensor[comp]
    return sym_tensor


class IntList(click.ParamType):
    name = "int_list"

    def convert(self, value, param, ctx):
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            return [int(v.strip()) for v in value.split(",")]
        except ValueError:
            self.fail(f"Expected comma-separated integers, got '{value}'", param, ctx)


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

tensor_arg = click.argument(
    "tensor_selection",
    type=click.Choice(list(recursion_deps.keys()), case_sensitive=False),
)


layer_width_option = click.option(
    "--layer_width",
    default=10,
    type=int,
    help="Width of the hidden layers",
    show_default=True,
)


cubature_options = [
    click.option(
        "--rtol", default=1e-2, type=float, help="Relative error tolerance for numerical"
            "integration method (cubature)", show_default=True,
    ),
    click.option(
        "--max_subdiv",
        default=10_000,
        type=int,
        help=("Maximum number of subdivisions for numerical integration method (see scipy's cubature)"),
        show_default=True,
    ),
]

input_options = [
    click.option(
        "--input_file", default=None, type=str, help="File containing input vectors for the NN "
            "(one row per sample)",
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


def setup_general_recursion(
    input_file: str | None,
    input_rescale: float,
    c_w: float,
    c_b: float | None,
    rtol: float,
    max_subdiv: int,
    force_recompute: bool = False,
):
    data_seed = 205252134345
    key = jax.random.key(data_seed)
    if input_file is not None:
        with open(input_file, "r") as f:
            input = np.array(json.load(f)["input"])
    else:
        key, subkey = jax.random.split(key)
        input = jax.random.normal(subkey, shape=(4, 4))

    input = input_rescale * input

    cubature_conf = CubatureConf(rtol, max_subdiv)
    gaussexpec_numeric = GaussExpecNumeric(cubature_conf)
    if not force_recompute:
        gaussexpec_numeric.load_cache()

    act_fn = Gelu
    act_fn_name = "Gelu"
    meta = {
        "C_w": c_w,
        "C_b": c_b,
        "nonlin": act_fn_name,
        "rtol": rtol,
        "max_subdiv": max_subdiv,
        "data_seed": data_seed,
        "input": input.tolist(),
        "input_hash": ntkunlimited.utils.hash_array(np.array(input)),
        "input_rescale": input_rescale,
    }
    return act_fn, input, gaussexpec_numeric, meta


def setup_fixed_width_recursion(
    n_layers: int,
    layer_width: int,
    input: ArrayLike,
    meta: dict,
):
    parameterization_setup = ParameterizationSetup(
        meta["parameterization"],
        meta["C_w"],
        meta["C_b"],
    )

    act_fn_name = meta["nonlin"]

    layers = create_network(
        n_layers=n_layers,
        layer_width=layer_width,
        nonlin=act_fn_name,
        parameterization_setup=parameterization_setup,
    )
    kernel_fns = [layer[2] for layer in layers]
    init_fns = [layer[0] for layer in layers]
    layer_widths = get_layer_widths(init_fns, input.shape)

    Gs, Hs = jax.tree.map(
        np.array,
        comp_infinite_kernels(
            kernel_fns,
            input,
            layer_widths,
            parameterization_setup.parameterization,
            diag_reg=None,
        ),
    )
    return Gs, Hs, layer_widths


def recursion_dispatcher(
    target,
    computed_tensors,
    act_fn,
    hidden_layer_width,
    layer_widths,
    gaussexpec_numeric,
    meta,
    force_recompute: bool = False,
    output_dir: Path = Path("tensors"),
):
    if target == "V4":
        computed_tensors[target] = comp_V4s(
            computed_tensors["G"],
            act_fn,
            hidden_layer_width,
            layer_widths,
            gaussexpec_numeric,
            meta,
            force_recompute,
            output_dir,
        )
    elif target == "G1":
        computed_tensors[target] = comp_G1s(
            computed_tensors["G"],
            computed_tensors["V4"],
            act_fn,
            hidden_layer_width,
            layer_widths,
            gaussexpec_numeric,
            meta,
            force_recompute,
            output_dir,
        )
    elif target == "F":
        computed_tensors[target] = comp_Fs(
            computed_tensors["G"],
            computed_tensors["H"],
            act_fn,
            hidden_layer_width,
            layer_widths,
            gaussexpec_numeric,
            meta,
            force_recompute,
            output_dir,
        )
    elif target == "D":
        computed_tensors[target] = comp_Ds(
            computed_tensors["V4"],
            computed_tensors["G"],
            computed_tensors["H"],
            act_fn,
            hidden_layer_width,
            layer_widths,
            gaussexpec_numeric,
            meta,
            force_recompute,
            output_dir,
        )
    elif target == "H1":
        computed_tensors[target] = comp_H1s(
            computed_tensors["G"],
            computed_tensors["H"],
            computed_tensors["G1"],
            computed_tensors["V4"],
            computed_tensors["F"],
            computed_tensors["D"],
            act_fn,
            hidden_layer_width,
            layer_widths,
            gaussexpec_numeric,
            meta,
            force_recompute,
            output_dir,
        )
    elif target == "A":
        computed_tensors[target] = comp_As(
            computed_tensors["G"],
            computed_tensors["H"],
            computed_tensors["V4"],
            computed_tensors["D"],
            act_fn,
            hidden_layer_width,
            layer_widths,
            gaussexpec_numeric,
            meta,
            force_recompute,
            output_dir,
        )
    elif target == "B":
        computed_tensors[target] = comp_Bs(
            computed_tensors["G"],
            computed_tensors["H"],
            act_fn,
            hidden_layer_width,
            layer_widths,
            gaussexpec_numeric,
            meta,
            force_recompute,
            output_dir,
        )

    else:
        raise ValueError(f"Unknown tensor target: {target}")

    return computed_tensors


@cli.command()
@add_cmd_options(
    [
        tensor_arg,
        *network_options,
        *cubature_options,
        *input_options,
    ]
)
@click.option(
    "--force-recompute",
    is_flag=True,
    default=False,
    help="Ignore cached tensors and gaussexpec results and recompute from scratch.",
)
@click.option(
    "--output_dir",
    default=".",
    type=str,
    help="Directory under which a 'tensors/' subdirectory will be created for output files.",
    show_default=True,
)
@click.option(
    "--layer_widths",
    type=IntList(),
    default="6,11,19,32",
    help="Comma-separated list of hidden layer widths to compute tensors for.",
    show_default=True,
)
def recursion_width_study(
    tensor_selection: str,
    n_layers: int,
    c_w: float,
    c_b: float | None,
    rtol: float,
    max_subdiv: int,
    input_file: str,
    input_rescale: float,
    force_recompute: bool,
    output_dir: str,
    layer_widths: list[int],
):
    hidden_layer_widths = layer_widths

    run_tensor_dir = Path(output_dir) / "tensors"
    run_tensor_dir.mkdir(parents=True, exist_ok=True)

    act_fn, input, gaussexpec_numeric, meta = setup_general_recursion(
        input_file, input_rescale, c_w, c_b, rtol, max_subdiv, force_recompute
    )
    meta["parameterization"] = "ntk" if tensor_selection in [] else "standard"
    # pr.enable()
    for hidden_layer_width in tqdm(
        hidden_layer_widths,
        desc=f"Computing {tensor_selection} for all widths",
        position=0,
    ):
        Gs, Hs, layer_widths = setup_fixed_width_recursion(
            n_layers, hidden_layer_width, input, meta
        )

        G_filename = (
            run_tensor_dir
            / f"G_analytic_{ntkunlimited.utils.create_identifier_string(meta, G_deps)}.json"
        )
        save_sym_full_tensor_json(
            Gs, FULL_2D_SYM, hidden_layer_width, meta, G_filename, layer_start=0
        )
        H_filename = (
            run_tensor_dir
            / f"H_analytic_{ntkunlimited.utils.create_identifier_string(meta, H_deps)}.json"
        )
        save_sym_full_tensor_json(
            Hs, FULL_2D_SYM, hidden_layer_width, meta, H_filename, layer_start=1
        )

        computed_tensors = {"G": Gs, "H": Hs}

        compute_order = find_compute_order(tensor_selection, recursion_deps)
        for t in compute_order:
            computed_tensors = recursion_dispatcher(
                t,
                computed_tensors,
                act_fn,
                hidden_layer_width,
                layer_widths,
                gaussexpec_numeric,
                meta,
                force_recompute,
                run_tensor_dir,
            )

    print(f"Output written to {run_tensor_dir.resolve()}")


@cli.command()
@click.option(
    "--tensor-dir",
    default=None,
    type=str,
    help="Also delete tensor files inside this directory (typically the 'tensors/' subdirectory of your output directory).",
)
def clear_cache(tensor_dir: str | None):
    """Clears the cache directory."""
    if cache_dir.exists() and cache_dir.is_dir():
        for entry in cache_dir.iterdir():
            if entry.is_file():
                entry.unlink()
            elif entry.is_dir():
                shutil.rmtree(entry)

        logger.info(f"Cleared cache directory: {cache_dir}")
    else:
        logger.info(f"No cache directory found at: {cache_dir}")

    if tensor_dir is not None:
        tensor_path = Path(tensor_dir)
        if tensor_path.exists() and tensor_path.is_dir():
            for entry in tensor_path.iterdir():
                if entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)

            logger.info(f"Cleared tensor directory: {tensor_path}")
        else:
            logger.info(f"No tensor directory found at: {tensor_path}")

    found = [
        p for p in Path(".").rglob("tensors")
        if p.is_dir()
        and (tensor_dir is None or p != Path(tensor_dir))
        and any(p.glob("*.json"))
    ]
    if found:
        logger.warning("The following 'tensors/' directories were found in the current working directory and were NOT deleted:")
        for p in found:
            logger.warning(f"  {p}")
        logger.warning("Use --tensor-dir <path> to delete one of them.")


if __name__ == "__main__":
    cli()
