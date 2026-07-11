from neural_tangents import stax
from math import sqrt
import jax
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterizationSetup:
    parameterization: str
    C_w: float
    C_b: float | None
    convert_to_book: bool = True


def make_ntk_book_compatible(kernel, parameterization_setup: ParameterizationSetup):
    # WARNING: This is a very specific hack to make the NTK match the one in the NTK book
    # for standard parameterization
    C_w = parameterization_setup.C_w
    C_b = parameterization_setup.C_b

    if parameterization_setup.parameterization == 'standard':
        input_width = kernel.shape1[-1]
        compensating_term = kernel.nngp * (1 - input_width) / C_w
        ntk = kernel.ntk + compensating_term
        return kernel.replace(ntk=ntk)
    else:
        nngp = kernel.nngp
        ntk = kernel.ntk
        ntk += nngp * (1 / C_w - 1)
        if parameterization_setup.C_b is not None:
            ntk += (1 - C_b) / C_w

        kernel = kernel.replace(ntk=ntk)
        return kernel


def create_network(n_layers: int, layer_width: int, nonlin: str, parameterization_setup:
                   ParameterizationSetup):
    parameterization = parameterization_setup.parameterization
    c_w = parameterization_setup.C_w
    c_b = parameterization_setup.C_b

    def Dense(out_dim, **kwargs):
        init_fn, apply_fn, kernel_fn = stax.Dense(out_dim, **kwargs)

        if not parameterization_setup.convert_to_book:
            return init_fn, apply_fn, kernel_fn

        def new_kernel_fn(k, **kwargs):
            k = make_ntk_book_compatible(k, parameterization_setup)
            return kernel_fn(k, **kwargs)
        return init_fn, apply_fn, new_kernel_fn

    nonlin_fn = getattr(stax, nonlin)
    W_std = sqrt(c_w)
    b_std = sqrt(c_b) if c_b is not None else None
    dense_hid_layer = Dense(layer_width, parameterization=parameterization, W_std=W_std, b_std=b_std)
    output_layer = Dense(2, parameterization=parameterization, W_std=W_std,
                         b_std=b_std)

    if n_layers == 1:
        layers = [output_layer]
    else:
        layers = [dense_hid_layer]
        if n_layers >= 3:
            layers += [stax.serial(nonlin_fn(), dense_hid_layer)] * (n_layers - 2)
        layers.append(stax.serial(nonlin_fn(), output_layer))
    return layers


def get_layer_widths(init_fns, input_shape):
    layer_widths = [input_shape[1]]
    pseudo_key = jax.random.key(0)
    for init_fn in init_fns:
        output_shape, _ = init_fn(pseudo_key, input_shape=input_shape)
        if len(output_shape) != 2:
            raise NotImplementedError("Only layers with 1d output supported right now")
        layer_widths.append(output_shape[1])
    return layer_widths
