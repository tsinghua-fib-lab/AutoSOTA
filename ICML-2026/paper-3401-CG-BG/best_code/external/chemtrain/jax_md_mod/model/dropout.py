# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom function for enabling dropout applications in haiku."""
from functools import wraps

import haiku as hk
from jax import random, vmap, numpy as jnp


# Note: This implementation currently stores the RNG key as float32
#  rather than uint32. This way, both energy_params and the RNG key
#  can be treated analogously in grad and optimizer of trainers.
#  Therefore, the RNG key is updated twice per parameter update:
#  By optimizer via a (arbitrary) gradient and by the standard RNG update.
#  This allows a more unified treatment in trainers, without frequent
#  branching if dropout is used or not.


def _dropout_wrapper(fun):
    """Wraps Haiku apply fuctions such that combined dropout and models params
     are split and supplied seperately as expected by model.
     """
    # first argument is params for Haiku models
    @wraps(fun)
    def dropout_fn(params, *args, **kwargs):
        if dropout_is_used(params):
            params, drop_key = split_dropout_params(params)
            kwargs['dropout_key'] = drop_key
        return fun(params, *args, **kwargs)
    return dropout_fn


def model_init_apply(model, model_kwargs):
    """Returns Haiku model.init and model.apply that adapivly use dropout if
    'dropout_key' is provided alongside parameters. If not, no dropout is
    applied.
    """
    dropout_mode = model_kwargs.get('dropout_mode', None)
    if dropout_mode is None:
        return model.init, model.apply
    else:
        return model.init, _dropout_wrapper(model.apply)


def split_dropout_params(meta_params):
    """Splits up meta params, built up by energy_params and
    dropout_key.
    """
    return (meta_params['haiku_params'],
            jnp.uint32(meta_params['Dropout_RNG_key']))


def next_dropout_params(meta_params):
    """Steps dropout_key and re-packes it in meta_params."""
    _, old_dropout_key = split_dropout_params(meta_params)
    new_dropout_key, _ = random.split(old_dropout_key, 2)
    return build_dropout_params(meta_params['haiku_params'], new_dropout_key)


def dropout_is_used(meta_params):
    """A function that returns whether dropout is used by
    checking if the 'Dropout_RNG_key' is set or exists at all.
    """
    try:
        return 'Dropout_RNG_key' in meta_params.keys()
    except AttributeError:
        return False


def build_dropout_params(energy_params, dropout_key):
    """Combines meta_params, built up by energy_params and
    dropout_key.
    """

    return {'haiku_params': energy_params,
            'Dropout_RNG_key': jnp.float32(dropout_key)}


def construct_dropout_params(dropout_key, dropout_setup):
    """Splits and distributes the random key for all dropout.Linear
    layers.
    """
    dropout_key_dict = {}
    if dropout_key is not None and len(dropout_setup) != 0:  # dropout
        n_keys = len(dropout_setup)
        split = random.split(dropout_key, n_keys)
        for i, (layer_name, do_rate) in enumerate(dropout_setup.items()):
            dropout_key_dict[layer_name] = {'key': split[i], 'do_rate': do_rate}

    return dropout_key_dict


class Linear(hk.Module):
    """Wrapper function for hk.Linear that applies dropout if
    name exists as key in dropout_dict during call.
    """
    def __init__(self, embed_size, name, with_bias=True, **init_kwargs):
        super().__init__(name=name)
        self.linear = hk.Linear(embed_size, with_bias=with_bias, **init_kwargs)
        # False: independent dropout of output features
        # True: same parts of feature vector are dropped for all edge embeddings
        self.shared = False  # TODO check which is most common mode

    def __call__(self, inputs, dropout_dict=None):
        linear_output = self.linear(inputs)
        if dropout_dict is None or self.module_name not in dropout_dict.keys():
            return linear_output
        else:  # apply dropout
            dropout_key = dropout_dict[self.module_name]['key']
            if self.shared:
                vectorized_dropout = vmap(hk.dropout, in_axes=(None, None, 0))
                dropped_array = vectorized_dropout(dropout_key, dropout_dict[
                    self.module_name]['do_rate'], linear_output)
            else:
                dropped_array = hk.dropout(dropout_key, dropout_dict[
                    self.module_name]['do_rate'], linear_output)
            return dropped_array


def dimenetpp_setup(setup_dict,
                    num_dense_out,
                    n_interaction_blocks,
                    num_res_before_skip,
                    num_res_after_skip,
                    overall_dropout_rate=None):
    """Builds the Dropout hyperparameters for DimeNet++.

    Args:
        setup_dict: Dict containing the block name to be dropouted
                    alongside the target dropout rate for each block.
                    Available blocks are 'output', 'interaction' and
                    'embedding'. If None, no layers will be dropped.
        num_dense_out: Number of fully-connected layers in output block
        n_interaction_blocks: Number of interaction blocks
        num_res_before_skip: Number of residual blocks before the skip
        num_res_after_skip: Number of residual blocks after the skip connection
        overall_dropout_rate: If given, will override all droput rates
                              with this global rate.

    Returns: dropout_setup - a dict encoding dropout structure of DimeNet++
    """
    # TODO replace this and only select per-block? Use haiku.MLP dropout?
    def index_to_layer_name(i):
        if i == 0:
            return ''
        else:
            return '_' + str(i)

    # For maximum flexibility, each layer can be addressed seperately.
    # These names are hardcoded to naming of layers in DimeNetPP
    # If you change layer names, make sure to adjust these
    delimiter = '/~/'
    net_prefix = 'DimeNetPP'
    output_prefix = 'Output'
    interaction_prefix = 'Interaction'
    embedding_prefix = 'Embedding'
    residual_prefix = 'ResLayer'
    output_layers = ['Dense_Series' + index_to_layer_name(i)
                     for i in range(num_dense_out)]
    output_layers.extend(['RBF_Dense', 'Upprojection'])
    interaction_layers = ['Dense_ji', 'Dense_kj', 'Downprojection',
                          'Upprojection', 'FinalBeforeSkip', 'rbf1', 'rbf2',
                          'sbf1', 'sbf2']
    residual_layers = ['ResidualSubLayer' + index_to_layer_name(i)
                       for i in range(2)]
    embedding_layers = ['Concat_Dense', 'RBF_Dense']

    if setup_dict is None:  # no dropout
        setup_dict = {}

    dropout_setup = {}
    if 'output' in setup_dict.keys():
        layer_prefix = net_prefix + delimiter + output_prefix
        droprate = setup_dict['output']

        for i_outblock in range(n_interaction_blocks + 1):
            i_interaction_prefix = layer_prefix + \
                                   index_to_layer_name(i_outblock) + delimiter
            for layer_name in output_layers:
                name = i_interaction_prefix + layer_name
                dropout_setup[name] = droprate

    if 'interaction' in setup_dict.keys():
        layer_prefix = net_prefix + delimiter + interaction_prefix
        droprate = setup_dict['interaction']

        for i_interaction in range(n_interaction_blocks):
            i_interaction_prefix = layer_prefix + \
                                   index_to_layer_name(i_interaction) \
                                   + delimiter
            for layer_name in interaction_layers:
                name = i_interaction_prefix + layer_name
                dropout_setup[name] = droprate
            for i_res_block in range(num_res_before_skip + num_res_after_skip):
                res_block_prefix = i_interaction_prefix + residual_prefix + \
                                   index_to_layer_name(i_res_block) + delimiter
                for res_layer_name in residual_layers:
                    name = res_block_prefix + res_layer_name
                    dropout_setup[name] = droprate

    if 'embedding' in setup_dict.keys():
        layer_prefix = net_prefix + delimiter + embedding_prefix
        droprate = setup_dict['embedding']
        for layer_name in embedding_layers:
            name = layer_prefix + delimiter + layer_name
            dropout_setup[name] = droprate

    if overall_dropout_rate is not None:  # override dropout rate

        for layer in dropout_setup:
            dropout_setup[layer] = overall_dropout_rate

    return dropout_setup
