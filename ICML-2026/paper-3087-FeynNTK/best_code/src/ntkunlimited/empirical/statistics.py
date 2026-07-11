import jax.numpy as jnp
import jax
from functools import partial
from collections import namedtuple
import json
from ntkunlimited.utils import _dicts_agree, ArrayShape


MeanVar = namedtuple('MeanVar', ['mean', 'var'])


def _update_stats(mean, var, new_data, n_samples, batch_dim):
    n_new_samples = new_data.shape[batch_dim]
    n_tot_samples = n_samples + n_new_samples

    new_mean = jnp.mean(new_data, axis=batch_dim)
    # updated_mean = (n_new_samples * new_mean + n_samples * mean) / n_tot_samples
    delta = new_mean - mean
    updated_mean = mean + n_new_samples / n_tot_samples * delta

    if n_tot_samples <= 1:
        updated_var = jnp.zeros_like(updated_mean)
    else:
        new_var = jnp.var(new_data, ddof=0, axis=batch_dim)
        # updated_var = n_new_samples * n_samples / n_tot_samples * (new_mean - mean)**2
        updated_var = (
            (n_new_samples * new_var + (n_samples - 1) * var + n_new_samples * n_samples / n_tot_samples * (new_mean - mean)**2) / (n_tot_samples - 1)
        )

    return MeanVar(updated_mean, updated_var)


def _is_array_shape(node):
    return isinstance(node, ArrayShape)


def _is_array_key(node):
    return isinstance(node, dict) and "array" in node


def _is_meanvar(node):
    return isinstance(node, MeanVar)


class StatsRecorder:
    """
    A utility class for recording and updating statistics (mean and variance) 
    for arraylike data in a PyTree structure.

    Attributes:
        mean (PyTree): A PyTree containing the mean values for the data.
        var (PyTree): A PyTree containing the Bessel-corrected sample variance of the data.
        n_samples (int): The total number of samples processed so far.
        batch_dim (int): The dimension along which batches are processed for all leaves.

    Methods:
        update(new_data):
            Updates the mean and variance with new incoming data.
    """

    def __init__(self, mean, var, n_samples, batch_dim=0):
        self.mean = mean
        self.var = var
        self.n_samples = n_samples
        self.batch_dim = batch_dim

    @classmethod
    def clean_init(cls, data_shape, batch_dim=0):
        """
        Initializes the StatsRecorder with the given data shape and batch dimension.

        Args:
            data_shape (PyTree): A PyTree structure describing the shape of the leaves.
            batch_dim (int): The dimension along which batches are processed. Defaults to 0.
        """
        output_shape = jax.tree.map(lambda x: x.remove_batch_dim(batch_dim), data_shape, is_leaf=_is_array_shape)
        instance = cls(
            mean=jax.tree.map(jnp.zeros, output_shape, is_leaf=_is_array_shape),
            var=jax.tree.map(jnp.zeros, output_shape, is_leaf=_is_array_shape),  # Bessel-corrected sample variance
            n_samples=0,
            batch_dim=batch_dim
        )
        return instance

    @classmethod
    def load_from_file(cls, filename: str, relevant_meta: dict = None):
        with open(filename, 'r') as f:
            json_str = f.read()
            data = json.loads(json_str)
            mean = jax.tree_map(lambda x: jnp.array(x['array']), data['mean'], is_leaf=_is_array_key)
            var = jax.tree_map(lambda x: jnp.array(x['array']), data['var'], is_leaf=_is_array_key)
            n_samples = data['n_samples_stats']
            meta = data['meta']

        if relevant_meta is not None:
            if not _dicts_agree(meta, relevant_meta):
                raise RuntimeError(f"Meta information in the file {filename} does not match "
                                   "the relevant meta information of this run.\n\tmeta from "
                                   f"file:\n{meta}\n\n\tmeta from this run:\n{relevant_meta}")

        instance = cls(
            mean=mean,
            var=var,
            n_samples=n_samples,
            batch_dim=data['batch_dim']
        )

        return instance

    def update(self, new_data):
        """
        Updates the statistics (mean and variance) with new data.

        Args:
            new_data (PyTree): A PyTree containing the new data to update the statistics.
        """
        update_this_stats = partial(_update_stats, n_samples=self.n_samples, batch_dim=self.batch_dim)
        stats = jax.tree.map(update_this_stats, self.mean, self.var, new_data)

        self.mean = jax.tree.map(lambda x: x.mean, stats, is_leaf=_is_meanvar)
        self.var = jax.tree.map(lambda x: x.var, stats, is_leaf=_is_meanvar)

        self.n_samples += jax.tree.leaves(new_data)[0].shape[self.batch_dim]

    def _to_dict(self, meta: dict):
        """
        Converts the statistics to a dictionary format.

        Returns:
            dict: A dictionary containing the mean, variance, and number of samples.
        """
        class_dict = {
            'meta': meta,
            'mean': jax.tree_map(lambda x: {'array': x.tolist()}, self.mean),
            'var': jax.tree_map(lambda x: {'array': x.tolist()}, self.var),
            'n_samples_stats': self.n_samples,
            'batch_dim': self.batch_dim
        }
        return class_dict

    def save_to_file(self, filename: str, meta: dict):
        with open(filename, 'w') as f:
            tmp_dict = self._to_dict(meta)
            json_str = json.dumps(tmp_dict)
            f.write(json_str)
