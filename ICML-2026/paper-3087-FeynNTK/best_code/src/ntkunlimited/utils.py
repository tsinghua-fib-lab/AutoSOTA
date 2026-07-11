import hashlib
import dill
import dill.detect
from pathlib import Path

import jax
import numpy as np


dill.settings["recurse"] = True


class ArrayShape(tuple):
    def __new__(self, *args):
        return super().__new__(self, args)

    def remove_batch_dim(self, batch_dim):
        if batch_dim < 0:
            batch_dim += len(self)
        self = ArrayShape(*(self[:batch_dim] + self[batch_dim + 1 :]))
        return self


def _dicts_agree(big_dict, relevant_dict):
    for key, value in relevant_dict.items():
        if key not in big_dict:
            return False
        if isinstance(value, dict):
            if not _dicts_agree(big_dict[key], value):
                return False
        else:
            if big_dict[key] != value:
                return False
    return True


def add_cmd_options(options):
    def _add_options(func):
        if isinstance(options, list):
            for option in reversed(options):
                func = option(func)
        else:
            func = options(func)
        return func

    return _add_options


def _get_pytree_array_shapes(tree):
    return jax.tree.map(lambda x: ArrayShape(*x.shape), tree)


def hash_array(arr: np.ndarray):
    h = hashlib.md5()
    arr = arr.astype(np.float32)
    h.update(arr.data.tobytes())
    return h.hexdigest()


def create_identifier_string(meta: dict, selection: list):
    output = ""
    for key in selection:
        if key in meta:
            value = meta[key]
            if isinstance(value, float):
                value = f"{value:.3f}"
            output += f"{key}-{value}_"
        else:
            output += f"{key}-None_"
    return output[:-1]


def cacheable_function(filename, args_hasher, overwrite=False):
    def decorator(func):
        if not overwrite and Path(filename).exists():
            with open(filename, "rb") as f:
                cache = dill.load(f)
        else:
            cache = {}

        class wrapper:
            def __init__(self, cache):
                self.cache = cache

            def __call__(self, *args):
                if args_hasher is not None:
                    arg_ids = args_hasher(args)
                else:
                    arg_ids = args
                if arg_ids not in cache:
                    self.cache[arg_ids] = func(*args)
                return self.cache[arg_ids]

            def cache_info(self):
                return {
                    "size": len(cache),
                }

            def clear_cache(self):
                self.cache = {}

            def save_cache(self):
                with open(filename, "wb") as f:
                    dill.dump(self.cache, f, byref=True)

        return wrapper(cache)

    return decorator
