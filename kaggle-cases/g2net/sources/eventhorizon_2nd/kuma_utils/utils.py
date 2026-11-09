import numpy as np


def is_env_notebook():
    if 'get_ipython' not in globals():
        return False
    env_name = get_ipython().__class__.__name__
    if env_name == 'TerminalInteractiveShell':
        return False
    return True


def vector_normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord=order, axis=axis, keepdims=True)
    l2[l2 == 0] = 1
    return v/l2


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -709, 100000)))
