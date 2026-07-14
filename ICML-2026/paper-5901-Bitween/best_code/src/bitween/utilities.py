import numpy as np


def pp(x):
    """
    Pretty print a number.
    """
    return np.format_float_positional(x, trim="-")
