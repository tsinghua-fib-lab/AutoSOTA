# state parameter set
import numpy as np


def state_parameter_set(state):
    sps = state / np.sum(state)

    return sps

