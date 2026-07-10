import random
import numpy as np

USED_FEATURES = [1] # e.g. [1, 10, 77]

PARAMS = [0.5]

BOUNDS = [[0, 1]]

def score_function(variable_features, params):

    return variable_features[:, 43] * params[0]