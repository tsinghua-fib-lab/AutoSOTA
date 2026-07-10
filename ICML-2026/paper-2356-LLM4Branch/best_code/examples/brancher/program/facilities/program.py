USED_FEATURES = [0, 7, 8, 9, 37, 39, 40, 43, 44, 45]
PARAMS = [0.04705795345254364, 0.6785051301493801, 0.2969679650822195, 0.6343660825577095, 0.48452965740137316, 1.0, 0.33402312976578125, 0.16108144646419523, 0.38212310014780954, 0.40656551874413094]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

import numpy as np

def score_function(variable_features, params):
    # Extract core branching features
    obj_coef = variable_features[:, 0]  # objective coefficient
    reduced_cost = variable_features[:, 7]  # reduced cost
    lp_value = variable_features[:, 8]  # LP solution value
    fractional = variable_features[:, 9]  # fractional part
    frac_distance = variable_features[:, 37]  # fractional distance
    pseudo_up = variable_features[:, 39]  # pseudocost up
    pseudo_down = variable_features[:, 40]  # pseudocost down
    pseudo_product = variable_features[:, 43]  # pseudocost product
    cutoff_up = variable_features[:, 44]  # cutoff up count
    cutoff_down = variable_features[:, 45]  # cutoff down count
    
    # Core branching indicators with robust transformations
    centrality = 4.0 * fractional * (1.0 - fractional)  # Strong centrality measure
    pseudo_reliability = np.minimum(pseudo_up, pseudo_down) / (pseudo_up + pseudo_down + 1e-8)
    bound_tightening = np.log1p(cutoff_up + cutoff_down)
    
    # Strategic feature interactions
    obj_centrality = obj_coef * centrality
    reliability_tightening = pseudo_reliability * bound_tightening
    reduced_centrality = reduced_cost * centrality
    pseudo_robust = np.sqrt(pseudo_product + 1e-8)
    
    # Additional strategic components
    distance_penalty = np.sqrt(frac_distance + 1e-8)
    cutoff_asymmetry = np.tanh(cutoff_up / (cutoff_down + 1e-8) - 1.0)
    lp_magnitude = np.abs(lp_value)
    pseudo_sum = pseudo_up + pseudo_down
    
    # Combined scoring with balanced interactions
    scores = (
        params[0] * centrality +
        params[1] * pseudo_robust +
        params[2] * pseudo_reliability +
        params[3] * obj_centrality +
        params[4] * reliability_tightening +
        params[5] * reduced_centrality +
        params[6] * distance_penalty +
        params[7] * cutoff_asymmetry +
        params[8] * lp_magnitude +
        params[9] * pseudo_sum
    )
    
    return scores