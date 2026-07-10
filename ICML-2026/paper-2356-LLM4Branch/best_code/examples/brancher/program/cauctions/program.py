USED_FEATURES = [7, 8, 9, 22, 39, 40, 42]
PARAMS = [0.22751823079556868, 0.8964043626580492, 0.4137359440043567, 0.0, 1.0, 0.031697459892304686, 0.5957606966815607]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

import numpy as np

def score_function(variable_features, params):
    # Extract key features for branching decisions
    reduced_cost = variable_features[:, 7]      # Normalized reduced cost
    lp_value = variable_features[:, 8]          # LP solution value
    fractional = variable_features[:, 9]        # Fractional part of LP solution
    constraint_count = variable_features[:, 22] # Number of participating constraints
    pseudocost_up = variable_features[:, 39]    # Pseudocost up value
    pseudocost_down = variable_features[:, 40]  # Pseudocost down value  
    pseudocost_sum = variable_features[:, 42]   # Pseudocost sum
    
    # Improved feature transformations
    # 1. Strong fractionality preference (peaks at 0.5)
    fractionality_score = 4.0 * fractional * (1.0 - fractional)
    
    # 2. Weighted pseudocost based on fractional distance
    weighted_pseudocost = fractional * pseudocost_down + (1.0 - fractional) * pseudocost_up
    
    # 3. Pseudocost reliability (minimum of both directions)
    pseudocost_reliability = np.minimum(pseudocost_up, pseudocost_down)
    
    # 4. Constraint involvement with normalization
    constraint_importance = constraint_count
    
    # 5. Reduced cost magnitude as tiebreaker
    reduced_cost_magnitude = np.abs(reduced_cost)
    
    # Compute weighted score with improved feature combinations
    score = (
        params[0] * weighted_pseudocost +
        params[1] * fractionality_score +
        params[2] * pseudocost_reliability +
        params[3] * pseudocost_sum +
        params[4] * constraint_importance +
        params[5] * reduced_cost_magnitude +
        params[6] * np.abs(lp_value)  # LP solution magnitude as secondary tiebreaker
    )
    
    return score