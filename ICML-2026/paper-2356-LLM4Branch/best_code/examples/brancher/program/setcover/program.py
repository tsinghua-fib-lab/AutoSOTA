USED_FEATURES = [0, 7, 9, 43, 48, 67]
PARAMS = [1.0, 0.5887010792086566, 0.31746091541407373, 0.9398783973248053, 0.6031768166959277, 0.2645470326979516, 0.6762821726601128]
BOUNDS = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

import numpy as np

def score_function(variable_features, params):
    # Extract core features with semantic clarity
    obj_coef = variable_features[:, 0]        # Objective coefficient
    reduced_cost = variable_features[:, 7]    # Reduced cost (dual information)
    fractional_part = variable_features[:, 9] # Fractional part of LP solution
    pseudocost_product = variable_features[:, 43] # Pseudocost product (reliability)
    dyn_degree_mean = variable_features[:, 48] # Dynamic constraint degree mean
    active_constraint_count = variable_features[:, 67] # Active constraint count
    
    # Enhanced nonlinear transformations with improved numerical stability
    # Strong branching centrality with quadratic emphasis on 0.5 fractionality
    centrality = 4.0 * fractional_part * (1.0 - fractional_part)
    
    # Enhanced constraint participation with proper scaling and saturation
    constraint_activity = np.tanh(dyn_degree_mean * 0.4) * np.tanh(active_constraint_count * 0.06)
    
    # Strategic feature interactions with better normalization
    obj_fraction_synergy = np.tanh(np.abs(obj_coef)) * centrality
    
    # Dual information with controlled saturation
    dual_importance = np.tanh(np.abs(reduced_cost) * 2.5)
    
    # Pseudocost strength with logarithmic scaling for stability
    pseudocost_strength = np.log1p(np.abs(pseudocost_product) + 1e-8)
    
    # Constraint importance with proper scaling
    constraint_importance = np.log1p(active_constraint_count) / np.log(2.0)
    
    # Optimized scoring with hierarchical structure and clear separation
    fractional_component = (
        params[0] * centrality +                    # Primary: fractionality centrality
        params[1] * fractional_part                 # Secondary: raw fractional value
    )
    
    pseudocost_component = (
        params[2] * pseudocost_strength
    )
    
    structural_component = (
        params[3] * constraint_activity +           # Constraint participation
        params[4] * obj_fraction_synergy +          # Objective-fraction synergy
        params[5] * dual_importance +               # Dual information
        params[6] * constraint_importance           # Scaled constraint importance
    )
    
    # Combined score with clear component separation
    raw_score = fractional_component + pseudocost_component + structural_component
    
    # Improved output transformation using centered softplus for better gradient flow
    return np.log1p(np.exp(raw_score - 2.0))  # Centered softplus for stable outputs