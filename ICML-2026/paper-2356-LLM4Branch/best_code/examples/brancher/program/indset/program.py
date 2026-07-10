USED_FEATURES = [0, 7, 9, 22, 43]
PARAMS = [1.0786976161645492, 0.23806897907821495, 1.2220001238555878, 0.9458262526360364, 0.01]
BOUNDS = [[0, 2], [0, 2], [0, 2], [0, 2], [0, 2]]

def score_function(variable_features, params):
    # Extract essential features for branching decision
    obj_coeff = variable_features[:, 0]        # Objective coefficient
    reduced_cost = variable_features[:, 7]     # Reduced cost
    fractional_part = variable_features[:, 9]  # Fractional part of LP solution
    constraint_count = variable_features[:, 22] # Number of participating constraints
    pseudocost_product = variable_features[:, 43] # Pseudocost product
    
    # Core branching preference - prefer variables near 0.5 (peaks at 0.5)
    branching_preference = 4.0 * fractional_part * (1.0 - fractional_part)
    
    # Optimized linear combination focusing on most impactful features
    score = (
        params[0] * obj_coeff +
        params[1] * reduced_cost +
        params[2] * branching_preference +
        params[3] * constraint_count +
        params[4] * pseudocost_product
    )
    
    return score
