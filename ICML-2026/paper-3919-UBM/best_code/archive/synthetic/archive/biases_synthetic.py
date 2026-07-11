''' 
Selection bias: 
-  Non-random selection of participants into the study leads to a 
non-representative sample.
'''
import logging
import numpy as np

def apply_selection_bias(data, selection_dependence='Y', beta_S_A=0.6, beta_S_Y=0.6):
    """
    Applies selection bias by selecting samples based on specified dependence.
    
    Parameters:
    - data: DataFrame, observational study da   ta
    - selection_dependence: str, 'Y', 'A', or 'both' 
        - 'Y': selection probability increases with higher Y values
        - 'A': selection probability increases with higher A values
        - 'both': selection probability depends on both A and Y
    
    Returns:
    - DataFrame after applying selection bias
    """
    if selection_dependence == 'Y':
        # Higher probability to include based on outcome Y
        selection_prob = 0.7 * (data['Y'] > data['Y'].median()) + 0.3
    elif selection_dependence == 'A':
        # Higher probability to include based on exposure A
        selection_prob = 0.6 * data['A'] + 0.4
    elif selection_dependence == 'both':
        # Dependence on both A and Y
        selection_prob = 0.5 * data['A'] + 0.5 * (data['Y'] > data['Y'].median())
    elif selection_dependence == 'berksons': 
        linear_S = beta_S_A * data['A'] + beta_S_Y * data['Y']
        selection_prob = 1 / (1 + np.exp(-linear_S))
    else:
        raise ValueError("Unsupported selection_dependence. Choose 'Y', 'A', or 'both'.")

    S = np.random.binomial(1, selection_prob, len(data))
    import pdb; pdb.set_trace()
    data_selection_bias = data[S == 1].copy()
    data_selection_bias = data_selection_bias.sample(n=data.shape[0], \
                                                     replace=True, \
                                                     random_state=42)
    return data_selection_bias

def apply_exposure_misclassification(data, misclassification_prob=0.1):
    """
    Applies misclassification to exposure A.

    Parameters:
    - data: DataFrame, observational study data
    - misclassification_prob: float, probability to flip A

    Returns:
    - DataFrame with misclassified A
    """
    A_misclassified = data['A'].apply(lambda x: 1 - x if np.random.rand() < misclassification_prob else x)
    data_info_bias_A = data.copy()
    data_info_bias_A['A'] = A_misclassified
    return data_info_bias_A

def apply_bias_wrapper(data, bias_dict):
    """
    Applies a type of bias (based on type given in bias_dict) 
    to the observational data.
    
    Parameters:
    - data: DataFrame, observational study data
    - bias_dict: dict, dictionary containing the type of bias to apply 
    and the parameters of the bias.
    
    Returns:
    - DataFrame after applying bias
    """ 
    bias_type = bias_dict["type"]

    if bias_type == "selection_bias":
        logging.info("Bias applied successfully")
        return apply_selection_bias(data, bias_dict["params"]["selection_dependence"])
    elif bias_type == "misclassification_bias":
        logging.info("Bias applied successfully")
        return apply_exposure_misclassification(data, bias_dict["params"]["misclassification_prob"])
    else:
        logging.warning(f"No bias applied")
        return data
