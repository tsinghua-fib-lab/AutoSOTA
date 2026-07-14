import pandas as pd
import numpy as np

def compute_joint_probabilities(df, columns, sample_size=None):
    """
    Compute joint probabilities for the given columns in the dataframe using a single `groupby` operation.
    Returns a DataFrame with `columns` and an additional 'probability' column.
    """
    if sample_size is not None:
        df = df.sample(n=sample_size)
    
    # Compute joint probabilities using a single `groupby` operation
    total_count = len(df)
    joint_prob_df = df.groupby(columns).size().reset_index(name='counts')
    joint_prob_df['probability'] = joint_prob_df['counts'] / total_count
    return joint_prob_df

def mutual_information(df, X, Y, sample_size=None):
    """
    Compute Mutual Information I(X; Y) using vectorized operations.
    """
    # Compute joint and marginal probabilities
    joint_prob_df = compute_joint_probabilities(df, [X, Y], sample_size)
    x_prob_df = compute_joint_probabilities(df, [X], sample_size)
    y_prob_df = compute_joint_probabilities(df, [Y], sample_size)

    # Merge to align the probabilities for MI calculation
    merged_df = pd.merge(joint_prob_df, x_prob_df, on=[X], suffixes=('', '_x'))
    merged_df = pd.merge(merged_df, y_prob_df, on=[Y], suffixes=('', '_y'))

    # Calculate Mutual Information using vectorized numpy operations
    merged_df['mi_contrib'] = merged_df['probability'] * np.log2(merged_df['probability'] / (merged_df['probability_x'] * merged_df['probability_y']))
    
    # Return the sum of mutual information contributions
    return merged_df['mi_contrib'].sum()

def conditional_mutual_information(df, X, Y, Z=None, sample_size=None):
    """
    Compute Conditional Mutual Information I(X; Y | Z) for large discrete data.
    """
    if Z is None or len(Z) == 0:
        return mutual_information(df, X, Y, sample_size)

    # Compute joint and marginal probabilities for X, Y, and Z combinations
    joint_prob_df = compute_joint_probabilities(df, [X, Y] + Z, sample_size)
    xz_prob_df = compute_joint_probabilities(df, [X] + Z, sample_size)
    yz_prob_df = compute_joint_probabilities(df, [Y] + Z, sample_size)
    z_prob_df = compute_joint_probabilities(df, Z, sample_size)

    # Merge to align joint and marginal probabilities for CMI calculation
    merged_df = pd.merge(joint_prob_df, xz_prob_df, on=[X] + Z, suffixes=('', '_xz'))
    merged_df = pd.merge(merged_df, yz_prob_df, on=[Y] + Z, suffixes=('', '_yz'))
    merged_df = pd.merge(merged_df, z_prob_df, on=Z, suffixes=('', '_z'))

    # Calculate Conditional Mutual Information using vectorized numpy operations
    merged_df['cmi_contrib'] = merged_df['probability'] * np.log2(
        merged_df['probability'] * merged_df['probability_z'] / 
        (merged_df['probability_xz'] * merged_df['probability_yz'])
    )
    
    # Return the sum of conditional mutual information contributions
    return merged_df['cmi_contrib'].sum()
