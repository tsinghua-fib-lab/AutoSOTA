# ./algorithms/circa.py
"""Wrapper function for running the "Cholesky" algorithm as described in 
Li et.al. (2025) "Root cause discovery via permutations and Cholesky decomposition"
https://arxiv.org/abs/2410.12151, with the original code found in https://github.com/Jinzhou-Li/RootCauseDiscovery.
Code written by Sergio Hernan Garrido Mejia, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0

source: https://github.com/amazon-science/RCAWithMissingStructuralKnowledgeCode/blob/main/algorithms/cholesky.py
"""

from typing import Dict

import pandas as pd

from RCAEval.e2e.RCAWithMissingStructuralKnowledgeCode.algorithms.RootCauseDiscovery.funcs.root_cause_discovery_funcs import root_cause_discovery_highdim_parallel, root_cause_discovery_main

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)


def apply_cholesky(normal_data: pd.DataFrame,
                   anomaly_data: pd.DataFrame,
                   cholesky_type: str="highdim") -> Dict[str, float]:
    """
    This is a wrapper around the method in 
    Li et.al. (2025) "Root cause discovery via permutations and Cholesky decomposition"
    https://arxiv.org/abs/2410.12151
    
    The code for this method can be found in https://github.com/Jinzhou-Li/RootCauseDiscovery
    """
    if anomaly_data.shape[0] > 1:
        anomaly_data = anomaly_data.iloc[[0]]
    
    variable_names = normal_data.columns
    
    if cholesky_type == "highdim":
        chol_scores = root_cause_discovery_highdim_parallel(
            X_obs=normal_data.to_numpy(),
            X_int=anomaly_data.to_numpy().flatten(),
            n_jobs=-1,
            y_idx_z_threshold=1.5,
            nshuffles=1,
            verbose=False,
            Precision_mat=None
        )
        result = {variable_names[i]: float(chol_scores[0][i]) for i in range(len(chol_scores[0]))}
    elif cholesky_type == "main":
        n_shuffles = 10
        chol_scores = root_cause_discovery_main(X_obs=normal_data.to_numpy(),
                                                X_int=anomaly_data.to_numpy().flatten(),
                                                nshuffles=n_shuffles,
                                                verbose=False)
        result = {variable_names[i]: float(chol_scores[i]) for i in range(len(chol_scores))}

    return result


def cholesky(
    data, inject_time=None, dataset=None, cholesky_type="highdim",**kwargs
):
    normal_df = data[data["time"] < inject_time]
    anomal_df = data[data["time"] >= inject_time]
    normal_df = normal_df.drop(columns=["time"])
    anomal_df = anomal_df.drop(columns=["time"])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]


    result = apply_cholesky(normal_df, anomal_df, cholesky_type=cholesky_type)
    ranks = sorted(result.items(), key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }