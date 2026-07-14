# ./algorithms/score_ordering.py
"""Function for running the SCORE ORDERING algorithm described in the paper.
Code written by Patrick Blöbaum, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0

source: https://github.com/amazon-science/RCAWithMissingStructuralKnowledgeCode/blob/main/algorithms/score_ordering.py
"""


from typing import Dict, Callable

import networkx as nx
import pandas as pd

from dowhy.gcm import RescaledMedianCDFQuantileScorer, ITAnomalyScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)



def apply_score_ordering(normal_data: pd.DataFrame,
                      anomaly_data: pd.DataFrame,
                      anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer) -> Dict[str, float]:
    """
    This is the implementation of the score ordering algorithm (algorithm 2 in the paper)
    """
    all_nodes = list(normal_data.columns)

    # Training the anomaly scorers with 2k observations of not anomalous data.
    scorers = {}
    scores = {}
    for n in all_nodes:
        scorers[n] = ITAnomalyScorer(anomaly_scorer())
        scorers[n].fit(normal_data[n].to_numpy())
        
        # Scoring the anomalous samples.
        scores[n] = float(scorers[n].score(anomaly_data[n].iloc[0:1].to_numpy()[0]))

    return scores


def score_ordering(
    data, inject_time=None, dataset=None,**kwargs
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


    result = apply_score_ordering(normal_df, anomal_df)
    ranks = sorted(result.items(), key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }