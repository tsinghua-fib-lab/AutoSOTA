from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.gcm import ITAnomalyScorer
import numpy as np
import time
import pandas as pd
from typing import Union, Callable, List, Dict
from dowhy.gcm import RescaledMedianCDFQuantileScorer, ITAnomalyScorer


def score_ordering(
    normal_data: pd.DataFrame,
    failure_data: pd.DataFrame,
    anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer,
):

    t0 = time.time()
    all_nodes = list(normal_data.columns)
    # Prepare outputs
    scorers = {}
    scores = {}

    failure_data = failure_data.iloc[2:3]

    # Train per-node scorers on normal data; score ALL anomalous rows (one-by-one)
    for n in all_nodes:
        scorers[n] = ITAnomalyScorer(anomaly_scorer())
        scorers[n].fit(normal_data[n].to_numpy())

        # Scoring the anomalous samples.
        scores[n] = (
            scorers[n].score(failure_data[n].iloc[0:1].to_numpy()[0]).flatten()[0]
        )

    # Rank nodes by descending score
    ranking = sorted(scores, key=scores.get, reverse=True)

    return {
        "ranking": ranking,
        "time": time.time() - t0,
    }
