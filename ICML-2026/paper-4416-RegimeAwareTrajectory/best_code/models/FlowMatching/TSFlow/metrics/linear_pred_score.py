# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Tuple

import numpy as np
import sys
sys.path.extend(['./', '../', '../../'])
from gluonts.dataset.split import slice_data_entry
from gluonts.evaluation import Evaluator
from gluonts.transform import AdhocTransform, Chain

from models.FlowMatching.TSFlow.model import LinearEstimator
from models.FlowMatching.TSFlow.utils.transforms import ScaleAndAddLongMeanFeature, ScaleAndAddMinMaxFeature
from models.FlowMatching.TSFlow.utils.util import GluonTSNumpyDataset, make_evaluation_predictions_with_scaling


def linear_pred_score(
    samples: np.ndarray,
    context_length: int,
    prediction_length: int,
    test_dataset,
    num_samples: int = 1,
    scaling_type: str = "mean",
) -> Tuple[dict, list, list]:
    """Compute the linear predictive score.
    Uses the `samples` to to fit a LinearRegression model
    and evaluate the forecast performance on the provided
    `test_dataset`.

    Parameters
    ----------
    samples
        The samples used to fit the linear regression model.
        A numpy array of shape [N, T].
        Assumed to be already scaled.
    context_length
        The context length for the linear model.
    prediction_length
        The prediction length for the linear model.
        Must be the same as the prediction length of the
        target `test_dataset`.
    test_datastet
        The test dataset on which the linear model will
        be evaluated.
    num_samples, optional
        Number of samples to draw from the linear model.
        Since the linear model is a point forecaster,
        `num_samples` > 1 would just result in the forecast
        being repeated `num_samples` times, by default 1
    scaling_type, optional
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Returns
    -------
        Evaluation metrics, target test time series and forecasts
    """
    min_past = context_length + prediction_length
    assert samples.shape[1] >= min_past
    dataset = GluonTSNumpyDataset(samples)

    linear_predictor = LinearEstimator(
        freq="H",  # Not actually used in the estimator
        prediction_length=prediction_length,
        context_length=context_length,
        num_train_samples=len(dataset),
        # Since `samples` are synthetic samples, they are assumed to be already scaled
        scaling=False,
    ).train(dataset)

    # The linear predictor has been trained on scaled samples,
    # however, the test dataset is still in the original space.
    # Therefore, the test time series need to be sliced and
    # scaled before being fed into the predictor.
    # After prediction, the time series must be scaled back to
    # the original space for metric computation.
    # The following lines of code perform this custom evaluation.

    # Slice test set to be of the same length as context_length + prediction_length
    slice_func = partial(slice_data_entry, slice_=slice(-min_past, None))
    if scaling_type == "mean":
        ScaleAndAddScaleFeature = ScaleAndAddLongMeanFeature
    elif scaling_type == "min-max":
        ScaleAndAddScaleFeature = ScaleAndAddMinMaxFeature
    # elif scaling_type == "zscore":
    #    ScaleAndAddScaleFeature = ScaleAndAddZScoreFeature
    # breakpoint()
    transformation = Chain(
        [
            AdhocTransform(slice_func),
            # Add scale to data entry for use later during evaluation
            ScaleAndAddScaleFeature("target", "scale", prediction_length),
        ]
    )
    sliced_test_set = transformation.apply(test_dataset)

    evaluator = Evaluator(num_workers=1)
    forecast_it, ts_it = make_evaluation_predictions_with_scaling(
        dataset=sliced_test_set,
        predictor=linear_predictor,
        num_samples=num_samples,
        scaling_type=scaling_type,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    metrics, _ = evaluator(tss, forecasts)

    return metrics, tss, forecasts
