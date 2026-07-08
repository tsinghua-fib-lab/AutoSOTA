import torch
import numpy as np
from abc import ABC, abstractmethod
from crepes.extras import hinge
from substantive.faircp.conformity.raps import raps_scores
from substantive.faircp.conformity.saps import saps_scores
from substantive.faircp.structs.enums import ScoreFunctionType


class ScoreFunction(ABC):
    @abstractmethod
    def get_scores(
        self,
        calib_probs: torch.Tensor,
        calib_targets: torch.Tensor,
        test_probs: torch.Tensor,
        h_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class HingeScoring(ScoreFunction):
    def get_scores(
        self,
        calib_probs: torch.Tensor,
        calib_targets: torch.Tensor,
        test_probs: torch.Tensor,
        h_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        # For calib data, consider targets and classes
        # Class labels are remapped from 0 to n_classes in `get_loader` method
        classes = torch.arange(calib_probs.shape[1])
        calib_non_conf_scores = hinge(calib_probs, classes, calib_targets).numpy()
        test_non_conf_scores = hinge(test_probs).numpy()
        return calib_non_conf_scores, test_non_conf_scores


class RapsScoring(ScoreFunction):
    def get_scores(
        self,
        calib_probs: torch.Tensor,
        calib_targets: torch.Tensor,
        test_probs: torch.Tensor,
        h_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        calib_non_conf_scores = raps_scores(
            calib_probs, h_params, targets=calib_targets
        )
        test_non_conf_scores = raps_scores(test_probs, h_params)
        return calib_non_conf_scores, test_non_conf_scores


class SapsScoring(ScoreFunction):
    def get_scores(
        self,
        calib_probs: torch.Tensor,
        calib_targets: torch.Tensor,
        test_probs: torch.Tensor,
        h_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        calib_non_conf_scores = saps_scores(
            calib_probs, h_params, targets=calib_targets
        )
        test_non_conf_scores = saps_scores(test_probs, h_params)
        return calib_non_conf_scores, test_non_conf_scores


def get_score_fn(type: ScoreFunctionType) -> ScoreFunction:
    match type:
        case ScoreFunctionType.HINGE:
            return HingeScoring()
        case ScoreFunctionType.RAPS:
            return RapsScoring()
        case ScoreFunctionType.SAPS:
            return SapsScoring()
