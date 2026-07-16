# =================================================================
# tpr.py
# Description: Shared helpers for TPR-at-fixed-FPR (detectability)
#              evaluation. Centralizes human-text-results caching and
#              the DynamicThresholdSuccessRateCalculator save logic.
# =================================================================

import os
import json

from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.pipelines.detection import UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType


def load_or_compute_human_text_results(my_watermark, human_text_dataset, human_text_result_save_dir, algorithm):
    """Load cached human-text (negative-distribution) detection scores, or compute and cache them.

    If `{human_text_result_save_dir}/{algorithm}.json` exists, load and return it.
    Otherwise compute the scores via UnWatermarkedTextDetectionPipeline, save (indent=4),
    and return them.
    """
    os.makedirs(human_text_result_save_dir, exist_ok=True)

    if os.path.exists(os.path.join(human_text_result_save_dir, f"{algorithm}.json")):
        print(f"\ncache exists for human-text results")
        with open(os.path.join(human_text_result_save_dir, f"{algorithm}.json"), 'r') as f:
            human_text_results = json.load(f)
    else:
        return_type = DetectionPipelineReturnType.SCORES

        human_text_pipline = UnWatermarkedTextDetectionPipeline(dataset=human_text_dataset, text_editor_list=[],
                                            show_progress=True, return_type=return_type)
        human_text_results = human_text_pipline.evaluate(my_watermark)
        with open(os.path.join(human_text_result_save_dir, f"{algorithm}.json"), 'w') as f:
            json.dump(human_text_results, f, indent=4)

    return human_text_results


DETECTABILITY_KEY = "detectability"


def evaluate_detectability(watermarked_scores, human_text_results, labels,
                           rules=("target_fpr", "best"), target_fprs=(0.01, 0.1)):
    """Compute every requested detectability metric for one attack in a single pass.

    The ``target_fpr`` rule is swept over each value in ``target_fprs`` (one TPR-at-fixed-FPR
    result per FPR); the ``best`` rule yields a single best-threshold F1 result. Each metric is
    computed with DynamicThresholdSuccessRateCalculator and holds every requested label (e.g.
    TPR and F1). Returns an ordered ``{metric_name: metric_dict}`` mapping, e.g.
    ``{"tpr_target_fpr_0.01": {...}, "tpr_target_fpr_0.1": {...}, "f1_best": {...}}``.
    Pure computation — the caller decides where the mapping is stored.
    """
    results = {}
    for rule in rules:
        sweep = target_fprs if rule == "target_fpr" else (None,)
        for target_fpr in sweep:
            calculator = DynamicThresholdSuccessRateCalculator(labels=labels, rule=rule, target_fpr=target_fpr)
            name = "f1_best" if rule == "best" else f"tpr_{rule}_{target_fpr}"
            results[name] = calculator.calculate(watermarked_scores, human_text_results)

    return results


def attach_detectability(attack_results, detectability):
    """Return `attack_results` (the list of per-sample records + summary dicts an attack saves)
    with the detectability mapping embedded as a single ``{"detectability": {...}}`` entry,
    replacing any existing one so re-sweeps stay idempotent.
    """
    cleaned = [r for r in attack_results
               if not (isinstance(r, dict) and set(r) == {DETECTABILITY_KEY})]
    return cleaned + [{DETECTABILITY_KEY: detectability}]
