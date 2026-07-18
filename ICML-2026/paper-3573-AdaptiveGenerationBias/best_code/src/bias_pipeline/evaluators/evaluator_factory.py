from typing import TYPE_CHECKING

from src.configs import RUNConfig, JudgeModelConfig

if TYPE_CHECKING:
    from .bias_evaluator import BiasEvaluator

from .comparative_bias_evaluator import ComparativeBiasEvaluator
from .indiv_comparative_bias_evaluator import IndivComparativeBiasEvaluator


def get_evaluator(config: JudgeModelConfig) -> "BiasEvaluator":
    if config.judge_type == "comparative":
        return ComparativeBiasEvaluator(config.judge_model, config.judge_attribute)
    elif config.judge_type == "indiv_comparative":
        return IndivComparativeBiasEvaluator(config.judge_model, config.judge_attribute)
    else:
        raise NotImplementedError
