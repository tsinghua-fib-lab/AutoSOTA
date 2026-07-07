import logging
from typing import Any

from transformers import PreTrainedModel

from entquant.eval.evaluator import ModelEvaluator
from entquant.model.entquant_model import EntQuantModel
from entquant.utils import clear_cache
from run.hydra_zen import register_workflow

logger = logging.getLogger(__name__)


@register_workflow("eval")
def evaluate_model(
    model: PreTrainedModel | EntQuantModel,
    evaluator: ModelEvaluator = "${cfg.eval}",
) -> dict[str, Any]:
    if isinstance(model, EntQuantModel):
        model = model.model
    results = evaluator(model)
    clear_cache()
    return results
