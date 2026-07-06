"""
Zero-cost proxy evaluators for Neural Architecture Search.

Ported from VLDB_code/TRAILS phase-1 evaluators.
All evaluators share the same interface:
    evaluator.evaluate(arch, device, batch_data, batch_labels, space_name) -> float

For pTProxy, use ptproxy.ptproxy_score() directly.
"""

from proxies.alg_base import Evaluator
from proxies.grad_norm import GradNormEvaluator
from proxies.grad_plain import GradPlainEvaluator
from proxies.jacob_cov import JacobConvEvaluator
from proxies.knas import KNASEvaluator
from proxies.nas_wot import NWTEvaluator
from proxies.ntk_condition_num import NTKCondNumEvaluator
from proxies.ntk_trace import NTKTraceEvaluator
from proxies.ntk_trace_approx import NTKTraceApproxEvaluator
from proxies.prune_fisher import FisherEvaluator
from proxies.prune_grasp import GraspEvaluator
from proxies.prune_snip import SnipEvaluator
from proxies.prune_synflow import SynFlowEvaluator
from proxies.weight_norm import WeightNormEvaluator
from proxies.ptproxy import ptproxy_score

from common.constant import Config

# Mapping: proxy name -> evaluator class
PROXY_EVALUATORS = {
    Config.GRAD_NORM: GradNormEvaluator,
    Config.GRAD_PLAIN: GradPlainEvaluator,
    Config.JACOB_COV: JacobConvEvaluator,
    Config.KNAS: KNASEvaluator,
    Config.NAS_WOT: NWTEvaluator,
    Config.NTK_COND_NUM: NTKCondNumEvaluator,
    Config.NTK_TRACE: NTKTraceEvaluator,
    Config.NTK_TRACE_APPROX: NTKTraceApproxEvaluator,
    Config.PRUNE_FISHER: FisherEvaluator,
    Config.PRUNE_GRASP: GraspEvaluator,
    Config.PRUNE_SNIP: SnipEvaluator,
    Config.SYNFLOW: SynFlowEvaluator,
    Config.WEIGHT_NORM: WeightNormEvaluator,
}


def get_evaluator(proxy_name: str):
    """Get an evaluator instance by proxy name."""
    if proxy_name not in PROXY_EVALUATORS:
        raise ValueError(f"Unknown proxy: {proxy_name}. "
                         f"Available: {list(PROXY_EVALUATORS.keys())}")
    return PROXY_EVALUATORS[proxy_name]()


__all__ = [
    "Evaluator",
    "GradNormEvaluator",
    "GradPlainEvaluator",
    "JacobConvEvaluator",
    "KNASEvaluator",
    "NWTEvaluator",
    "NTKCondNumEvaluator",
    "NTKTraceEvaluator",
    "NTKTraceApproxEvaluator",
    "FisherEvaluator",
    "GraspEvaluator",
    "SnipEvaluator",
    "SynFlowEvaluator",
    "WeightNormEvaluator",
    "ptproxy_score",
    "PROXY_EVALUATORS",
    "get_evaluator",
]
