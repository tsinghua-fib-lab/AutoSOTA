"""AttnLRP baseline API."""

from flashtrace.attribution import AttnLRPSpanAggregate, LLMLRPAttribution, MultiHopAttnLRPResult
from flashtrace.lrp_patches import detect_model_type, lrp_context

__all__ = [
    "AttnLRPSpanAggregate",
    "LLMLRPAttribution",
    "MultiHopAttnLRPResult",
    "detect_model_type",
    "lrp_context",
]
