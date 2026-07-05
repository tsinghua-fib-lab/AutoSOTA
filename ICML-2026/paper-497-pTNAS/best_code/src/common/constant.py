"""
Constants ported from VLDB_code/TRAILS common/constant.py.
Only the subset needed by pTNAS proxy evaluators.
"""


class Config:
    # Search space identifiers
    MLPSP = "mlp_sp"

    # Proxy / evaluator names (used as keys in score JSON files)
    PTPROXY = "pTProxy"
    LEGACY_EXPRESS_FLOW = "express_flow"
    # Backward-compatible alias for older scripts.
    ExpressFlow = PTPROXY
    GRAD_NORM = "grad_norm"
    GRAD_PLAIN = "grad_plain"
    NAS_WOT = "nas_wot"
    JACOB_COV = "jacob_cov"
    KNAS = "knas"
    SYNFLOW = "synflow"
    PRUNE_FISHER = "fisher"
    PRUNE_GRASP = "grasp"
    PRUNE_SNIP = "snip"
    NTK_COND_NUM = "ntk_cond_num"
    NTK_TRACE = "ntk_trace"
    NTK_TRACE_APPROX = "ntk_trace_approx"
    WEIGHT_NORM = "weight_norm"
