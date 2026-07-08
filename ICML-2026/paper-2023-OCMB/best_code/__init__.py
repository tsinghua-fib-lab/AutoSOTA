"""
OCMB: Ordering-Constrained Markov Blanket Discovery

纯净独立版本 - 包含CAPS和SciNO两个backbone

使用示例:
    from OCMB.ocmb import run_ocmb, OCMB_CaPS, OCMB_SciNO

    # 使用便捷函数
    graph, ocmb = run_ocmb(X, backbone='caps', verbose=True)

    # 或直接使用类
    ocmb = OCMB_CaPS(max_parents=10, k_mb=5, alpha_mb=0.01)
    ocmb.fit(X)
    graph = ocmb.get_adjacency_matrix()
"""

from .ocmb import (
    # 基类
    OCMB_Base,
    # Oracle变体
    OCMB_Oracle,
    OCMB_Random,
    OCMB_OracleOrderOnly,
    OCMB_OracleCandPa,
    OCMB_OracleCandNei,
    OCMB_Precomputed,
    # 实用backbone
    OCMB_SciNO,
    OCMB_CaPS,
    OCMB_CaPS_LGBMScore,
    # 便捷函数
    run_ocmb,
    # 工具函数
    get_topological_order_from_dag,
    get_random_order,
    order_divergence,
    order_kendall_tau,
    calculate_metrics,
)

__version__ = '1.0.0'
__all__ = [
    'OCMB_Base',
    'OCMB_Oracle',
    'OCMB_Random',
    'OCMB_OracleOrderOnly',
    'OCMB_OracleCandPa',
    'OCMB_OracleCandNei',
    'OCMB_Precomputed',
    'OCMB_SciNO',
    'OCMB_CaPS',
    'OCMB_CaPS_LGBMScore',
    'run_ocmb',
    'get_topological_order_from_dag',
    'get_random_order',
    'order_divergence',
    'order_kendall_tau',
    'calculate_metrics',
]
