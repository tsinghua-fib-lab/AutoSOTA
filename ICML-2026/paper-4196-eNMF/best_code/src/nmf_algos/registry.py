"""Algorithm registry for NMF and NMFC methods."""

from nmf_algos.algorithms.NMF_ADMM import NMF_ADMM
from nmf_algos.algorithms.NMF_ALS import NMF_ALS
from nmf_algos.algorithms.NMF_AOADMM import NMF_AOADMM
from nmf_algos.algorithms.NMF_ENMF import NMF_ENMF
from nmf_algos.algorithms.NMF_GRADMUL import NMF_GRADMUL
from nmf_algos.algorithms.NMF_HALS import NMF_HALS
from nmf_algos.algorithms.NMF_MUL import NMF_MUL
from nmf_algos.algorithms.NMFC_ADM import NMFC_ADM
from nmf_algos.algorithms.NMFC_ENMF import NMFC_ENMF
from nmf_algos.algorithms.NMFC_MUL import NMFC_MUL
from nmf_algos.algorithms.NMFC_SCD import NMFC_SCD

NMF_METHOD_REGISTRY = {
    "ENMF": NMF_ENMF,
    "HALS": NMF_HALS,
    "AOADMM": NMF_AOADMM,
    "MUL": NMF_MUL,
    "GRADMUL": NMF_GRADMUL,
    "ALS": NMF_ALS,
    "ADMM": NMF_ADMM,
}


NMFC_METHOD_REGISTRY = {
    "ENMFC": NMFC_ENMF,
    "ADM": NMFC_ADM,
    "MUL": NMFC_MUL,
    "SCD": NMFC_SCD,
}


METHOD_REGISTRY = {
    **NMF_METHOD_REGISTRY,
    **{f"NMFC_{name}": cls for name, cls in NMFC_METHOD_REGISTRY.items()},
}


def get_algorithm_class(method_name):
    """Return the algorithm class for a method name."""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unsupported method: {method_name}. "
            f"Available methods: {list(METHOD_REGISTRY)}"
        )
    return METHOD_REGISTRY[method_name]
