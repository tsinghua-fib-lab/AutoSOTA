from .algorithms.NMF_ENMF import NMF_ENMF
from .algorithms.NMF_AOADMM import NMF_AOADMM
from .algorithms.NMF_HALS import NMF_HALS
from .algorithms.NMF_MUL import NMF_MUL
from .algorithms.NMF_GRADMUL import NMF_GRADMUL
from .algorithms.NMF_ALS import NMF_ALS
from .algorithms.NMFC_ADM import NMFC_ADM
from .algorithms.NMFC_ENMF import NMFC_ENMF
from .algorithms.NMFC_MUL import NMFC_MUL
from .algorithms.NMFC_SCD import NMFC_SCD

# snake_case aliases
nmf_enmf = NMF_ENMF
nmf_aoadmm = NMF_AOADMM
nmf_hals = NMF_HALS
nmf_mul = NMF_MUL
nmf_gradmul = NMF_GRADMUL
nmf_als = NMF_ALS

nmfc_adm = NMFC_ADM
nmfc_enmf = NMFC_ENMF
nmfc_mul = NMFC_MUL
nmfc_scd = NMFC_SCD


__all__ = [
    "NMF_ENMF",
    "NMF_AOADMM",
    "NMF_HALS",
    "NMF_MUL",
    "NMF_GRADMUL",
    "NMF_ALS",
    "NMFC_ADM",
    "NMFC_ENMF",
    "NMFC_MUL",
    "NMFC_SCD",
    "nmf_enmf",
    "nmf_aoadmm",
    "nmf_hals",
    "nmf_mul",
    "nmf_gradmul",
    "nmf_als",
    "nmfc_adm",
    "nmfc_enmf",
    "nmfc_mul",
    "nmfc_scd",
]
