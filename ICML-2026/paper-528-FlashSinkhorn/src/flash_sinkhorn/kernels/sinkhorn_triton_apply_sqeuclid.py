"""Backward-compatible re-exports for apply kernels.

After Phase C reorganization, the actual implementations live in:
- apply_ott.py: OTT-convention kernels (mat5, deprecated vec/mat wrappers)
- apply_flash.py: FlashStyle kernels (vec/mat with shifted potentials)

All existing imports from this module continue to work.
"""
from flash_sinkhorn.kernels.apply_ott import (  # noqa: F401
    apply_plan_vec_sqeuclid,
    apply_plan_mat_sqeuclid,
    mat5_sqeuclid,
)
from flash_sinkhorn.kernels.apply_flash import (  # noqa: F401
    apply_plan_mat_flashstyle,
    apply_plan_vec_flashstyle,
)
