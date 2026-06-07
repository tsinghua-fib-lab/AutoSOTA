# Patched for eval - lazy imports
from .dist import _init_dist_pytorch, get_dist_info, master_only, get_rank, collect_results_cpu, _init_dist_slurm, barrier
from .utils_refcoco import AverageMeter, Summary, intersectionAndUnionGPU

# Optional
try:
    from .refcoco_refer import REFER
except ImportError:
    REFER = None
