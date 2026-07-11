# ----------------------- Config -----------------------
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

@dataclass
class CFG:
    N: int = 2048
    steps: int = 5000
    step_size: float = 1e-3
    sigma: float = 1.0
    sigma_start: float = 0.01   # starting sigma for cosine schedule
    sigma_end: float = 0.001    # ending sigma for cosine schedule
    noise_schedule: str = "cosine"  # "cosine" or "fixed" 
    zeta : float = 1e-2
    seed: int = 0
    kernel: str = "sobolev"  # "sobolev" or "gaussian"
    g : int = 0  # parameter for KT thinning
    bandwidth: float = 1.0  # for Gaussian kernel
    # Return full trajectory if True (memory heavy): (steps+1, N, d)
    return_path: bool = False
    kt_function: str = "compresspp_kt"
    skip_swap: bool = True
    alpha: float = 0.0  # SGHMC momentum decay
    nesterov: bool = False  # Nesterov accelerated gradient (look-ahead) (0.0 = no momentum, 0.9 = default SGHMC)

