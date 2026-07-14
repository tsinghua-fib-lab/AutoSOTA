from dataclasses import dataclass, field
from typing import Literal, Optional, List

@dataclass
class Arguments:
    
    # data related
    data_type: Literal['traj'] = field(default='traj')
    dataset: str = field(default='halfcheetah-medium-expert-v2')
    task: List[int] = field(default_factory=lambda: [1,])
    method: str = field(default='tfg')   

    # diffusion related
    train_steps: int = field(default=1000)
    inference_steps: int = field(default=15)
    eta: float = field(default=0.4)
    clip_x0: bool = field(default=True)
    clip_sample_range: float = field(default=1.0)

    # inference related:
    seed: int = field(default=3)
    device: str = field(default='cuda')
    logging_dir: str = field(default='logs')
    per_sample_batch_size: int = field(default=1)
    num_samples: int = field(default=10)
    batch_id: int = field(default=0)    # start from the zero

    # guidance related
    guidance_name: str = field(default='no')
    guidance_strength: float = field(default=0.0)  # guidance scale for time dependent guidance in baseline
    clip_scale: float = field(default=100)

    # doob guidance
    doob_M: int = field(default=16)
    doob_gamma: float = field(default=1)
    doob_tau: float = field(default=0.5)
    doob_taus: Optional[List[float]] = field(default=None)
    doob_t_star_idx: Optional[int] = field(default=10)
    doob_antithetic_sampling: bool = field(default=True)
    doob_use_reward_gate: bool = field(default=False)
    doob_reward_threshold: float = field(default=200)
    doob_clip_scale: Optional[float] = field(default=None)
    doob_greedy_step: Optional[int] = field(default=None)
    doob_store_best: bool = field(default=False)
    doob_save_rewards: bool = field(default=False)

    
