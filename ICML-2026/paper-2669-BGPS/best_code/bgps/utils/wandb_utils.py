from typing import Dict, Optional
import os
import platform
import wandb

from utils.dist_utils import is_master, broadcast_objects, is_distributed_set, barrier

__all__ = ["set_wandb"]


def set_wandb(cfg: Dict, force_mode: Optional[str] = None) -> Optional[str]:
    """Setup WandB as logging tool.
    :return:        WandB save directory.
    """
    save_dir = cfg["save_dir"]  # root save dir

    wandb_mode = cfg["wandb"]["mode"].lower()
    if force_mode is not None:
        wandb_mode = force_mode.lower()
    if wandb_mode == "disable":  # common mistake
        wandb_mode = "disabled"

    if wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError(f"WandB mode {wandb_mode} invalid.")

    if is_master():  # wandb init only at master
        os.makedirs(save_dir, exist_ok=True)

        wandb_project = cfg["project"]
        wandb_name = cfg["name"]

        wandb_note = cfg["wandb"]["notes"] if ("notes" in cfg["wandb"]) else None
        wandb_id = cfg["wandb"]["id"] if ("id" in cfg["wandb"]) else None
        wandb_group = cfg["wandb"]["group"] if ("group" in cfg["wandb"]) else None
        server_name = platform.node()
        wandb_note = server_name + (f"-{wandb_note}" if (wandb_note is not None) else "")

        wandb.init(
            dir=save_dir,
            config=cfg,
            project=wandb_project,
            name=wandb_name,
            notes=wandb_note,
            mode=wandb_mode,
            resume="allow",
            id=wandb_id,
            group=wandb_group,
        )

        wandb_path = wandb.run.dir if (wandb_mode != "disabled") else save_dir
    else:
        wandb_path = None
    
    # barrier()
    if is_distributed_set():  # sync save path to every thread.
        wandb_path = broadcast_objects([wandb_path], src_rank=0)[0]

    return wandb_path
