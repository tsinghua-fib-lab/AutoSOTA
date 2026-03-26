import torch

from rl4co.envs import RL4COEnvBase


def get_select_start_nodes_fn(select_start_nodes, **kwargs):
    if select_start_nodes == "random":
        return RandomStartNodes(**kwargs)
    elif select_start_nodes == "all":  # default from POMO
        return AllSelectStartNodes(**kwargs)
    else:
        raise ValueError(f"Unknown select_start_nodes: {select_start_nodes}")


class SelectStartNodes:
    def __init__(self, num_starts=None):
        self.num_starts = num_starts

    def __call__(self, td, num_starts, backup_n_starts, **kwargs):
        if self.num_starts is not None:
            num_starts = self.num_starts
        if isinstance(
            num_starts, RL4COEnvBase
        ):  # to avoid current RL4CO problematic API, to fix
            num_starts = backup_n_starts
        return self._select(td, num_starts)

    def _select(self, td, num_starts):
        raise NotImplementedError("Implement this method in a subclass")

    def get_num_starts(self, td):
        return (
            self.num_starts if self.num_starts is not None else td["locs"].shape[-2] - 1
        )  # exclude depot


class RandomStartNodes(SelectStartNodes):
    def _select(self, td, num_starts):
        return torch.randint(0, td["locs"].size(1), (td["locs"].size(0), num_starts))


class AllSelectStartNodes(SelectStartNodes):
    def _select(self, td, num_starts):
        num_loc = td["locs"].shape[-2] - 1
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
            + 1
        )
        return selected
