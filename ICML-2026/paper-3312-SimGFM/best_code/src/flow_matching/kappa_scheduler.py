import torch

from src.flow_matching.time_distorter import TimeDistorter  # reuse distortion specs for training


class KappaScheduler:
    """Computes κ(t) and κ'(t), and samples training t according to train distortion."""

    def __init__(self, train_distortion: str, sample_distortion: str):
        self.train_distortion = train_distortion
        self.sample_distortion = sample_distortion
        self.td = TimeDistorter(
            train_distortion=self.train_distortion,
            sample_distortion=self.sample_distortion,
        )

    def sample_train_t(self, batch_size: int, device: torch.device):
        return self.td.train_ft(batch_size, device)

    def kappa_and_grad(self, t: torch.Tensor):
        return self.td.compute_kappa_and_grad(t, self.sample_distortion)


