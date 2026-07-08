import torch


class AlphaSampler:

    def __init__(self, mode, pred_alphas, sample_boundary=0.05):
        self._max_alpha = min(0.5, max(pred_alphas) + sample_boundary)
        self._min_alpha = max(0, min(pred_alphas) - sample_boundary)
        self._no_of_pred_alpha = len(pred_alphas)
        if mode == 'uniform':
            self.sample = self.uniform_sample
        else:
            raise ValueError("Ivalid Mode!")

    def uniform_sample(self, alpha_batch):
        return (torch.rand_like(alpha_batch) * (self._max_alpha - self._min_alpha)) + self._min_alpha

    def sample_inference(self, alpha: float, device, no_of_alpha=None):
        if no_of_alpha is None:
            no_of_alpha = self._no_of_pred_alpha
        return self.sample(torch.empty(no_of_alpha, device=device))

