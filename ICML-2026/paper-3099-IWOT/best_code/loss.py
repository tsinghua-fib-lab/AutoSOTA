import torch
from torch import nn


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()
        self.loss = MultiClassHingeLoss()

    # Assumes it gets one-hot-labels (all other loss functions do)
    def forward(self, x, y, reduction="mean"):
        return self.loss(x, torch.argmax(y, dim=1), reduction)

    # Assume it gets integer labels
    def forward_int(self, x, y, reduction="mean"):
        return self.loss(x, y, reduction)


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, x, y, reduction="mean"):
        losses = torch.sqrt(torch.sum((x - y) ** 2, dim=1))
        if reduction == "mean":
            return torch.mean(losses)
        return losses


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x, y, reduction="mean"):
        losses = self.loss(x, y)
        if reduction == "mean":
            return torch.mean(losses)
        return losses


# 3rd party library for multi class hinge loss
# Note that we modified it for reduction!
class MultiClassHingeLoss(nn.Module):
    r"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input `x` (a 2D mini-batch `Tensor`) and
    output `y` (which is a 1D tensor of target class indices,
    :math:`0 \leq y \leq \text{x.size}(1)`):

    This implements a Crammer & Singer formulation, which penalizes the maximal margin violation.
    Note that `torch.nn.MultiMarginLoss` uses the Weston & Watkins formulation,
    which penalizes the sum of the margin violations and performs significantly worse in our experience.
    """

    smooth = False

    def __init__(self):
        super(MultiClassHingeLoss, self).__init__()
        self.smooth = False
        self._range = None

    def forward(self, x, y, reduction="mean"):
        aug = self._augmented_scores(x, y)
        xi = self._compute_xi(x, aug, y)

        if reduction == "none":
            loss = torch.sum(aug * xi, dim=1)
        else:  # assuming it's mean
            loss = torch.sum(aug * xi) / x.size(0)
        return loss

    def _augmented_scores(self, s, y):
        if self._range is None:
            delattr(self, "_range")
            self.register_buffer(
                "_range", torch.arange(s.size(1), device=s.device)[None, :]
            )

        delta = torch.ne(y[:, None], self._range).detach().float()
        return s + delta - s.gather(1, y[:, None])

    @torch.autograd.no_grad()
    def _compute_xi(self, s, aug, y):

        # find argmax of augmented scores
        _, y_star = torch.max(aug, 1)
        # xi_max: one-hot encoding of maximal indices
        xi_max = torch.eq(y_star[:, None], self._range).float()

        if MultiClassHingeLoss.smooth:
            # find smooth argmax of scores
            xi_smooth = nn.functional.softmax(s, dim=1)
            # compute for each sample whether it has a positive contribution to the loss
            losses = torch.sum(xi_smooth * aug, 1)
            mask_smooth = torch.ge(losses, 0).float()[:, None]
            # keep only smoothing for positive contributions
            xi = mask_smooth * xi_smooth + (1 - mask_smooth) * xi_max
        else:
            xi = xi_max

        return xi

    def __repr__(self):
        return "MultiClassHingeLoss()"


class set_smoothing_enabled(object):
    def __init__(self, mode):
        """
        Context-manager similar to the torch.set_grad_enabled(mode).
        Within the scope of the manager the MultiClassHingeLoss is smoothed (it is not otherwise).
        """
        MultiClassHingeLoss.smooth = bool(mode)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        MultiClassHingeLoss.smooth = False
