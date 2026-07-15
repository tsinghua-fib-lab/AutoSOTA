import torch
from torch.nn.modules.loss import _Loss


def get_dice_score(output, target, epsilon=1e-9):
    """Get dice score to evaluate the semantic segmentation model.
    Its value lies between 0 and 1. The more the score is close to 1,
    the more the performance is good.

    Parameters
    ----------
    output : torch.Tensor
        Predicted values

    target : torch.Tensor
        Ground truth.

    epsilon : float
        Small value to avoid zero division error.

    Returns
    -------
    torch.Tensor
        A torch tensor containing the respective dice scores.
    """
    SPATIAL_DIMENSIONS = 2, 3, 4
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score


class BaselineLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """Get dice loss to evaluate the semantic segmentation model.
        Its value lies between 0 and 1. The more the loss is close to 0,
        the more the performance is good.

        Parameters
        ----------
        output : torch.Tensor
            Predicted values

        target : torch.Tensor
            Ground truth.

        Returns
        -------
        torch.Tensor
            A torch tensor containing the respective dice losses.
        """
        return torch.mean(1 - get_dice_score(output, target))


if __name__ == "__main__":
    a = BaselineLoss()
    print(
        a(
            torch.ones((10, 1, 10, 10, 10)),
            (torch.rand((10, 1, 10, 10, 10)) > 0.5).long(),
        )
    )
