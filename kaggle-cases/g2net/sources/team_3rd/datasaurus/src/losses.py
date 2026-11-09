import torch
import torch.nn.functional as F


# https://github.com/iridiumblue/roc-star
# class RocStar(torch.nn.Module):
#     def __init__(self, gamma=0.5):
#         super().__init__()
#         self.g = gamma
#         self.prev_x, self.prev_y = None, None

#     def forward(self, x, y):
#         if self.prev_x is not None and self.prev_x.shape == x.shape:
#             loss = roc_star_loss(
#                 y.view(-1),
#                 torch.sigmoid(x.view(-1)),
#                 self.g,
#                 self.prev_y.view(-1),
#                 torch.sigmoid(self.prev_x.view(-1)),
#             )
#         # else: loss = F.binary_cross_entropy_with_logits(x.view(-1), y.view(-1).float())
#         else:
#             loss = 0
#         loss = loss + 0.1 * F.binary_cross_entropy_with_logits(
#             x.view(-1), 0.95 * y.view(-1).float()
#         )
#         self.prev_x, self.prev_y = torch.clone(x.detach()), torch.clone(y)
#         return loss


def rank_loss_soft(input, target):
    m, s = 5.0, 5.0
    input = input.view(-1).float()
    target = target.view(-1)
    w = (target[:, None] - target[None, :]).view(-1)
    x0 = (input[:, None] - input[None, :]).view(-1)
    x = x0[w.nonzero()]
    w = w[w.nonzero()]
    if len(x) == 0:
        loss = F.binary_cross_entropy_with_logits(
            input.view(-1), target.view(-1).float()
        )
    else:
        loss = torch.where(
            w < 0, -F.softplus(s * x + m) * w, F.softplus(-s * x + m) * w
        ) + 1e-2 * F.smooth_l1_loss(x0, x0)
    return loss.mean()


def rank_loss(input, target):
    input = input.view(-1).float()
    target = target.view(-1)
    p = input[target == 1]
    n = input[target == 0]
    if len(p) == 0:
        p = torch.Tensor([1]).to(input.device)
    if len(n) == 0:
        n = torch.Tensor([-1]).to(input.device)
    x = p[:, None] - n[None, :]
    return -F.logsigmoid(x).mean() + 1e-4 * (input ** 2).mean()
