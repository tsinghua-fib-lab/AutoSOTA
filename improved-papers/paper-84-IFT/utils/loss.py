from environment import *


class MSELoss(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

    def forward(self, pred, true):
        i = -1 if self.CFG.mode == 'MS' else 0
        pred, true = pred[:, :, i:], true[:, :, i:]
        diff = (pred - true) ** 2
        coef = torch.arange(1, self.CFG.pred_len + 1, dtype=torch.float, device=true.device).unsqueeze(1) ** (-0.5)
        coef = coef if self.CFG.decay else 1.0
        if self.CFG.reduction == 'mean':
            loss = torch.mean(coef * diff)
        elif self.CFG.reduction == 'sum':
            loss = torch.sum(coef * diff)
        elif self.CFG.reduction == 'mix':
            loss = torch.mean(torch.sum(coef * diff, dim=1))
        else:
            assert False
        return loss


class MAELoss(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

    def forward(self, pred, true):
        i = -1 if self.CFG.mode == 'MS' else 0
        pred, true = pred[:, :, i:], true[:, :, i:]
        diff = torch.abs(pred - true)
        coef = torch.arange(1, self.CFG.pred_len + 1, dtype=torch.float, device=true.device).unsqueeze(1) ** (-0.5)
        coef = coef if self.CFG.decay else 1.0
        if self.CFG.reduction == 'mean':
            loss = torch.mean(coef * diff)
        elif self.CFG.reduction == 'sum':
            loss = torch.sum(coef * diff)
        elif self.CFG.reduction == 'mix':
            loss = torch.mean(torch.sum(coef * diff, dim=1))
        else:
            assert False
        return loss
