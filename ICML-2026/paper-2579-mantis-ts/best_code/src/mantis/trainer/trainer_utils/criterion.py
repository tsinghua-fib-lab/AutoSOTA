import torch

from torch import nn


class ContrastiveLoss(nn.Module):
    """
    A contrastive loss used for pre-training.

    Parameters
    ----------
    temperature: float, default=0.1
        Temperature scaling parameter used to regulate the sharpness of the softmax operator.
    device: {'cpu', 'cuda'}, default='cuda'
        On which device the model is located.
    """
    def __init__(self, temperature=0.1, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.device = device
    
    def forward(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum('nc,ck->nk', [q, k.t()])
        logits /= self.temperature
        labels = torch.arange(q.shape[0], dtype=torch.long).to(self.device)
        return nn.CrossEntropyLoss()(logits, labels)
