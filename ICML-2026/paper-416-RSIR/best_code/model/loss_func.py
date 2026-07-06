import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score, reduce=True):
        weight = self._cal_weight(neg_score)
        padding_mask = torch.isinf(pos_score)
        pos_loss = F.logsigmoid(pos_score)
        pos_loss.masked_fill_(padding_mask, 0.0)
        if reduce:
            pos_loss = pos_loss.sum() / (~padding_mask).sum()
        else:
            pos_loss = pos_loss / (~padding_mask).sum()
        neg_loss = F.softplus(neg_score) * weight
        neg_loss = neg_loss.sum(-1)
        if pos_score.dim() == neg_score.dim()-1:
            neg_loss.masked_fill_(padding_mask, 0.0)
            if reduce:
                neg_loss = neg_loss.sum() / (~padding_mask).sum()
            else:
                neg_loss = neg_loss / (~padding_mask).sum()
        else:
            neg_loss = torch.mean(neg_loss)
        return -pos_loss + neg_loss

    def _cal_weight(self, neg_score):
        return torch.ones_like(neg_score) / neg_score.size(-1)
    
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score):
        padding_mask = torch.isinf(pos_score)
        loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
        loss.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
        weight = F.softmax(torch.ones_like(neg_score), -1)
        return -(loss * weight).sum(-1).sum() / (~padding_mask).sum()

class SampledSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score, reduce=True):
        if pos_score.dim() == neg_score.dim() - 1:
            pos_score_exp = pos_score.unsqueeze(-1)
        else:
            pos_score_exp = pos_score

        logits = torch.cat([pos_score_exp, neg_score], dim=-1)
        
        labels = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        
        loss = loss.view(labels.shape)
        
        padding_mask = torch.isinf(pos_score)
        loss.masked_fill_(padding_mask, 0.0)
        
        if reduce:
            return loss.sum() / (~padding_mask).sum().clamp(min=1)
        else:
            return loss
       