import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class Default:
    def __init__(self):
        self.crossent = torch.nn.CrossEntropyLoss()
    
    def __call__(self, logits, labels, training: bool):
        return self.crossent(logits.view(-1, logits.size(-1)), labels.view(-1))


class NSL:
    def __init__(self, num_classes):
        self.crossent = torch.nn.CrossEntropyLoss()
        self.s = torch.tensor(math.sqrt(2) * math.log(num_classes - 1), requires_grad=False)

    def __call__(self, logits, labels, training: bool):
        logits = logits * self.s
        return self.crossent(logits.view(-1, logits.size(-1)), labels.view(-1))


class AdaCos(NSL):
    def __call__(self, logits_org, labels_org, training: bool):
        if not training:
            return super().__call__(logits_org, labels_org, training)
        # Filter out ignored (negative) tokens
        logits_org = logits_org.view(-1, logits_org.size(-1))
        labels_org = labels_org.view(-1)
        logits = logits_org[labels_org >= 0]
        labels = labels_org[labels_org >= 0]
        with torch.no_grad():
            one_hot = F.one_hot(labels, num_classes=logits.size(-1))
            max_logit = torch.max(logits)
            assert max_logit < 1.00001, "All logits must be < 1 for AdaCos (each logit must be cosine of an angle)"
            # Will use logsumexp trick
            s_logits = logits * self.s
            s_max_logit = max_logit * self.s
            log_B_avg = torch.where(one_hot < 1, torch.exp(s_logits - s_max_logit), 0.0)
            log_B_avg = torch.log(torch.mean(torch.sum(log_B_avg, dim=-1))) + s_max_logit
            theta = torch.acos(torch.clamp(logits[one_hot == 1], -1.0 + 1e-7, 1.0 - 1e-7))
            theta_med = torch.median(theta)
            self.s = log_B_avg / torch.cos(torch.min(torch.tensor(math.pi/4), theta_med))
            # Clamp to avoid nan
            self.s = torch.clamp(self.s, max=100)
            assert torch.isfinite(self.s).any()

        return super().__call__(logits, labels, training)


def get_loss_fct(string: Optional[str], model):
    string = string.lower() if string is not None else None
    if string is None or string == "default":
        return Default()
    elif string == "nsl":
        return NSL(model.config.vocab.num_classes())
    elif string == "adacos":
        return AdaCos(model.config.vocab.num_classes())