import torch
import torch.nn as nn
import torch.nn.functional as F

def regularized_loss(model, n_samples, logits, y_onehot, p=4, lam=0.01):
    """
    Cross-entropy + multi-class inter-class regulariser.
    """
    # classification loss: cross-entropy on logits
    cls = -torch.mean((y_onehot * torch.log_softmax(logits, dim=1)).sum(dim=1))
    
    # inter-class regulariser: ||w_k - w_l||_2^p summed
    W = model.linear.weight            # shape [C, d]
    diffs = W.unsqueeze(1) - W.unsqueeze(0)  # [C, C, d]
    pair_norms = torch.norm(diffs, p=2, dim=2)  # [C, C]
    rg = pair_norms.pow(p).sum() / n_samples
    
    return cls + lam * rg

def smooth_hinge(delta: torch.Tensor, h: float = 0.1) -> torch.Tensor:
    """
    Huber–style smoothed hinge loss:
      loss = |δ|              if δ > h
           = (δ²)/(4h)+δ/2+(h/4)  if |δ| ≤ h
           = 0                if δ < -h
    """
    loss = torch.zeros_like(delta)
    big   = delta >  h
    small = delta < -h
    mid   = ~(big | small)

    # δ >  h: loss=δ
    loss[big] = delta[big]
    # |δ| ≤ h: quadratic region
    md = delta[mid]
    loss[mid] = md**2/(4*h) + md/2 + h/4
    # δ < -h: loss stays 0
    return loss
    
def rbf_feature_map(X, X_test, n_components=100, gamma=1.0):
    sampler = RBFSampler(gamma=gamma,
                    n_components=n_components,
                    random_state=42)
    return sampler.fit_transform(X), sampler.transform(X_test)


class SVM_Linear(nn.Module):
    def forward(self, x, y, C=1.0):
        margin = y * self.fc(x).flatten()
        delta  = 1.0 - margin
        hinge  = smooth_hinge(delta, self.h)
        reg    = torch.norm(self.fc.weight, p=2)
        return reg + C * hinge


class M3SVM(nn.Module):
    """
    Multi-class SVM as a single linear layer.
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        # returns raw scores (logits)
        return self.linear(x)



