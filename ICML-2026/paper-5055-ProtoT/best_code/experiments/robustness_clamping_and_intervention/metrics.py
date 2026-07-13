import torch
import numpy as np
import torch.nn.functional as F

def js_divergence(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    return 0.5*F.kl_div((p+eps).log(), m, reduction="sum") + \
           0.5*F.kl_div((q+eps).log(), m, reduction="sum")

# === Top-k overlap ===
def topk_overlap(p, q, k=10):
    pk = set(torch.topk(p, k).indices.tolist())
    qk = set(torch.topk(q, k).indices.tolist())
    return len(pk & qk) / k


# === Spearman rank correlation over union ===
def spearman_over_union(p, q, k=20):
    pk = torch.topk(p, k).indices.tolist()
    qk = torch.topk(q, k).indices.tolist()
    U = list(set(pk) | set(qk))
    if len(U) < 2:
        return 1.0

    vp, vq = p[U].cpu().numpy(), q[U].cpu().numpy()

    # Compute ranks (higher values → lower ranks → use -v)
    rp = np.argsort(np.argsort(-vp)) + 1
    rq = np.argsort(np.argsort(-vq)) + 1

    rp_m, rq_m = rp.mean(), rq.mean()
    num = ((rp - rp_m) * (rq - rq_m)).sum()
    den = np.sqrt(((rp - rp_m)**2).sum() * ((rq - rq_m)**2).sum())

    return float(num/den) if den > 0 else 0.0