import torch
import math
import json
import os

def fisher_guided_lora_min_from_W(W, Sx, Sy, rank=32, scale=1.0):
    """
    Fisher-guided low-rank approximation without dtype casting or SVD.
    Works in bfloat16 on CUDA.
    """
    WT_W = torch.matmul(W.T, W)
    V_approx = torch.nn.functional.normalize(WT_W, dim=0)
    U_approx = torch.matmul(W, V_approx)
    U_approx = torch.nn.functional.normalize(U_approx, dim=0)

    proj_x = torch.matmul(V_approx.T, Sx @ V_approx)
    proj_y = torch.matmul(U_approx.T, Sy @ U_approx)

    # Joint 2D scoring: captures off-diagonal covariance (IDEA-005)
    align_score = torch.sum(proj_x * proj_y, dim=1)
    # Normalize by weight Frobenius norm to remove scale bias (IDEA-003)
    weight_norm = torch.norm(W, "fro")
    align_score = align_score / (weight_norm + 1e-8)

    _, sel_idx = torch.topk(-align_score, rank)
    V_sel = V_approx[:, sel_idx]
    U_sel = U_approx[:, sel_idx]

    s_vals = torch.sqrt(torch.clamp(align_score[sel_idx], min=1e-8))
    A = (s_vals[:, None] * V_sel.T) * math.sqrt(1/scale)
    B = (U_sel * s_vals[None, :]) * math.sqrt(1/scale)
    W_res = W - (B @ A) * scale

    return A, B, W_res

