import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from matplotlib import pyplot as plt

from .Estimator.CosineHammingParallel import CosineHammingParallel
from .Estimator.Performer import Performer


def unit_hamming_distance_array(size_n):
    if size_n == 1:
        return torch.tensor([0, 1], dtype=torch.long)
    a = unit_hamming_distance_array(size_n - 1)
    return torch.concat([a, torch.flip(a, dims=[0]) + 2 ** (size_n - 1)], 0)


def power_method(A):
    itr_num = 32
    x = torch.randn(A.shape[0], A.shape[1], A.shape[3], device=A.device, dtype=A.dtype)
    x = x / torch.linalg.norm(x, dim=2).unsqueeze(-1)
    for i in range(itr_num):
        y = torch.einsum('bhnm,bhm->bhn', A, x)
        x = y / torch.linalg.norm(y, dim=2).unsqueeze(-1)
    return torch.linalg.norm(y, dim=2)


class Angular_LSH:
    def __init__(self, num_projs, dim, device, dtype):
        self.num_projs = num_projs
        self.proj_dir = torch.randn(dim + (self.num_projs,), dtype=dtype, device=device)
        self.perm = unit_hamming_distance_array(self.num_projs).to(device)

    def Hash(self, Q):
        projected_Q = torch.einsum('bhnd,bhdr -> bhnr', Q, self.proj_dir) > 0
        return self.perm[torch.round(torch.matmul(projected_Q.float(),
                                                  (2. ** torch.arange(self.num_projs, device=Q.device)))).long()]


class CosineHammingAttention(torch.nn.Module):
    def __init__(self, softmax_temp=None, attention_dropout=0.0, m1=512, m2=256, sample_size=800, rep_D=7, num_projs=7,
                 Bucket_size=256, mode="CalcMetrics"):
        super().__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.m1 = m1
        self.m2 = m2
        self.sample_size = sample_size
        self.rep_D = rep_D
        self.num_projs = num_projs
        self.Bucket_size = Bucket_size
        self.mode = mode

    def calc_A_res(self, key, query, Q_sort_idx, value, batch_size, head_size):
        Gram_V = torch.einsum('bhnt,bhnd->bhtd', value, value)
        V_norm = power_method(Gram_V).unsqueeze(2)

        P = torch.linalg.norm(value, dim=3) / V_norm
        P += torch.ones_like(P) / key.shape[2]

        P = P.clip(0)

        P = torch.nn.functional.normalize(P, p=1, dim=2)

        Pflat = P.view(-1, P.shape[2])
        index = Pflat.multinomial(num_samples=self.sample_size, replacement=True)

        num_blocks = key.shape[2] // self.Bucket_size
        bucket_size_query = query.shape[2] // num_blocks

        sampled_set = index.view(batch_size, head_size, -1)

        Offset = torch.zeros_like(sampled_set)
        n = key.shape[2]
        Offset += n * torch.arange(Offset.shape[1], device=query.device).unsqueeze(0).unsqueeze(-1)
        Offset += n * Offset.shape[1] * torch.arange(Offset.shape[0], device=query.device).unsqueeze(-1).unsqueeze(-1)
        sampled_set = (sampled_set + Offset).view(-1)

        block_id = torch.div(index, self.Bucket_size, rounding_mode='floor')  # bh * s
        # Q_sort_idx : b * h * q
        # Q_sort_idx, _ = torch.sort(Q_sort_idx, dim=2)
        bucket_member = Q_sort_idx.view(-1, bucket_size_query)  # b h num_block * q_block

        Offset = torch.zeros_like(block_id)
        Offset += num_blocks * torch.arange(Offset.shape[0], device=query.device).unsqueeze(-1)
        block_sample = (block_id + Offset).view(-1)
        query_sample_collision = bucket_member[block_sample, :]

        Offset = torch.zeros_like(query_sample_collision)

        Offset += query.shape[2] * torch.arange(Offset.shape[0], device=query.device).unsqueeze(-1)
        query_sample_collision_flat = (query_sample_collision + Offset).view(-1)
        
        # if batch_size <= 0 or head_size <= 0 or self.sample_size <= 0 or query.shape[2] <= 0:
        #     import pdb; pdb.set_trace();

        # mask_matrix = torch.ones(batch_size, head_size, self.sample_size, query.shape[2]).view(-1).to(query.device)
        mask_matrix = torch.ones(batch_size * head_size * self.sample_size * query.shape[2]).to(query.device)
        mask_matrix[query_sample_collision_flat] = 0
        # if max(query_sample_collision_flat) >= mask_matrix.numel():
        #     import pdb; pdb.set_trace();

        # try:
        #     mask_matrix[query_sample_collision_flat] = 0
        # except:
        #     import pdb; pdb.set_trace();
        mask_matrix = mask_matrix.view(batch_size, head_size, self.sample_size, query.shape[2])
        mask_matrix = torch.transpose(mask_matrix, 2, 3)

        Vpi = value.view(-1, value.shape[3])
        Vpi = Vpi[sampled_set, :].view(batch_size, head_size, self.sample_size, value.shape[3])

        Kpi = key.view(-1, key.shape[3])
        Kpi = Kpi[sampled_set, :].view(batch_size, head_size, self.sample_size, key.shape[3])

        Ppi = P.view(-1)
        Ppi = Ppi[sampled_set].view(batch_size, head_size, self.sample_size)
        sig = 1.0 / (Ppi * self.sample_size)

        # if torch.any(torch.isnan(sig)) or torch.any(torch.isnan(Kpi))  or torch.any(torch.isinf(Kpi)):
        #     import pdb; pdb.set_trace();

        Api = torch.exp(torch.einsum('bhnd,bhsd->bhns', query, Kpi)) * mask_matrix

        att_res = torch.einsum('bhns,bhsp->bhnp', Api, sig.unsqueeze(-1) * Vpi)
        # if torch.any(torch.isnan(att_res)):
        #     import pdb; pdb.set_trace();
        return att_res

    def select(self, matrix, indices):
        Offset = torch.zeros_like(indices)
        n = matrix.shape[2]
        Offset += n * torch.arange(Offset.shape[1], device=matrix.device).unsqueeze(0).unsqueeze(-1)
        Offset += n * Offset.shape[1] * torch.arange(Offset.shape[0], device=matrix.device).unsqueeze(-1).unsqueeze(-1)
        indices_flat = (indices + Offset).view(-1)

        return torch.index_select(matrix.view(-1, matrix.shape[3]), 0, indices_flat).view(matrix.shape)

    def forward(self, query, key, value):
        query = rearrange(query, 'b t h e -> b h t e').contiguous()
        key = rearrange(key, 'b s h e -> b h s e').contiguous()
        value = rearrange(value, 'b s h d -> b h s d').contiguous()

        proj_shape = (query.shape[0], query.shape[1], query.shape[3])

        lsh = Angular_LSH(self.num_projs, proj_shape, device=query.device, dtype=query.dtype)
        K_hash, K_sort_idx = torch.sort(lsh.Hash(key), dim=2)
        Q_hash, Q_sort_idx = torch.sort(lsh.Hash(query), dim=2)

        value_aug = torch.cat((value, torch.ones(value.shape[0], value.shape[1], value.shape[2], 1).to(value.device)), dim=3)
        att_sparse, _ = CosineHammingParallel(Bucket_size=self.Bucket_size).forward(query=query, key=key,
                                                                                    weight=value_aug,
                                                                                    K_sort_idx=K_sort_idx,
                                                                                    Q_sort_idx=Q_sort_idx)
        batch_size, head_size = query.shape[0], query.shape[1]
        value_sorted = self.select(value_aug, K_sort_idx)
        key_sorted = self.select(key, K_sort_idx)

        if self.sample_size == 0:
            att_res = torch.zeros_like(att_sparse)
        else:
            att_res = self.calc_A_res(key=key_sorted, query=query, Q_sort_idx=Q_sort_idx,
                                      value=value_sorted, batch_size=batch_size, head_size=head_size)

        # if torch.any(torch.isnan(att_sparse)) or torch.any(torch.isnan(att_res)):
        #     import pdb; pdb.set_trace();

        att_final = att_sparse + att_res

        D_tilde = att_final[:, :, :, value_aug.shape[3] - 1]

        est = att_final[:, :, :, :value_aug.shape[3] - 1] / D_tilde.unsqueeze(-1)
        # if torch.any(torch.isnan(att_final[:, :, :, :value_aug.shape[3] - 1])):
        #     import pdb; pdb.set_trace();

        return est


if __name__ == "__main__":
    key = torch.tensor([[1, 2], [1, -2], [1, 0], [0, 1]], dtype=torch.float64)
    key = key.unsqueeze(0).unsqueeze(0)
    query = key
    value = torch.ones_like(key)
    Att_hamming = CosineHammingAttention(rep_D=1, num_projs=7, Bucket_size=2, sample_size=1).forward(
        key=key,
        query=query,
        value=value
    )
