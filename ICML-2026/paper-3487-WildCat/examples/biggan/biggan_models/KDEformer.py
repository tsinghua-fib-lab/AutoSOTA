import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class CosineHammingLSH(torch.nn.Module):

    def __init__(self, Bucket_size=64):
        super().__init__()
        self.Bucket_size = Bucket_size

    def select(self, matrix, indices):
        Offset = torch.zeros_like(indices)
        n = matrix.shape[2]
        Offset += n * torch.arange(Offset.shape[1], device=matrix.device).unsqueeze(0).unsqueeze(-1)
        Offset += n * Offset.shape[1] * torch.arange(Offset.shape[0], device=matrix.device).unsqueeze(-1).unsqueeze(-1)
        indices_flat = (indices + Offset).view(-1)
        return torch.index_select(matrix.view(-1, matrix.shape[3]), 0, indices_flat).view(matrix.shape)

    def forward(self, query, key, weight, K_sort_idx, Q_sort_idx, input_attn_mask=None):
        num_blocks = key.shape[2] // self.Bucket_size
        query_Bucket_size = query.shape[2] // num_blocks

        query_sorted = self.select(query, Q_sort_idx)
        key_sorted = self.select(key, K_sort_idx)
        weight_sorted = self.select(weight, K_sort_idx)

        key_split_per_block = key_sorted.view(-1, self.Bucket_size, key.shape[3])
        query_split_per_block = query_sorted.view(-1, query_Bucket_size, query.shape[3])

        weight_split_per_block = weight_sorted.view(-1, self.Bucket_size, weight.shape[3])
        A_sparse = torch.exp(torch.einsum('bnd,bmd->bnm', query_split_per_block, key_split_per_block))

        if input_attn_mask is not None:
            mask_split_per_block = input_attn_mask.view(-1, self.Bucket_size, weight.shape[3]).unsqueeze(0).unsqueeze(0)
            A_sparse *= mask_split_per_block

        result = torch.bmm(A_sparse, weight_split_per_block)
        result = result.view(query.shape[0], query.shape[1], query.shape[2], weight.shape[3])

        Q_sort_idx_new = torch.argsort(Q_sort_idx, dim=2)
        result = self.select(result, Q_sort_idx_new)
        
        return result


def unit_hamming_distance_array(size_n):
    if size_n == 1:
        return torch.tensor([0, 1], dtype=torch.long)
    a = unit_hamming_distance_array(size_n - 1)
    return torch.concat([a, torch.flip(a, dims=[0]) + 2 ** (size_n - 1)], 0)


def power_method(A, itr_num=20, rng=None):
    x = torch.randn(A.shape[0], A.shape[1], A.shape[3], device=A.device, dtype=A.dtype, generator=rng)
    x = x / torch.linalg.norm(x, dim=2).unsqueeze(-1)
    for _ in range(itr_num):
        y = torch.einsum('bhnm,bhm->bhn', A, x)
        x = y / torch.linalg.norm(y, dim=2).unsqueeze(-1)
    return torch.linalg.norm(y, dim=2)


class Angular_LSH(torch.nn.Module):

    def __init__(self, num_projs, dim, device, dtype, rng=None):
        super().__init__()
        self.num_projs = num_projs
        self.proj_dir = torch.randn(dim + (num_projs,), device=device, dtype=dtype, generator=rng)
        self.perm = unit_hamming_distance_array(self.num_projs).to(device)

    def hash(self, mat):
        proj_mat = (torch.einsum('...nd,...dr -> ...nr', mat, self.proj_dir) > 0.).type(mat.dtype)
        enc_vec = 2.** torch.arange(self.num_projs, dtype=mat.dtype, device=mat.device).reshape(1,1,-1)
        bin_ids = torch.einsum("...nr,...r->...n", proj_mat, enc_vec).long()
        return self.perm[bin_ids]


class KDEformer(torch.nn.Module):
    
    def __init__(self, sample_size=800, num_projs=7, Bucket_size=-1, **kwargs):
        super().__init__()
        self.sample_size = sample_size
        self.num_projs = num_projs
        self.Bucket_size = Bucket_size

        if 'mask_size' in kwargs.keys():
            self.mask_matrix = torch.ones(kwargs['mask_size'])
        else:
            self.mask_matrix = None


    def calc_A_res(self, key, query, Q_sort_idx, value, batch_size, head_size):
        Gram_V = torch.einsum('bhnt,bhnd->bhtd', value, value)
        V_norm = power_method(Gram_V, rng=torch.Generator(query.device).manual_seed(1234)).unsqueeze(2)

        P = torch.linalg.norm(value, dim=3) / V_norm
        P += torch.ones_like(P) / key.shape[2]

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
        bucket_member = Q_sort_idx.view(-1, bucket_size_query)  # b h num_block * q_block

        Offset = torch.zeros_like(block_id)
        Offset += num_blocks * torch.arange(Offset.shape[0], device=query.device).unsqueeze(-1)
        block_sample = (block_id + Offset).view(-1)
        query_sample_collision = bucket_member[block_sample, :]

        Offset = torch.zeros_like(query_sample_collision)

        Offset += query.shape[2] * torch.arange(Offset.shape[0], device=query.device).unsqueeze(-1)
        query_sample_collision_flat = (query_sample_collision + Offset).view(-1)
        if self.mask_matrix is None:
            # Note: commented out original implementation
            ###mask_matrix = torch.ones(batch_size, head_size, self.sample_size, query.shape[2]).view(-1).to(query.device)
            mask_matrix = torch.ones(batch_size, head_size, self.sample_size, query.shape[2],device=query.device).view(-1)
        else:
            mask_matrix = self.mask_matrix.to(query.device)
        # mask_matrix = torch.ones(batch_size, head_size, self.sample_size, query.shape[2]).view(-1).to(query.device)
        mask_matrix[query_sample_collision_flat] = 0
        mask_matrix = mask_matrix.view(batch_size, head_size, self.sample_size, query.shape[2])
        mask_matrix = torch.transpose(mask_matrix, 2, 3)

        Vpi = value.view(-1, value.shape[3])
        Vpi = Vpi[sampled_set, :].view(batch_size, head_size, self.sample_size, value.shape[3])

        Kpi = key.view(-1, key.shape[3])
        Kpi = Kpi[sampled_set, :].view(batch_size, head_size, self.sample_size, key.shape[3])

        Ppi = P.view(-1)
        Ppi = Ppi[sampled_set].view(batch_size, head_size, self.sample_size)
        sig = 1.0 / (Ppi * self.sample_size)

        Api = torch.exp(torch.einsum('bhnd,bhsd->bhns', query, Kpi)) * mask_matrix

        att_res = torch.einsum('bhns,bhsp->bhnp', Api, sig.unsqueeze(-1) * Vpi)

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
        lsh = Angular_LSH(self.num_projs, proj_shape, device=query.device, dtype=query.dtype, rng=torch.Generator(query.device).manual_seed(1234))
        _, K_sort_idx = torch.sort(lsh.hash(key), dim=2)
        _, Q_sort_idx = torch.sort(lsh.hash(query), dim=2)

        # Note: commented out original implementation
        ###value_aug = torch.cat((value, torch.ones(value.shape[0], value.shape[1], value.shape[2], 1).to(value.device)), dim=3)
        value_aug = torch.cat((value, torch.ones(value.shape[0], value.shape[1], value.shape[2], 1, device=value.device)), dim=3)
        att_sparse = CosineHammingLSH(Bucket_size=self.Bucket_size)(query=query, key=key, weight=value_aug, K_sort_idx=K_sort_idx, Q_sort_idx=Q_sort_idx)

        batch_size, head_size = query.shape[0], query.shape[1]
        value_sorted = self.select(value_aug, K_sort_idx)
        key_sorted = self.select(key, K_sort_idx)

        if self.sample_size == 0:
            att_res = torch.zeros_like(att_sparse)
        else:
            att_res = self.calc_A_res(key=key_sorted, query=query, Q_sort_idx=Q_sort_idx, value=value_sorted, batch_size=batch_size, head_size=head_size)

        att_final = att_sparse + att_res
        D_tilde = att_final[:, :, :, value_aug.shape[3] - 1]
        est = att_final[:, :, :, :value_aug.shape[3] - 1] / D_tilde.unsqueeze(-1)
        return est

