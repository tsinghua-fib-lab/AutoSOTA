import torch
from einops import rearrange
from torch.nn.functional import normalize


class CosineHammingParallel:
    def __init__(self, Bucket_size=64):
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
        
        query_split_per_block = query_sorted.view(-1, query_Bucket_size,
                                                  query.shape[3])
        weight_split_per_block = weight_sorted.view(-1, self.Bucket_size,
                                                    weight.shape[3])

        A_sparse = torch.exp(torch.einsum('bnd,bmd->bnm', query_split_per_block, key_split_per_block))

        if input_attn_mask is not None:
            mask_split_per_block = input_attn_mask.view(-1, self.Bucket_size,
                                             weight.shape[3]).unsqueeze(0).unsqueeze(0)
            A_sparse *= mask_split_per_block

        result = torch.bmm(A_sparse, weight_split_per_block)
        result = result.view(query.shape[0], query.shape[1], query.shape[2], weight.shape[3])

        # if result.max() == torch.inf:
        #     import pdb; pdb.set_trace();

        Q_sort_idx_new = torch.argsort(Q_sort_idx, dim=2)
        result = self.select(result, Q_sort_idx_new)

        sample_complexity = query.shape[2] * self.Bucket_size

        return result, sample_complexity


if __name__ == "__main__":
    print("test hamming")
    key = torch.tensor([[1, -1, 2], [-1, 1, 2]], dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    query = torch.tensor([[1, -1, 0], [-1, 1, 2]], dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    value = torch.tensor([[1, 1, 1, 1], [-1, 1, 2, 1]], dtype=torch.float64).unsqueeze(0).unsqueeze(0)

    Q_sorted = torch.tensor([0, 1]).unsqueeze(0).unsqueeze(0)
    K_sorted = torch.tensor([0, 1]).unsqueeze(0).unsqueeze(0)

    print(Q_sorted)
    D_exact = torch.sum(torch.exp(torch.einsum('bhnd,bhmd->bhnm', query, key)), dim=3)

    csh = CosineHammingParallel(Bucket_size=1)
    x, _ = csh.forward(query, key, value, K_sorted, Q_sorted)
    D_tilde = x[:, :, :, 2]
    print("D hamming err : ", torch.mean(torch.abs(D_exact - D_tilde) / D_exact))
