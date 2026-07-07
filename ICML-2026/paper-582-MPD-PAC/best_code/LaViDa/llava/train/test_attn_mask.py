from torch.nn.attention.flex_attention import flex_attention,create_block_mask
import torch
prefix_length = torch.tensor([2,2,3,3], dtype=torch.int32, device="cuda")
_seq_len = 128
past_len = 3
_bsz = len(prefix_length)
def prefix_lm_dllm(b, h, q_idx, kv_idx):
    #return kv_idx >= prefix_length[b]
    q_idx_2 = q_idx + past_len
    return (kv_idx >= prefix_length[b]) & (q_idx < prefix_length[b])
block_mask = create_block_mask(prefix_lm_dllm, B=_bsz, H=None, Q_LEN=_seq_len, KV_LEN=_seq_len+past_len)
            
            