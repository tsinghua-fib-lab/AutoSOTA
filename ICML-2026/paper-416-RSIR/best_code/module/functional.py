import torch


def seq_pooling_function(batch_seq_embeddings: torch.Tensor, seq_len: torch.Tensor, weight=None, mask_token=None, pooling_type='mean', keepdim=False):
    # batch_seq_embeddings: [B, L, D] or [B, Neg, L, D]
    # seq_len: [B] or [B,Neg], weight: [B,L] or [B,Neg,L]
    B = batch_seq_embeddings.size(0)
    _need_reshape = False
    if batch_seq_embeddings.dim() == 4:
        _need_reshape = True
        batch_seq_embeddings = batch_seq_embeddings.view(
            -1, *batch_seq_embeddings.shape[2:])
        seq_len = seq_len.view(-1)
        if weight is not None:
            weight = weight.view(-1, weight.size(-1))

    N, L, D = batch_seq_embeddings.shape

    if weight is not None:
        batch_seq_embeddings = weight.unsqueeze(-1) * batch_seq_embeddings
    
    if pooling_type == 'mask':
        # Data type of mask_token should be bool and 
        # the shape of mask_token should be [B, L]
        assert mask_token != None, "mask_token can be None when pooling_type is 'mask'."
        result = batch_seq_embeddings[mask_token]

    elif pooling_type in ['origin', 'concat', 'mean', 'sum', 'max']:
        mask = torch.arange(L).unsqueeze(0).unsqueeze(2).to(batch_seq_embeddings.device)
        mask = mask.expand(N, -1,  D)
        seq_len = seq_len.unsqueeze(1).unsqueeze(2)
        seq_len_ = seq_len.expand(-1, mask.size(1), -1)
        mask = mask >= seq_len_

        batch_seq_embeddings = batch_seq_embeddings.masked_fill(mask, 0.0)

        if pooling_type == 'origin':
            return batch_seq_embeddings
        elif pooling_type in ['origin', 'concat', 'max']:
            if not keepdim: 
                if pooling_type == 'concat':
                    result = batch_seq_embeddings.reshape(N, -1)
                else:
                    result = batch_seq_embeddings.max(dim=1)
            else:
                if pooling_type == 'concat':
                    result = batch_seq_embeddings.reshape(N, -1).unsqueeze(1)
                else:
                    result = batch_seq_embeddings.max(dim=1).unsqueeze(1)
        elif pooling_type in ['mean', 'sum']:
            batch_seq_embeddings_sum = batch_seq_embeddings.sum(dim=1, keepdim=keepdim)
            if pooling_type == 'sum':
                result = batch_seq_embeddings_sum
            else:
                result = batch_seq_embeddings_sum / (seq_len + torch.finfo(torch.float32).eps if keepdim else seq_len.squeeze(2))

    elif pooling_type == 'last':
        gather_index = (seq_len-1).view(-1, 1, 1).expand(-1, -1, D)  # B x 1 x D
        output = batch_seq_embeddings.gather(
            dim=1, index=gather_index).squeeze(1)  # B x D
        result = output if not keepdim else output.unsqueeze(1)

    if _need_reshape:
        return result.reshape(B, N//B, *result.shape[1:])
    else:
        return result