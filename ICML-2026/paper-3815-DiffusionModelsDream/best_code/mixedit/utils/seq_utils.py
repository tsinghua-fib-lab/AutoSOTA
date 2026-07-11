import torch
from functools import partial


def base_n(x, base, max_length):
    powers = base ** torch.arange(max_length, device=x.device)
    digits = (x.unsqueeze(-1) // powers) % base  # shape: (B, L, max_length)
    return digits.reshape(x.shape[0], -1)


def base_n_inv(x, base, max_length):
    x = x.reshape(x.shape[0], -1, max_length)
    powers = base ** torch.arange(max_length, device=x.device)
    return (x * powers).sum(dim=-1)


def find_bos_and_shift_fn(base, max_length, tokenizer):
    _base_n = partial(base_n, base=base, max_length=max_length)
    _base_n_inv = partial(base_n_inv, base=base, max_length=max_length)
    
    if tokenizer.bos_token_id is not None:
        bos_idx = _base_n(torch.tensor([tokenizer.bos_token_id])).squeeze(0) # Shape: N

        def fn(seq):
            # seq: B x L
            windows = seq.unfold(dimension=1, size=max_length, step=1)
            match = (windows == bos_idx.view(1, 1, -1))
            match_all = match.all(dim=-1)

            # Find the first occurrence of a match in each batch
            idx = torch.argmax(match_all.int(), dim=-1)  # Shape: B

            # If there's no match, set the result to -1
            no_match = ~match_all.any(dim=-1)
            idx[no_match] = -1

            sentences = []
            shifted_samples = []
            for i in range(len(seq)):
                shift_idx = idx[i]
                if shift_idx == -1:
                    shift_idx = 0
                fin = - (max_length - shift_idx % max_length) if shift_idx > 0 else len(seq[i])
                shifted = seq[i][shift_idx:fin]
                shifted = _base_n_inv(shifted.reshape(1, 1, -1)).squeeze(0)
                sentences.append(tokenizer.decode(shifted))
                shifted_samples.append(shifted)
            return sentences, shifted_samples
    else:
        def fn(seq):
            sample = _base_n_inv(seq)
            sentences = tokenizer.batch_decode(sample)
            return sentences, sample.tolist()
        
    return fn