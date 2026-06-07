import torch

def move_masked_to_left_brute_force(tensor, mask):
    results = []
    new_mask = []
    for i in range(tensor.shape[0]):
        t = torch.cat([tensor[i][mask[i]], torch.zeros_like(tensor[i][~mask[i]])])
        results.append(t)
        l = [True] * mask[i].sum() + [False] * (~mask[i]).sum()
        new_mask.append(l)
    results = torch.stack(results)
    new_mask = torch.tensor(new_mask, dtype=torch.bool)
    max_length = mask.sum(dim=1).max()

    results = results[:, :max_length]
    new_mask = new_mask[:, :max_length]
    return results, new_mask


def move_masked_to_left_ids(tensor, mask, pad_zero=True):
    masked_index = mask.cumsum(dim=1) - 1
    unmasked_index = (~mask).cumsum(dim=1) - 1
    unmasked_index += mask.sum(dim=1).unsqueeze(1)
    s2t_index = torch.where(mask, masked_index, unmasked_index)
    t2s_index = torch.argsort(s2t_index, dim=1)
    result = torch.gather(tensor, 1, t2s_index)
    
    length = mask.sum(dim=1)
    result = result[:, :length.max()]
    new_mask = torch.arange(result.shape[1], device=length.device).unsqueeze(0) < length.unsqueeze(1)
    if pad_zero:
        result[~new_mask] = 0
    return result, new_mask


def move_masked_to_left(tensor, mask, pad_zero=True):
    masked_index = mask.cumsum(dim=1) - 1
    unmasked_index = (~mask).cumsum(dim=1) - 1
    unmasked_index += mask.sum(dim=1).unsqueeze(1)
    s2t_index = torch.where(mask, masked_index, unmasked_index)
    t2s_index = torch.argsort(s2t_index, dim=1)
    result = torch.gather(tensor, 1, t2s_index.unsqueeze(2).expand(-1, -1, tensor.shape[2]))
    
    length = mask.sum(dim=1)
    result = result[:, :length.max()]
    new_mask = torch.arange(result.shape[1], device=length.device).unsqueeze(0) < length.unsqueeze(1)
    if pad_zero:
        result[~new_mask] = 0
    return result, new_mask

def get_mask_of_last_masked_index_brute_force(mask, length):
    results = []
    for i in range(mask.shape[0]):
        len = length if isinstance(length, int) else length[i].item()
        l = [False] * mask.shape[1]
        for j in range(mask.shape[1] - 1, -1, -1):
            if mask[i][j] and len > 0:
                l[j] = True
                len -= 1
            else:
                l[j] = False
        results.append(l)
    return torch.tensor(results, dtype=torch.bool)

def get_mask_of_last_masked_index(mask, length):
    cumsum = mask.cumsum(dim=1)
    new_length = mask.sum(dim=1) - length
    last_masked_index = (cumsum > new_length.unsqueeze(1)) & mask
    return last_masked_index


def test_move_masked_to_left():
    b = 10
    n = 20
    tensor = torch.randn(b, n, 5)
    mask = torch.randint(0, 2, (b, n)).bool()
    # print(tensor)
    # print(mask)
    result_1, mask_1 = move_masked_to_left(tensor, mask)
    result_2, mask_2 = move_masked_to_left_brute_force(tensor, mask)
    # assert (result_1[mask_1] == result_2[mask_2]).all()
    assert (result_1 == result_2).all()
    assert (mask_1 == mask_2).all()
    assert (mask.sum(dim=1) == mask_1.sum(dim=1)).all()

    for i in range(b):
        l = mask[i].sum()
        assert (mask_1[i][:l].all() == True)
        assert (mask_1[i][l:].any() == False)

def test_get_mask_of_last_masked_index():
    b = 10
    n = 20
    mask = torch.randint(0, 2, (b, n)).bool()
    length = torch.randint(0, n//2, (b,))
    last_masked_index_1 = get_mask_of_last_masked_index(mask, length)
    last_masked_index_2 = get_mask_of_last_masked_index_brute_force(mask, length)
    assert (last_masked_index_1 == last_masked_index_2).all()

if __name__ == '__main__':
    test_move_masked_to_left()
    test_get_mask_of_last_masked_index()