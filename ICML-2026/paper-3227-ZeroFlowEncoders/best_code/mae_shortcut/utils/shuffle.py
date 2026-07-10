import torch
from einops import rearrange


def patchify(imgs, patch_size):
    """
    (B, C, H, W) -> (B, n_patches, patch_dim)
    imgs: (B, C, H, W)
    patch_size: int
    return:
        patches: (B, n_patches, patch_dim)
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0

    patches = rearrange(imgs, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch_size, pw=patch_size) # (B, n_patches, patch_dim); h * w = n_patches, patch_dim = c * ph * pw
    return patches


def unpatchify(patches, img_size, patch_ch=3):
    """
    (B, n_patches, patch_dim) -> (B, C, H, W)
    patches: (B, n_patches, patch_dim)
    img_size: int
    return:
        imgs: (B, C, H, W)
    """
    B, N, P = patches.shape
    H = W = img_size
    p = int((P // patch_ch) ** 0.5)
    h = H // p
    w = W // p
    assert N == h * w

    imgs = rearrange(patches, 'b (h w) (c ph pw) -> b c (h ph) (w pw)', h=h, w=w, ph=p, pw=p) # (B, C, H, W)
    return imgs

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    B, N, D = x.shape  # batch, length, dim
    len_keep = int(N * (1 - mask_ratio))
    
    noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore) # (B, n_patches)

    return x_masked, mask, ids_keep, ids_restore

# --------------------------------------------------------
# MAE components
# --------------------------------------------------------












# --------------------------------------------------------
# ZFE components
# --------------------------------------------------------

def split_patches_with_ids(img, img_p, ids_keep, ids_restore, patch_size):
    """
    img, img_p: (B, C, H, W) img and its independent copy
    ids_keep: (B, N_keep)
    ids_restore: (B, N)
    return:
        X  : masked patches from img
        Y  : visible patches from img
        Xp : masked patches from img_p
        Yp : visible patches from img_p
    """
    patches  = patchify(img, patch_size)    # (B, N, P)
    patches_p = patchify(img_p, patch_size) # (B, N, P)

    B, N, P = patches.shape
    N_keep = ids_keep.shape[1]

    # visible
    Y  = torch.gather(patches,   1, ids_keep.unsqueeze(-1).expand(-1, -1, P))
    Yp = torch.gather(patches_p, 1, ids_keep.unsqueeze(-1).expand(-1, -1, P))

    # masked: use ids_restore to get masked indices
    ids_mask = ids_restore[:, N_keep:]   # (B, N_mask)

    X  = torch.gather(patches,   1, ids_mask.unsqueeze(-1).expand(-1, -1, P))
    Xp = torch.gather(patches_p, 1, ids_mask.unsqueeze(-1).expand(-1, -1, P))

    return X, Y, Xp, Yp