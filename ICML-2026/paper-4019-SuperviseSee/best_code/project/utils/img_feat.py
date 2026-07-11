import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans

def split_image(image, tile_size=128, stride=64):
    w, h = image.size
    patches = []
    for top in range(0, h - tile_size + 1, stride):
        for left in range(0, w - tile_size + 1, stride):
            box = (left, top, left + tile_size, top + tile_size)
            patches.append(image.crop(box))
    return patches

def stitch_masks(masks, patch_size, stride):
    device = masks[0].device
    C, ph, pw = masks[0].shape
    N = len(masks)
    n = int(N ** 0.5)
    assert n * n == N, "len(masks) must be a perfect square"

    ratio = stride / patch_size
    new_stride = int(ratio * ph)

    H = (n - 1) * new_stride + ph
    W = (n - 1) * new_stride + ph

    vote_map  = torch.zeros((C, H, W), device=device)
    count_map = torch.zeros((C, H, W), device=device)

    for idx, mask in enumerate(masks):
        i = idx // n
        j = idx %  n
        top  = i * new_stride
        left = j * new_stride

        vote_map[:, top:top+ph, left:left+pw]  += mask
        count_map[:, top:top+ph, left:left+pw] += 1.0

    # avoid division by zero
    count_map = torch.clamp(count_map, min=1.0)
    soft_map = vote_map / count_map

    return soft_map


def extract_feat(img, model, transform, tile_size, stride, device):
    patches = split_image(img, tile_size, stride)
    features = []
    # prefix tokens
    n_prefix = getattr(model, "num_prefix_tokens", 1)

    for patch in patches:
        t = transform(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.forward_features(t)
        feat = F.normalize(out[:, n_prefix:, :], dim=-1)[0]
        num_patches = feat.shape[0]  # 196
        dim = feat.shape[1]          # 1024
        grid_size = int(num_patches ** 0.5)  # 14
        feat = feat.permute(1, 0).reshape(dim, grid_size, grid_size) # [1024, 14, 14]
        features.append(feat)

    feature_map = stitch_masks(features, tile_size, stride)  # [1024, 112, 112]
    return feature_map

def cal_sim(feature_map, mask_fg, mask_bg, K):
    """
    feature_map: [D, Hf, Wf]
    binary_mask: [H_orig, W_orig], values in {0,1}

    returns: refined_heatmap [2, Hf, Wf]  # [background, foreground]
    """

    D, Hf, Wf = feature_map.shape
    fg = torch.from_numpy(mask_fg).unsqueeze(0).unsqueeze(0).float()  # [1,1,Href_orig,Wref_orig]
    bg = torch.from_numpy(mask_bg).unsqueeze(0).unsqueeze(0).float()  # [1,1,Href_orig,Wref_orig]
    
    fg = F.interpolate(fg.float(), size=(Hf, Wf), mode='nearest').long().squeeze(0).squeeze(0)
    bg = F.interpolate(bg.float(), size=(Hf, Wf), mode='nearest').long().squeeze(0).squeeze(0)
    
    features = feature_map.view(D, -1).permute(1,0).contiguous()  # [M, D], M = Hf*Wf
    
    prototypes = []
    for mask in [bg, fg]:  # 0=background,1=foreground
        idx = torch.nonzero(mask.view(-1)==1, as_tuple=True)[0]
        feats_cls = features[idx]  # [Nc, D]

        if feats_cls.size(0) < K:
            mean = feats_cls.mean(dim=0, keepdim=True)
            centers = mean.repeat(K,1)  # [K, D]
        else:
            _, centers = kmeans(
                X=feats_cls, num_clusters=K, distance='cosine', device=feats_cls.device
            )
            centers = centers.to(features.device)
        prototypes.append(centers)

    prototypes = torch.cat(prototypes, dim=0)  # [2*K, D]
    
    features_norm = F.normalize(features, dim=1)
    prototypes_norm  = F.normalize(prototypes,  dim=1)
    
    S = features_norm @ prototypes_norm.t()  # cosine similarities in [-1,1]
    return S