import torch
import torchvision
import common_utils
from common_utils.common import flatten
from common_utils.image import get_ssim_pairs_kornia


def normalize_batch(x, ret_all=False):
    """ Normalize each element in batch x --> (x-mean)/std"""
    n, c, h, w = x.shape
    means = x.reshape(n * c, h * w).mean(dim=1).reshape(n, c, 1, 1)
    stds = x.reshape(n * c, h * w).std(dim=1).reshape(n, c, 1, 1)
    if ret_all:
        return x.sub(means).div(stds), means, stds
    else:
        return x.sub(means).div(stds)


def l2_dist(x, y, div_dim=False):
    """ L2 distance between x and y """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    
    # Ensure both tensors have the same dtype
    if x.dtype != y.dtype:
        y = y.to(x.dtype)

    xx = x.pow(2).sum(1).view(-1, 1)
    yy = y.pow(2).sum(1).view(1, -1)
    xy = torch.einsum('id,jd->ij', x, y)
    # dists = xx + yy - 2 * xy
    squared_dists = xx + yy - 2 * xy
    dists = torch.sqrt(squared_dists.clamp(min=0.0))
    if div_dim:
        N, D = x.shape
        dists /= D

    return dists


def ncc_dist(x, y, div_dim=False):
    """ Normalized Cross-Correlation distacne between x and y """
    return l2_dist(normalize_batch(x), normalize_batch(y), div_dim)


def transform_vmin_vmax_batch(x, min_max=None):
    """ Transform each image in x: [min, max] --> [0, 1]"""
    if min_max is None:
        vmin = x.data.reshape(x.shape[0], -1).min(dim=1)[0][:, None, None, None]
        vmax = x.data.reshape(x.shape[0], -1).max(dim=1)[0][:, None, None, None]
    else:
        vmin, vmax = min_max
    return (x - vmin).div(vmax - vmin)


def viz_nns(x, y, max_per_nn=None, metric='ncc', ret_all=False):
    """
    return a batch, for each image in x, its nn in y
    sorted according to closest nn
    metric: NCC
    max_per_nn: filter duplicates (leave only max_per_nn elements of y-elements)
    """

    if metric == 'ncc':
        dists = ncc_dist(x, y)
    elif metric == 'l2':
        dists = l2_dist(x, y)
    else:
        raise ValueError(f'Unknown metric={metric}')

    v, nn_idx = dists.min(dim=1)

    keep = None
    if max_per_nn is not None:
        nn_idx_vals_i = torch.stack([nn_idx, v, torch.arange(v.shape[0], device=v.device)])
        nn_idx_vals_i = [(int(a), b, int(c)) for a, b, c in nn_idx_vals_i.t().tolist()]  # bring indexes back to int
        sorted_stuff = sorted(nn_idx_vals_i)

        # filter duplicates (leave only max_per_nn from each image from y)
        counter = 0
        cur_idx = sorted_stuff[0][0]
        keep = []
        for e in sorted_stuff:
            if e[0] != cur_idx:
                cur_idx = e[0]
                counter = 0
            if counter < max_per_nn:
                keep.append(e)
                counter += 1
        # sort by best value first
        keep = sorted(keep, key=lambda q: q[1])
        # keep is now: (nn idx in y, value, idx of x)
        xx = x[torch.tensor([e[2] for e in keep])]
        yy = y[torch.tensor([e[0] for e in keep])]
        v = torch.tensor([e[1] for e in keep])
    else:
        _, sidxs = v.sort()
        xx = x[sidxs]
        yy = y[nn_idx[sidxs]]

    qq = torch.stack(flatten(list(zip(xx, yy))))
    qq = transform_vmin_vmax_batch(qq)

    if ret_all:
        return qq, v, xx, yy, keep

    return qq, v

def get_evaluation_score_dssim(xxx, yyy, ds_mean, vote=None, show=False, top=10):
    xxx = xxx.clone()
    yyy = yyy.clone()
    device = xxx.device

    x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic')
    y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic')
    D = ncc_dist(y2search, x2search, div_dim=True)

    dists, idxs = D.sort(dim=1, descending=False)

    if vote is not None:
        # Ignore distant nearest-neighbours
        xs_idxs = []
        for i in range(dists.shape[0]):
            x_idxs = [idxs[i, 0].item()]
            for j in range(1, dists.shape[1]):
                if (dists[i, j] / dists[i, 0]) < 1.1:
                    x_idxs.append(idxs[i, j].item())
                else:
                    break
            xs_idxs.append(x_idxs)

        # Voting
        xs = []
        for x_idxs in xs_idxs:
            if vote == 'min':
                x_voted = xxx[x_idxs[0]].unsqueeze(0)
            elif vote == 'mean':
                x_voted = xxx[x_idxs].mean(dim=0, keepdim=True)
            elif vote == 'median':
                x_voted = xxx[x_idxs].median(dim=0, keepdim=True).values
            elif vote == 'mode':
                x_voted = xxx[x_idxs].mode(dim=0, keepdim=True).values
            else:
                raise
            xs.append(x_voted)
        xx = torch.cat(xs, dim=0).clone()
        yy = yyy
    else:
        xx = xxx[idxs[:, 0]]
        yy = yyy

    # Scale to images
    yy += ds_mean
    xx = transform_vmin_vmax_batch(xx + ds_mean)

    # Score
    ssims = get_ssim_pairs_kornia(xx, yy)
    dssim = (1 - ssims) / 2
    dssims, sort_idxs = dssim.sort(descending=False)

    # Sort & Show
    xx = xx[sort_idxs]
    yy = yy[sort_idxs]

    qq = torch.stack(common_utils.common.flatten(list(zip(xx, yy))))
    grid = torchvision.utils.make_grid(qq[:100], normalize=False, nrow=20)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(80 * 2, 10 * 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

    ev_score = dssims[:top].mean()
    N_x = xxx.shape[0]
    N_y = yyy.shape[0]
    inv_sort = torch.empty_like(sort_idxs)
    inv_sort[sort_idxs] = torch.arange(N_y, device=device)
    dssim_y = dssims[inv_sort]           
    nn_x_for_y = idxs[:, 0].to(device)
    binary_x = torch.zeros(N_x, dtype=torch.long, device=device)
    good_y = (dssim_y < 0.15).long() 
    binary_x[nn_x_for_y] = good_y
    return ev_score.item(), grid, dssims, binary_x

def get_psnr_pairs(x, y, max_val=1.0):
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10((max_val ** 2) / (mse + 1e-10))  
    return psnr

def get_evaluation_score_psnr(xxx, yyy, ds_mean, show=False,top=10):
    xxx = xxx.clone()
    yyy = yyy.clone()

    x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic')
    y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic')
    D = ncc_dist(y2search, x2search, div_dim=True)
    _, idxs = D.sort(dim=1, descending=False)
    xx = xxx[idxs[:, 0]]
    yy = yyy

    yy += ds_mean
    xx = transform_vmin_vmax_batch(xx + ds_mean)

    # calculate PSNR
    psnrs = get_psnr_pairs(xx, yy, max_val=1.0)
    psnrs_sorted, sort_idxs = psnrs.sort(descending=True)

    xx = xx[sort_idxs]
    yy = yy[sort_idxs]
    qq = torch.stack(common_utils.common.flatten(list(zip(xx, yy))))
    grid = torchvision.utils.make_grid(qq[:100], normalize=False, nrow=20)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(80 * 2, 10 * 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

    ev_score = psnrs_sorted[:top].mean()
    return ev_score.item(), grid

def get_model_outputs_on_grid(model, lim=1.5, n=1000):
    x_coord = torch.linspace(-lim, lim, n)
    y_coord = torch.linspace(-lim, lim, n)
    grid = torch.stack(torch.meshgrid([x_coord, y_coord], indexing=None))
    zi = model(grid.reshape(2, -1).t().to('cuda')).reshape(n, n).cpu().data
    xi = grid[0,:,:].cpu()
    yi = grid[1,:,:].cpu()
    return xi, yi, zi



