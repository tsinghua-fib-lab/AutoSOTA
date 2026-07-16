import torch
import torchvision
from tqdm.auto import tqdm

import common_utils
from common_utils.image import get_ssim_all, get_ssim_pairs_kornia
from evaluations import l2_dist, ncc_dist, normalize_batch, transform_vmin_vmax_batch
import kornia

@torch.no_grad()
def get_dists(x, y, search, use_bb):
    """D: x -> y"""
    xxx = x.clone()
    yyy = y.clone()

    # Search Real --> Extracted
    if search == 'l2':
        D = l2_dist(xxx, yyy, div_dim=True)
    if search == 'ncc':
        D = ncc_dist(xxx, yyy, div_dim=True)
    elif search == 'ncc2':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        D = ncc_dist(x2search, y2search, div_dim=True)
    elif search == 'ncc4':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1/4, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1/4, mode='bicubic', align_corners=False)
        D = ncc_dist(x2search, y2search, div_dim=True)

    # Consider each reconstruction for only one train-samples
    if use_bb:
        bb_mask = D.mul(-100000000).softmax(dim=0).mul(10).round().div(10).round()
        assert bb_mask.sum(dim=0).abs().sum() == D.shape[1]
        D[bb_mask != 1] = torch.inf

    dists, idxs = D.sort(dim=1, descending=False)
    return dists, idxs


@torch.no_grad()
def find_nearest_neighbour(X, x0, search='ncc', vote='mean', use_bb=True, nn_threshold=None, ret_idxs=False):
    xxx = X.clone()
    yyy = x0.clone()

    # Search Real --> Extracted
    if search == 'l2':
        D = l2_dist(yyy, xxx, div_dim=True)
    if search == 'ncc':
        D = ncc_dist(yyy, xxx, div_dim=True)
    elif search == 'ncc2':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        D = ncc_dist(y2search, x2search, div_dim=True)
    elif search == 'ncc4':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1/4, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1/4, mode='bicubic', align_corners=False)
        D = ncc_dist(y2search, x2search, div_dim=True)
    elif search == 'dssim':
        D_ssim = get_ssim_all(yyy, xxx)
        D_dssim = (1 - D_ssim)/2
        D = D_dssim

    # Only consider Best-Bodies
    if use_bb:
        bb_mask = D.mul(-100000000).softmax(dim=0).mul(10).round().div(10).round()
        assert bb_mask.sum(dim=0).abs().sum() == D.shape[1]
        D[bb_mask != 1] = torch.inf

    dists, idxs = D.sort(dim=1, descending=False)

    # yy = yyy
    if vote == 'min' or vote is None:
        xx = xxx[idxs[:, 0]]
    else:
        # Ignore distant nearest-neighbours
        if nn_threshold is None:
            xs_idxs = idxs[:, :int(0.01*x0.shape[0])]
        else:
            xs_idxs = []
            for i in range(dists.shape[0]):
                x_idxs = [idxs[i, 0].item()]
                for j in range(1, dists.shape[1]):
                    if (dists[i, j] / dists[i, 0]) < nn_threshold:
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

    if ret_idxs:
        return xx, idxs[:, 0]

    return xx


@torch.no_grad()
def scale(xx, x0, ds_mean, xx_add_ds_mean=True):
    xx = xx.clone()
    x0 = x0.clone()
    ds_mean = ds_mean.clone()
    # Scale to images
    yy = x0 + ds_mean
    if xx_add_ds_mean:
        xx = transform_vmin_vmax_batch(xx + ds_mean)
    else:
        xx = transform_vmin_vmax_batch(xx)

    return xx, yy


@torch.no_grad()
def sort_by_metric(xx, yy, sort='ssim'):
    xx = xx.clone()
    yy = yy.clone()

    # Score
    psnr = lambda a, b: 20 * torch.log10(1.0 / (a - b).pow(2).reshape(a.shape[0], -1).mean(dim=1).sqrt())

    # Sort
    if sort == 'ssim':
        dists = get_ssim_pairs_kornia(xx, yy)
        dssim = (1 - dists) / 2
        _, sort_idxs = dists.sort(descending=True)
    elif sort == 'ncc':
        dists = (normalize_batch(xx) - normalize_batch(yy)).reshape(xx.shape[0], -1).norm(dim=1)
        _, sort_idxs = dists.sort()
    elif sort == 'l2':
        dists = (xx - yy).reshape(xx.shape[0], -1).norm(dim=1)
        _, sort_idxs = dists.sort()
    elif sort == 'psnr':
        dists = psnr(xx, yy)
        _, sort_idxs = dists.sort(descending=True)
    else:
        raise

    xx = xx[sort_idxs]
    yy = yy[sort_idxs]
    return xx, yy, dists, sort_idxs


@torch.no_grad()
def plot_table(xx, yy, fig_elms_in_line, fig_lines_per_page, fig_type='side_by_side',
               figpath=None, show=False, dpi=100, color_by_labels=None):
    # PRINT TABLES
    import matplotlib.pyplot as plt
    xx = xx.clone()
    yy = yy.clone()

    RED = torch.tensor([1, 0, 0])[None, :, None, None]
    BLUE = torch.tensor([0, 1, 0])[None, :, None, None]
    def add_colored_margin(x, labels, p=1):
        n, c, h, w = x.shape
        bg = torch.zeros(n, c, h + 2 * p, w + 2 * p)
        bg[labels == 0] += RED
        bg[labels == 1] += BLUE
        bg[:, :, p:-p, p:-p] = x
        return bg

    if color_by_labels is not None:
        yy = add_colored_margin(yy, color_by_labels, p=2)
        xx = add_colored_margin(xx, color_by_labels, p=2)

    if fig_type == 'side_by_side':
        qq = torch.stack(common_utils.common.flatten(list(zip(xx, yy))))
    elif fig_type == 'one_above_another':
        q_zip = common_utils.common.flatten(list(zip(torch.split(xx, fig_elms_in_line), torch.split(yy, fig_elms_in_line))))
        if len(q_zip) > 2:
            q_zip = q_zip[:-2]
            print('CUT the end of the zipped bla because it might have different shape before torch.cat')
        qq = torch.cat(q_zip)
    else:
        raise

    lines_num = qq.shape[0] // fig_elms_in_line
    print(qq.shape, lines_num)
    for page_num, line_num in enumerate(tqdm(range(0, lines_num, fig_lines_per_page))):
        s = line_num * fig_elms_in_line
        e = (line_num + fig_lines_per_page) * fig_elms_in_line
        print(page_num, s, e)
        grid = torchvision.utils.make_grid(qq[s:e], normalize=False, nrow=fig_elms_in_line, pad_value=1)
        if figpath is not None:
            plt.imsave(figpath, grid.permute(1, 2, 0).cpu().numpy(), dpi=dpi)
            print('Saved fig at:', figpath)
        if show:
            plt.figure(figsize=(80 * 2, 10 * 2))
            plt.axis('off')
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.show()
        plt.close('all')
        break
    print('DONE!')


def plot_table_cifar(xx, xx2, yy, ssim, ssim2, fig_elms_in_line, fig_lines_per_page, 
               figpath=None, show=False, dpi=100, color_by_labels=None):
    # PRINT TABLES
    import matplotlib.pyplot as plt
    xx = xx.clone()
    xx2 = xx2.clone()
    yy = yy.clone()

    RED = torch.tensor([1, 0, 0])[None, :, None, None]
    BLUE = torch.tensor([0, 0, 1])[None, :, None, None]
    def add_colored_margin(x, labels, p=1):
        n, c, h, w = x.shape
        bg = torch.zeros(n, c, h + 2 * p, w + 2 * p)
        bg[ssim >= ssim2] += RED
        bg[ssim < ssim2] += BLUE
        bg[:, :, p:-p, p:-p] = x
        return bg

    if color_by_labels is not None:
        yy = add_colored_margin(yy, color_by_labels, p=2)
        xx = add_colored_margin(xx, color_by_labels, p=2)
        xx2 = add_colored_margin(xx2, color_by_labels, p=2)

    q_zip = common_utils.common.flatten(list(zip(torch.split(xx, fig_elms_in_line),torch.split(xx2, fig_elms_in_line), torch.split(yy, fig_elms_in_line))))
    # if len(q_zip) > 2:
    #     q_zip = q_zip[:-2]
    #     print('CUT the end of the zipped bla because it might have different shape before torch.cat')
    qq = torch.cat(q_zip)


    lines_num = qq.shape[0] // fig_elms_in_line
    print(lines_num)
    for page_num, line_num in enumerate(tqdm(range(0, lines_num, fig_lines_per_page))):
        s = line_num * fig_elms_in_line
        e = (line_num + fig_lines_per_page) * fig_elms_in_line
        print(page_num, s, e)
        
        captions = []
        for i in range(s, e, fig_elms_in_line):
            end_idx = min(i + fig_elms_in_line, e-s)
            current_batch = end_idx - i
            
            captions.extend([f"{val:.2f}" for val in ssim[i:end_idx].tolist()])
            captions.extend([f"{val:.2f}" for val in ssim2[i:end_idx].tolist()])
            captions.extend(["  "] * current_batch)
        fig, axes = plt.subplots(
            nrows=(len(qq[s:e]) + fig_elms_in_line - 1) // fig_elms_in_line,
            ncols=fig_elms_in_line,
            figsize=(fig_elms_in_line, fig_lines_per_page * 1.3),
            constrained_layout=True
        )
        axes = axes.flatten()

        for idx, (ax, caption) in enumerate(zip(axes, captions)):
            if idx < len(qq[s:e]):
                img = qq[s + idx]
                if isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0).cpu().numpy()
                
                ax.imshow(img)
                ax.text(0.5, -0.04, caption, transform=ax.transAxes,ha='center', va='top', fontsize=20)
                ax.axis('off')
            else:
                ax.axis('off') 
        
        if figpath is not None:
            plt.savefig(figpath, dpi=dpi)
            print('Saved fig at:', figpath)
        break
    print('DONE!')

def plot_table_mnist(xx, xx2, yy, metric, metric2, fig_elms_in_line, fig_lines_per_page, 
               figpath=None, show=False, dpi=100, color_by_labels=None):
    # PRINT TABLES
    import matplotlib.pyplot as plt
    xx = xx.clone()
    xx2 = xx2.clone()
    yy = yy.clone()

    RED = torch.tensor([1, 0, 0])[None, :, None, None]
    BLUE = torch.tensor([0, 0, 1])[None, :, None, None]
    def add_colored_margin(x, labels, p=1):
        n, c, h, w = x.shape
        bg = torch.zeros(n, c, h + 2 * p, w + 2 * p)
        bg[metric >= metric2] += RED
        bg[metric < metric2] += BLUE
        bg[:, :, p:-p, p:-p] = x
        return bg

    if color_by_labels is not None:
        yy = add_colored_margin(yy, color_by_labels, p=2)
        xx = add_colored_margin(xx, color_by_labels, p=2)
        xx2 = add_colored_margin(xx2, color_by_labels, p=2)

    q_zip = common_utils.common.flatten(list(zip(torch.split(xx, fig_elms_in_line),torch.split(xx2, fig_elms_in_line), torch.split(yy, fig_elms_in_line))))
    # if len(q_zip) > 2:
    #     q_zip = q_zip[:-2]
    #     print('CUT the end of the zipped bla because it might have different shape before torch.cat')
    qq = torch.cat(q_zip)


    lines_num = qq.shape[0] // fig_elms_in_line
    print(lines_num)
    for page_num, line_num in enumerate(tqdm(range(0, lines_num, fig_lines_per_page))):
        s = line_num * fig_elms_in_line
        e = (line_num + fig_lines_per_page) * fig_elms_in_line
        print(page_num, s, e)
    
        captions = []
        for i in range(s, e, fig_elms_in_line):
            end_idx = min(i + fig_elms_in_line, e-s)
            current_batch = end_idx - i

            captions.extend([f"{val:.2f}" for val in metric[i:end_idx].tolist()])
            captions.extend([f"{val:.2f}" for val in metric2[i:end_idx].tolist()])
            captions.extend(["  "] * current_batch)
        fig, axes = plt.subplots(
            nrows=(len(qq[s:e]) + fig_elms_in_line - 1) // fig_elms_in_line,
            ncols=fig_elms_in_line,
            figsize=(fig_elms_in_line, fig_lines_per_page*1.3),
            constrained_layout=True
        )
        axes = axes.flatten()

        for idx, (ax, caption) in enumerate(zip(axes, captions)):
            if idx < len(qq[s:e]):
                img = qq[s + idx]
                if isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0).cpu().numpy()
                
                ax.imshow(img)
                ax.text(0.5, -0.04, caption, transform=ax.transAxes,ha='center', va='top', fontsize=20)
                ax.axis('off')
            else:
                ax.axis('off') 
        if figpath is not None:
            plt.savefig(figpath, dpi=dpi)
            print('Saved fig at:', figpath)
        break
    print('DONE!')

@torch.no_grad()
def find_best_ssim_scores_batch(train_x, recon_x, batch_size_r=64, window_size=11):
    device = train_x.device
    N = train_x.shape[0]
    best_scores = torch.full((N,), -1.0, device=device)

    for i in tqdm(range(N), desc="Processing train samples"):
        train_img = train_x[i:i+1]  # [1, C, H, W]
        max_score = -1.0

        for r_start in range(0, recon_x.shape[0], batch_size_r):
            r_end = min(r_start + batch_size_r, recon_x.shape[0])
            batch_recon = recon_x[r_start:r_end]

            scores = kornia.metrics.ssim(
                train_img.expand(batch_recon.shape[0], -1, -1, -1),
                batch_recon,
                window_size=window_size
            )  # [B_r, C, H, W]
            scores = scores.mean(dim=(1, 2, 3))  # [B_r]
            max_score = max(max_score, scores.max().item())

        best_scores[i] = max_score

    return best_scores