import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import shutil
import warnings
from matplotlib.colors import Normalize
from matplotlib import cm

DENSITY_CMAP = "viridis"
DENSITY_VMIN = 0.0
DENSITY_VMAX = 1.0


def density_color_norm():
    """Canonical density color mapping used by every density plot."""
    return Normalize(vmin=DENSITY_VMIN, vmax=DENSITY_VMAX, clip=True)


# Publication-quality Matplotlib defaults (Computer Modern look, large readable fonts)
# These are intentionally slightly conservative so they work across systems even
# if the Computer Modern font isn't installed as a system font; mathtext will
# still use the CM glyphs via 'mathtext.fontset'. Adjust sizes as needed.
# If a system LaTeX installation is available, use it for publication-quality
# rendering. Otherwise fall back to Matplotlib's internal mathtext to avoid
# subprocess calls to `latex` (which may not be available on compute nodes).
latex_bin = shutil.which('latex')
if latex_bin is None:
    warnings.warn("LaTeX not found on PATH; falling back to Matplotlib mathtext (text.usetex=False).",
                  RuntimeWarning)

rc = {
    # High-quality raster/vector output
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    # Font selection (use default matplotlib font)
    # 'font.family': 'serif',
    # 'font.serif': ['DejaVu Serif', 'STIXGeneral', 'Times', 'TeX Gyre Termes', 'Times New Roman'],

    # Sizes/styles tuned for publication readability
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.6,
    'figure.figsize': (6, 4),
}

if latex_bin is not None:
    # Use LaTeX for all text rendering so figures exactly match LaTeX documents.
    # NOTE: this requires a working LaTeX installation with the referenced
    # packages available (amsmath, newtxtext, newtxmath).
    rc['text.usetex'] = True
    rc['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{newtxtext}\usepackage{newtxmath}'
else:
    # Fall back to mathtext with Computer Modern glyphs for math. This avoids
    # calling external LaTeX binaries (latex/dvips/dvipng) which may be absent.
    rc['text.usetex'] = False
    rc['mathtext.fontset'] = 'cm'

mpl.rcParams.update(rc)

# Use a colorblind-friendly cycle by default for consistent publication colors
try:
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=plt.get_cmap('tab10').colors)
except Exception:
    pass


def _svg_path(path):
    if isinstance(path, str) and path.lower().endswith('.png'):
        return path[:-4] + '.svg'
    return path


def _pdf_png_paths(path):
    if isinstance(path, str) and path.lower().endswith('.pdf'):
        return path, path[:-4] + '.png'
    return path, None

def filter_valid_samples(x):
    """Return a torch.Tensor with rows that contain no NaN/Inf.

    Accepts numpy arrays or torch tensors. Returns a torch tensor.
    """
    if x is None:
        return torch.tensor([])
    if isinstance(x, np.ndarray):
        arr = x
        if arr.size == 0:
            return torch.tensor([])
        mask = np.isfinite(arr).all(axis=1)
        return torch.tensor(arr[mask])
    if torch.is_tensor(x):
        if x.numel() == 0:
            return x
        mask = torch.isfinite(x).all(dim=1)
        return x[mask]
    # try to coerce
    try:
        t = torch.tensor(x)
        return filter_valid_samples(t)
    except Exception:
        return torch.tensor([])


def ensure_tensor_2d(x, D):
    """Ensure samples are a 2D torch tensor with second dimension D.

    If x is empty or cannot be reshaped, returns an empty tensor.
    """
    if x is None:
        return torch.tensor([])
    t = x
    if isinstance(x, np.ndarray):
        t = torch.tensor(x)
    if not torch.is_tensor(t):
        try:
            t = torch.tensor(x)
        except Exception:
            return torch.tensor([])
    if t.numel() == 0:
        return t
    try:
        return t.view(-1, D)
    except Exception:
        # fallback: attempt convert to float and reshape
        try:
            flat = t.reshape(-1)
            nrows = flat.numel() // D
            return flat[: nrows * D].view(-1, D)
        except Exception:
            return torch.tensor([])


def save_grid(samples, fname, nrow=5, ncol=5, cmap='gray'):
    """Save a grid of square images stored as flattened vectors (e.g. MNIST).

    samples: numpy array or torch tensor with shape (N, D)
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    if samples is None or len(samples) == 0:
        return
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.5))
    for i, ax in enumerate(axs.flat):
        if i >= len(samples):
            ax.axis('off')
            continue
        img = samples[i].reshape(int(np.sqrt(samples.shape[1])), -1)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
    plt.savefig(fname)
    plt.close(fig)


def save_metrics_table_paper(general_metrics, intrinsic_metrics=None, out_tex_path='metrics.tex', caption='Metrics'):
    """Save a simple LaTeX table (and CSV) summarizing metrics.

    `general_metrics` should be a dict mapping method->dict of named values.
    """
    import csv

    # CSV path
    csv_path = os.path.splitext(out_tex_path)[0] + '.csv'
    methods = list(general_metrics.keys())
    # gather columns
    cols = set()
    for v in general_metrics.values():
        cols.update(list(v.keys()))
    # never include the auxiliary 'Avg. Dist. to M' column in outputs
    if 'Avg. Dist. to M' in cols:
        cols.remove('Avg. Dist. to M')
    # Ensure MMD column is present so it's included in the table
    if 'MMD' not in cols:
        cols.add('MMD')
    cols = sorted(list(cols))

    # Write CSV
    os.makedirs(os.path.dirname(out_tex_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method'] + cols)
        for m in methods:
            # Format MMD column into scientific notation for CSV if numeric
            row_vals = []
            for c in cols:
                val = general_metrics[m].get(c, '')
                if c == 'MMD':
                    try:
                        import numpy as _np

                        if isinstance(val, (float, int)) and _np.isfinite(val):
                            v = float(val)
                            s = f"{v:.3e}"
                            row_vals.append(s)
                            continue
                    except Exception:
                        pass
                row_vals.append(val)
            row = [m] + row_vals
            writer.writerow(row)

    # Write a minimal LaTeX table
    try:
        with open(out_tex_path, 'w') as f:
            f.write('\begin{table}[ht]\centering\small\begin{tabular}{l' + 'r' * len(cols) + '}\toprule\n')
            f.write('Method & ' + ' & '.join(cols) + ' \\midrule\n')
            for m in methods:
                tex_vals = []
                for c in cols:
                    v = general_metrics[m].get(c, '')
                    if c == 'MMD':
                        # If numeric, format into LaTeX scientific notation
                        try:
                            import numpy as _np

                            if isinstance(v, (float, int)) and _np.isfinite(v):
                                s = f"{float(v):.3e}"
                                mant, exp = s.split('e')
                                mant = mant.rstrip('0').rstrip('.') if '.' in mant else mant
                                tex_vals.append(f"${mant}\\times10^{{{int(exp)}}}$")
                                continue
                        except Exception:
                            pass
                    # otherwise write the raw string (already LaTeX-formatted if desired)
                    tex_vals.append(str(v))
                f.write(m + ' & ' + ' & '.join(tex_vals) + ' \\\n+')
            f.write('\bottomrule\end{tabular}\caption{' + caption + '}\end{table}\n')
    except Exception:
        # best-effort: ignore LaTeX failures
        pass


def compute_naninf_and_avg_dist(orig, proj=None, projector=None):
    """Compute (Num NaN/Inf rows, Avg distance to manifold) for sample arrays.

    preferrs using the provided `projector`'s per-sample distances when available
    and computes the mean over the same finite-filtered sample set used for
    extrinsic metrics. Accepts numpy arrays or torch tensors for `orig` and
    `proj`. Returns (n_bad, avg_dist) where n_bad is int (number of rows with
    NaN/Inf) or "n/a" and avg_dist is float mean distance or "n/a".
    """
    import numpy as _np
    import torch as _torch

    if orig is None:
        return "n/a", "n/a"

    # Coerce orig to numpy + torch forms
    try:
        if _torch.is_tensor(orig):
            orig_t = orig.view(-1, orig.shape[-1])
            orig_np = orig_t.cpu().numpy()
        else:
            orig_np = _np.array(orig)
            orig_t = _torch.tensor(orig_np)
    except Exception:
        try:
            orig_np = _np.array(orig)
            orig_t = _torch.tensor(orig_np)
        except Exception:
            return "n/a", "n/a"

    if orig_np.size == 0:
        return 0, "n/a"

    # finite mask and bad count
    try:
        finite_mask = _np.isfinite(orig_np).all(axis=1)
        n_bad = int((_np.asarray(finite_mask) == False).sum())
    except Exception:
        try:
            fm = _torch.isfinite(orig_t).all(dim=1)
            n_bad = int((~fm).sum().item())
            finite_mask = fm.cpu().numpy()
        except Exception:
            return "n/a", "n/a"

    # Prefer projector per-sample distances computed on the same finite subset
    if projector is not None:
        try:
            idx = _np.where(finite_mask)[0]
            if idx.size == 0:
                return n_bad, "n/a"
            X = orig_t[idx]
            # Move to projector device if possible
            try:
                dev = getattr(projector, 'device', None)
                if dev is not None and isinstance(dev, _torch.device):
                    X = X.to(dev)
            except Exception:
                pass
            # Try return_details=True first
            try:
                res = projector.project(X, return_details=True)
            except TypeError:
                try:
                    res = projector.project(X)
                except Exception:
                    res = None
            except Exception:
                res = None

            if isinstance(res, tuple) and len(res) >= 2:
                dist = res[1]
                try:
                    if _torch.is_tensor(dist):
                        dist_np = dist.detach().cpu().numpy()
                    else:
                        dist_np = _np.array(dist)
                    if dist_np.size == 0:
                        return n_bad, "n/a"
                    # filter non-finite distances (protect against inf/NaN)
                    dist_finite = dist_np[_np.isfinite(dist_np)]
                    if dist_finite.size == 0:
                        return n_bad, "n/a"
                    mean_val = float(_np.mean(dist_finite))
                    if not _np.isfinite(mean_val):
                        return n_bad, "n/a"
                    return int(n_bad), mean_val
                except Exception:
                    return n_bad, "n/a"
        except Exception:
            # fall through to proj-based fallback
            pass

    # Fallback: if proj provided, compute mean Euclidean distance between orig and proj
    if proj is not None:
        try:
            if _torch.is_tensor(proj):
                proj_np = proj.view(-1, proj.shape[-1]).cpu().numpy()
            else:
                proj_np = _np.array(proj)
        except Exception:
            try:
                proj_np = _np.array(proj)
            except Exception:
                return n_bad, "n/a"

        # align rows if shapes differ
        try:
            if orig_np.shape != proj_np.shape:
                minr = min(orig_np.shape[0], proj_np.shape[0])
                if minr == 0:
                    return n_bad, "n/a"
                orig_al = orig_np[:minr]
                proj_al = proj_np[:minr]
            else:
                orig_al = orig_np
                proj_al = proj_np
        except Exception:
            return n_bad, "n/a"

        try:
            mask_both = _np.isfinite(orig_al).all(axis=1) & _np.isfinite(proj_al).all(axis=1)
            if mask_both.sum() == 0:
                return n_bad, "n/a"
            diffs = orig_al[mask_both] - proj_al[mask_both]
            dists = _np.linalg.norm(diffs, axis=1)
            # filter non-finite distances
            dists_finite = dists[_np.isfinite(dists)]
            if dists_finite.size == 0:
                return n_bad, "n/a"
            mean_val = float(_np.mean(dists_finite))
            if not _np.isfinite(mean_val):
                return n_bad, "n/a"
            return int(n_bad), mean_val
        except Exception:
            return n_bad, "n/a"

    return n_bad, "n/a"

def save_standalone_colorbar(norm=None, cmap=DENSITY_CMAP, filename='colorbar.svg', label='Density', dpi=200, height_in=3.0, width_in=0.5, orientation='vertical'):
    norm = density_color_norm()
    cmap = DENSITY_CMAP
    fig = plt.figure(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.4)
    cax = fig.add_axes([0.05, 0.05, 0.2, 0.9])
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation=orientation)
    cb.set_label(label)
    filename = _svg_path(filename)
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    _, png_filename = _pdf_png_paths(filename)
    if png_filename is not None:
        plt.savefig(png_filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _crop_whitespace_image(path, margin_px=4):
    """Crop whitespace (near-white background) from an image file in-place.

    margin_px: number of pixels to leave as margin around the crop.
    """
    try:
        from PIL import Image
        import numpy as _np
    except Exception:
        return
    try:
        im = Image.open(path).convert('RGBA')
        a = _np.array(im)
        h, w = a.shape[0], a.shape[1]
        rgb = a[:, :, :3]
        alpha = a[:, :, 3]
        # Visible pixel: alpha>10 and not near-white
        visible = (alpha > 10) & (_np.any(rgb < 250, axis=2))
        if not visible.any():
            return
        ys, xs = _np.nonzero(visible)
        xmin, xmax = max(0, xs.min() - margin_px), min(w - 1, xs.max() + margin_px)
        ymin, ymax = max(0, ys.min() - margin_px), min(h - 1, ys.max() + margin_px)
        cropped = im.crop((xmin, ymin, xmax + 1, ymax + 1))
        cropped.save(path)
    except Exception:
        return


def plot_2d(points, fname, title=None, point_alpha=0.6, cmap='viridis'):
    if torch.is_tensor(points):
        pts = points.cpu().numpy()
    else:
        pts = np.array(points)
    # Ensure numeric dtype (float64) to avoid object arrays with weird large integers
    try:
        pts = np.asarray(pts, dtype=np.float64)
    except Exception:
        try:
            if torch.is_tensor(points):
                pts = points.cpu().numpy().astype(np.float64)
        except Exception:
            # leave pts as-is if coercion fails
            pass
    if pts.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=point_alpha, rasterized=True)
    if title:
        ax.set_title(title)
    filename = _svg_path(fname)
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    plt.savefig(filename)
    _, png_filename = _pdf_png_paths(filename)
    if png_filename is not None:
        plt.savefig(png_filename)
    plt.close(fig)


def plot_2d_density_no_cbar(points, filename, title=None, gridsize=200, cmap=DENSITY_CMAP, point_alpha=0.4, dpi=300, norm=None):
    norm = density_color_norm()
    cmap = DENSITY_CMAP
    if torch.is_tensor(points):
        pts = points.cpu().numpy()
    else:
        pts = np.array(points)
    if pts.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    h = ax.hist2d(pts[:, 0], pts[:, 1], bins=gridsize, cmap=cmap, norm=norm)
    try:
        h[3].set_rasterized(True)
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    filename = _svg_path(filename)
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    _, png_filename = _pdf_png_paths(filename)
    if png_filename is not None:
        plt.savefig(png_filename, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    if filename.lower().endswith('.png'):
        try:
            _crop_whitespace_image(filename, margin_px=4)
        except Exception:
            pass
    if png_filename is not None:
        try:
            _crop_whitespace_image(png_filename, margin_px=4)
        except Exception:
            pass
    plt.close(fig)

def compute_avg_stats(method_names, trainers_map, n_trials, num_samples=None, external_proj_time=None):
    """Compute simple average timing stats from trainer objects.

    Returns dict name->{'m':mean_model,'p':mean_proj,'s':mean_sampling}
    """
    import numpy as _np
    stats = {}
    for name in method_names:
        tr = trainers_map.get(name)
        m = _np.nan
        p = _np.nan
        s = _np.nan
        if tr is not None:
            # try several attribute names that may exist
            try:
                m = float(getattr(tr, 'total_model_forward_sample_time', _np.nan))
            except Exception:
                m = _np.nan
            try:
                if hasattr(tr, 'model_forward_times') and len(getattr(tr, 'model_forward_times', [])) > 0:
                    m = float(_np.nansum([float(x) for x in tr.model_forward_times]))
            except Exception:
                pass
            try:
                p = float(getattr(tr, 'total_projection_sample_time', _np.nan))
            except Exception:
                p = _np.nan
            try:
                if hasattr(tr, 'projection_sample_times') and len(getattr(tr, 'projection_sample_times', [])) > 0:
                    p = float(_np.nansum([float(x) for x in tr.projection_sample_times]))
            except Exception:
                pass
            try:
                s = float(getattr(tr, 'sampling_time', _np.nan))
            except Exception:
                s = _np.nan
        # allow adding an externally measured projection time (for DDPM_projected)
        if name == 'DDPM_projected' and external_proj_time is not None:
            try:
                p = float(external_proj_time)
            except Exception:
                pass
        stats[name] = {'m': m, 'p': p, 's': s}
    return stats


def plot_scores_vs_time(
    scores_list=None,
    scores_plain=None,
    sigma_list=None,
    output_path="results/scores_vs_time.svg",
    figsize=(10, 5),
    cmap="viridis",
    logscale=True,
    xlabel=r'$t$ (time steps in reverse order)',
    ylabel=r"Average Score $\nabla_x(t) \log p_t(x(t))$ Across $x(t)$",
    dpi=300,
):
    """
    Unified plotting for score curves across timesteps.

    - If `scores_list` is provided and `sigma_list` is provided, each entry in
      `scores_list` is plotted with a color mapped from `sigma_list` and a
      horizontal colorbar added.
    - If `scores_plain` is provided it is plotted as a dashed red line.
    - Handles NaN/Inf by marking them with 'x'.
    """
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=figsize)

    nan_label_plotted = False

    # plot varied-sigma curves (colored by sigma)
    if scores_list is not None and sigma_list is not None and len(scores_list) == len(sigma_list):
        try:
            norm = mpl.colors.LogNorm(vmin=float(min(sigma_list)), vmax=float(max(sigma_list)))
        except Exception:
            norm = mpl.colors.Normalize(vmin=float(min(sigma_list)), vmax=float(max(sigma_list)))
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        for raw, sigma in zip(scores_list, sigma_list):
            try:
                arr = np.array([float(x) for x in raw])
            except Exception:
                arr = np.array(raw)
            invalid_mask = ~np.isfinite(arr)
            valid_mask = np.isfinite(arr)
            color = mpl.cm.get_cmap(cmap)(norm(sigma)) if hasattr(norm, 'vmin') else mpl.cm.get_cmap(cmap)(0.5)
            ax.plot(arr, color=color)
            if np.any(invalid_mask):
                top_y = np.nanmax(arr[valid_mask]) * 1.1 if np.any(valid_mask) else 1.0
                label = 'NaN or Inf' if not nan_label_plotted else ""
                ax.plot(np.where(invalid_mask)[0], [top_y] * np.sum(invalid_mask), 'x', color=color, markersize=5, label=label)
                nan_label_plotted = True
        # colorbar
        try:
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.12)
            cbar.set_label('Noise level (sigma)', fontsize=mpl.rcParams['axes.labelsize'])
            cbar.ax.tick_params(labelsize=mpl.rcParams['xtick.labelsize'])
        except Exception:
            pass

    # plot plain/projection line if present
    if scores_plain is not None:
        try:
            arr_plain = np.array([float(x) for x in scores_plain])
        except Exception:
            arr_plain = np.array(scores_plain)
        invalid_mask_plain = ~np.isfinite(arr_plain)
        valid_mask_plain = np.isfinite(arr_plain)
        if np.any(valid_mask_plain):
            top_y_plain = np.nanmax(arr_plain[valid_mask_plain]) * 1.1
        else:
            top_y_plain = 1.0
        ax.plot(arr_plain, label='Proj. DDPM', linestyle='--', color='red')
        if np.any(invalid_mask_plain):
            ax.plot(np.where(invalid_mask_plain)[0], [top_y_plain] * np.sum(invalid_mask_plain), 'x', color='red', markersize=5, label='NaN or Inf' if not nan_label_plotted else "")

    if logscale:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # flip tick labels so that the leftmost corresponds to latest timestep as some scripts prefer
    try:
        num_points = arr_plain.shape[0] if scores_plain is not None else (len(scores_list[0]) if scores_list else 0)
        if num_points > 0:
            xticks = np.linspace(0, num_points - 1, num=5, dtype=int)
            xtick_labels = [f'{num_points - 1 - x}' for x in xticks]
            xtick_labels[0] = '100' if len(xtick_labels) >= 1 else xtick_labels[0]
            xtick_labels[-1] = '0'
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
    except Exception:
        pass

    # Make tick labels large and readable
    ax.tick_params(axis='both', which='major', labelsize=mpl.rcParams['xtick.labelsize'])

    # Ensure legend uses the publication-sized font
    if ax.get_legend() is not None:
        ax.legend(fontsize=mpl.rcParams['legend.fontsize'])
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    _, png_output_path = _pdf_png_paths(output_path)
    if png_output_path is not None:
        fig.savefig(png_output_path, dpi=dpi)
    plt.close(fig)


def plot_metric_vs_sigma(
    sigma_list,
    mean_arr,
    std_arr,
    plain_value=None,
    plain_std=None,
    ylabel='Metric',
    title=None,
    output_path='results/metric_vs_sigma.svg',
    logx=True,
    marker='o',
    figsize=(10, 5),
    dpi=300,
):
    """Generic plotter for mean +/- std vs sigma with optional horizontal plain baseline."""
    fig, ax = plt.subplots(figsize=figsize)
    mean_arr = np.array(mean_arr)
    std_arr = np.array(std_arr)
    ax.plot(sigma_list, mean_arr, marker=marker, markersize=8, label=r'$p_{\sigma}$')
    ax.fill_between(sigma_list, mean_arr - std_arr, mean_arr + std_arr, alpha=0.25)
    if plain_value is not None:
        ax.axhline(y=plain_value, color='r', linestyle='--', label='Proj. DDPM')
        if plain_std is not None:
            ax.fill_between(sigma_list, [plain_value - plain_std] * len(sigma_list), [plain_value + plain_std] * len(sigma_list), color='r', alpha=0.1)
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel('Noise Level (sigma)')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=mpl.rcParams['xtick.labelsize'])
    if ax.get_legend() is not None:
        ax.legend(fontsize=mpl.rcParams['legend.fontsize'])
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    _, png_output_path = _pdf_png_paths(output_path)
    if png_output_path is not None:
        fig.savefig(png_output_path, dpi=dpi)
    plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from utils.constraints import *
import os
import csv
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from scipy.stats import gaussian_kde

def save_mesh_point_plot(
    data_points, mesh, vertices,
    output_path="mesh_with_points.svg",
    view='isometric',
    point_size=8.0,
    camera_zoom=1.4,
):
    """
    Plots the given data points on the surface of the mesh with no density computation.
    """
    import numpy as np
    import pyvista as pv
    import os

    # Ensure numpy arrays and coerce numeric dtype
    vertices = np.asarray(vertices)
    try:
        data_points = np.asarray(data_points, dtype=np.float64)
    except Exception:
        # fallback: try torch conversion if needed
        try:
            if torch.is_tensor(data_points):
                data_points = data_points.cpu().numpy().astype(np.float64)
            else:
                data_points = np.asarray(data_points)
        except Exception:
            data_points = np.asarray(data_points)

    # Filter invalid rows (NaN/Inf)
    try:
        if data_points is None:
            data_points = np.empty((0, 3), dtype=float)
        else:
            if data_points.size == 0:
                data_points = np.empty((0, 3), dtype=float)
            else:
                mask = np.isfinite(data_points).all(axis=1)
                n_before = data_points.shape[0]
                data_points = data_points[mask]
                n_after = data_points.shape[0]
                if n_after < n_before:
                    print(f"[Points] Filtered {n_before-n_after} NaN/Inf rows from data_points")
    except Exception:
        pass

    # Magnitude-based outlier filtering: default based on mesh scale
    try:
        if data_points.size > 0:
            if vertices is not None and getattr(vertices, 'size', 0) > 0:
                vnorms = np.linalg.norm(vertices, axis=1)
                mesh_scale = np.nanmax(vnorms) if vnorms.size>0 else 1.0
                mag_thresh_use = max(2.0, float(mesh_scale) * 3.0)
            else:
                mag_thresh_use = 2.0
            norms = np.linalg.norm(data_points, axis=1)
            keep = (np.isfinite(norms)) & (norms <= mag_thresh_use)
            n_before = data_points.shape[0]
            data_points = data_points[keep]
            n_after = data_points.shape[0]
            if n_after < n_before:
                print(f"[Points] Filtered {n_before-n_after} outlier points with norm > {mag_thresh_use:.3f}")
    except Exception:
        pass
    faces = mesh.faces
    faces_pv = np.hstack([[3, *face] for face in faces]).astype(np.int32)

    # Create mesh
    pv_mesh = pv.PolyData(vertices, faces_pv)

    # Create PyVista plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.8, show_edges=False)
    # Ensure points are 2D/3D array for PyVista
    try:
        if data_points.size > 0:
            plotter.add_points(data_points, color='red', point_size=point_size, render_points_as_spheres=True)
    except Exception:
        # best-effort: skip adding points if shape incompatible
        pass

    # Set view
    if view == 'xy': 
        plotter.view_xy()
    elif view == 'xz': 
        plotter.view_xz()
    elif view == 'yz': 
        plotter.view_yz()
    elif view == 'isometric': 
        plotter.view_isometric()
        plotter.camera.up = (0, -1, 0)  # flips vertically
    try:
        if camera_zoom is not None and float(camera_zoom) > 0:
            plotter.camera.zoom(float(camera_zoom))
    except Exception:
        pass

    # Save
    output_path = _svg_path(output_path)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    try:
        plotter.save_graphic(output_path)
    except Exception:
        try:
            plotter.show(screenshot=output_path)
        except Exception:
            pass
    _, png_output_path = _pdf_png_paths(output_path)
    if png_output_path is not None:
        try:
            plotter.show(screenshot=png_output_path)
        except Exception:
            pass
    plotter.close()
    if output_path.lower().endswith('.png'):
        try:
            _crop_whitespace_image(output_path, margin_px=4)
        except Exception:
            pass
    if png_output_path is not None:
        try:
            _crop_whitespace_image(png_output_path, margin_px=4)
        except Exception:
            pass
    print(f"[Points] Saved mesh with points to {output_path}")

def save_metrics_table_paper(general_metrics, intrinsic_metrics=None, out_tex_path='results/metrics_table.tex',
                             caption='Metrics', float_format='%.4f', bold_best=True, display_name_map=None):
    """Write a LaTeX table in two tabulars suitable for direct paste into a paper.

    This implementation ensures actual newline characters are written (no literal "\\n" sequences),
    uses a canonical column ordering favoring JSD (not MMD), and keeps LaTeX backslashes intact.
    
    Args:
        display_name_map: dict mapping method names to display names (e.g., {"ProjectedDDPM": "DDPM (proj.)"})
    """
    if display_name_map is None:
        display_name_map = {}
    os.makedirs(os.path.dirname(out_tex_path) or '.', exist_ok=True)

    def _write_table(f, metrics, col_order, col_modes, caption_star=None):
        # column spec
        f.write('\\begin{tabular}{l')
        for _ in col_order:
            f.write('     >{\\centering\\arraybackslash}p{2.0cm}')
        f.write('}\n')
        # top rule + header
        f.write('\\toprule\n')
        header = ['Method'] + col_order
        f.write(' & '.join(header) + ' \\\\ \n')
        f.write('\\midrule\n')

        # compute best per column
        bests = {}
        for c, mode in zip(col_order, col_modes):
            best_val = None
            for m in metrics:
                v = metrics[m].get(c, None)
                if v is None:
                    continue
                if best_val is None:
                    best_val = v
                else:
                    try:
                        if mode == 'max' and float(v) > float(best_val):
                            best_val = v
                        if mode == 'min' and float(v) < float(best_val):
                            best_val = v
                    except Exception:
                        pass
            bests[c] = best_val

        for m in metrics:
            vals = []
            for c in col_order:
                v = metrics[m].get(c, '')
                if isinstance(v, float) or isinstance(v, int):
                    try:
                        s = float_format % float(v)
                    except Exception:
                        s = str(v)
                else:
                    s = str(v)
                if bold_best and (bests.get(c) is not None):
                    try:
                        if float(s) == float(bests[c]):
                            s = '\\textbf{' + s + '}'
                    except Exception:
                        pass
                vals.append(s)
            # Use display name if available
            display_m = display_name_map.get(m, m)
            f.write(display_m + ' & ' + ' & '.join(vals) + ' \\\\ \n')

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        if caption_star:
            f.write('\\caption*{%s}\n' % caption_star)

    # Build the file
    with open(out_tex_path, 'w') as f:
        f.write('\\begin{table}[h]\n')
        f.write('\\centering\n')

        # General table: pick canonical columns (include Train/Sampling times if present)
        if general_metrics and len(general_metrics) > 0:
            sample = next(iter(general_metrics.values()))
            # Ordering rationale: times first, then coverage (maximize), then other metrics (minimize)
            # Include MMD as a canonical column so it's shown in paper tables
            possible = ['Train time (s/epoch)', 'Sampling time (s)', 'COV', 'Pairwise RMSD', 'MMD', 'Ramachandran JSD', 'FID', 'Class JSD', 'JSD', 'TVD']
            cols = [c for c in possible if c in sample]
            if not cols:
                cols = list(sample.keys())
            modes = []
            for c in cols:
                if 'COV' in c.upper():
                    modes.append('max')
                else:
                    modes.append('min')
            _write_table(f, general_metrics, cols, modes, caption_star='General metrics.')

        # Intrinsic table
        if intrinsic_metrics and len(intrinsic_metrics) > 0:
            # prefer canonical ordering COV, JSD, TVD
            sample2 = next(iter(intrinsic_metrics.values()))
            cols2 = [c for c in ['COV', 'JSD', 'TVD'] if c in sample2]
            if not cols2:
                cols2 = list(sample2.keys())
            modes2 = ['max' if 'COV' in c.upper() else 'min' for c in cols2]
            f.write('\\begin{tabular}{l}\n')
            f.write('\\toprule\n')
            header2 = ['Method'] + cols2
            f.write(' & '.join(header2) + ' \\\\ \n')
            f.write('\\midrule\n')

            # compute bests
            bests2 = {}
            for c, mode in zip(cols2, modes2):
                best_val = None
                for m in intrinsic_metrics:
                    v = intrinsic_metrics[m].get(c, None)
                    if v is None:
                        continue
                    if best_val is None:
                        best_val = v
                    else:
                        try:
                            if mode == 'max' and float(v) > float(best_val):
                                best_val = v
                            if mode == 'min' and float(v) < float(best_val):
                                best_val = v
                        except Exception:
                            pass
                bests2[c] = best_val

            for m in intrinsic_metrics:
                vals = []
                for c in cols2:
                    v = intrinsic_metrics[m].get(c, '')
                    # Keep raw value for numeric comparisons
                    raw_v = intrinsic_metrics[m].get(c, '')
                    display = ''
                    if isinstance(raw_v, float) or isinstance(raw_v, int):
                        try:
                            # Special formatting for MMD: scientific notation in LaTeX
                            if c == 'MMD':
                                s = f"{float(raw_v):.3e}"
                                try:
                                    mant, exp = s.split('e')
                                    mant = mant.rstrip('0').rstrip('.') if '.' in mant else mant
                                    display = '\\text{$%s\\times10^{%d}$}' % (mant, int(exp))
                                except Exception:
                                    display = float_format % float(raw_v)
                            else:
                                display = float_format % float(raw_v)
                        except Exception:
                            display = str(raw_v)
                    else:
                        display = str(raw_v)

                    # Bold the best value (use raw numeric comparison when possible)
                    if bold_best and (bests2.get(c) is not None):
                        try:
                            if isinstance(raw_v, (float, int)) and bests2.get(c) is not None and float(raw_v) == float(bests2[c]):
                                # strip outer $ when bolding LaTeX math to avoid nested $...
                                display = '\\textbf{' + display + '}'
                        except Exception:
                            pass
                    vals.append(display)
                # Use display name if available
                display_m = display_name_map.get(m, m)
                f.write(display_m + ' & ' + ' & '.join(vals) + ' \\\\ \n')

            f.write('\\bottomrule\n')
            f.write('\\end{tabular}\n')
            f.write('\\caption*{Intrinsic metrics (restricted to the constrained manifold).}\n')

        f.write('\\caption{%s}\n' % caption)
        f.write('\\end{table}\n')

    print(f"Saved paper-style metrics table to {out_tex_path}")

from datasets import *
from trainers import *
from utils.constraints import *
from utils.metrics import *

from typing import Optional, Tuple

def _orthonormal_basis_from_pole(n: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a (3,) pole direction n (not necessarily unit), return an orthonormal frame (e1, e2, n_hat)
    with e1, e2 spanning the tangent plane orthogonal to n_hat.
    """
    n_hat = n / (n.norm() + eps)
    # pick a helper vector not parallel to n_hat
    helper = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)
    if torch.abs((helper @ n_hat)).item() > 0.9:
        helper = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype, device=n.device)
    e1 = helper - (helper @ n_hat) * n_hat
    e1 = e1 / (e1.norm() + eps)
    e2 = torch.cross(n_hat, e1)
    return e1, e2, n_hat


def to_intrinsic_2d(
    X: torch.Tensor,
    center: torch.Tensor,
    radius: torch.Tensor,
    method: str = "lambert",              # "lambert" | "stereographic" | "equirectangular"
    pole: Optional[torch.Tensor] = None,  # direction in R^3 defining the map's center
    return_projected: bool = False,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Map 3D points X (N,3) on or near the sphere S(center, radius) to 2D coordinates.

    Steps:
      1) Project X onto the sphere: x_s = c + r * (X - c) / ||X - c||.
      2) Apply a chosen 2D projection with respect to a pole direction.

    Args:
        X: (N,3) tensor of points.
        center: (3,) sphere center.
        radius: scalar tensor or float radius (>0).
        method: one of {"lambert", "stereographic", "equirectangular"}.
        pole: (3,) direction setting the projection center/pole. Defaults to +z.
        return_projected: if True, also return the (N,3) on-sphere projections.
        eps: numerical epsilon.

    Returns:
        uv: (N,2) tensor of 2D coordinates.
        X_proj (optional): (N,3) projected points on the sphere.
    """
    if not torch.is_tensor(radius):
        radius = torch.tensor(radius, dtype=X.dtype, device=X.device)

    center = center.reshape(3).to(dtype=X.dtype, device=X.device)
    if pole is None:
        pole = torch.tensor([0.0, 0.0, 1.0], dtype=X.dtype, device=X.device)
    else:
        pole = pole.reshape(3).to(dtype=X.dtype, device=X.device)

    # 1) Project onto the sphere
    V = X - center  # (N,3)
    norms = V.norm(dim=-1, keepdim=True).clamp_min(eps)  # (N,1)
    Xs = center + (radius * V / norms)  # (N,3), on-sphere

    # Build local tangent frame at the chosen pole direction
    e1, e2, n_hat = _orthonormal_basis_from_pole(pole, eps=eps)

    # Decompose each point on the sphere in this frame
    # p_n is the component along the pole (like "cos colatitude")
    p_n = (Xs @ n_hat) / radius  # (N,)
    p_e1 = (Xs @ e1) / radius    # (N,)
    p_e2 = (Xs @ e2) / radius    # (N,)
    # Now (p_e1, p_e2, p_n) live on the unit sphere: p_e1^2 + p_e2^2 + p_n^2 = 1

    method = method.lower()
    if method == "stereographic":
        # Stereographic from the +n_hat pole onto the plane tangent at -n_hat.
        # Unit-sphere formula: u = (2 * p_perp) / (1 - p_n).
        denom = (1.0 - p_n).clamp_min(eps)
        u = (2.0 * radius) * (p_e1 / denom)
        v = (2.0 * radius) * (p_e2 / denom)
        uv = torch.stack([u, v], dim=-1)

    elif method == "lambert":
        # Lambert azimuthal equal-area centered at n_hat (plane tangent at n_hat).
        # Unit-sphere formula: y = sqrt(2 / (1 + p_n)) * p_perp
        factor = torch.sqrt(2.0 / (1.0 + p_n).clamp_min(eps)) * radius  # (N,)
        u = factor * p_e1
        v = factor * p_e2
        uv = torch.stack([u, v], dim=-1)

    elif method == "equirectangular":
        # Longitude/latitude, then unwrap onto a plane:
        # lon = atan2(p_e2, p_e1), lat = asin(p_n)
        lon = torch.atan2(p_e2, p_e1)    # (-pi, pi]
        lat = torch.asin(p_n.clamp(-1.0, 1.0))  # [-pi/2, pi/2]
        # Map to "meters": u = r * lon, v = r * lat
        # (This is the classic plate carrée; large distortion near poles.)
        u = radius * lon
        v = radius * lat
        uv = torch.stack([u, v], dim=-1)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'lambert', 'stereographic', or 'equirectangular'.")

    return (uv, Xs) if return_projected else (uv, None)


def get_plane_basis(A):
    """
    Given normal vector A (shape: [1, 3]), returns a (3, 2) matrix whose columns are
    orthonormal vectors spanning the plane orthogonal to A.
    """
    A = A.squeeze()
    # Find one vector not parallel to A
    if abs(A[0]) < 0.9:
        v = torch.tensor([1.0, 0.0, 0.0])
    else:
        v = torch.tensor([0.0, 1.0, 0.0])
    # First basis vector: orthogonalize v to A
    u1 = v - torch.dot(v, A) / torch.dot(A, A) * A
    u1 = u1 / torch.norm(u1)
    # Second basis vector: cross product
    u2 = torch.cross(A, u1)
    u2 = u2 / torch.norm(u2)
    # Stack into basis matrix (3 x 2)
    basis = torch.stack([u1, u2], dim=1)
    return basis

def to_intrinsic_2d_plane(X, A, b):
    """
    Given points X (N, 3), a plane normal A, and offset b, returns (N, 2) coordinates in plane's basis.
    """
    basis = get_plane_basis(A)
    # Project onto plane (if needed)
    # Plane: { x | <A, x> = b }
    A = A.squeeze()
    b = b.squeeze()
    # Move points to the plane by subtracting their distance from the plane along A
    # For each x: x_proj = x - (<A, x> - b) * (A / ||A||^2)
    dots = torch.matmul(X, A)
    correction = ((dots - b) / torch.dot(A, A)).unsqueeze(1) * A
    X_proj = X - correction
    # Express in plane basis
    intrinsic = torch.matmul(X_proj, basis)
    return intrinsic


def plot_2d(points, filename, title):
    points = points.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.savefig(filename)
    _, png_filename = _pdf_png_paths(filename)
    if png_filename is not None:
        plt.savefig(png_filename)
    plt.close()

def kde_on_grid(pointsnp, gridsize=200, margin_frac=0.05):
    """
    pointsnp: shape (2, N) numpy array
    Returns: X, Y, Z on a square grid (same x/y extent).
    """
    kde = gaussian_kde(pointsnp)

    x_min, y_min = pointsnp[0].min(), pointsnp[1].min()
    x_max, y_max = pointsnp[0].max(), pointsnp[1].max()

    x_c = 0.5 * (x_min + x_max)
    y_c = 0.5 * (y_min + y_max)

    span_x = max(x_max - x_min, 1e-12)
    span_y = max(y_max - y_min, 1e-12)
    span = max(span_x, span_y)

    half = 0.5 * span * (1.0 + 2.0 * margin_frac)  # symmetric padding

    xs = np.linspace(x_c - half, x_c + half, gridsize)
    ys = np.linspace(y_c - half, y_c + half, gridsize)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z

def compute_shared_norm(
    all_point_sets,
    gridsize=200,
    margin_frac=0.05,
    vmin=DENSITY_VMIN,
    vmax_quantile=0.995,
):
    """
    Pass a list of torch.Tensors shaped (N,2).
    Computes a shared KDE color normalization.

    Density plots intentionally use a repository-wide fixed color mapping:
    viridis with vmin=0.0 and vmax=1.0. The point sets are accepted for
    backwards-compatible call sites, but do not change the color scale.
    """
    return density_color_norm()

def save_standalone_colorbar(
    norm: Normalize | None = None,
    cmap: str = DENSITY_CMAP,
    filename: str = "colorbar.svg",
    label: str = "Density",
    dpi: int = 300,
    height_in: float = 3.5,
    width_in: float = 0.5,
    orientation: str = "vertical",
    tick_params: dict | None = None,
):
    """
    Save a standalone colorbar image using the provided norm + cmap.
    Note: orientation is forced to horizontal regardless of input to ensure consistency across plots.
    """
    norm = density_color_norm()
    cmap = DENSITY_CMAP
    if tick_params is None:
        tick_params = dict(labelsize=9, length=3)

    # Force horizontal orientation regardless of input; if dimensions look vertical swap them.
    orientation = "horizontal"
    if height_in > width_in:
        width_in, height_in = height_in, width_in
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = fig.add_axes([0.05, 0.35, 0.9, 0.3])  # layout tuned for horizontal bar

    ColorbarBase(
        ax,
        cmap=plt.get_cmap(cmap),
        norm=norm,
        orientation=orientation,
        label=label,
    )
    ax.tick_params(**tick_params)
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    _, png_filename = _pdf_png_paths(filename)
    if png_filename is not None:
        fig.savefig(png_filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_2d_density_no_cbar(
    points: torch.Tensor,
    filename: str,
    title: str,
    gridsize: int = 200,
    cmap: str = DENSITY_CMAP,
    point_alpha: float = 0.4,
    dpi: int = 300,
    norm: Normalize | None = None,
    margin_frac: float = 0.05,
    scatter_color: str = "white",
):
    """
    Plot a 2D KDE with scatter overlay, **without** a colorbar, using a shared norm if provided.
    Ensures a perfectly square figure and equal aspect.
    """
    norm = density_color_norm()
    cmap = DENSITY_CMAP
    # Convert to numpy and transpose for gaussian_kde input
    pointsnp = points.detach().cpu().numpy().T  # shape (2, N)

    # Compute KDE on its own grid
    X, Y, Z = kde_on_grid(pointsnp, gridsize=gridsize, margin_frac=margin_frac)

    # Figure: square
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    # Density map with shared norm (uniform color scale across plots).
    # Use an embedded raster image in SVG output instead of a vector QuadMesh;
    # vector mesh cells render as visible hairline grids in many SVG viewers.
    ax.imshow(
        Z,
        extent=(float(X.min()), float(X.max()), float(Y.min()), float(Y.max())),
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="equal",
        zorder=0,
    )

    # Overlay scatter points
    ax.scatter(
        pointsnp[0],
        pointsnp[1],
        s=4,
        alpha=point_alpha,
        color=scatter_color,
        edgecolor="none",
        rasterized=True,
        zorder=1,
    )

    # Ensure square limits: make x/y spans equal with margin
    try:
        x_min = float(pointsnp[0].min())
        x_max = float(pointsnp[0].max())
        y_min = float(pointsnp[1].min())
        y_max = float(pointsnp[1].max())
        x_c = 0.5 * (x_min + x_max)
        y_c = 0.5 * (y_min + y_max)
        span = max(x_max - x_min, y_max - y_min)
        pad = span * float(margin_frac)
        half = 0.5 * span + pad
        ax.set_xlim(x_c - half, x_c + half)
        ax.set_ylim(y_c - half, y_c + half)
    except Exception:
        pass

    # Axis formatting
    # ax.set_title(title, fontsize=14, fontweight="bold")
    # ax.set_xlabel("x", fontsize=14)
    # ax.set_ylabel("y", fontsize=14)
    ax.set_aspect("equal", "box")
    # turn off x ticks
    ax.set_xticks([])
    # turn off y ticks
    ax.set_yticks([])
    ax.grid(False)

    # Save without colorbar
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    _, png_filename = _pdf_png_paths(filename)
    if png_filename is not None:
        fig.savefig(png_filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
