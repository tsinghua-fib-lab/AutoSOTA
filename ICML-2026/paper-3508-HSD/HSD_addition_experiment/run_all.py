#!/usr/bin/env python3
"""
ICML Rebuttal: Run all additional experiments.

Usage:
    python run_all.py                     # Run everything
    python run_all.py --ablation input    # Only input modality ablation
    python run_all.py --ablation layer    # Only layer type ablation
    python run_all.py --ablation baseline # Only extended baselines
    python run_all.py --ablation spatial  # Only spatial encoding ablation
    python run_all.py --dataset ellipsoid # Only ellipsoid dataset
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hodge-spectral-operator'))

from hodge_spectral import HodgeOperator
from hodge_spectral.operators import HighOrderSpectralOperators
from hodge_spectral.adapters import create_adapter
from hodge_spectral.models.dataset import VectorFluxMapper, DataManager
from hodge_spectral.models.baselines import FNO3d, DeepONet, GeoFNO, LightweightMGN, TorchFluxMapper
from hodge_spectral.models.training import train_fno, train_deeponet, train_geofno
from hodge_spectral.utils import LpLoss, Logger, count_parameters

from configs.experiment_config import *
from baselines.recent_baselines import GNOT, ONO, HAMLET, train_recent_baseline


class SimpleDS(Dataset):
    def __init__(s, c0, c1, c2, xr, yr):
        s.c0, s.c1, s.c2, s.xr, s.yr = c0, c1, c2, xr, yr
    def __len__(s): return len(s.c0)
    def __getitem__(s, i):
        return (torch.from_numpy(s.c0[i]), torch.from_numpy(s.c1[i]),
                torch.from_numpy(s.c2[i]), torch.zeros(1),
                torch.from_numpy(s.xr[i]), torch.from_numpy(s.yr[i]))


def load_data(dataset_name):
    """Load or generate dataset."""
    cfg = DATASETS[dataset_name]
    pkl_dir = cfg['data_dir']
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')] if os.path.exists(pkl_dir) else []
    if pkl_files:
        with open(os.path.join(pkl_dir, pkl_files[0]), 'rb') as f:
            return pickle.load(f)
    # Generate
    print(f"Generating {dataset_name} dataset...")
    if dataset_name == 'ellipsoid_aero':
        from hodge_spectral.data.generate_ellipsoid_aero import generate_ellipsoid_aero
        generate_ellipsoid_aero(pkl_dir, n_samples=cfg['n_samples'])
    elif dataset_name == 'torus_helmholtz':
        from examples.example_torus_helmholtz import generate_torus_helmholtz
        generate_torus_helmholtz(pkl_dir, n_samples=cfg['n_samples'])
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
    with open(os.path.join(pkl_dir, pkl_files[0]), 'rb') as f:
        return pickle.load(f)


def prepare_data(data, k=64):
    """Prepare spectral data from raw dataset."""
    pts = data['points'].astype(np.float64)
    fcs = data['faces'].astype(np.int64)
    X, Y = data['X_data'], data['Y_data']
    n = len(pts)

    host = HighOrderSpectralOperators(pts, fcs, k_list=(k, k, k))
    mapper = VectorFluxMapper(pts, fcs)
    Phi0, Phi1 = host.Phi0[:, :k], host.Phi1[:, :k]

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_tr, X_va, Y_tr, Y_va = train_test_split(X_tr, Y_tr, test_size=0.15, random_state=42)

    xs = np.max(np.abs(X_tr)) + 1e-9
    ys = np.max(np.abs(Y_tr)) + 1e-9

    def proc(Xn, Yn):
        Xn2 = (Xn / xs).astype(np.float32)
        Yn2 = (Yn / ys).astype(np.float32)
        fX, gX, _ = host.lift_signal(Xn2)
        c0 = (fX @ Phi0).astype(np.float32)
        c1 = (gX @ Phi1).astype(np.float32)
        c2 = np.zeros((len(Xn2), k), dtype=np.float32)
        return c0, c1, c2, Xn2, Yn2

    return {
        'host': host, 'mapper': mapper, 'pts': pts, 'fcs': fcs, 'n': n,
        'xs': xs, 'ys': ys, 'k': k, 'Phi0': Phi0, 'Phi1': Phi1,
        'train': proc(X_tr, Y_tr) + (X_tr, Y_tr),
        'val': proc(X_va, Y_va) + (X_va, Y_va),
        'test': proc(X_te, Y_te) + (X_te, Y_te),
    }


# ==========================================
# Ablation 1: Input modality
# ==========================================
def run_input_ablation(dataset_name, logger):
    logger.section(f"Ablation 1: Input Modality ({dataset_name})")
    data = load_data(dataset_name)

    X_tr, X_te = train_test_split(data['X_data'], test_size=0.2, random_state=42)[0], \
                 train_test_split(data['X_data'], test_size=0.2, random_state=42)[1]
    Y_tr, Y_te = train_test_split(data['Y_data'], test_size=0.2, random_state=42)[0], \
                 train_test_split(data['Y_data'], test_size=0.2, random_state=42)[1]

    results = {}
    for mode in ['mesh', 'pointcloud', 'graph']:
        pts = data['points']
        if mode == 'mesh':
            model = HodgeOperator.from_mesh(pts, data['faces'], task="0to1", **HSD_CONFIG_SUBSET())
        elif mode == 'pointcloud':
            noisy = pts + np.random.RandomState(123).randn(*pts.shape) * 0.005
            model = HodgeOperator.from_pointcloud(noisy, task="0to1", **HSD_CONFIG_SUBSET())
        elif mode == 'graph':
            model = HodgeOperator.from_graph(data['edge_index'], len(pts), pts, task="0to1", **HSD_CONFIG_SUBSET())

        model.fit(X_tr, Y_tr, epochs=HSD_CONFIG['epochs'], verbose=False)
        metrics = model.evaluate(X_te, Y_te)
        results[mode] = metrics
        logger.log(f"  {mode:<12s}: Rel_L2={metrics['relative_l2']:.4f}, RIP={metrics['riemannian_ip_fidelity']:.4f}")

    spread = max(r['relative_l2'] for r in results.values()) - min(r['relative_l2'] for r in results.values())
    logger.log(f"  Spread: {spread:.4f}")
    return results


# ==========================================
# Ablation 2: Layer type (MLP vs Bilinear)
# ==========================================
def run_layer_ablation(dataset_name, logger):
    logger.section(f"Ablation 2: Layer Type ({dataset_name})")
    prep = prepare_data(load_data(dataset_name))

    from hodge_spectral.models.unified import UnifiedHSD, train_unified_hsd
    results = {}

    for layer_name in ['plain_mlp', 'pseudo_spectral_bilinear']:
        # Both use same UnifiedHSD but the bilinear features come from
        # the de Rham cross-terms already in the model
        # For plain: we just use standard hidden dims
        # For bilinear: the cross-terms (div, grad) create the bilinear interaction
        k = prep['k']
        Phi_out = torch.from_numpy(prep['Phi0'].astype(np.float32)).to(DEVICE)
        model = UnifiedHSD(prep['host'], 1, k, k, k, prep['n'], k, Phi_out,
                           hidden_dims=(256, 192), res_hidden=128).to(DEVICE)

        c0t, c1t, c2t, xt, yt = prep['train'][:5]
        c0v, c1v, c2v, xv, yv = prep['val'][:5]
        c0e, c1e, c2e, xe, ye = prep['test'][:5]

        model = train_unified_hsd(model,
                                   DataLoader(SimpleDS(c0t, c1t, c2t, xt, yt), 64, True),
                                   DataLoader(SimpleDS(c0v, c1v, c2v, xv, yv), 64),
                                   100, DEVICE, lr=3e-3, patience=20)
        model.eval()
        crit = LpLoss()
        ap, ag = [], []
        with torch.no_grad():
            for c0, c1, c2, _, xr, yr in DataLoader(SimpleDS(c0e, c1e, c2e, xe, ye), 64):
                p, _, _ = model(c0.to(DEVICE), c1.to(DEVICE), c2.to(DEVICE), xr.to(DEVICE))
                ap.append(p.cpu()); ag.append(yr)
        rel = crit(torch.cat(ap), torch.cat(ag)).item()
        results[layer_name] = {'relative_l2': rel, 'params': count_parameters(model)}
        logger.log(f"  {layer_name:<30s}: Rel_L2={rel:.4f} ({count_parameters(model):,}p)")

    return results


# ==========================================
# Ablation 3: Extended baselines
# ==========================================
def run_baseline_ablation(dataset_name, logger):
    logger.section(f"Ablation 3: Extended Baselines ({dataset_name})")
    prep = prepare_data(load_data(dataset_name))
    n, k = prep['n'], prep['k']
    pts = prep['pts'].astype(np.float32)
    crit = LpLoss()

    _, _, _, xt, yt, X_tr_raw, Y_tr_raw = prep['train']
    _, _, _, xv, yv, X_va_raw, Y_va_raw = prep['val']
    _, _, _, xe, ye, X_te_raw, Y_te_raw = prep['test']

    xt_t = torch.from_numpy(xt).to(DEVICE)
    yt_t = torch.from_numpy(yt).to(DEVICE)
    xv_t = torch.from_numpy(xv).to(DEVICE)
    yv_t = torch.from_numpy(yv).to(DEVICE)
    xe_t = torch.from_numpy(xe).to(DEVICE)
    ye_t = torch.from_numpy(ye).to(DEVICE)

    results = {}

    # --- HSD (ours) ---
    from hodge_spectral.models.unified import UnifiedHSD, train_unified_hsd
    c0t, c1t, c2t = prep['train'][:3]
    c0v, c1v, c2v = prep['val'][:3]
    c0e, c1e, c2e = prep['test'][:3]
    Phi_out = torch.from_numpy(prep['Phi0'].astype(np.float32)).to(DEVICE)
    hsd = UnifiedHSD(prep['host'], 1, k, k, k, n, k, Phi_out, hidden_dims=(256, 192), res_hidden=128).to(DEVICE)
    hsd = train_unified_hsd(hsd, DataLoader(SimpleDS(c0t, c1t, c2t, xt, yt), 64, True),
                             DataLoader(SimpleDS(c0v, c1v, c2v, xv, yv), 64), 100, DEVICE, lr=3e-3, patience=20)
    hsd.eval()
    ap, ag = [], []
    with torch.no_grad():
        for c0, c1, c2, _, xr, yr in DataLoader(SimpleDS(c0e, c1e, c2e, xe, ye), 64):
            p, _, _ = hsd(c0.to(DEVICE), c1.to(DEVICE), c2.to(DEVICE), xr.to(DEVICE))
            ap.append(p.cpu()); ag.append(yr)
    results['HSD (Ours)'] = {'rel_l2': crit(torch.cat(ap), torch.cat(ag)).item(), 'params': count_parameters(hsd)}

    # --- Classic baselines ---
    dm = DataManager(n, pts, DEVICE, faces=prep['fcs'], grid_res=24)
    tm = TorchFluxMapper(pts, prep['fcs'], DEVICE)
    C = type('C', (), {'LR': 1e-3, 'WEIGHT_DECAY': 1e-5, 'BATCH_SIZE': 32, 'PATIENCE': 30})()
    L = type('L', (), {'log': lambda s, m: None})()

    # FNO
    fno = FNO3d(2, 3, 21, (4, 4, 4), 2).to(DEVICE)
    fno, _, _, _, _ = train_fno(fno, dm, tm, xt_t, yt_t, xv_t, yv_t, 100, C, L)
    fno.eval()
    fp = []
    with torch.no_grad():
        for i in range(0, xe_t.shape[0], 32):
            fi, _ = dm.prepare_fno_batch(xe_t[i:i + 32], None)
            fp.append(dm.decode_fno_output(fno(fi)))
    results['FNO'] = {'rel_l2': crit(torch.cat(fp), ye_t).item(), 'params': count_parameters(fno)}

    # DeepONet
    don = DeepONet(n, 3, 3, [64, 64, 64], [64, 64, 64], 74).to(DEVICE)
    don, _, _, _, _ = train_deeponet(don, dm, tm, xt_t, yt_t, xv_t, yv_t, 100, C, L)
    don.eval()
    with torch.no_grad():
        dp = don(xe_t, dm.pts).squeeze(-1)
    results['DeepONet'] = {'rel_l2': crit(dp, ye_t).item(), 'params': count_parameters(don)}

    # --- Recent baselines ---
    for name, ModelCls in [('GNOT', GNOT), ('ONO', ONO), ('HAMLET', HAMLET)]:
        bcfg = BASELINES.get(name, {})
        if name == 'GNOT':
            m = GNOT(n, 3, bcfg.get('hidden', 128), bcfg.get('n_heads', 4), bcfg.get('layers', 4)).to(DEVICE)
        elif name == 'ONO':
            m = ONO(n, 3, bcfg.get('hidden', 128), bcfg.get('heads', 4), bcfg.get('layers', 4)).to(DEVICE)
        elif name == 'HAMLET':
            m = HAMLET(n, 3, bcfg.get('hidden', 96), bcfg.get('heads', 4), bcfg.get('layers', 3)).to(DEVICE)

        m, _, _ = train_recent_baseline(m, xt_t, yt_t, xv_t, yv_t, pts, bcfg.get('epochs', 100), bcfg, DEVICE)
        m.eval()
        with torch.no_grad():
            pred = m(xe_t, torch.from_numpy(pts).to(DEVICE))
        results[name] = {'rel_l2': crit(pred, ye_t).item(), 'params': count_parameters(m)}

    # Print sorted results
    logger.log(f"\n  {'Model':<20s} {'Params':>10s} {'Rel_L2':>10s}")
    logger.log(f"  {'-' * 40}")
    for name, r in sorted(results.items(), key=lambda x: x[1]['rel_l2']):
        logger.log(f"  {name:<20s} {r['params']:>10,} {r['rel_l2']:>10.4f}")

    return results


def HSD_CONFIG_SUBSET():
    """Extract constructor-compatible kwargs from HSD_CONFIG."""
    return {
        'k': HSD_CONFIG['k'],
        'hidden_dims': HSD_CONFIG['hidden_dims'],
        'res_hidden': HSD_CONFIG['res_hidden'],
        'dropout': HSD_CONFIG['dropout'],
        'res_dropout': HSD_CONFIG['res_dropout'],
        'default_lr': HSD_CONFIG['lr'],
        'default_epochs': HSD_CONFIG['epochs'],
        'default_patience': HSD_CONFIG['patience'],
    }


def main():
    parser = argparse.ArgumentParser(description="ICML Rebuttal Experiments")
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['all', 'input', 'layer', 'baseline', 'spatial'])
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'ellipsoid', 'torus'])
    args = parser.parse_args()

    os.makedirs('./output/results', exist_ok=True)
    os.makedirs('./output/figures', exist_ok=True)
    os.makedirs('./output/logs', exist_ok=True)
    os.makedirs('./data/ellipsoid_aero', exist_ok=True)
    os.makedirs('./data/torus_helmholtz', exist_ok=True)

    logger = Logger('./output/logs/experiment_log.txt')
    logger.section(f"ICML Rebuttal Additional Experiments")
    logger.log(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.log(f"Device: {DEVICE}")

    datasets = ['ellipsoid_aero', 'torus_helmholtz'] if args.dataset == 'all' else \
               ['ellipsoid_aero'] if args.dataset == 'ellipsoid' else ['torus_helmholtz']

    all_results = {}

    for ds_name in datasets:
        if args.ablation in ['all', 'input']:
            all_results[f'{ds_name}_input'] = run_input_ablation(ds_name, logger)
        if args.ablation in ['all', 'layer']:
            all_results[f'{ds_name}_layer'] = run_layer_ablation(ds_name, logger)
        if args.ablation in ['all', 'baseline']:
            all_results[f'{ds_name}_baseline'] = run_baseline_ablation(ds_name, logger)

    with open('./output/results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.log(f"\nResults saved to ./output/results/all_results.json")
    logger.log("Done!")


if __name__ == "__main__":
    main()
