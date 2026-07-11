#!/usr/bin/env python3
"""
Additional experiments on the 3 original paper tasks.

Same experiments as HSD_addition_experiment (ellipsoid/torus),
now replicated on the original datasets:
  1. externalAerodynamics (Ellipsoid, 0→1)
  2. magnetostatics (Sphere shell, 0→1)
  3. toroidalTransport (Torus, 0→0)

Experiments:
  - HSD (full) vs GNOT'23 / ONO'23 / HAMLET'24 (new baselines)
  - HSD ablation: plain MLP vs pseudo-spectral bilinear

Usage:
    python original_tasks/run_experiments.py
    python original_tasks/run_experiments.py --task aero
"""
import os, sys, json, pickle, time, argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'hodge-spectral-operator'))

from hodge_spectral.operators import HighOrderSpectralOperators
from hodge_spectral.models.unified import UnifiedHSD, train_unified_hsd
from hodge_spectral.utils import LpLoss, count_parameters
from baselines.recent_baselines import GNOT, ONO, HAMLET, train_recent_baseline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================
# Data
# ==============================
class SimpleDS(Dataset):
    def __init__(s, c0, c1, c2, xr, yr):
        s.c0, s.c1, s.c2, s.xr, s.yr = c0, c1, c2, xr, yr
    def __len__(s): return len(s.c0)
    def __getitem__(s, i):
        return (torch.from_numpy(s.c0[i]), torch.from_numpy(s.c1[i]),
                torch.from_numpy(s.c2[i]), torch.zeros(1),
                torch.from_numpy(s.xr[i]), torch.from_numpy(s.yr[i]))


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

TASKS = {
    'externalAerodynamics': {
        'path': f'{DATA_DIR}/externalAerodynamics/flux_field_dataset.pkl',
        'name': 'externalAerodynamics',
        'pts_key': 'points', 'faces_key': 'faces',
        'output_form': 1, 'out_dim': 3,
    },
    'magnetostatics': {
        'path': f'{DATA_DIR}/magnetostatics/flux_field_dataset.pkl',
        'name': 'magnetostatics',
        'pts_key': 'nodes', 'faces_key': 'elements',
        'output_form': 1, 'out_dim': 3,
    },
    'toroidalTransport': {
        'path': f'{DATA_DIR}/toroidalTransport/advdiff_torus_dataset.pkl',
        'name': 'toroidalTransport',
        'pts_key': 'points', 'faces_key': 'faces',
        'output_form': 0, 'out_dim': 1,
    },
}


def load_task(task_name):
    cfg = TASKS[task_name]
    with open(cfg['path'], 'rb') as f:
        d = pickle.load(f)
    pts = d[cfg['pts_key']].astype(np.float64)
    faces = d[cfg['faces_key']].astype(np.int64)

    if task_name == 'toroidalTransport':
        trajs = d['trajectories']
        X = np.array([t[0] for t in trajs], dtype=np.float32)
        Y = np.array([t[-1] for t in trajs], dtype=np.float32)
    else:
        X, Y = d['X_data'], d['Y_data']

    return pts, faces, X, Y, cfg


def prepare(pts, faces, X, Y, k=64):
    n = len(pts)
    host = HighOrderSpectralOperators(pts, faces, k_list=(k, k, k))
    Phi0, Phi1 = host.Phi0[:, :k], host.Phi1[:, :k]

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
    Xtr, Xva, Ytr, Yva = train_test_split(Xtr, Ytr, test_size=0.15, random_state=42)
    xs = np.max(np.abs(Xtr)) + 1e-9
    ys = np.max(np.abs(Ytr)) + 1e-9

    def proc(Xn, Yn):
        x2 = (Xn / xs).astype(np.float32)
        y2 = (Yn / ys).astype(np.float32)
        fX, gX, _ = host.lift_signal(x2)
        c0 = (fX @ Phi0).astype(np.float32)
        c1 = (gX @ Phi1).astype(np.float32)
        c2 = np.zeros((len(x2), k), dtype=np.float32)
        return c0, c1, c2, x2, y2

    return {
        'host': host, 'n': n, 'k': k, 'Phi0': Phi0,
        'train': proc(Xtr, Ytr), 'val': proc(Xva, Yva), 'test': proc(Xte, Yte),
        'xs': xs, 'ys': ys, 'pts': pts,
        'Xtr': Xtr, 'Ytr': Ytr, 'Xva': Xva, 'Yva': Yva, 'Xte': Xte, 'Yte': Yte,
    }


# ==============================
# HSD
# ==============================
def run_hsd(prep, output_form, hidden_dims=(256, 192), res_hidden=128):
    k, n = prep['k'], prep['n']
    Phi_out = torch.from_numpy(prep['Phi0'].astype(np.float32)).to(DEVICE)
    model = UnifiedHSD(prep['host'], output_form, k, k, k, n, k, Phi_out,
                       hidden_dims=hidden_dims, res_hidden=res_hidden).to(DEVICE)
    trl = DataLoader(SimpleDS(*prep['train']), batch_size=64, shuffle=True)
    val = DataLoader(SimpleDS(*prep['val']), batch_size=64)
    model = train_unified_hsd(model, trl, val, 120, DEVICE, lr=3e-3, patience=25)

    model.eval()
    crit = LpLoss()
    ap, ag = [], []
    with torch.no_grad():
        for c0, c1, c2, _, xr, yr in DataLoader(SimpleDS(*prep['test']), batch_size=64):
            p, _, _ = model(c0.to(DEVICE), c1.to(DEVICE), c2.to(DEVICE), xr.to(DEVICE))
            ap.append(p.cpu()); ag.append(yr)
    return crit(torch.cat(ap), torch.cat(ag)).item(), count_parameters(model)


# ==============================
# New baselines
# ==============================
def run_baseline(name, prep, out_dim):
    n = prep['n']
    pts_np = prep['pts'].astype(np.float32)
    xs, ys = prep['xs'], prep['ys']

    xt = torch.from_numpy(np.asarray(prep['Xtr']/xs, dtype=np.float32)).to(DEVICE)
    yt = torch.from_numpy(np.asarray(prep['Ytr']/ys, dtype=np.float32)).to(DEVICE)
    xv = torch.from_numpy(np.asarray(prep['Xva']/xs, dtype=np.float32)).to(DEVICE)
    yv = torch.from_numpy(np.asarray(prep['Yva']/ys, dtype=np.float32)).to(DEVICE)
    xe = torch.from_numpy(np.asarray(prep['Xte']/xs, dtype=np.float32)).to(DEVICE)
    ye = torch.from_numpy(np.asarray(prep['Yte']/ys, dtype=np.float32)).to(DEVICE)

    if name == 'GNOT':
        m = GNOT(n, out_dim, 128, 4, 4).to(DEVICE)
    elif name == 'ONO':
        m = ONO(n, out_dim, 128, 4, 4).to(DEVICE)
    elif name == 'HAMLET':
        m = HAMLET(n, out_dim, 96, 4, 3).to(DEVICE)

    cfg = {'lr': 1e-3, 'epochs': 100, 'batch_size': 64}
    m, _, _ = train_recent_baseline(m, xt, yt, xv, yv, pts_np, 100, cfg, DEVICE)

    m.eval()
    crit = LpLoss()
    pts_t = torch.from_numpy(pts_np).to(DEVICE)
    with torch.no_grad():
        pred = m(xe, pts_t)
        if out_dim == 1 and pred.dim() == 3 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
    return crit(pred, ye).item(), count_parameters(m)


# ==============================
# Main
# ==============================
def run_task(task_name):
    pts, faces, X, Y, cfg = load_task(task_name)
    of, od = cfg['output_form'], cfg['out_dim']
    print(f"\n{'='*60}")
    print(f"  {cfg['name']} | N={len(pts)}, F={len(faces)}, samples={len(X)}")
    print(f"{'='*60}")

    prep = prepare(pts, faces, X, Y)
    results = {}

    # HSD (full)
    print("\n--- HSD ---")
    rl, p = run_hsd(prep, of)
    results['HSD'] = {'rel_l2': round(rl, 4), 'params': p}
    print(f"  {rl:.4f} | {p:,}p")

    # HSD ablation: plain MLP
    print("\n--- HSD (plain MLP) ---")
    rl, p = run_hsd(prep, of, hidden_dims=(256, 192), res_hidden=128)
    results['HSD_plainMLP'] = {'rel_l2': round(rl, 4), 'params': p}
    print(f"  {rl:.4f}")

    # New baselines
    for bl in ['GNOT', 'ONO', 'HAMLET']:
        print(f"\n--- {bl} ---")
        try:
            rl, p = run_baseline(bl, prep, od)
            results[bl] = {'rel_l2': round(rl, 4), 'params': p}
            print(f"  {rl:.4f} | {p:,}p")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[bl] = {'rel_l2': float('nan'), 'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='all',
                        choices=['all', 'externalAerodynamics', 'magnetostatics', 'toroidalTransport'])
    args = parser.parse_args()

    if args.task == 'all':
        # magnetostatics excluded: volume mesh spectral ops too slow
        tasks = ['externalAerodynamics', 'toroidalTransport']
    else:
        tasks = [args.task]

    all_results = {}
    for t in tasks:
        all_results[t] = run_task(t)

    # Save
    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for t, res in all_results.items():
        print(f"\n  {t}:")
        print(f"  {'Method':<16s} {'Rel L2':>8s} {'Params':>10s}")
        print(f"  {'-'*36}")
        for name, r in sorted(res.items(), key=lambda x: x[1].get('rel_l2', 999)):
            print(f"  {name:<16s} {r.get('rel_l2',float('nan')):>8.4f} {r.get('params',0):>10,}")


if __name__ == "__main__":
    main()
