"""
Addon Metrics Wrapper (epoch-based plotting, real-time console prints)
---------------------------------------------------------------------
- Non-intrusive, optimizer-agnostic.
- Logs BEFORE each optimizer step: grad_L1, p_sat_all, w_l2, w_inf (plus time).
- Tracks non-fault weight stats: nf_mean, nf_std, nf_l2.
- Logs per-layer avg grad L1 norm via grad_per_layer JSON, with fallback to detect model automatically.
- CSV has no 'step' column.
- Console prints include per-layer stats.
- When --plot is set, produces epoch-averaged figures for all metrics and per-layer gradients.

Usage:
    python addon_metrics_wrapper.py \
      --metrics_csv mnist_fault.csv --limit 0.5 --sat_eps 1e-3 \
      --print_every 10 --plot --fig_dir figs -- \
      simple_snn.py --data_path propdata/MNIST --batch_size 100 --num_epochs 15 --Fault True --fault_ratio 0.3
"""

import sys, argparse, csv, math, time, runpy, json, torch, os
import pandas as pd
import torch.nn as nn

from pathlib import Path
from collections import defaultdict

# ── addon_metrics_wrapper.py 최상단 가까이 ──
_name_cache = {}  # id(module) -> pretty name

def _find_module_path(mod):
    if id(mod) in _name_cache:
        return _name_cache[id(mod)]

    import __main__
    roots = []

    maybe_model = getattr(__main__, 'model', None)
    maybe_net = getattr(__main__, 'net', None)
    for m in (maybe_model, maybe_net):
        if isinstance(m, torch.nn.Module):
            roots.append(m)

    for obj in __main__.__dict__.values():
        if isinstance(obj, torch.nn.Module) and obj is not maybe_model:
            roots.append(obj)

    for root in roots:
        for name, m in root.named_modules():
            if m is mod:
                _name_cache[id(mod)] = name or root.__class__.__name__
                return _name_cache[id(mod)]

    return None

# --- live container for current‑step z stats (cleared every optimiser.step) ---
_z_cur = {}
_z_cnt_cur = {}

def _install_z_hook():
    """Monkey‑patch nn.Module.__call__ so every Linear / Conv layer logs mean |z|."""
    orig_call = torch.nn.Module.__call__

    def patched(self, *args, **kwargs):
        out = orig_call(self, *args, **kwargs)

        if isinstance(self, (torch.nn.Linear,
                             torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            lname = getattr(self, "_pretty_name", None)
            if lname is None:
                lname = _find_module_path(self)  # NEW: 정식 경로 시도
                if lname is None:  # Fallback
                    lname = f"{self._get_name()}_{id(self):x}"
                self._pretty_name = lname

            # (a) 기존: mean |z|
            _z_cur[lname] = out.detach().mean().item()

            # (b) NEW: threshold-proximal count
            thr = float(getattr(self, wrapper_args.thr_attr, 1.0))
            delta = wrapper_args.z_delta
            cnt = ((out >= thr - delta) & (out <= thr + delta)).sum().item()
            _z_cnt_cur[lname] = int(cnt)  # ← Tensor → int 변환

        return out

    torch.nn.Module.__call__ = patched

def _flush_z_snapshot():
    snap = dict(_z_cur); _z_cur.clear()
    return json.dumps(snap)

def _flush_zcnt_snapshot():
    snap = dict(_z_cnt_cur); _z_cnt_cur.clear()
    return json.dumps(snap)

def _grad_per_z_layers(model) -> dict[str, float]:
    out = {}
    for name, m in model.named_modules():
        if not name:
            continue
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM, nn.GRU)):
            vals = []
            for p in m.parameters(recurse=False):
                if p.grad is None:
                    continue
                vals.append(p.grad.detach().abs().mean())
            out[name] = float(torch.stack(vals).mean().item()) if vals else 0.0
    return out

_param_whitelist = None  # set of id(p) we will count in metrics

def _build_param_whitelist_from_model(model):
    import torch.nn as nn
    allow = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM, nn.GRU)
    allow_ids = set()
    for m in model.modules():
        if isinstance(m, allow):
            # weight / bias (둘 다 포함)
            for name, p in m.named_parameters(recurse=False):
                if name in ("weight", "bias") and (p is not None):
                    allow_ids.add(id(p))
    return allow_ids

# ---------- Parse wrapper-specific args ----------
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--metrics_csv", type=str, default="metrics_log.csv",
                    help="CSV file to write metrics per optimizer step")
parser.add_argument("--sat_eps", type=float, default=1e-3,
                    help="Epsilon margin for saturation: |w| >= limit - sat_eps")
parser.add_argument("--limit", type=float, default=1.0,
                    help="Absolute weight limit used by the target script")
parser.add_argument("--quiet", type=bool, default=False, help="Silence console prints from addon")
parser.add_argument("--print_every", type=int, default=50,
                    help="Print every N optimizer iteration (1 = print every iteration)")
parser.add_argument("--plot", type=bool, default=False, help="Plot metrics after run finishes")
parser.add_argument("--fig_dir", type=str, default="figs", help="Directory to save figures")
parser.add_argument("--steps_per_epoch", type=int, default=None,
                    help="Override: number of optimizer steps per epoch (drop_last assumed in target)")
parser.add_argument("--target", type=str, default="vgg_snn.py",
                    help="Path to the original training script to run")
parser.add_argument('--z_delta', type=float, default=0.05,
                    help='|z – threshold| ≤ delta')
parser.add_argument('--thr_attr', type=str, default='v_threshold',
                    help='threshold attribute name in the model')
parser.add_argument('--wrapper_off', type=bool, default=True,
                    help='If True, do NOT collect/print/plot any metrics; just run the target script.')

wrapper_args, passthrough = parser.parse_known_args()

if not wrapper_args.wrapper_off:
    _install_z_hook()

# Determine target path
target_path = Path(wrapper_args.target).as_posix()
if not Path(target_path).exists():
    if len(passthrough) > 0 and Path(passthrough[0]).exists():
        target_path = Path(passthrough[0]).as_posix()
        passthrough = passthrough[1:]
    else:
        raise SystemExit(f"[addon] Target script not found: {target_path}")

# ---------- Extract basic training args from passthrough ----------
def _get_arg(flag, default=None, cast=str):
    if flag in passthrough:
        i = passthrough.index(flag)
        if i + 1 < len(passthrough):
            try:
                return cast(passthrough[i+1])
            except Exception:
                return default
    return default

batch_size = _get_arg("--batch_size", 100, int)
data_path  = _get_arg("--data_path", "propdata/MNIST", str)


# Default train set size assumptions
if ("MNIST" in data_path) or ("FMNIST" in data_path) or ("FashionMNIST" in data_path):
    total_train = 60000
elif ("CIFAR" in data_path) or ("CIFAR10" in data_path) or ("CIFAR100" in data_path):
    total_train = 50000
else:
    total_train = None

# steps_per_epoch (drop_last=True)
if wrapper_args.steps_per_epoch is not None and wrapper_args.steps_per_epoch > 0:
    steps_per_epoch = int(wrapper_args.steps_per_epoch)
else:
    steps_per_epoch = max(1, total_train // int(batch_size))

# ---------- Monkey-patch .step() for ALL Optimizers ----------
csv_path = Path(wrapper_args.metrics_csv).as_posix()
sat_eps = float(wrapper_args.sat_eps)
wlimit = float(wrapper_args.limit)
print_every = max(1, int(wrapper_args.print_every))

# Prepare CSV header (NO 'step' column)
if not wrapper_args.wrapper_off:
    _fieldnames = [
        "time", "grad_L1", "p_sat_all", "w_l2", "w_inf",
        "nf_mean", "nf_std", "nf_l2",
        "grad_per_layer", "z_per_layer", "z_cnt_per_layer"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames)
        writer.writeheader()

if not wrapper_args.wrapper_off:
    def _compute_stats_from_groups(param_groups, wlimit, sat_eps, whitelist=None):
        total_elems = 0
        sat_hits_abs = 0
        w2 = 0.0
        winf = 0.0
        grad_L1 = 0.0
        weights = []

        for g in param_groups:
            if g.get("ignore_in_metrics", False):
                continue

            for p in g.get("params", []):
                if p is None or p.data is None:
                    continue
                # ★ 여기! Conv/Linear 화이트리스트 이외는 스킵
                if (whitelist is not None) and (id(p) not in whitelist):
                    continue

                W = p.data
                if W.numel() == 0:
                    continue

                if p.grad is not None:
                    grad_L1 += p.grad.data.abs().sum().item()

                A = W.abs()
                sat_hits_abs += (A >= (wlimit - sat_eps)).sum().item()
                total_elems += W.numel()
                weights.append(W.view(-1))

                w2 += float((W ** 2).sum().item())
                wmax = float(A.max().item())
                if wmax > winf:
                    winf = wmax

        p_sat_all = (sat_hits_abs / total_elems) if total_elems > 0 else 0.0
        w_l2 = math.sqrt(w2)

        w_cat = torch.cat(weights) if weights else torch.tensor([])
        mask = (w_cat.abs() < (wlimit - sat_eps))

        if mask.any():
            nf = w_cat[mask]
            nf_mean = nf.mean().item()
            nf_std = nf.std().item()
            nf_l2 = nf.square().sum().sqrt().item()
        else:
            nf_mean = nf_std = nf_l2 = 0.0

        return grad_L1, p_sat_all, w_l2, winf, nf_mean, nf_std, nf_l2


    _original_steps = {}
    _local_step = 0


    def _get_root_model_from_main():
        import __main__, torch

        for key in ("model", "net"):
            m = getattr(__main__, key, None)
            if isinstance(m, torch.nn.Module):
                return m

        cand = [obj for obj in __main__.__dict__.values() if isinstance(obj, torch.nn.Module)]
        if cand:
            return max(cand, key=lambda m: sum(p.numel() for p in m.parameters()))
        return None


    def _make_step_wrapper(cls):
        orig = cls.step
        _original_steps[cls] = orig

        def _wrapped_step(self, *args, **kwargs):
            global _local_step, _param_whitelist

            # Compute metrics BEFORE update
            model = _get_root_model_from_main()
            if (_param_whitelist is None) and (model is not None):
                _param_whitelist = _build_param_whitelist_from_model(model)

            grad_L1, p_sat_all, w_l2, w_inf, nf_mean, nf_std, nf_l2 = \
                _compute_stats_from_groups(self.param_groups, wlimit, sat_eps, whitelist=_param_whitelist)

            z_snapshot_json = _flush_z_snapshot()
            zcnt_snapshot_json = _flush_zcnt_snapshot()

            # Compute gradient per layer (avg L1) with fallback
            model = _get_root_model_from_main()

            layer_stats = _grad_per_z_layers(model) if model is not None else {}
            grad_per_layer_json = json.dumps(layer_stats)

            # CSV append
            row = {
                "time": time.time(),
                "grad_L1": grad_L1,
                "p_sat_all": p_sat_all,
                "w_l2": w_l2,
                "w_inf": w_inf,
                "nf_mean": nf_mean,
                "nf_std": nf_std,
                "nf_l2": nf_l2,
                "grad_per_layer": grad_per_layer_json,
                "z_per_layer": z_snapshot_json,
                "z_cnt_per_layer": zcnt_snapshot_json
            }
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_fieldnames)
                writer.writerow(row)

            # Real-time console print
            if not wrapper_args.quiet:
                epoch_idx = _local_step // steps_per_epoch
                step_in_ep = _local_step % steps_per_epoch
                if _local_step % print_every == 0 or step_in_ep == steps_per_epoch - 1:
                    print(f"[addon] epoch={epoch_idx} step={step_in_ep}/{steps_per_epoch - 1} | "
                          f"grad_L1={grad_L1:.3e} | p_sat_all={p_sat_all:.3f} | "
                          f"w_l2={w_l2:.3f} | w_inf={w_inf:.3f} | "
                          f"nf_mean={nf_mean:.3f} | nf_std={nf_std:.3f} | nf_l2={nf_l2:.3f} | "
                          f"grad_per_layer={grad_per_layer_json} | z_per_layer={z_snapshot_json} | z_cnt_per_layer={zcnt_snapshot_json}")

            _local_step += 1
            return orig(self, *args, **kwargs)

        return _wrapped_step


    # Patch all optimizer classes
    patched = 0
    for name, obj in torch.optim.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, torch.optim.Optimizer) and obj is not torch.optim.Optimizer:
            if hasattr(obj, 'step'):
                try:
                    obj.step = _make_step_wrapper(obj); patched += 1
                except:
                    pass

    if not wrapper_args.quiet:
        print(f"[addon] Patched step() for {patched} optimizer classes.")
        print(f"[addon] steps_per_epoch={steps_per_epoch}, print_every={print_every}")

# ---------- Run the original script ----------
sys.argv = [target_path] + passthrough
runpy.run_path(target_path, run_name="__main__")
print(f"[addon] Metrics logged to: {csv_path}")

# ---------- Z per layer plotting function ----------
def plot_z_per_layer(csv_path: str, fig_dir: str = "figs",
                     steps_per_epoch: int | None = None):
    """
    Read the `z_per_layer` JSON column produced by the wrapper and
    drop one PNG per layer showing the mean |z| trend.

    Args
    ----
    csv_path : str
        CSV file generated during training.
    fig_dir : str, default 'figs'
        Output directory for PNG files.
    steps_per_epoch : int | None
        If provided, also plots an epoch-averaged curve (dotted).
    """
    # ── 0. sanity ───────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if "z_per_layer" not in df.columns:
        print("[plot_z_per_layer] No `z_per_layer` column found – skipping.")
        return

    # ── 1. JSON → dict[list]로 풀기 ──────────────────────────────
    per_layer: dict[str, list[float]] = defaultdict(list)
    for snap_json in df["z_per_layer"].dropna():
        snap = json.loads(snap_json)          # e.g. {"conv1":0.023, "fc":0.004}
        for layer, val in snap.items():
            per_layer[layer].append(val)

    os.makedirs(fig_dir, exist_ok=True)

    # ── 2. 레이어별 플롯 ─────────────────────────────────────────
    for layer, series in per_layer.items():
        plt.figure()
        plt.plot(series, label="step-wise")
        if steps_per_epoch:
            # epoch 단위로 평균 내면 더 매끄러운 곡선 확인 가능
            epochs   = len(series) // steps_per_epoch
            epochavg = [
                sum(series[i*steps_per_epoch:(i+1)*steps_per_epoch]) / steps_per_epoch
                for i in range(epochs)
            ]
            plt.plot(
                [i*steps_per_epoch + steps_per_epoch/2 for i in range(epochs)],
                epochavg, linestyle="--", marker="o", label="epoch-avg"
            )

        plt.title(f"mean z per step  –  {layer}")
        plt.xlabel("step")
        plt.ylabel("mean z")
        plt.legend()
        out_path = os.path.join(fig_dir, f"z_{layer}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_path}")

def plot_zcnt_per_layer(csv_path: str, fig_dir: str = "figs",
                        steps_per_epoch: int | None = None):
    """
    Read the `z_cnt_per_layer` JSON column produced by the wrapper and
    drop one PNG per layer showing the count of |z| within delta of threshold.

    Args
    ----
    csv_path : str
        CSV file generated during training.
    fig_dir : str, default 'figs'
        Output directory for PNG files.
    steps_per_epoch : int | None
        If provided, also plots an epoch-averaged curve (dotted).
    """
    # ── 0. sanity ───────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if "z_cnt_per_layer" not in df.columns:
        print("[plot_zcnt_per_layer] No `z_cnt_per_layer` column found – skipping.")
        return

    # ── 1. JSON → dict[list]로 풀기 ──────────────────────────────
    per_layer: dict[str, list[int]] = defaultdict(list)
    for snap_json in df["z_cnt_per_layer"].dropna():
        snap = json.loads(snap_json)          # e.g. {"conv1":23, "fc":4}
        for layer, val in snap.items():
            per_layer[layer].append(val)

    os.makedirs(fig_dir, exist_ok=True)

    # ── 2. 레이어별 플롯 ─────────────────────────────────────────
    for layer, series in per_layer.items():
        plt.figure()
        plt.plot(series, label="step-wise")
        if steps_per_epoch:
            # epoch 단위로 평균 내면 더 매끄러운 곡선 확인 가능
            epochs   = len(series) // steps_per_epoch
            epochavg = [
                sum(series[i*steps_per_epoch:(i+1)*steps_per_epoch]) / steps_per_epoch
                for i in range(epochs)
            ]
            plt.plot(
                [i*steps_per_epoch + steps_per_epoch/2 for i in range(epochs)],
                epochavg, linestyle="--", marker="o", label="epoch-avg"
            )

        plt.title(f"count of z within delta  –  {layer}")
        plt.xlabel("step")
        plt.ylabel("count")
        plt.legend()
        out_path = os.path.join(fig_dir, f"zcnt_{layer}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_path}")

# ---------- Optional plotting ----------
if (not wrapper_args.wrapper_off) and wrapper_args.plot:
    try:
        import matplotlib.pyplot as plt

        fig_dir = Path(wrapper_args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        # Epoch aggregation for scalar metrics
        n_steps = len(rows)
        n_epochs_est = max(1, (n_steps + steps_per_epoch - 1) // steps_per_epoch)
        epoch_x = list(range(n_epochs_est))
        agg_keys = ["grad_L1", "p_sat_all", "w_l2", "w_inf", "nf_mean", "nf_std", "nf_l2"]
        agg = {k: [0.0]*n_epochs_est for k in agg_keys}
        cnt = [0]*n_epochs_est
        for idx, row in enumerate(rows):
            e = min(idx // steps_per_epoch, n_epochs_est - 1)
            for k in agg_keys:
                try: agg[k][e] += float(row[k])
                except: pass
            cnt[e] += 1
        for k in agg_keys:
            for e in range(n_epochs_est):
                agg[k][e] = agg[k][e]/cnt[e] if cnt[e]>0 else float('nan')

        # Plot scalar metrics
        for metric in agg_keys:
            plt.figure(); plt.plot(epoch_x, agg[metric])
            plt.title(metric); plt.xlabel("epoch"); plt.ylabel(metric)
            plt.tight_layout(); plt.savefig((fig_dir/f"{metric}.png").as_posix())
            plt.close()

        # Epoch aggregation for per-layer gradients
        layer_names = set()
        for row in rows:
            data = json.loads(row["grad_per_layer"])
            layer_names.update(data.keys())
        agg_layer = {ln: [0.0]*n_epochs_est for ln in layer_names}
        cnt_layer = [0]*n_epochs_est
        for idx, row in enumerate(rows):
            e = min(idx // steps_per_epoch, n_epochs_est - 1)
            data = json.loads(row["grad_per_layer"])
            for ln, val in data.items(): agg_layer[ln][e] += val
            cnt_layer[e] += 1
        for ln in agg_layer:
            for e in range(n_epochs_est):
                agg_layer[ln][e] = agg_layer[ln][e]/cnt_layer[e] if cnt_layer[e]>0 else float('nan')

        # Plot per-layer gradients
        for ln, vals in agg_layer.items():
            plt.figure(); plt.plot(epoch_x, vals)
            plt.title(f"grad_per_layer_{ln}"); plt.xlabel("epoch"); plt.ylabel("avg grad L1 norm")
            plt.tight_layout(); plt.savefig((fig_dir/f"grad_per_layer_{ln}.png").as_posix())
            plt.close()

        print(f"[addon] Epoch-based figures saved to: {fig_dir.as_posix()}")

        plot_z_per_layer(wrapper_args.metrics_csv,
                         fig_dir=wrapper_args.fig_dir,
                         steps_per_epoch=wrapper_args.steps_per_epoch)

        plot_zcnt_per_layer(wrapper_args.metrics_csv,
                            fig_dir=wrapper_args.fig_dir,
                            steps_per_epoch=wrapper_args.steps_per_epoch)

    except Exception as e:
        print(f"[addon] Plotting failed: {e}")
