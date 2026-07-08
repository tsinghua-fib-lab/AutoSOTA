#!/usr/bin/env python3
"""Compare frameworks and plot per-seed curves (no aggregation).

Motivation: aggregated mean+std plots make it hard to see what each seed looks like.
This script renders a multi-panel figure (one panel per seed) for each (fid, dim).

- Supports per-variant SSFT (same CLI conventions as eval_compare_frameworks.py)
- Supports per-variant optimizer config override via --variant-config
- Optionally saves raw traces to JSON for auditing.

Notes:
- The plotted metric is best-gen by default (best fitness in current population per generation).
- The y-axis is log10 scale with an automatic positive shift if values are <= 0.
"""

import argparse
import io
import json
import os
import pickle
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.storage
from torch_basic_settings import DEVICE, DTYPE

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

from tasks import TaskProblem, bbob
from tasks.utils import genOffset, setOffset


def _cpu_load_from_bytes(data):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


torch.storage._load_from_bytes = _cpu_load_from_bytes


def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_bbob_task(
    fid: int,
    dim: int,
    bounds: List[float],
    seed: int,
    offset_seed: Optional[int] = None,
    fixed_offsets: Optional[Dict[int, Dict]] = None,
) -> Tuple[TaskProblem, Dict]:
    func = dict(bbob.FUNCTIONS[int(fid)])
    func["xlb"], func["xub"] = float(bounds[0]), float(bounds[1])

    if fixed_offsets is not None:
        if int(fid) not in fixed_offsets:
            raise ValueError(f"Missing fixed offset for fid={fid} (dim={dim}).")
        setOffset(fun=func, kwargs=fixed_offsets[int(fid)])
        _set_global_seed(seed)
        return TaskProblem(fun=func, repaire=True, dim=dim), func

    # BBOB offset depends on RNG. Decouple task instance (offset_seed) from evaluation randomness (seed).
    off_seed = int(seed) if offset_seed is None else int(offset_seed)
    _set_global_seed(off_seed)
    genOffset(fun=func, dim=dim)
    _set_global_seed(seed)

    return TaskProblem(fun=func, repaire=True, dim=dim), func


def _safe_log10(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y_min = np.min(y)
    if not np.isfinite(y_min):
        return np.log10(np.maximum(y, 1e-30))
    if y_min <= 0:
        y = y + (-y_min) + 1e-12
    else:
        y = y + 1e-12
    return np.log10(y)


def _get_best_f(pop: torch.Tensor, minimize: bool = True) -> float:
    f = pop[..., 0].view(-1)
    return float(torch.min(f).item() if minimize else torch.max(f).item())


def _load_json_if_exists(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_base_config(name: str, config_path: Optional[str] = None) -> Dict:
    cfg_path = os.path.join(REPO_ROOT, "configs", f"{name}_config.json")
    if config_path:
        cfg_path = config_path if os.path.isabs(config_path) else os.path.join(REPO_ROOT, config_path)
    return _load_json_if_exists(cfg_path)


def _load_learned_ckpt(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if (
        isinstance(obj, dict)
        and "best_params" in obj
        and isinstance(obj["best_params"], torch.Tensor)
        and obj["best_params"].ndim == 1
    ):
        return "metabbo_best_params", obj["best_params"]
    return "state_dict", obj


def _load_les_best_params_into_gradbased(algo: torch.nn.Module, best_params: torch.Tensor) -> None:
    if not hasattr(algo, "net"):
        raise ValueError("Expected GradBasedLES-like module with `.net` attribute.")

    net = getattr(algo, "net")
    dk = int(net.attn_hidden_dims)
    mh = int(net.mlp_hidden_dims)
    mlp_in_dim = int(net.fc1.in_features)

    params = best_params.detach().clone().to(device=DEVICE, dtype=DTYPE).view(-1)
    idx = 0

    def take(num: int) -> torch.Tensor:
        nonlocal idx
        out = params[idx : idx + num]
        idx += num
        return out

    wq = take(3 * dk).view(3, dk)
    bq = take(dk).view(dk)
    wk = take(3 * dk).view(3, dk)
    bk = take(dk).view(dk)
    wv = take(3 * 1).view(3, 1)
    bv = take(1).view(1)

    w1 = take(mlp_in_dim * mh).view(mlp_in_dim, mh)
    b1 = take(mh).view(mh)
    w_mu = take(mh * 1).view(mh, 1)
    b_mu = take(1).view(1)
    w_sigma = take(mh * 1).view(mh, 1)
    b_sigma = take(1).view(1)

    if idx != params.numel():
        raise RuntimeError(f"LES best_params slicing mismatch: consumed={idx}, total={params.numel()}")

    with torch.no_grad():
        net.wq.weight.copy_(wq.T)
        net.wq.bias.copy_(bq)
        net.wk.weight.copy_(wk.T)
        net.wk.bias.copy_(bk)
        net.wv.weight.copy_(wv.T)
        net.wv.bias.copy_(bv)

        net.fc1.weight.copy_(w1.T)
        net.fc1.bias.copy_(b1)
        net.mu_head.weight.copy_(w_mu.T)
        net.mu_head.bias.copy_(b_mu)
        net.sigma_head.weight.copy_(w_sigma.T)
        net.sigma_head.bias.copy_(b_sigma)


def _build_optimizer(
    *,
    name: str,
    dim: int,
    popsize: int,
    seed: int,
    ckpt: str,
    needs_grad: bool,
    action_mode: Optional[str] = None,
    config_path: Optional[str] = None,
) -> torch.nn.Module:
    name = name.lower()

    if name == "les":
        config = _load_base_config("les", config_path)
        config.update({"popsize": popsize, "problemdim": dim, "minimize": True, "seed": seed})

        ckpt_kind, payload = _load_learned_ckpt(ckpt)
        if ckpt_kind == "metabbo_best_params":
            if needs_grad:
                from optimizers.les import GradBasedLES

                algo = GradBasedLES(config)
                _load_les_best_params_into_gradbased(algo, payload)
                return algo

            from optimizers.les import GradFreeLES

            algo = GradFreeLES(config)
            algo.update_params(payload)
            return algo

        from optimizers.les import GradBasedLES

        algo = GradBasedLES(config)
        algo.load_state_dict(payload, strict=True)
        return algo

    if name == "lde":
        from optimizers.lde import LDE

        config = _load_base_config("lde", config_path)
        config.update({"popsize": popsize, "problemdim": dim, "minimize": True, "seed": seed})
        if action_mode is not None:
            config["action_mode"] = str(action_mode)

        algo = LDE(config)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        algo.load_state_dict(state, strict=True)
        return algo

    if name == "lga":
        ckpt_kind, payload = _load_learned_ckpt(ckpt)
        config = _load_base_config("lga", config_path)
        config.update({"popsize": popsize, "problemdim": dim, "minimize": True, "seed": seed})

        if ckpt_kind == "metabbo_best_params":
            from optimizers.lga import LGA

            algo = LGA(config)
            algo.update_params(payload)
            return algo

        from optimizers.lga import GradBasedLGA

        algo = GradBasedLGA(config)
        algo.load_state_dict(payload, strict=True)
        return algo

    if name == "pom":
        config = _load_base_config("pom", config_path)
        config.update({"popsize": popsize, "problemdim": dim, "minimize": True, "seed": seed})

        from optimizers import POM

        algo = POM(config)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        algo.load_state_dict(state, strict=True)
        return algo

    raise ValueError(f"Unsupported optimizer: {name}")


def run_single(
    *,
    optimizer: str,
    ckpt: str,
    fid: int,
    dim: int,
    popsize: int,
    budget: int,
    seed: int,
    bounds: List[float],
    do_adapt: bool,
    adapt_lr: float,
    ss_interval: int,
    loss_eps: float,
    adapt_steps: int,
    action_mode: Optional[str] = None,
    config_path: Optional[str] = None,
    init_seed_offset: int = 0,
    offset_seed: Optional[int] = None,
    fixed_offsets: Optional[Dict[int, Dict]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    task, _func = _make_bbob_task(
        fid=fid,
        dim=dim,
        bounds=bounds,
        seed=seed,
        offset_seed=offset_seed,
        fixed_offsets=fixed_offsets,
    )

    init_seed = int(seed) + int(init_seed_offset)
    _set_global_seed(init_seed)
    x = task.genRandomPop((1, popsize, dim))
    pop, _ = task.calfitness(x)
    _set_global_seed(seed)
    if not bool(torch.isfinite(pop[..., 0]).all()):
        raise ValueError("Non-finite fitness in initial population.")

    do_adapt = bool(do_adapt)
    ss_interval = max(1, int(ss_interval))
    adapt_steps = int(adapt_steps)
    if do_adapt and adapt_steps != 1:
        raise ValueError("Only adapt_steps=1 is supported (keeps FE accounting consistent).")

    algo = _build_optimizer(
        name=optimizer,
        dim=dim,
        popsize=popsize,
        seed=seed,
        ckpt=ckpt,
        needs_grad=do_adapt,
        action_mode=action_mode,
        config_path=config_path,
    )

    if hasattr(algo, "reset"):
        algo.reset()
    if hasattr(algo, "set_seed"):
        algo.set_seed(seed)

    if do_adapt:
        if hasattr(algo, "train"):
            algo.train()
        if not hasattr(algo, "parameters"):
            raise ValueError("SSFT requires a torch.nn.Module with parameters().")
        opt = torch.optim.Adam(algo.parameters(), lr=float(adapt_lr))
    else:
        if hasattr(algo, "eval"):
            algo.eval()
        opt = None

    pop_history: deque = deque(maxlen=ss_interval)
    pop_history.append(pop.detach().clone())

    total_fe = int(pop.shape[1])
    fe_trace = [total_fe]
    best_gen = [_get_best_f(pop, minimize=True)]

    budget = int(budget)

    while True:
        if not bool(torch.isfinite(pop[..., 0]).all()):
            raise ValueError("Population contains NaN/Inf fitness.")

        gen_fe = int(pop.shape[1])
        if total_fe + gen_fe > budget:
            break

        if do_adapt:
            hist_pop = pop_history[0]
            l_hist = torch.mean(hist_pop[..., 0].view(-1))

            out = algo(pop, task)
            pop2 = out[0] if isinstance(out, tuple) else out
            if not bool(torch.isfinite(pop2[..., 0]).all()):
                raise ValueError("Non-finite fitness after optimizer step.")

            l_cur = torch.mean(pop2[..., 0].view(-1))
            denom = torch.abs(l_hist).clamp_min(float(loss_eps))
            loss = (l_cur - l_hist) / denom
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite SSFT loss: fid={fid} dim={dim} seed={seed} loss={loss}")

            assert opt is not None
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Detach internal state to avoid graph growth.
            if hasattr(algo, "export_state") and hasattr(algo, "import_state"):
                algo.import_state(algo.export_state())

            pop = pop2.detach()
            pop_history.append(pop.clone())
        else:
            with torch.no_grad():
                out = algo(pop, task)
            pop2 = out[0] if isinstance(out, tuple) else out
            if not bool(torch.isfinite(pop2[..., 0]).all()):
                raise ValueError("Non-finite fitness after optimizer step.")
            pop = pop2
            pop_history.append(pop.clone())

        total_fe += gen_fe
        best_gen.append(_get_best_f(pop, minimize=True))
        fe_trace.append(total_fe)

    return np.array(fe_trace, dtype=np.int64), np.array(best_gen, dtype=np.float64)


def _parse_kv_list(items: List[str], what: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {what} (expected name=value): {item}")
        name, val = item.split("=", 1)
        name = name.strip()
        val = val.strip()
        if not name or not val:
            raise ValueError(f"Invalid {what}: {item}")
        out[name] = val
    return out


def _parse_ssft_variant_j(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --ssft-variant (expected name=j): {item}")
        name, j = item.split("=", 1)
        name = name.strip()
        j = int(j.strip())
        if j < 1:
            raise ValueError(f"SSFT interval j must be >= 1, got {j} (from {item})")
        out[name] = j
    return out


def _parse_bbob_offsets_spec(items: List[str]) -> Dict[int, str]:
    """Parse fixed BBOB offsets specs.

    Format: dim=path (repeatable). Example: --bbob-offsets 30=offsets/bbob_offsets_dim30.pkl
    """
    out: Dict[int, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --bbob-offsets (expected dim=path): {item}")
        dim_s, path = item.split("=", 1)
        dim_s = dim_s.strip()
        path = path.strip()
        if not dim_s or not path:
            raise ValueError(f"Invalid --bbob-offsets: {item}")
        try:
            dim = int(dim_s)
        except Exception as e:
            raise ValueError(f"Invalid --bbob-offsets dim: {item}") from e
        out[dim] = path
    return out


def _load_bbob_offsets(path: str) -> Dict[int, Dict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in offsets pkl, got {type(obj)}: {path}")
    return obj


def _layout(n: int) -> Tuple[int, int]:
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, 3
    cols = 4
    rows = (n + cols - 1) // cols
    return rows, cols


def _label_for_variant(vname: str, ssft_variant_j: Dict[str, int], ss_interval_default: int) -> str:
    if vname in ssft_variant_j:
        return f"{vname}(SSFT j={int(ssft_variant_j[vname])})"
    return vname


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", type=str, required=True, choices=["les", "lde", "lga", "pom"])
    p.add_argument("--dims", type=int, nargs="+", default=[30])
    p.add_argument("--popsize", type=int, default=100)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--bounds", type=float, nargs=2, default=[-10.0, 10.0])
    p.add_argument("--fids", type=int, nargs="+", default=[i for i in range(1, 25)])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--offset-seed", type=int, default=None, help="If set, use this seed to generate BBOB offset for all evaluation seeds (keeps task instance fixed across seeds).")
    p.add_argument("--init-seed-offset", type=int, default=0, help="Additive offset applied to evaluation seed when generating the initial population (breaks correlation with genOffset/xopt when bounds match).")
    p.add_argument(
        "--bbob-offsets-dir",
        type=str,
        default=None,
        help="If set, load fixed BBOB offsets from this directory. Expected files: bbob_offsets_dim{dim}.pkl",
    )
    p.add_argument(
        "--bbob-offsets",
        type=str,
        action="append",
        default=[],
        help="Repeatable. Format: dim=path/to/bbob_offsets_dim{dim}.pkl. Overrides --bbob-offsets-dir for that dim.",
    )
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--curve", type=str, default="best-gen", choices=["best-gen", "best-so-far"])

    p.add_argument(
        "--variant",
        type=str,
        action="append",
        required=True,
        help="Repeatable. Format: name=/path/to/ckpt.pth",
    )
    p.add_argument(
        "--variant-config",
        type=str,
        action="append",
        default=[],
        help="Optional. Repeatable. Format: name=path/to/config.json (per-variant optimizer config override).",
    )

    p.add_argument("--ssft-variant", type=str, action="append", default=[], help="Repeatable. Format: name=j")
    p.add_argument("--adapt-lr", type=float, default=1e-4)
    p.add_argument("--adapt-steps", type=int, default=1)
    p.add_argument("--ss-interval", type=int, default=1)
    p.add_argument("--loss-eps", type=float, default=1e-12)
    p.add_argument(
        "--action-mode",
        type=str,
        default=None,
        choices=["sample", "rsample", "mean"],
        help="For LDE only. Use rsample/mean for differentiable SSFT.",
    )

    p.add_argument("--save-traces", type=str, default=None, help="Optional. Save per-seed traces JSON to this path")

    args = p.parse_args()

    variants = _parse_kv_list(args.variant, "--variant")
    variant_cfgs = _parse_kv_list(args.variant_config, "--variant-config") if args.variant_config else {}
    ssft_variant_j = _parse_ssft_variant_j(args.ssft_variant)
    ssft_variants = set(ssft_variant_j.keys())

    bbob_offsets_spec = _parse_bbob_offsets_spec(args.bbob_offsets)
    fixed_offsets_by_dim: Dict[int, Dict[int, Dict]] = {}
    fixed_offsets_paths: Dict[int, str] = {}
    for dim in args.dims:
        dim = int(dim)
        path: Optional[str] = None
        if dim in bbob_offsets_spec:
            path = bbob_offsets_spec[dim]
        elif args.bbob_offsets_dir:
            path = os.path.join(str(args.bbob_offsets_dir), f"bbob_offsets_dim{dim}.pkl")

        if path is None:
            continue

        pth = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Offsets file not found for dim={dim}: {pth}")
        fixed_offsets_by_dim[dim] = _load_bbob_offsets(pth)
        fixed_offsets_paths[dim] = pth

    os.makedirs(args.outdir, exist_ok=True)

    # Keep a JSON audit of traces if requested.
    traces_out: Dict[str, object] = {
        "meta": {
            "optimizer": args.optimizer,
            "variants": variants,
            "variant_configs": variant_cfgs,
            "ssft_variant_j": {k: int(v) for k, v in ssft_variant_j.items()},
            "adapt_lr": float(args.adapt_lr),
            "adapt_steps": int(args.adapt_steps),
            "ss_interval": int(args.ss_interval),
            "budget": int(args.budget),
            "popsize": int(args.popsize),
            "seeds": [int(s) for s in args.seeds],
            "offset_seed": (None if args.offset_seed is None else int(args.offset_seed)),
            "init_seed_offset": int(args.init_seed_offset),
            "bbob_offsets_dir": (None if args.bbob_offsets_dir is None else str(args.bbob_offsets_dir)),
            "bbob_offsets": {str(k): str(v) for k, v in bbob_offsets_spec.items()},
            "bbob_offsets_loaded": {str(k): str(v) for k, v in fixed_offsets_paths.items()},
            "fids": [int(f) for f in args.fids],
            "dims": [int(d) for d in args.dims],
            "curve": str(args.curve),
        },
        "data": {},
    }

    for dim in args.dims:
        dim = int(dim)
        dim_key = str(dim)
        fixed_offsets = fixed_offsets_by_dim.get(dim)
        traces_out["data"][dim_key] = {}

        for fid in args.fids:
            fid = int(fid)

            # traces[vname][seed] = (fe, y)
            per_seed_fe: Dict[int, np.ndarray] = {}
            traces: Dict[str, Dict[int, np.ndarray]] = {vname: {} for vname in variants.keys()}

            # collect
            for vname, ckpt in variants.items():
                do_adapt = vname in ssft_variants
                v_ss_interval = int(ssft_variant_j.get(vname, args.ss_interval))
                for seed in args.seeds:
                    seed = int(seed)
                    fe, best_gen = run_single(
                        optimizer=args.optimizer,
                        ckpt=ckpt,
                        fid=fid,
                        dim=dim,
                        popsize=int(args.popsize),
                        budget=int(args.budget),
                        seed=seed,
                        bounds=list(args.bounds),
                        offset_seed=(None if args.offset_seed is None else int(args.offset_seed)),
                        init_seed_offset=int(args.init_seed_offset),
                        do_adapt=do_adapt,
                        adapt_lr=float(args.adapt_lr),
                        ss_interval=int(v_ss_interval),
                        loss_eps=float(args.loss_eps),
                        adapt_steps=int(args.adapt_steps),
                        action_mode=args.action_mode,
                        config_path=variant_cfgs.get(vname),
                        fixed_offsets=fixed_offsets,
                    )
                    if str(args.curve).lower() == "best-so-far":
                        best_gen = np.minimum.accumulate(best_gen)
                    traces[vname][seed] = best_gen
                    if seed not in per_seed_fe:
                        per_seed_fe[seed] = fe

            # shift based on ALL seeds+variants
            all_vals = []
            for vname in variants.keys():
                for seed in args.seeds:
                    all_vals.append(traces[vname][int(seed)])
            all_vals = np.concatenate(all_vals, axis=0)
            min_val = float(np.min(all_vals))
            shift = (-min_val + 1e-12) if min_val <= 0 else 0.0

            seeds_sorted = [int(s) for s in args.seeds]
            rows, cols = _layout(len(seeds_sorted))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.6), sharex=True, sharey=True)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes_flat = axes.reshape(-1)

            # consistent colors per variant
            color_cycle = plt.get_cmap("tab10")
            vnames = list(variants.keys())
            color_map = {v: color_cycle(i % 10) for i, v in enumerate(vnames)}

            handles_labels = None
            for idx, seed in enumerate(seeds_sorted):
                ax = axes_flat[idx]
                fe = per_seed_fe[seed]
                for vname in vnames:
                    y = traces[vname][seed] + shift
                    log_y = _safe_log10(y)
                    label = _label_for_variant(vname, ssft_variant_j, int(args.ss_interval))
                    ax.plot(fe, log_y, label=label, linewidth=1.25, color=color_map[vname])
                ax.set_title(f"seed={seed}")
                if handles_labels is None:
                    handles_labels = ax.get_legend_handles_labels()
                ax.grid(True, alpha=0.2)

            # turn off unused panels
            for k in range(len(seeds_sorted), len(axes_flat)):
                axes_flat[k].axis("off")

            fig.suptitle(
                f"{args.optimizer.upper()}  BBOB f{fid}  dim={dim}  pop={args.popsize}  B={args.budget}  ({args.curve})",
                fontsize=12,
            )
            fig.text(0.5, 0.04, "Function Evaluations", ha="center")
            fig.text(0.03, 0.5, "log10(Best Fitness)", va="center", rotation="vertical")

            if handles_labels is not None:
                handles, labels = handles_labels
                fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=9, frameon=False)

            fig.tight_layout(rect=[0.05, 0.08, 1, 0.93])

            outpath = os.path.join(
                args.outdir,
                f"{args.optimizer}_framework_compare_byseed_f{fid}_dim{dim}_pop{args.popsize}_B{args.budget}_" + ("bestsofar" if str(args.curve).lower() == "best-so-far" else "bestgen") + ".png",
            )
            fig.savefig(outpath, dpi=150)
            plt.close(fig)

            # save traces summary
            if args.save_traces:
                traces_out["data"][dim_key][str(fid)] = {
                    "shift": shift,
                    "fe": {str(s): per_seed_fe[int(s)].tolist() for s in seeds_sorted},
                    "traces": {
                        v: {str(s): traces[v][int(s)].tolist() for s in seeds_sorted}
                        for v in vnames
                    },
                    "plot": outpath,
                }

    if args.save_traces:
        out_path = args.save_traces if os.path.isabs(args.save_traces) else os.path.join(REPO_ROOT, args.save_traces)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(traces_out, f, indent=2, ensure_ascii=False)
        print(f"[OK] wrote traces: {out_path}")


if __name__ == "__main__":
    main()
