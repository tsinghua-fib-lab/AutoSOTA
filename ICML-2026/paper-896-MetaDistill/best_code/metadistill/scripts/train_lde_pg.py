import argparse
import json
import os
import sys
from typing import Dict, List

# Allow running as `python scripts/train_lde_pg.py` from repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from meta_trainers.pg_trainer import LDEPolicyGradientTrainer
from optimizers.lde import LDE
from tasks import bbob, cec, ef


def _load_funcs(function_set: str, fids: List[int], bounds: List[float]) -> List[Dict]:
    fs = function_set.lower()
    funcs: List[Dict] = []
    for fid in fids:
        if fs == "bbob":
            func = dict(bbob.FUNCTIONS[int(fid)])
            func["xlb"], func["xub"] = float(bounds[0]), float(bounds[1])
        elif fs == "cec":
            func = dict(cec.FUNCTIONS[f"cecf{int(fid)}"])
        elif fs == "ef":
            func = dict(ef.FUNCTIONS[f"ef{int(fid)}"])
        else:
            raise ValueError(f"Unsupported function set: {function_set}")
        funcs.append(func)
    return funcs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lde_config.json")
    parser.add_argument("--function-set", type=str, default="cec", choices=["cec", "bbob", "ef"])
    parser.add_argument("--fids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--bounds", type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument("--ckpt-dir", type=str, default="ckpt/lde_pg")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)

    # Trainer hyperparams default to config, but CLI seed/ckpt-dir/function-set/fids are authoritative.
    n_epochs = int(cfg.get("train_epochs", 150))
    n_trajectories = int(cfg.get("n_trajectories", 20))
    traj_len = int(cfg.get("traj_len", 50))
    lr = float(cfg.get("train_lr", 0.005))

    funcs = _load_funcs(args.function_set, args.fids, bounds=args.bounds)
    model = LDE(cfg)
    trainer = LDEPolicyGradientTrainer(
        model=model,
        training_funcs=funcs,
        n_epochs=n_epochs,
        n_trajectories=n_trajectories,
        traj_len=traj_len,
        lr=lr,
        seed=args.seed,
        ckpt_dir=args.ckpt_dir,
        minimize=bool(cfg.get("minimize", True)),
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    trainer.train()


if __name__ == "__main__":
    main()

