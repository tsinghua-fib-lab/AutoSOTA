"""Entry point for the Bayesian-inference (SBI) benchmark (paper §6.1, Table 1).

Runs Calibrated Bayesian Guidance on the 5 tasks (Lueckmann et al. 2021) using the
shared ``calibrated_guidance`` estimator, and scores C2ST against the reference
posteriors. Paper grad-free targets: 0.505 / 0.513 / 0.584 / 0.507 / 0.525
(average 0.527).

    python run.py sbi                                  # paper config (N=100, K=1000)
    python run.py sbi --estimator reparam              # gradient-based
    python run.py sbi --num-steps 30 --num-particles 200 --quick   # fast check
"""

from __future__ import annotations

import argparse

from experiments.common import seed_everything
from experiments.common.wandb_logging import WandbRun
from experiments.sbi.benchmark import run_benchmark

EXPERIMENT = "sbi"


def _parse_observations(spec: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(out)


def run(args) -> int:
    seed_everything(args.seed)
    observations = _parse_observations(args.observations)
    config = {
        "estimator": args.estimator,
        "num_steps": args.num_steps,
        "num_particles": args.num_particles,
        "observations": list(observations),
        "seed": args.seed,
    }
    results = run_benchmark(
        estimator=args.estimator,
        num_steps=args.num_steps,
        num_particles=args.num_particles,
        observations=observations,
        seed=args.seed,
        device=args.device,
    )

    header = f"{'task':8s} {'C2ST':>8s} {'std':>7s} {'paper':>7s}"
    print(header)
    print("-" * len(header))
    metrics = {}
    for tk in ["task1", "task2", "task3", "task4", "task5"]:
        r = results[tk]
        print(f"{tk:8s} {r['c2st_mean']:8.3f} {r.get('c2st_std', 0.0):7.3f} {r['paper']:7.3f}")
        metrics[f"c2st/{tk}"] = r["c2st_mean"]
    avg = results["average"]
    print(f"{'average':8s} {avg['c2st_mean']:8.3f} {'':7s} {avg['paper']:7.3f}")
    metrics["c2st/average"] = avg["c2st_mean"]

    with WandbRun(EXPERIMENT, config, run_name=args.run_name or f"sbi_{args.estimator}") as wb:
        wb.log(metrics)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py sbi", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--estimator", choices=["reinforce", "reparam"], default="reinforce",
                   help="gradient-free (reinforce) or gradient-based (reparam) CBG.")
    p.add_argument("--num-steps", type=int, default=100, help="outer flow-matching steps N (paper 100).")
    p.add_argument("--num-particles", type=int, default=1000, help="candidates per step K (paper 1000).")
    p.add_argument("--observations", default="1",
                   help="sbibm observation id(s), e.g. '1' or '1-10' (Table 1 averages over 1-10).")
    p.add_argument("--quick", action="store_true", help="(no-op flag; use small --num-steps/--num-particles)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p


def cli(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(cli())
