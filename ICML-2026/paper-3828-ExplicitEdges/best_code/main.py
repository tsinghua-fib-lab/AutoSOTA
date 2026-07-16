"""RCDP-UCB — single entry point for every experiment in the paper.

Examples
--------
  python main.py list
  python main.py run post-serving --phi abs --d 10 --K 10 --delay strategic
  python main.py run linear       --delay stochastic --mean 100 --std 100
  python main.py reproduce post-serving --n_runs 10
  python main.py reproduce --all        --n_runs 10
  python main.py figures                            # regenerate the README figures (assets/)

Experiments (see `python main.py list`):
  linear          Fig. 1 — linear setting, no post-serving context
  post-serving    Fig. 2 — all methods learn the post-serving mapping
  representative  Fig. 3 — only RCDP-UCB exploits the post-serving mapping
  std-ablation    Appendix — RCDP-UCB vs stochastic-delay variance
"""
import argparse
import experiments as X


def _build_cfg(a, exp):
    cfg = dict(d=a.d, K=a.K, corruption_budget=a.corruption_budget,
               mean=a.mean, std=a.std, delay_mag=a.delay_mag, delay_type=a.delay)
    if a.phi is not None:
        cfg["phi"] = a.phi
    if a.e is not None:
        cfg["e"] = a.e
    return cfg


def cmd_list(_a):
    print("Available experiments:\n")
    for name, spec in X.EXPERIMENTS.items():
        n = "ablation" if spec["sweep"] is None else f"{len(spec['sweep'])} configs"
        print(f"  {name:15s} {spec['paper']}")
        print(f"  {'':15s}   phis={spec.get('phis', ['-'])}  |  sweep: {n}\n")
    print("Run one:        python main.py run <experiment> [options]")
    print("Reproduce all:  python main.py reproduce <experiment> | --all")


def cmd_run(a):
    if a.experiment not in X.EXPERIMENTS:
        raise SystemExit(f"unknown experiment '{a.experiment}'. Try: python main.py list")
    if a.experiment == "std-ablation":
        X._run_std_ablation(a.T, a.n_runs)
        return
    cfg = _build_cfg(a, a.experiment)
    print(f"[{a.experiment}] {X.EXPERIMENTS[a.experiment]['paper']}")
    X.run_one(a.experiment, cfg, T=a.T, n_runs=a.n_runs)


def cmd_reproduce(a):
    targets = list(X.EXPERIMENTS) if a.all else [a.experiment]
    if not a.all and a.experiment not in X.EXPERIMENTS:
        raise SystemExit(f"unknown experiment '{a.experiment}'. Try: python main.py list")
    for exp in targets:
        X.reproduce(exp, T=a.T, n_runs=a.n_runs)


def cmd_figures(a):
    """Regenerate the README teaser images into assets/."""
    X.make_figures(n_runs=a.n_runs, T=a.T, rerun=not a.no_rerun)


def _add_common(p):
    p.add_argument("--T", type=int, default=2000, help="horizon")
    p.add_argument("--n_runs", type=int, default=5, help="independent seeded repetitions")


def _add_run_opts(p):
    p.add_argument("--phi", default=None,
                   choices=["polynomial", "abs", "cosine", "sinusoidal", "piecewise", "linear", "interaction"],
                   help="pre->post-serving mapping (default: experiment's first)")
    p.add_argument("--d", type=int, default=10, help="pre-serving dimension")
    p.add_argument("--K", type=int, default=10, help="number of arms")
    p.add_argument("--e", type=int, default=None, help="post-serving dimension (default: experiment's)")
    p.add_argument("--delay", default="stochastic", choices=["stochastic", "strategic", "clean"])
    p.add_argument("--mean", type=float, default=100.0, help="stochastic-delay mean")
    p.add_argument("--std", type=float, default=100.0, help="stochastic-delay std")
    p.add_argument("--delay_mag", type=int, default=10000, help="strategic-delay budget")
    p.add_argument("--corruption_budget", type=int, default=25, help="corruption budget C")


def main():
    parser = argparse.ArgumentParser(prog="main.py", description="RCDP-UCB experiments (ICML 2026).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available experiments").set_defaults(func=cmd_list)

    pr = sub.add_parser("run", help="run a single configuration")
    pr.add_argument("experiment", help="experiment name (see `list`)")
    _add_run_opts(pr); _add_common(pr); pr.set_defaults(func=cmd_run)

    pp = sub.add_parser("reproduce", help="run an experiment's full paper sweep")
    pp.add_argument("experiment", nargs="?", default=None, help="experiment name")
    pp.add_argument("--all", action="store_true", help="reproduce every experiment")
    _add_common(pp); pp.set_defaults(func=cmd_reproduce)

    pf = sub.add_parser("figures", help="regenerate the README teaser images (assets/)")
    pf.add_argument("--n_runs", type=int, default=10, help="independent seeded repetitions")
    pf.add_argument("--T", type=int, default=2000, help="horizon")
    pf.add_argument("--no-rerun", dest="no_rerun", action="store_true",
                    help="re-plot from existing results/ without re-running the configs")
    pf.set_defaults(func=cmd_figures)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
