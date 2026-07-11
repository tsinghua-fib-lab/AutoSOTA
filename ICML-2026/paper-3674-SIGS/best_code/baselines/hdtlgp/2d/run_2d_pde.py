# run_2d_pde.py — HD-TLGP baseline
# Re-implementation of the method from:
# Cao et al. "An Interpretable Approach to High-Dimensional PDEs." AAAI 2024.
# doi:10.1609/aaai.v38i18.30050
# 2-D runner for: damped wave & Poisson–Gauss (2/3/4 centers), with Protocol-1 KB vs Protocol-2 primitives.
import argparse, time, math, random, numpy as np, torch, sympy as sp, csv
from Tr_GPSR2D_patched import define_gp, tr_gp, str_to_individual
from Dataset2D_patched import (
    dataset_damped_wave_2d,
    dataset_poisson_gauss_2d_2centers,
    dataset_poisson_gauss_2d_3centers,
    dataset_poisson_gauss_2d_4centers,
)
from evaluate2D_patched import make_evaluator_2d
from deap import gp

def deap_tree_to_sympy(ind, pset):
    x = sp.Symbol("x"); y = sp.Symbol("y"); t = sp.Symbol("t")
    stack = []
    for node in reversed(ind):
        if isinstance(node, gp.Terminal):
            v = getattr(node, "value", None)
            if isinstance(v, str):
                if v in ("x","x1","ARG0"): stack.append(x); continue
                if v in ("y","x2","ARG1"): stack.append(y); continue
                if v in ("t","ARG2"):      stack.append(t); continue
            stack.append(sp.Float(v))
        else:
            ar = node.arity
            args = [stack.pop() for _ in range(ar)][::-1]
            name = node.name
            if name == "add":  stack.append(args[0] + args[1])
            elif name == "sub":stack.append(args[0] - args[1])
            elif name == "mul":stack.append(args[0] * args[1])
            elif name == "pdiv": stack.append(args[0] / args[1])
            elif name == "sin": stack.append(sp.sin(args[0]))
            elif name == "cos": stack.append(sp.cos(args[0]))
            elif name == "exp": stack.append(sp.exp(args[0]))
            elif name == "tanh":stack.append(sp.tanh(args[0]))
            elif name == "sqrt":stack.append(sp.sqrt(args[0]))
            else: raise ValueError(f"Unknown primitive {name}")
    assert len(stack) == 1
    return sp.simplify(stack[0])

# ---------- Protocol-1 KB motifs ----------
def kb_exprs_for_2d(problem, pparams):
    if problem == "damped_wave2d":
        k     = float(pparams.get("k", 0.5))
        omega = float(pparams.get("omega", 0.4))
        gamma = float(pparams.get("gamma", 0.3))
        x0    = float(pparams.get("x0", 0.0))
        y0    = float(pparams.get("y0", 0.0))
        r = f"sqrt(add(mul(sub(x,{x0:.12g}),sub(x,{x0:.12g})), mul(sub(y,{y0:.12g}),sub(y,{y0:.12g}))))"
        # Manufactured radial motif + separable sinusoid-time block
        radial = f"cos(sub(mul({k:.12g}, {r}), mul({omega:.12g}, t)))"
        sep = f"mul(mul(sin(mul({math.pi:.6g}, x)), sin(mul({math.pi:.6g}, y))), mul(cos(mul({omega:.12g}, t)), exp(mul({-gamma:.12g}, t))))"
        
        return [radial, sep]

    # Poisson–Gauss (multiple centers)
    if problem.startswith("poisson_gauss2d_"):
        centers = pparams.get("centers", [(0.5,0.5)])
        sigma   = float(pparams.get("sigma", 0.12))
        motifs = []
        # boundary mask motif (only as a factor; final evaluator masks again)
        mask = f"mul(sin(mul({math.pi:.6g}, x)), sin(mul({math.pi:.6g}, y)))"
        motifs.append(mask)
        # include each Gaussian (unmasked)
        for (cx,cy) in centers:
            quad = f"add(mul(sub(x,{cx:.12g}),sub(x,{cx:.12g})), mul(sub(y,{cy:.12g}),sub(y,{cy:.12g})))"
            gauss = f"exp(pdiv(mul({-1.0:.12g}, {quad}), {2.0*sigma*sigma:.12g}))"
            motifs.append(gauss)
        # also a combined sum-of-Gaussians primitive
        if len(centers) > 1:
            # sum gaussians in prefix: add(g1, add(g2, g3)) etc.
            g_exprs = []
            for (cx,cy) in centers:
                quad = f"add(mul(sub(x,{cx:.12g}),sub(x,{cx:.12g})), mul(sub(y,{cy:.12g}),sub(y,{cy:.12g})))"
                g_exprs.append(f"exp(pdiv(mul({-1.0:.12g}, {quad}), {2.0*sigma*sigma:.12g}))")
            # fold with add
            s = g_exprs[0]
            for gi in g_exprs[1:]:
                s = f"add({s}, {gi})"
            motifs.append(s)
        return motifs

    return []

def build_kb(pset, problem, pparams):
    return [str_to_individual(s, pset) for s in kb_exprs_for_2d(problem, pparams)]

def main():
    ap = argparse.ArgumentParser(description="2D PDE GP (damped wave & Poisson–Gauss 2/3/4 centers) with protocol-1/2")
    ap.add_argument("--problem",  choices=[
        "damped_wave2d",
        "poisson_gauss2d_2c",
        "poisson_gauss2d_3c",
        "poisson_gauss2d_4c",
    ], required=True)
    ap.add_argument("--protocol", type=int, choices=[1,2], required=True,
                    help="1 = motif KB + primitives; 2 = primitives only")
    ap.add_argument("--max-seconds", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-interval", type=float, default=5.0)
    ap.add_argument("--csv-log", type=str, default=None,
                    help="optional CSV path; default is {problem}_protocol_{protocol}.csv")
    # dataset overrides
    ap.add_argument("--dw-k", type=float, default=0.5)
    ap.add_argument("--dw-omega", type=float, default=0.4)
    ap.add_argument("--dw-gamma", type=float, default=0.3)
    ap.add_argument("--dw-x0", type=float, default=0.0)
    ap.add_argument("--dw-y0", type=float, default=0.0)
    ap.add_argument("--pg-sigma", type=float, default=0.12)
    ap.add_argument("--max-gens", type=int, default=25, help="Generation cap (optional)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # dataset + meta
    if args.problem == "damped_wave2d":
        X_train, y_train, pparams, F_str, ptype = dataset_damped_wave_2d(
            k=args.dw_k, omega=args.dw_omega, gamma=args.dw_gamma, x0=args.dw_x0, y0=args.dw_y0, return_meta=True,n_x=12, n_y=12, n_t=12
        )
        d = 2; kb_tag = "damped_wave2d"
    elif args.problem == "poisson_gauss2d_2c":
        X_train, y_train, pparams, F_str, ptype = dataset_poisson_gauss_2d_2centers(
            sigma=args.pg_sigma, return_meta=True, n_x=30, n_y=30
        )
        d = 2; kb_tag = "poisson_gauss2d_2c"
    elif args.problem == "poisson_gauss2d_3c":
        X_train, y_train, pparams, F_str, ptype = dataset_poisson_gauss_2d_3centers(
            sigma=args.pg_sigma, return_meta=True, n_x=30, n_y=30
        )
        d = 2; kb_tag = "poisson_gauss2d_3c"
    else:  # "poisson_gauss2d_4c"
        X_train, y_train, pparams, F_str, ptype = dataset_poisson_gauss_2d_4centers(
            sigma=args.pg_sigma, return_meta=True,
             n_x=30, n_y=30
        )
        d = 2; kb_tag = "poisson_gauss2d_4c"

    toolbox, pset = define_gp(X_train, y_train, d=d, protocol=args.protocol)
    KB = build_kb(pset, kb_tag, pparams) if args.protocol == 1 else []

    evaluator = make_evaluator_2d(ptype=ptype, pparams=pparams, F_str=F_str)
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", evaluator, toolbox=toolbox, X_train=X_train, y=y_train, d=d)

    out_csv = args.csv_log or f"{args.problem}_protocol_{args.protocol}.csv"
    csv_f = open(out_csv, "w", newline=""); csv_w = csv.writer(csv_f)
    csv_w.writerow(["elapsed_s","gen","best_fitness","popnum","size","height"]); csv_f.flush()

    def progress_cb(info):
        csv_w.writerow([f"{info['elapsed']:.3f}", info['gen'], f"{info['best_fitness']:.6e}",
                        info['popnum'], info['size'], info['height']])
        csv_f.flush()

    t0 = time.time()
    best = tr_gp(KB, toolbox, pset, X_train, y_train,
                 max_seconds=args.max_seconds,
                 log_interval=args.log_interval,
                 progress_cb=progress_cb,
                  max_gens=args.max_gens)
    wall = time.time() - t0
    csv_f.close()

    # reporting (rel L2 on raw supervised pairs)
    expr_fun = toolbox.compile(expr=best)
    xs, ys, ts = X_train
    Xv = torch.tensor(xs, dtype=torch.float64)
    Yv = torch.tensor(ys, dtype=torch.float64)
    Tv = torch.tensor(ts, dtype=torch.float64)
    y_pred = expr_fun(Xv, Yv, Tv)
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    y_true = np.array(y_train, dtype=np.float64)
    rL2 = float(np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true)+1e-12))

    sym = deap_tree_to_sympy(best, pset)
    summary_path = f"{args.problem}_protocol_{args.protocol}_result.txt"
    with open(summary_path, "w") as f:
        f.write("========== RESULT ==========\n")
        f.write(f"Problem        : {args.problem}\n")
        f.write(f"Protocol       : {args.protocol} (1=motif KB, 2=primitives only)\n")
        f.write(f"Time budget    : {args.max_seconds}s (used ~{wall:.1f}s)\n")
        f.write(f"Best (prefix)  : {str(best)}\n")
        f.write(f"Best (infix )  : {sym}\n")
        f.write(f"Rel L2 (u vs y): {rL2:.6e}\n")
        f.write("============================\n")

if __name__ == "__main__":
    main()
