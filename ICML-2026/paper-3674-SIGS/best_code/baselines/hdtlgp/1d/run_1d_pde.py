# HD-TLGP baseline — re-implementation of the method from:
# Cao et al. "An Interpretable Approach to High-Dimensional PDEs." AAAI 2024.
# doi:10.1609/aaai.v38i18.30050
import argparse, time, math, random, numpy as np, torch, sympy
from Tr_GPSRHeat_patched import define_gp, tr_gp, str_to_individual
from HeatDataset_patched import dataset_diffusion_1d, dataset_wave_1d, dataset_burgers_1d
from evaluate_patched import make_evaluator_1d, _torch_Diffusion1d, _torch_Wave1d, _torch_Burgers1d
import sympy as sp
from deap import gp
import csv


def deap_tree_to_sympy(ind, pset):
    x = sp.Symbol("x"); t = sp.Symbol("t")
    stack = []
    for node in reversed(ind):
        if isinstance(node, gp.Terminal):
            v = getattr(node, "value", None)
            if isinstance(v, str):
                if v in ("x", "x1", "ARG0"):
                    stack.append(x); continue
                if v in ("t", "ARG1"):
                    stack.append(t); continue
            stack.append(sp.Float(v))
        else:
            ar = node.arity
            args = [stack.pop() for _ in range(ar)][::-1]
            name = node.name
            if name == "add":   stack.append(args[0] + args[1])
            elif name == "sub": stack.append(args[0] - args[1])
            elif name == "mul": stack.append(args[0] * args[1])
            elif name == "pdiv": stack.append(args[0] / args[1])  # safe even if never used
            elif name == "sin": stack.append(sp.sin(args[0]))
            elif name == "cos": stack.append(sp.cos(args[0]))
            elif name == "exp": stack.append(sp.exp(args[0]))
            elif name == "tanh": stack.append(sp.tanh(args[0]))
            else:
                raise ValueError(f"Unknown primitive {name}")
    assert len(stack) == 1
    return sp.simplify(stack[0])


# ---------- Protocol-1 KB motifs (PREFIX notation for DEAP) ----------
def kb_exprs_for(problem, pparams):
    """
    Protocol 1 KB motifs:
      • Diffusion/Wave: include the ACTUAL first mode (with dataset amplitude),
        plus potential higher modes (no amplitudes), AND 2–3 mode sum scaffolds.
      • Burgers: include the shock tanh core and one scaled copy (no offset m).
    Protocol 2: caller passes KB=[].
    """
    if problem == "diffusion":
        A = float(pparams.get("A", 3.974))
        D = float(pparams.get("D", 0.697))
        L = float(pparams.get("L", 1.397))

        k1   = math.pi / L
        lam0 = (math.pi**2 * D) / (L**2)

        # single-mode building blocks
        m1 = f"mul(sin(mul({1*k1:.4f}, x)), exp(mul({-(1**2)*lam0:.4f}, t)))"
        m3 = f"mul(sin(mul({3*k1:.4f}, x)), exp(mul({-(3**2)*lam0:.4f}, t)))"
        m5 = f"mul(sin(mul({5*k1:.4f}, x)), exp(mul({-(5**2)*lam0:.4f}, t)))"

        # (a) ACTUAL first mode with dataset amplitude
        first_mode = f"mul({A:.4f}, {m1})"

        # (b) potential single modes (no amplitude)
        singles = [m3, m5]

        # (c) 2–3 mode sum scaffolds (no amplitudes)
        sums = [
            f"add({m1}, {m3})",
            f"add({m1}, {m5})",
            f"add({m3}, {m5})",
            f"add(add({m1}, {m3}), {m5})",
        ]

        # optional separated factors for composition
        extras = [f"sin(mul({k1:.4f}, x))", f"exp(mul({-lam0:.4f}, t))"]

        return [first_mode] + singles + sums + extras

    if problem == "wave":
        coeffs = pparams.get("coeffs", [1.4e-1, 4.6e-3, 2.3e-4, 1.1e-4])
        c2 = float(pparams.get("c2", 0.14**2))
        c  = math.sqrt(max(c2, 1e-16))
        a1 = float(coeffs[0])

        def mode(n):
            return f"mul(sin(mul({n*math.pi:.4f}, x)), cos(mul({c*n*math.pi:.4f}, t)))"

        m1, m2, m3, m4 = mode(1), mode(2), mode(3), mode(4)

        # (a) ACTUAL first mode with amplitude (π/16)*a1
        k1 = (math.pi/16.0)*a1
        first_mode = f"mul({k1:.4f}, {m1})"

        # (b) potential single modes (no amplitude)
        singles = [m2, m3, m4]

        # (c) 2–3 mode sum scaffolds (no amplitudes)
        sums = [
            f"add({m1}, {m2})",
            f"add({m1}, {m3})",
            f"add({m2}, {m3})",
            f"add(add({m1}, {m2}), {m3})",
        ]

        # separated factors
        extras = [f"sin(mul({math.pi:.4f}, x))", f"cos(mul({c*math.pi:.4f}, t))"]

        return [first_mode] + singles + sums + extras

    if problem == "burgers":
        uL = float(pparams.get("u_L", 1.46))
        uR = float(pparams.get("u_R", 0.26))
        x0 = float(pparams.get("x0", 0.33))
        nu = float(pparams.get("nu", 0.01))
        s  = 0.5*(uL + uR)
        A  = 0.5*(uL - uR)
        alpha = (uL - uR) / (4.0 * nu)

        core = f"tanh(mul({alpha:.4f}, add(add(x, {(-x0):.4f}), mul({(-s):.4f}, t))))"

        # No full analytic (m - A*tanh(...)). Just shape + one scaled copy.
        return [core, f"mul({A:.4f}, {core})"]

    return []




def build_kb(pset, problem, pparams):
    return [str_to_individual(s, pset) for s in kb_exprs_for(problem, pparams)]


def rel_l2(pred, truth):
    num = np.linalg.norm(pred - truth)
    den = np.linalg.norm(truth) + 1e-12
    return float(num / den)


def eval_expr_over_pairs(expr_callable, X_train):
    xs, ts = X_train
    Xv = torch.tensor(xs, dtype=torch.float64)
    Tv = torch.tensor(ts, dtype=torch.float64)
    y_pred_t = expr_callable(Xv, Tv)
    return (y_pred_t.detach().cpu().numpy()
            if isinstance(y_pred_t, torch.Tensor)
            else np.asarray(y_pred_t, dtype=np.float64))


def compute_pde_only(best_ind, toolbox, ptype, pparams, F_str, X_train):
    expr = toolbox.compile(expr=best_ind)
    # small uniform 1D grid for PDE residual
    xs, ts = X_train
    x_grid = np.linspace(min(xs), max(xs), 101, dtype=np.float64)
    t_grid = np.linspace(min(ts), max(ts), 101, dtype=np.float64)
    n = min(len(x_grid), len(t_grid))
    Xb = torch.tensor(x_grid[:n], dtype=torch.float64, requires_grad=True)
    Tb = torch.tensor(t_grid[:n], dtype=torch.float64, requires_grad=True)

    def expr_callable(X, T):
        return expr(X, T)  # torch → torch

    F_sym = None
    try:
        if F_str is not None and str(F_str).strip() != "0":
            F_sym = sympy.sympify(F_str)
    except Exception:
        F_sym = None

    if ptype == "diffusion-1d":
        D = float(pparams.get("D", 1.0))
        return float(_torch_Diffusion1d(expr_callable, Xb, Tb, D, F_sym).item())
    if ptype == "wave-1d":
        c2 = float(pparams.get("c2", 0.14**2))
        return float(_torch_Wave1d(expr_callable, Xb, Tb, c2, F_sym).item())
    if ptype == "burgers-1d":
        nu = float(pparams.get("nu", 0.01))
        return float(_torch_Burgers1d(expr_callable, Xb, Tb, nu, F_sym).item())
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description="Paper-faithful 1D PDE GP (protocols + time cap + pruning)")
    ap.add_argument("--problem",  choices=["diffusion","wave","burgers"], required=True)
    ap.add_argument("--protocol", type=int, choices=[1,2], required=True,
                    help="1 = motif KB + primitives; 2 = primitives only")
    ap.add_argument("--max-seconds", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gens", type=int, default=25, help="Generation cap (optional)")
    ap.add_argument("--log-interval", type=float, default=5.0, help="seconds between heartbeat prints")
    ap.add_argument("--csv-log", type=str, default=None,
                    help="optional CSV path; default is {problem}_protocol_{protocol}.csv")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # dataset + meta (domains/params/forcing)
    if args.problem == "diffusion":
        X_train, y_train, pparams, F_str, ptype = dataset_diffusion_1d(return_meta=True)
    elif args.problem == "wave":
        X_train, y_train, pparams, F_str, ptype = dataset_wave_1d(return_meta=True)
    elif args.problem == "burgers":
        X_train, y_train, pparams, F_str, ptype = dataset_burgers_1d(return_meta=True)
    else:
        raise ValueError("Unknown problem")
    #print(f"DEBUG: Dataset - X range [{np.min(X_train[0])}, {np.max(X_train[0])}], y range [{np.min(y_train)}, {np.max(y_train)}]")
    #print(f"DEBUG: Dataset size - {len(X_train[0])} points")


    toolbox, pset = define_gp(X_train, y_train, d=1, protocol=args.protocol)
    KB = build_kb(pset, args.problem, pparams) if args.protocol == 1 else []
    #print(f"DEBUG: KB size = {len(KB)}, popnum will be {max(200, 2*len(KB))}")
    evaluator = make_evaluator_1d(ptype=ptype, pparams=pparams, F_str=F_str)
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", evaluator, toolbox=toolbox, X_train=X_train, y=y_train, d=1)

    out_csv = args.csv_log or f"{args.problem}_protocol_{args.protocol}.csv"
    csv_f = open(out_csv, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["elapsed_s","gen","best_fitness","popnum","size","height"])
    csv_f.flush()

    def progress_cb(info):
        csv_w.writerow([
            f"{info['elapsed']:.3f}",
            info['gen'],
            f"{info['best_fitness']:.6e}",
            info['popnum'],
            info['size'],
            info['height'],
        ])
        csv_f.flush()

    t0 = time.time()
    best = tr_gp(KB, toolbox, pset, X_train, y_train,
                 max_seconds=args.max_seconds,
                 log_interval=args.log_interval,
                 progress_cb=progress_cb,
                  max_gens=args.max_gens)
    wall = time.time() - t0

    # reporting
    expr_fun = toolbox.compile(expr=best)
    y_pred = eval_expr_over_pairs(expr_fun, X_train)
    y_true = np.array(y_train, dtype=np.float64)
    rL2 = rel_l2(y_pred, y_true)
    pde_only = compute_pde_only(best, toolbox, ptype, pparams, F_str, X_train)

    sym = deap_tree_to_sympy(best, pset)
    summary_path = f"{args.problem}_protocol_{args.protocol}_result.txt"
    with open(summary_path, "w") as f:
        f.write("========== RESULT ==========\n")
        f.write(f"Problem        : {args.problem}\n")
        f.write(f"Protocol       : {args.protocol} (1=motif KB, 2=primitives only)\n")
        f.write(f"Time budget    : {args.max_seconds}s (used ~{wall:.1f}s)\n")
        f.write(f"Best (prefix)  : {str(best)}\n")
        f.write(f"Best (infix )  : {sym}\n")
        f.write(f"PDE residual   : {pde_only:.6e}\n")
        f.write(f"Rel L2 (u vs y): {rL2:.6e}\n")
        f.write("============================\n")

    # Append a compact result line to the CSV
    csv_w.writerow(["RESULT", "", f"{pde_only:.6e}", "", len(best), best.height])
    csv_f.flush()
    csv_f.close()


if __name__ == "__main__":
    main()
