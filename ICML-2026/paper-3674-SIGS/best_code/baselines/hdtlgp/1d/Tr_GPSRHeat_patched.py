# DEAP GP with protocol-aware primitives, time-bounded evolution, KB transfer,
# simple VAC pruning, and optional local constant tuning hook.
import math, random, time, copy, operator
import numpy as np
import torch
from deap import base, gp, creator, tools


# --- helpers for paper-style pruning ---
def _is_terminal_number(node):
    from deap import gp
    return isinstance(node, gp.Terminal) and not isinstance(node.value, str)

def _replace_slice(tree, slice_obj, repl_list):
    from deap import creator
    return creator.Individual(tree[:slice_obj.start] + repl_list + tree[slice_obj.stop:])

def _simplify_once(ind):
    from deap import gp
    changed = False
    i = 0
    while i < len(ind):
        node = ind[i]
        if isinstance(node, gp.Primitive):
            name, ar = node.name, node.arity
            sl = ind.searchSubtree(i)
            if ar == 1:
                child = ind[i+1:sl.stop]
                if name == "exp" and len(child) == 1 and _is_terminal_number(child[0]) and float(child[0].value) == 0.0:
                    ind = _replace_slice(ind, sl, [gp.Terminal(1.0, False, None)])
                    changed = True; i = 0; continue
            elif ar == 2:
                mid = i+1
                left_sl  = ind.searchSubtree(mid)
                right_sl = slice(left_sl.stop, sl.stop)
                left  = ind[left_sl]
                right = ind[right_sl]
                def is_num(treelet, v):
                    return len(treelet) == 1 and _is_terminal_number(treelet[0]) and float(treelet[0].value) == v
                if name == "mul":
                    if is_num(left, 0.0) or is_num(right, 0.0):  # 0*x or x*0 -> 0
                        ind = _replace_slice(ind, sl, [gp.Terminal(0.0, False, None)])
                        changed = True; i = 0; continue
                    if is_num(left, 1.0):                       # 1*x -> x
                        ind = _replace_slice(ind, sl, list(right))
                        changed = True; i = 0; continue
                    if is_num(right, 1.0):                      # x*1 -> x
                        ind = _replace_slice(ind, sl, list(left))
                        changed = True; i = 0; continue
                elif name == "add":
                    if is_num(left, 0.0):                       # 0+x -> x
                        ind = _replace_slice(ind, sl, list(right))
                        changed = True; i = 0; continue
                    if is_num(right, 0.0):                      # x+0 -> x
                        ind = _replace_slice(ind, sl, list(left))
                        changed = True; i = 0; continue
                elif name == "pow":
                    if is_num(right, 1.0):                      # x^1 -> x
                        ind = _replace_slice(ind, sl, list(left))
                        changed = True; i = 0; continue
                    if is_num(right, 0.0):                      # x^0 -> 1
                        ind = _replace_slice(ind, sl, [gp.Terminal(1.0, False, None)])
                        changed = True; i = 0; continue
        i += 1
    return ind, changed

def simplify_tree(ind, max_passes=5):
    for _ in range(max_passes):
        ind, changed = _simplify_once(ind)
        if not changed: break
    return ind

def prune_subtree_to_one(pop, prob=0.6):
    from deap import gp, creator
    out = []
    one = [gp.Terminal(1.0, False, None)]
    for ind in pop:
        new_ind = creator.Individual(ind[:])
        if random.random() < prob and len(new_ind) > 3:
            root = random.randrange(1, len(new_ind))
            sl = new_ind.searchSubtree(root)
            new_ind = _replace_slice(new_ind, sl, one)
            try: del new_ind.fitness.values
            except Exception: pass
        out.append(new_ind)
    return out

def simplify_zero_prune(pop, prob=0.2):
    from deap import gp, creator
    out = []
    for ind in pop:
        new_ind = creator.Individual(ind[:])
        if random.random() < prob:
            idx = [i for i, n in enumerate(new_ind) if _is_terminal_number(n)]
            if idx:
                i = random.choice(idx)
                new_ind[i].value = 0.0     # zero out one constant
                new_ind = simplify_tree(new_ind)  # peephole simplify
                try: del new_ind.fitness.values
                except Exception: pass
        out.append(new_ind)
    return out

# ---------- helpers to keep everything tensor-safe ----------
def _to_tensor(x, like=None, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(like, torch.Tensor):
        return torch.tensor(x, dtype=like.dtype, device=like.device)
    return torch.tensor(x, dtype=dtype)

def tsin(x):
    x = _to_tensor(x)
    return torch.sin(x)

def tcos(x):
    x = _to_tensor(x)
    return torch.cos(x)

def texp(x):
    x = _to_tensor(x)
    # Clip input to exp to prevent overflow/underflow
    x_clipped = torch.clamp(x, min=-20, max=20)
    return torch.exp(x_clipped)

def ttanh(x):
    x = _to_tensor(x)
    return torch.tanh(x)

def pdiv(a, b):
    # protected division that works for tensors or floats
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        a = _to_tensor(a)
        b = _to_tensor(b, like=a)
        return torch.where(torch.abs(b) > 1e-12, a / b, a)
    try:
        return a / b if abs(b) > 1e-12 else a
    except Exception:
        return a

def _ensure_individual_cls():
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# ---------- optional local optimize hook ----------
def attach_local_optimize(toolbox):
    try:
        import local_optimize as lo
        def _lo_method(self, X_train, y, pset):
            try:
                if hasattr(lo, "local_optimize"):
                    return lo.local_optimize(self, X_train, y, pset)
            except Exception:
                return
        if not hasattr(creator.Individual, "local_optimize"):
            setattr(creator.Individual, "local_optimize", _lo_method)
        toolbox.local_optimize_attached = True
    except Exception:
        def _noop(self, X_train, y, pset): 
            return
        if not hasattr(creator.Individual, "local_optimize"):
            setattr(creator.Individual, "local_optimize", _noop)
        toolbox.local_optimize_attached = False

def define_gp(X_train, y, d, protocol=1):
    _ensure_individual_cls()

    # ---- Primitive set ----
    pset = gp.PrimitiveSet("MAIN", d + 1)
    if d == 1:
        pset.renameArguments(ARG0="x");  pset.renameArguments(ARG1="t")
    elif d == 2:
        pset.renameArguments(ARG0="x1"); pset.renameArguments(ARG1="x2"); pset.renameArguments(ARG2="t")
    elif d == 3:
        pset.renameArguments(ARG0="x1"); pset.renameArguments(ARG1="x2"); pset.renameArguments(ARG2="x3"); pset.renameArguments(ARG3="t")

    # tensor-safe primitives (keep your set)
    pset.addPrimitive(operator.add, 2)      # add
    pset.addPrimitive(operator.sub, 2)      # sub
    pset.addPrimitive(operator.mul, 2)      # mul
    # pset.addPrimitive(operator.pow, 2)
    pset.addPrimitive(pdiv, 2)              # protected div
    pset.addPrimitive(tsin,  1, name="sin")
    pset.addPrimitive(tcos,  1, name="cos")
    pset.addPrimitive(texp,  1, name="exp")
    pset.addPrimitive(ttanh, 1, name="tanh")
    pset.addEphemeralConstant("c", lambda: random.uniform(-90.0, 90.0))
    # constants (no ephemeral constants, per request)
    pset.addTerminal(1.0); pset.addTerminal(0.5); pset.addTerminal(math.pi)

    # ---- Toolbox (original-like GA) ----
    toolbox = base.Toolbox()

    # init: ramped half-and-half depth 1..4 (original)
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)

    # mutation replacement subtrees from genFull depth 1..8 (original)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=1, max_=8)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def _stub_eval(*args, **kwargs):
        return (1e9,)
    toolbox.register("evaluate", _stub_eval)

    # selection: selBest (original)
    toolbox.register("select", tools.selBest)

    # crossover: cxOnePoint with height cap 8 (original)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    # mutation: mutUniform using expr_mut, height cap 8 (original)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    # keep protocol + local-opt hook
    toolbox.protocol = protocol
    attach_local_optimize(toolbox)
    toolbox.pset = pset
    return toolbox, pset


def str_to_individual(expr_str, pset):
    # DEAP expects prefix strings with registered primitive names
    s = expr_str.replace("^","**").replace("pi", f"{math.pi}")
    return gp.PrimitiveTree.from_string(s, pset)

# ---------- simple VAC pruning (1D): prune low-variance subtrees ----------
def _subtree_is_low_variance(subtree, toolbox, x_range, t_range, tol=1e-3, ns=32):
    f = toolbox.compile(expr=subtree)
    xs = np.random.uniform(*x_range, size=ns)
    ts = np.random.uniform(*t_range, size=ns)
    Xv = torch.tensor(xs, dtype=torch.float64)
    Tv = torch.tensor(ts, dtype=torch.float64)
    vals = f(Xv, Tv)
    if not isinstance(vals, torch.Tensor):
        vals = _to_tensor(vals)
    vals = vals.detach().cpu().numpy()
    return float(np.var(vals)) < tol

def vac_operator_1d(pop, toolbox, X_train, prob=0.5, tol=1e-3):
    xs, ts = X_train
    x_range = (min(xs), max(xs)); t_range = (min(ts), max(ts))
    const_node = str_to_individual("1.0", toolbox.pset)
    out = []
    for ind in pop:
        # keep Individual type (has .fitness)
        new_ind = creator.Individual(ind[:])
        if random.random() < prob and len(new_ind) > 3:
            root = random.randrange(1, len(new_ind))
            sl = new_ind.searchSubtree(root)
            subtree = gp.PrimitiveTree(new_ind[sl])
            try:
                if _subtree_is_low_variance(subtree, toolbox, x_range, t_range, tol=tol):
                    new_ind[:] = new_ind[:sl.start] + const_node[:] + new_ind[sl.stop:]
                    if hasattr(new_ind, "fitness"):
                        # mark as needing reevaluation
                        del new_ind.fitness.values
            except Exception:
                pass
        out.append(new_ind)
    return out

def tr_gp(knowledge_base, toolbox, pset, X_train, y,
          max_seconds=600, log_interval=5.0, progress_cb=None,
          enable_local_opt=True,max_gens=25):
    import numpy as _np
    import time as _time

    def safe_eval(ind):
        try:
            fit = toolbox.evaluate(ind)
            val = float(fit[0]) if isinstance(fit, tuple) and len(fit) else 1e12
            if not _np.isfinite(val) or abs(val) > 1e10: 
                val = 1e12
            ind.fitness.values = (val,)
        except (OverflowError, FloatingPointError):
            ind.fitness.values = (1e12,)

    def fval(ind):
        if getattr(ind.fitness, "valid", False) and len(ind.fitness.values) > 0:
            v = ind.fitness.values[0]
            return v if _np.isfinite(v) else float("inf")
        return float("inf")

    popnum = max(200, 2*len(knowledge_base))
    # cxpb, mutpb, trpb = 0.6, 0.6, 0.6
    # vacpb = 0.6
    cxpb, mutpb = 0.6, 0.6
    if getattr(toolbox, "protocol", 1) == 2:
        trpb  = 0.0   # Protocol 2: no KB splicing
        vacpb = 0.25  # more pruning is ok without KB
    else:
        trpb  = 0.6   # Protocol 1: enable KB splicing
        vacpb = 0.0  # gentle pruning so motifs survive


    pop = toolbox.population(n=popnum)
    if knowledge_base:
        for ind in pop:
            if random.random() < 0.5:
                root = random.randrange(len(ind))
                sl = ind.searchSubtree(root)
                kb_tree = random.choice(knowledge_base)
                ind[sl] = copy.deepcopy(kb_tree)
                try: del ind.fitness.values
                except Exception: pass

    # initial eval
    for ind in pop:
        if enable_local_opt and hasattr(creator.Individual, "local_optimize"):
            ind.local_optimize(X_train, y, pset)
        safe_eval(ind)

    best = min(pop, key=fval)
    t0 = _time.time()
    last_log = t0
    gen = 0
    print(f"[0.0s/{max_seconds}s] init  best={fval(best):.6e} size={len(best)} height={best.height}", flush=True)

    while True:
        if (_time.time() - t0 >= max_seconds) or (max_gens is not None and gen >= max_gens):
            break
        gen += 1

        elites = tools.selBest(pop, max(5, popnum//2))
        offspring = [creator.Individual(ind[:]) for ind in elites]

        # crossover/fill
        while len(offspring) < popnum:
            if _time.time() - t0 >= max_seconds: break
            c1, c2 = map(toolbox.clone, random.sample(pop, 2))
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                try: del c1.fitness.values
                except Exception: pass
                try: del c2.fitness.values
                except Exception: pass
            offspring += [c1, c2]
        offspring = offspring[:popnum]

        # mutation
        for m in offspring:
            if _time.time() - t0 >= max_seconds: break
            if random.random() < mutpb:
                toolbox.mutate(m)
                try: del m.fitness.values
                except Exception: pass

        # KB splice
        if knowledge_base:
            for ind in offspring:
                if _time.time() - t0 >= max_seconds: break
                if random.random() < trpb:
                    root = random.randrange(len(ind))
                    sl = ind.searchSubtree(root)
                    kb_tree = random.choice(knowledge_base)
                    ind[sl] = copy.deepcopy(kb_tree)
                    try: del ind.fitness.values
                    except Exception: pass

        # VAC prune (returns Individuals; may invalidate)
        # offspring = vac_operator_1d(offspring, toolbox, X_train, prob=vacpb, tol=1e-3)
        offspring = prune_subtree_to_one(offspring, prob=vacpb)
        # offspring = simplify_zero_prune(offspring, prob=vacpb)

        # evaluate invalids
        for ind in offspring:
            if _time.time() - t0 >= max_seconds: break
            if not getattr(ind.fitness, "valid", False):
                if enable_local_opt and hasattr(creator.Individual, "local_optimize"):
                    ind.local_optimize(X_train, y, pset)
                safe_eval(ind)

        pool = offspring + pop
        
        pop = tools.selBest(pool, popnum)

        cur = min(pop, key=fval)
        if fval(cur) < fval(best):
            best = cur

        now = _time.time()
        if log_interval and (now - last_log) >= log_interval:
            elapsed = now - t0
            print(f"[{elapsed:.1f}s/{max_seconds}s] gen={gen} best={fval(best):.6e} size={len(best)} height={best.height} pop={len(pop)}",
                  flush=True)
            if progress_cb:
                try:
                    progress_cb({"elapsed": elapsed, "gen": gen, "best_fitness": float(fval(best)),
                                 "popnum": len(pop), "size": len(best), "height": best.height})
                except Exception:
                    pass
            last_log = now

    elapsed = _time.time() - t0
    print(f"[{elapsed:.1f}s/{max_seconds}s] done best={fval(best):.6e} size={len(best)} height={best.height}", flush=True)
    return best
