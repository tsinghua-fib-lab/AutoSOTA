# Tr_GPSR2D_patched.py
# DEAP GP with protocol-aware primitives, time-bounded evolution, KB transfer,
# peephole simplifying, and optional local constant tuning. Supports d=1/2/3.
import math, random, copy, operator
import numpy as np
import time as _time
import torch
from deap import base, gp, creator, tools

# ---------- peephole simplify ----------
def _is_terminal_number(node):
    return isinstance(node, gp.Terminal) and not isinstance(node.value, str)

def _replace_slice(tree, slice_obj, repl_list):
    return creator.Individual(tree[:slice_obj.start] + repl_list + tree[slice_obj.stop:])

def tsqrt(x):
    x = _to_tensor(x)
    # guard against tiny negative noise before sqrt
    return torch.sqrt(torch.clamp(x, min=0.0))

def _simplify_once(ind):
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
                    if is_num(left, 0.0) or is_num(right, 0.0):
                        ind = _replace_slice(ind, sl, [gp.Terminal(0.0, False, None)])
                        changed = True; i = 0; continue
                    if is_num(left, 1.0):
                        ind = _replace_slice(ind, sl, list(right))
                        changed = True; i = 0; continue
                    if is_num(right, 1.0):
                        ind = _replace_slice(ind, sl, list(left))
                        changed = True; i = 0; continue
                elif name == "add":
                    if is_num(left, 0.0):
                        ind = _replace_slice(ind, sl, list(right))
                        changed = True; i = 0; continue
                    if is_num(right, 0.0):
                        ind = _replace_slice(ind, sl, list(left))
                        changed = True; i = 0; continue
        i += 1
    return ind, changed

def simplify_tree(ind, max_passes=5):
    for _ in range(max_passes):
        ind, changed = _simplify_once(ind)
        if not changed: break
    return ind

# ---------- tensor-safe primitives ----------
def _to_tensor(x, like=None, dtype=torch.float64):
    if isinstance(x, torch.Tensor): return x
    if isinstance(like, torch.Tensor):
        return torch.tensor(x, dtype=like.dtype, device=like.device)
    return torch.tensor(x, dtype=dtype)

def tsin(x):   return torch.sin(_to_tensor(x))
def tcos(x):   return torch.cos(_to_tensor(x))
def texp(x):
    x = _to_tensor(x)
    # Clip input to exp to prevent overflow/underflow
    x_clipped = torch.clamp(x, min=-20, max=20)
    return torch.exp(x_clipped)
def ttanh(x):  return torch.tanh(_to_tensor(x))
def tsqrt(x):  return torch.sqrt(torch.clamp(_to_tensor(x), min=0.0))

def pdiv(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        a = _to_tensor(a); b = _to_tensor(b, like=a)
        return torch.where(torch.abs(b) > 1e-12, a / b, a)
    try:    return a / b if abs(b) > 1e-12 else a
    except: return a

# ---------- optional local optimize hook ----------
def _ensure_individual_cls():
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

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
        def _noop(self, X_train, y, pset): return
        if not hasattr(creator.Individual, "local_optimize"):
            setattr(creator.Individual, "local_optimize", _noop)
        toolbox.local_optimize_attached = False

def define_gp(X_train, y, d, protocol=1):
    _ensure_individual_cls()

    pset = gp.PrimitiveSet("MAIN", d + 1)
    if d == 1:
        pset.renameArguments(ARG0="x");  pset.renameArguments(ARG1="t")
    elif d == 2:
        pset.renameArguments(ARG0="x");  pset.renameArguments(ARG1="y"); pset.renameArguments(ARG2="t")
    elif d == 3:
        pset.renameArguments(ARG0="x1"); pset.renameArguments(ARG1="x2"); pset.renameArguments(ARG2="x3"); pset.renameArguments(ARG3="t")

    # primitives
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(pdiv,         2, name="pdiv")
    pset.addPrimitive(tsin,  1, name="sin")
    pset.addPrimitive(tcos,  1, name="cos")
    pset.addPrimitive(texp,  1, name="exp")
    pset.addPrimitive(ttanh, 1, name="tanh")
    pset.addPrimitive(tsqrt, 1, name="sqrt")   # for radial motifs
    pset.addTerminal(math.pi)


    # numeric terminals
    from functools import partial
    pset.addEphemeralConstant("c", partial(random.uniform, -20.0, 20.0))

    pset.addTerminal(1.0); pset.addTerminal(0.5); pset.addTerminal(math.pi)

    toolbox = base.Toolbox()
    toolbox.register("expr",     gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("expr_mut", gp.genFull,        pset=pset, min_=1, max_=8)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", lambda *_args, **_kw: (1e9,))
    toolbox.register("select", tools.selBest)
    toolbox.register("mate",   gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    import operator as _op
    toolbox.decorate("mate",   gp.staticLimit(key=_op.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=_op.attrgetter("height"), max_value=10))

    toolbox.protocol = protocol
    attach_local_optimize(toolbox)
    toolbox.pset = pset
    return toolbox, pset

def str_to_individual(expr_str, pset):
    s = expr_str.replace("^","**").replace("pi", f"{math.pi}")
    return gp.PrimitiveTree.from_string(s, pset)

def tr_gp(knowledge_base, toolbox, pset, X_train, y,
          max_seconds=600, log_interval=5.0, progress_cb=None,
          enable_local_opt=True,max_gens=25):
    import numpy as _np

    def safe_eval(ind):
        try:
            fit = toolbox.evaluate(ind)
            val = float(fit[0]) if isinstance(fit, tuple) and len(fit) else 1e12
            if not _np.isfinite(val): val = 1e12
            ind.fitness.values = (val,)
        except Exception:
            ind.fitness.values = (1e12,)

    def fval(ind):
        if getattr(ind.fitness, "valid", False) and len(ind.fitness.values) > 0:
            v = ind.fitness.values[0]
            return v if _np.isfinite(v) else float("inf")
        return float("inf")

    popnum = max(50, 2*len(knowledge_base))
    cxpb, mutpb, trpb = 0.6, 0.6, 0.6
    if getattr(toolbox, "protocol", 1) == 1:
        vacpb = 0.0  # No pruning for Protocol 1 (preserve motifs)
    else:
        vacpb = 0.25  # Aggressive for Protocol 2


    pop = toolbox.population(n=popnum)

    # seed with KB snippets
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
        simplify_tree(ind)
        safe_eval(ind)

    best = min(pop, key=fval)
    t0 = _time.time(); last_log = t0; gen = 0
    print(f"[0.0s/{max_seconds}s] init best={fval(best):.6e} size={len(best)} height={best.height}", flush=True)

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
                for c in (c1, c2):
                    try: del c.fitness.values
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

        # KB splice (protocol-1)
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

        # evaluate
        for ind in offspring:
            if _time.time() - t0 >= max_seconds: break
            if not getattr(ind.fitness, "valid", False):
                if enable_local_opt and hasattr(creator.Individual, "local_optimize"):
                    ind.local_optimize(X_train, y, pset)
                simplify_tree(ind)
                safe_eval(ind)

        pool = offspring + pop
        pop = tools.selBest(pool, popnum)
        cur = min(pop, key=fval)
        if fval(cur) < fval(best): best = cur

        now = _time.time()
        if log_interval and (now - last_log) >= log_interval:
            elapsed = now - t0
            print(f"[{elapsed:.1f}s/{max_seconds}s] gen={gen} best={fval(best):.6e} size={len(best)} height={best.height} pop={len(pop)}", flush=True)
            if progress_cb:
                try:
                    progress_cb({"elapsed": elapsed, "gen": gen, "best_fitness": float(fval(best)),
                                 "popnum": len(pop), "size": len(best), "height": best.height})
                except Exception: pass
            last_log = now

    elapsed = _time.time() - t0
    print(f"[{elapsed:.1f}s/{max_seconds}s] done best={fval(best):.6e} size={len(best)} height={best.height}", flush=True)
    return best
