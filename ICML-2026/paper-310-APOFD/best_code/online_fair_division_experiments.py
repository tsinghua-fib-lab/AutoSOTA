from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12


@dataclass(frozen=True)
class Metrics:
    prop1: float
    welfare: float
    welfare_opt: float


@dataclass(frozen=True)
class Aggregate:
    family: str
    algorithm: str
    prop1_mean: float
    prop1_se: float
    welfare_mean: float
    welfare_se: float
    trials: int


def normalize_by_miv(V: np.ndarray) -> np.ndarray:
    """Normalize each agent's values by her maximum item value.

    Agents with zero maximum value are left as all zero.
    """
    V = np.asarray(V, dtype=float)
    maxima = V.max(axis=1)
    W = V.copy()
    positive = maxima > EPS
    W[positive, :] = W[positive, :] / maxima[positive, None]
    return W


def allocation_from_owner(owner: Sequence[int], n: int) -> List[List[int]]:
    alloc: List[List[int]] = [[] for _ in range(n)]
    for g, i in enumerate(owner):
        alloc[int(i)].append(g)
    return alloc


def metrics(V: np.ndarray, owner: Sequence[int], cap_prop1: bool = True) -> Metrics:
    """Return clipped realized PROP1 ratio and normalized utilitarian welfare.

    The realized PROP1 ratio is min_i n*(v_i(A_i)+max_{g not in A_i}v_i(g))/v_i(G),
    with ratio 1 for zero-total agents and agents receiving all goods. We clip at 1
    because the experiment measures satisfaction of the standard PROP1 target.
    """
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    owner_arr = np.asarray(owner, dtype=int)
    prop_ratios: List[float] = []
    for i in range(n):
        total = float(V[i].sum())
        if total <= EPS:
            prop_ratios.append(1.0)
            continue
        own_mask = owner_arr == i
        own_value = float(V[i, own_mask].sum())
        if own_mask.all():
            ratio = float("inf")
        else:
            best_outside = float(V[i, ~own_mask].max())
            ratio = n * (own_value + best_outside) / total
        if cap_prop1:
            ratio = min(1.0, ratio)
        prop_ratios.append(ratio)
    prop1 = float(min(prop_ratios)) if prop_ratios else 1.0

    welfare = float(sum(V[int(owner_arr[g]), g] for g in range(m)))
    welfare_opt = float(V.max(axis=0).sum())
    welfare_norm = 1.0 if welfare_opt <= EPS else welfare / welfare_opt
    return Metrics(prop1=prop1, welfare=welfare_norm, welfare_opt=welfare_opt)


def greedy1(V: np.ndarray) -> List[int]:
    """Greedy Strategy 1: allocate to max_i v_i(g_t)/v_i(G^(t))."""
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    owner: List[int] = []
    running_totals = np.zeros(n)
    for t in range(m):
        running_totals += V[:, t]
        ratios = np.full(n, -np.inf)
        pos = running_totals > EPS
        ratios[pos] = V[pos, t] / running_totals[pos]
        # If an agent has zero current total, the current value is also zero; keep -inf.
        owner.append(int(np.argmax(ratios)))
    return owner


def greedy2(V: np.ndarray) -> List[int]:
    """Greedy Strategy 2: allocate to min_i v_i(A_i^(t-1))/v_i(G^(t))."""
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    owner: List[int] = []
    running_totals = np.zeros(n)
    bundle_values = np.zeros(n)
    for t in range(m):
        running_totals += V[:, t]
        ratios = np.zeros(n)
        pos = running_totals > EPS
        ratios[pos] = bundle_values[pos] / running_totals[pos]
        ratios[~pos] = 0.0
        i = int(np.argmin(ratios))
        owner.append(i)
        bundle_values[i] += V[i, t]
    return owner


def greedy3(V: np.ndarray) -> List[int]:
    """Greedy Strategy 3: myopically optimize current PROP1 slack."""
    V = np.asarray(V, dtype=float)
    n, m = V.shape
    owner: List[int] = []
    running_totals = np.zeros(n)
    bundle_values = np.zeros(n)
    outside_max = np.zeros(n)  # c_i^(t-1): max value among arrived goods not allocated to i
    for t in range(m):
        running_totals += V[:, t]
        score = np.full(n, np.inf)
        pos = running_totals > EPS
        score[pos] = (bundle_values[pos] + np.maximum(outside_max[pos], V[pos, t])) / running_totals[pos]
        score[~pos] = 0.0
        i = int(np.argmin(score))
        owner.append(i)
        # Update c_j for agents who did not receive the current good; for the recipient,
        # g_t is not outside her bundle, so c_i remains the previous outside max.
        for j in range(n):
            if j == i:
                bundle_values[j] += V[j, t]
            else:
                outside_max[j] = max(outside_max[j], V[j, t])
    return owner


def algorithm1_miv(V: np.ndarray, normalize: bool = True, tol: float = 1e-10) -> List[int]:
    """Algorithm 1 from the paper, using perfect maximum-item-value predictions.

    If normalize=True, each agent is first scaled so her maximum item value is 1.
    The returned allocation is invariant under this per-agent scaling for the
    PROP1 guarantee, but welfare is always evaluated outside this function using
    the original V passed to metrics.
    """
    W = normalize_by_miv(V) if normalize else np.asarray(V, dtype=float).copy()
    n, m = W.shape
    owner: List[int] = []
    # r_i stores the earliest max-valued good index; -1 means not observed yet.
    r = np.full(n, -1, dtype=int)
    totals = np.zeros(n)

    def phi_for_agent(i: int, candidate_owner: int, t: int) -> float:
        """Potential for agent i after allocating current good t to candidate_owner."""
        total_after = totals[i] + W[i, t]
        # r has already been updated if t is the first max item for i.
        ri = r[i]
        gets = candidate_owner == i
        own_value_after = 0.0
        # Sum from owner list for previous goods; m is small in experiments. This
        # simple implementation prioritizes clarity and exact agreement with the
        # paper over micro-optimization.
        for g, o in enumerate(owner):
            if o == i:
                own_value_after += W[i, g]
        if gets:
            own_value_after += W[i, t]
        if ri < 0 or t < ri:
            x = 1.0 / (1.0 + total_after)
            y = own_value_after / (1.0 + total_after)
        else:
            # Remove g_{r_i} from the PROP1 accounting term if it is in i's bundle.
            own_without_ri = own_value_after
            if ri == t:
                if gets:
                    own_without_ri -= W[i, t]
            else:
                if ri < len(owner) and owner[ri] == i:
                    own_without_ri -= W[i, ri]
            if total_after <= EPS:
                # This should not occur once ri exists, but keep a safe value.
                return 0.0
            x = 1.0 / total_after
            y = own_without_ri / total_after
        denom = (n * n + n + 1) * x + n * n * y - 1.0
        if denom <= EPS:
            # Numerically, valid instances should keep denom positive. Returning
            # a large value avoids choosing numerically invalid candidates.
            return float("inf")
        return x / denom

    for t in range(m):
        # Update r_i at the arrival of the current good, before making the choice.
        for i in range(n):
            if r[i] < 0 and W[i, t] >= 1.0 - tol:
                r[i] = t
        # Evaluate the global potential after assigning t to each possible agent.
        candidate_phi = np.zeros(n)
        for k in range(n):
            candidate_phi[k] = sum(phi_for_agent(i, k, t) for i in range(n))
        k_star = int(np.argmin(candidate_phi))
        owner.append(k_star)
        totals += W[:, t]
    return owner


def _check_prop1_agent(V: np.ndarray, owner: np.ndarray, i: int, n: int) -> bool:
    """Check if agent i satisfies PROP1 >= 1.0 (uncapped)."""
    total = float(V[i].sum())
    if total <= EPS:
        return True
    own_mask = owner == i
    if own_mask.all():
        return True
    own_value = float(V[i, own_mask].sum())
    best_outside = float(V[i, ~own_mask].max())
    return n * (own_value + best_outside) / total >= 1.0 - 1e-12


def postprocess_welfare(V: np.ndarray, owner: Sequence[int],
                        n: int, m: int, max_passes: int = 100) -> List[int]:
    """Post-process allocation with PROP1-safe welfare-improving swaps and gives.

    Iteratively finds the best welfare-improving operation (swap or one-way give)
    that maintains PROP1 >= 1.0 for all affected agents. Converges when no such
    operation exists.
    """
    V = np.asarray(V, dtype=float)
    owner = np.array(owner, dtype=int)

    for _pass in range(max_passes):
        best_delta = 0.0
        best_op = None  # ('swap', gi, gj, i, j) or ('give', g, i, j)

        # --- Swap operations: exchange one good between two agents ---
        for i in range(n):
            i_goods = np.where(owner == i)[0]
            if len(i_goods) == 0:
                continue
            for j in range(i + 1, n):
                j_goods = np.where(owner == j)[0]
                if len(j_goods) == 0:
                    continue
                for gi in i_goods:
                    vi_gi = V[i, gi]
                    vj_gi = V[j, gi]
                    for gj in j_goods:
                        delta = (vj_gi + V[i, gj]) - (vi_gi + V[j, gj])
                        if delta <= best_delta:
                            continue
                        # Tentative swap
                        new_owner = owner.copy()
                        new_owner[gi] = j
                        new_owner[gj] = i
                        if (_check_prop1_agent(V, new_owner, i, n) and
                                _check_prop1_agent(V, new_owner, j, n)):
                            best_delta = delta
                            best_op = ('swap', gi, gj, i, j)

        # --- Give operations: one-way transfer (donor -> recipient) ---
        for i in range(n):
            i_goods = np.where(owner == i)[0]
            if len(i_goods) == 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                for g in i_goods:
                    delta = V[j, g] - V[i, g]
                    if delta <= best_delta:
                        continue
                    new_owner = owner.copy()
                    new_owner[g] = j
                    if _check_prop1_agent(V, new_owner, i, n):
                        best_delta = delta
                        best_op = ('give', g, i, j)

        if best_op is None:
            break

        if best_op[0] == 'swap':
            _, gi, gj, i, j = best_op
            owner[gi] = j
            owner[gj] = i
        else:
            _, g, i, j = best_op
            owner[g] = j

    return list(owner)


def algorithm1_postprocessed(V: np.ndarray, normalize: bool = True,
                             tol: float = 1e-10, max_passes: int = 100) -> List[int]:
    """Algorithm 1 followed by PROP1-safe welfare post-processing."""
    owner = algorithm1_miv(V, normalize=normalize, tol=tol)
    n, m = V.shape
    return postprocess_welfare(V, owner, n, m, max_passes=max_passes)


ALGORITHMS: Mapping[str, Callable[[np.ndarray], List[int]]] = {
    "Alg. 1": algorithm1_miv,
    "Alg. 1+": algorithm1_postprocessed,
    "Greedy-1": greedy1,
    "Greedy-2": greedy2,
    "Greedy-3": greedy3,
}


def instance_uniform(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    return rng.random((n, m))


def instance_dense(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    # Dense binary-interest instances: each good is valued by many agents, and
    # valued goods have comparable magnitudes. These create many competitive
    # ties/near-ties while keeping utilitarian losses easy to interpret.
    mask = rng.random((n, m)) < 0.60
    values = 0.80 + 0.20 * rng.random((n, m))
    return mask.astype(float) * values


def instance_correlated(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    # Correlated cardinal values: all agents share a common item-quality signal,
    # mixed with idiosyncratic noise.
    common = rng.random((1, m))
    idiosyncratic = rng.random((n, m))
    return 0.50 * common + 0.50 * idiosyncratic


def instance_specialist(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    # Each good has a natural specialist; non-specialists have small residual value.
    V = 0.03 * rng.random((n, m))
    specialists = rng.integers(0, n, size=m)
    for g, s in enumerate(specialists):
        V[s, g] = 0.70 + 0.30 * rng.random()
    return V


INSTANCE_FAMILIES: Mapping[str, Callable[[np.random.Generator, int, int], np.ndarray]] = {
    "uniform": instance_uniform,
    "dense": instance_dense,
    "correlated": instance_correlated,
    "specialist": instance_specialist,
}


def run_random_families(n: int, m: int, trials: int, seed: int) -> Tuple[List[Aggregate], Dict[Tuple[str, str], List[Metrics]]]:
    rng = np.random.default_rng(seed)
    raw: Dict[Tuple[str, str], List[Metrics]] = {}
    for family, gen in INSTANCE_FAMILIES.items():
        for _ in range(trials):
            # Following the MIV-normalized view used by Algorithm 1, all
            # algorithms and metrics are evaluated after scaling each agent so
            # that her maximum item value is 1.
            V = normalize_by_miv(gen(rng, n, m))
            for alg_name, alg in ALGORITHMS.items():
                owner = alg(V)
                raw.setdefault((family, alg_name), []).append(metrics(V, owner))

    aggregates: List[Aggregate] = []
    for (family, alg_name), vals in sorted(raw.items()):
        prop = np.array([v.prop1 for v in vals], dtype=float)
        welfare = np.array([v.welfare for v in vals], dtype=float)
        aggregates.append(
            Aggregate(
                family=family,
                algorithm=alg_name,
                prop1_mean=float(prop.mean()),
                prop1_se=float(prop.std(ddof=1) / math.sqrt(len(prop))) if len(prop) > 1 else 0.0,
                welfare_mean=float(welfare.mean()),
                welfare_se=float(welfare.std(ddof=1) / math.sqrt(len(welfare))) if len(welfare) > 1 else 0.0,
                trials=len(vals),
            )
        )
    return aggregates, raw


def greedy1_counterexample(m: int, n: int = 2) -> np.ndarray:
    V = np.zeros((n, m))
    V[:, 0] = 1.0
    if m > 1:
        V[0, 1:] = 1.0
        V[1, 1:] = 0.5
    return V


def greedy2_counterexample(m: int, n: int = 2) -> np.ndarray:
    V = np.zeros((n, m))
    V[:, 0] = 1.0
    if m > 1:
        V[0, 1:] = 1.0
        V[1, 1:] = 1.0 / (m * m)
    return V


def greedy3_counterexample(cycles: int, n: int = 2) -> np.ndarray:
    """Generate the n=2 adversarial sequence from the proof of Proposition B.2.

    The construction is adaptive to Greedy-3 with lexicographic tie-breaking. It
    returns a finite prefix after the requested number of drop cycles.
    """
    if n != 2:
        raise ValueError("This helper implements the n=2 construction.")
    goods: List[List[float]] = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    owner = greedy3(np.array(goods, dtype=float).T)
    # Ensure we are in the intended branch; with lexicographic ties, owner=[0,1,0].
    for _ in range(cycles):
        Vcur = np.array(goods, dtype=float).T
        owner = greedy3(Vcur)
        alloc = allocation_from_owner(owner, n)
        totals = Vcur.sum(axis=1)
        bundle = np.array([Vcur[i, alloc[i]].sum() if alloc[i] else 0.0 for i in range(n)])
        c = np.zeros(n)
        for i in range(n):
            outside = [g for g in range(Vcur.shape[1]) if owner[g] != i]
            c[i] = Vcur[i, outside].max() if outside else 0.0
        alpha_i = (bundle + c) / totals
        i_min = int(np.argmin(alpha_i))
        j = 1 - i_min
        if alpha_i[j] <= alpha_i[i_min] + 1e-14:
            # Already tied; no filler needed.
            tau = 0
        else:
            tau = max(0, math.ceil((2.0 / c[j]) * totals[j] * (alpha_i[j] / alpha_i[i_min] - 1.0)) - 1)
        for _f in range(tau):
            g = [0.0, 0.0]
            g[j] = c[j] / 2.0
            goods.append(g)
        # Drop good valued at c_r by both agents; here c remains 1 by construction.
        goods.append([c[0], c[1]])
    return np.array(goods, dtype=float).T


def stress_test(max_m: int = 160) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return ratios for targeted greedy constructions and Algorithm 1 baselines.

    Output maps construction name to (x_values, targeted_greedy_ratios, alg1_ratios).
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ms = np.arange(4, max_m + 1, 4)
    g1_ratios, a1_ratios = [], []
    for m in ms:
        V = greedy1_counterexample(int(m), n=2)
        g1_ratios.append(metrics(V, greedy1(V)).prop1)
        a1_ratios.append(metrics(V, algorithm1_miv(V)).prop1)
    out["G1 construction"] = (ms, np.array(g1_ratios), np.array(a1_ratios))

    g2_ratios, a2_ratios = [], []
    for m in ms:
        V = greedy2_counterexample(int(m), n=2)
        g2_ratios.append(metrics(V, greedy2(V)).prop1)
        a2_ratios.append(metrics(V, algorithm1_miv(V)).prop1)
    out["G2 construction"] = (ms, np.array(g2_ratios), np.array(a2_ratios))

    xs, g3_ratios, a3_ratios = [], [], []
    cycles = 1
    while True:
        V = greedy3_counterexample(cycles, n=2)
        xs.append(V.shape[1])
        g3_ratios.append(metrics(V, greedy3(V)).prop1)
        a3_ratios.append(metrics(V, algorithm1_miv(V)).prop1)
        if V.shape[1] >= max_m:
            break
        cycles += 1
    out["G3 construction"] = (np.array(xs), np.array(g3_ratios), np.array(a3_ratios))
    return out


def write_summary_csv(aggregates: Sequence[Aggregate], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "algorithm", "prop1_mean", "prop1_se", "welfare_mean", "welfare_se", "trials"])
        for a in aggregates:
            writer.writerow([
                a.family,
                a.algorithm,
                f"{a.prop1_mean:.6f}",
                f"{a.prop1_se:.6f}",
                f"{a.welfare_mean:.6f}",
                f"{a.welfare_se:.6f}",
                a.trials,
            ])


def print_summary(aggregates: Sequence[Aggregate]) -> None:
    family_order = list(INSTANCE_FAMILIES.keys())
    alg_order = list(ALGORITHMS.keys())
    lookup = {(a.family, a.algorithm): a for a in aggregates}
    for family in family_order:
        print(f"\n{family}")
        for alg in alg_order:
            a = lookup[(family, alg)]
            print(
                f"  {alg:8s}: PROP1 {a.prop1_mean:.3f} ± {1.96*a.prop1_se:.3f}, "
                f"welfare {a.welfare_mean:.3f} ± {1.96*a.welfare_se:.3f}"
            )



def write_stress_csv(stress: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["construction", "goods", "targeted_greedy_prop1", "algorithm1_prop1"])
        for name, (xs, greedy_ratios, alg1_ratios) in stress.items():
            for x, g, a in zip(xs, greedy_ratios, alg1_ratios):
                writer.writerow([name, int(x), f"{float(g):.6f}", f"{float(a):.6f}"])

def make_combined_figure(aggregates: Sequence[Aggregate], stress: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    family_order = list(INSTANCE_FAMILIES.keys())
    alg_order = list(ALGORITHMS.keys())
    marker = {"Alg. 1": "o", "Alg. 1+": "*", "Greedy-1": "s", "Greedy-2": "^", "Greedy-3": "D"}
    lookup = {(a.family, a.algorithm): a for a in aggregates}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ax = axes[0]
    for alg in alg_order:
        xs = [lookup[(fam, alg)].welfare_mean for fam in family_order]
        ys = [lookup[(fam, alg)].prop1_mean for fam in family_order]
        xerr = [1.96 * lookup[(fam, alg)].welfare_se for fam in family_order]
        yerr = [1.96 * lookup[(fam, alg)].prop1_se for fam in family_order]
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker=marker[alg], linestyle="", capsize=2, label=alg)
        for fam, x, y in zip(family_order, xs, ys):
            label = {"uniform": "U", "dense": "D", "correlated": "C", "specialist": "S"}[fam]
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.set_xlabel("Normalized utilitarian welfare")
    ax.set_ylabel("Realized PROP1 ratio")
    ax.set_title("Random instances (n=8, m=40)")
    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    for name, (xs, greedy_ratios, alg1_ratios) in stress.items():
        short = name.split()[0]
        ax.plot(xs, greedy_ratios, marker=".", linewidth=1.2, label=f"{short} targeted greedy")
    # Show the worst Algorithm 1 ratio across the stress-test prefixes with a single marker series.
    all_alg_points_x: List[float] = []
    all_alg_points_y: List[float] = []
    for _name, (xs, _greedy, alg1_ratios) in stress.items():
        all_alg_points_x.extend(xs.tolist())
        all_alg_points_y.extend(alg1_ratios.tolist())
    order = np.argsort(np.array(all_alg_points_x))
    ax.scatter(np.array(all_alg_points_x)[order], np.array(all_alg_points_y)[order], s=14, label="Alg. 1 on same prefixes")
    ax.axhline(0.5, linestyle="--", linewidth=1.0, label="1/n guarantee (n=2)")
    ax.set_xlabel("Number of goods")
    ax.set_ylabel("Realized PROP1 ratio")
    ax.set_title("Greedy counterexample stress tests")
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_pareto_figure(aggregates: Sequence[Aggregate], path: Path) -> None:
    family_order = list(INSTANCE_FAMILIES.keys())
    alg_order = list(ALGORITHMS.keys())
    marker = {"Alg. 1": "o", "Alg. 1+": "*", "Greedy-1": "s", "Greedy-2": "^", "Greedy-3": "D"}
    lookup = {(a.family, a.algorithm): a for a in aggregates}
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for alg in alg_order:
        xs = [lookup[(fam, alg)].welfare_mean for fam in family_order]
        ys = [lookup[(fam, alg)].prop1_mean for fam in family_order]
        xerr = [1.96 * lookup[(fam, alg)].welfare_se for fam in family_order]
        yerr = [1.96 * lookup[(fam, alg)].prop1_se for fam in family_order]
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker=marker[alg], linestyle="", capsize=2, label=alg)
        for fam, x, y in zip(family_order, xs, ys):
            label = {"uniform": "U", "dense": "D", "correlated": "C", "specialist": "S"}[fam]
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.set_xlabel("Normalized utilitarian welfare")
    ax.set_ylabel("Realized PROP1 ratio")
    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_stress_figure(stress: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for name, (xs, greedy_ratios, _alg1_ratios) in stress.items():
        short = name.split()[0]
        ax.plot(xs, greedy_ratios, marker=".", linewidth=1.2, label=f"{short} targeted greedy")
    for _name, (xs, _greedy, alg1_ratios) in stress.items():
        ax.scatter(xs, alg1_ratios, s=14, label="Alg. 1" if _name == next(iter(stress)) else None)
    ax.axhline(0.5, linestyle="--", linewidth=1.0, label="1/n guarantee (n=2)")
    ax.set_xlabel("Number of goods")
    ax.set_ylabel("Realized PROP1 ratio")
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def make_pareto_grid_figure(aggregates: Sequence[Aggregate], path: Path) -> None:
    """Two-column friendly 2x2 Pareto panels, one per instance family."""
    family_order = list(INSTANCE_FAMILIES.keys())
    alg_order = list(ALGORITHMS.keys())
    marker = {"Alg. 1": "o", "Alg. 1+": "*", "Greedy-1": "s", "Greedy-2": "^", "Greedy-3": "D"}
    lookup = {(a.family, a.algorithm): a for a in aggregates}
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), sharex=True, sharey=True)
    pretty = {"uniform": "Uniform", "dense": "Dense binary", "correlated": "Correlated", "specialist": "Specialist"}
    for ax, family in zip(axes.flat, family_order):
        for alg in alg_order:
            a = lookup[(family, alg)]
            ax.errorbar(
                [a.welfare_mean], [a.prop1_mean],
                xerr=[1.96 * a.welfare_se], yerr=[1.96 * a.prop1_se],
                marker=marker[alg], linestyle="", capsize=2, label=alg, markersize=6,
            )
        ax.set_title(pretty[family], fontsize=10)
        ax.set_xlim(0.0, 1.03)
        ax.set_ylim(0.0, 1.03)
        ax.grid(True, alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("Realized PROP1 ratio")
    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized utilitarian welfare")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1428)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    aggregates, raw = run_random_families(args.n, args.m, args.trials, args.seed)
    print_summary(aggregates)
    write_summary_csv(aggregates, args.outdir / "summary.csv")

    stress = stress_test(max_m=500)
    write_stress_csv(stress, args.outdir / "stress_summary.csv")
    # Print compact stress diagnostics.
    print("\nStress tests (last prefix):")
    for name, (xs, greedy_ratios, alg1_ratios) in stress.items():
        print(f"  {name:16s}: m={int(xs[-1]):3d}, targeted greedy={greedy_ratios[-1]:.3f}, Alg.1={alg1_ratios[-1]:.3f}")

    make_pareto_figure(aggregates, args.outdir / "fig_pareto.pdf")
    make_pareto_grid_figure(aggregates, args.outdir / "fig_pareto_grid.pdf")
    make_stress_figure(stress, args.outdir / "fig_stress.pdf")
    make_combined_figure(aggregates, stress, args.outdir / "fig_experiments.pdf")


if __name__ == "__main__":
    main()
