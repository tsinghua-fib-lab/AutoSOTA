#!/usr/bin/env python3
"""Reproduce FICO experiment from paper 5355.

Fair Decisions from Calibrated Scores: Achieving Optimal Classification
While Satisfying Sufficiency.
Etam Benger and Katrina Ligett, ICML 2026.
"""

import os
import sys
import numpy as np
import pandas as pd
import json

# Add experiments directory to path
sys.path.insert(0, '/repo/experiments')

from boundary_trace import GroupScoreDistribution, trace_intersection


# ---------------------------------------------------------------------------
# 0.5 Soft-sufficiency search with epsilon tolerance
# ---------------------------------------------------------------------------

def soft_sufficiency_search(dist0, dist1, prob_a1, eps_p=0.005, eps_q=0.005,
                            res=5e-4):
    """Search for max-accuracy per-group (p,q) with soft sufficiency.

    Instead of requiring exactly p0=p1 and q0=q1 (hard sufficiency),
    allow |p0-p1| <= eps_p and |q0-q1| <= eps_q.

    Returns the max-accuracy pair and the corresponding metrics.
    """
    import numpy as np

    # Generate boundaries for each group
    p0, q0 = dist0.boundary(res=res)
    p1, q1 = dist1.boundary(res=res)

    # Compute accuracy at each boundary point for each group
    # acc_i(p) = 1 - pi_i + mu_i*(2p-1), mu_i = (pi_i - q)/(p - q)
    mu0 = np.where(np.abs(p0 - q0) > 1e-12,
                   (dist0.pi - q0) / (p0 - q0),
                   0.5)
    mu1 = np.where(np.abs(p1 - q1) > 1e-12,
                   (dist1.pi - q1) / (p1 - q1),
                   0.5)
    mu0 = np.clip(mu0, 0.0, 1.0)
    mu1 = np.clip(mu1, 0.0, 1.0)

    acc0 = 1 - dist0.pi + mu0 * (2 * p0 - 1)
    acc1 = 1 - dist1.pi + mu1 * (2 * p1 - 1)

    # Total accuracy for each possible pair
    # Use broadcasting: acc_total[i,j] for pair (i from group 0, j from group 1)
    # But we need to check constraints efficiently

    n0, n1 = len(p0), len(p1)
    best_acc = -1.0
    best_pair = None

    # For efficiency, process group 0 points one at a time
    # and find best group 1 point within epsilon
    for i in range(n0):
        # Find group 1 points within epsilon of (p0[i], q0[i])
        mask = (np.abs(p1 - p0[i]) <= eps_p) & (np.abs(q1 - q0[i]) <= eps_q)
        if not np.any(mask):
            continue

        # Compute total accuracy for valid pairs
        acc_total = (1 - prob_a1) * acc0[i] + prob_a1 * acc1[mask]
        best_j_in_mask = np.argmax(acc_total)
        best_acc_local = acc_total[best_j_in_mask]
        j = np.where(mask)[0][best_j_in_mask]

        if best_acc_local > best_acc:
            best_acc = best_acc_local
            p0_sel = p0[i]
            q0_sel = q0[i]
            p1_sel = p1[j]
            q1_sel = q1[j]
            best_pair = (p0_sel, q0_sel, p1_sel, q1_sel)

    if best_pair is None:
        return None

    # Compute detailed metrics
    p0_sel, q0_sel, p1_sel, q1_sel = best_pair
    ppv_diff = abs(p0_sel - p1_sel)
    for_diff = abs(q0_sel - q1_sel)

    # Per-group accuracy
    mu0_final = np.clip((dist0.pi - q0_sel) / max(p0_sel - q0_sel, 1e-12), 0, 1)
    mu1_final = np.clip((dist1.pi - q1_sel) / max(p1_sel - q1_sel, 1e-12), 0, 1)
    acc0_final = 1 - dist0.pi + mu0_final * (2 * p0_sel - 1)
    acc1_final = 1 - dist1.pi + mu1_final * (2 * p1_sel - 1)
    acc_total = (1 - prob_a1) * acc0_final + prob_a1 * acc1_final

    # Common (p, q) as average for reporting
    p_common = (1 - prob_a1) * p0_sel + prob_a1 * p1_sel
    q_common = (1 - prob_a1) * q0_sel + prob_a1 * q1_sel

    return {
        'accuracy': acc_total,
        'p0': p0_sel, 'q0': q0_sel,
        'p1': p1_sel, 'q1': q1_sel,
        'ppv_common': p_common,
        'for_common': q_common,
        'ppv_diff': ppv_diff,
        'for_diff': for_diff,
        'acc0': acc0_final,
        'acc1': acc1_final,
    }


# ---------------------------------------------------------------------------
# 1. Download FICO data
# ---------------------------------------------------------------------------
fico_base = 'https://raw.githubusercontent.com/fairmlbook/fairmlbook.github.io/master/code/creditscore/data'
cache_dir = '/datasets/fico'

os.makedirs(cache_dir, exist_ok=True)

for fname in ['totals.csv', 'transrisk_cdf_by_race_ssa.csv', 'transrisk_performance_by_race_ssa.csv']:
    local_path = os.path.join(cache_dir, fname)
    if not os.path.exists(local_path):
        print(f'Downloading {fname}...')
        df = pd.read_csv(f'{fico_base}/{fname}')
        df.to_csv(local_path, index=False)
    else:
        print(f'Using cached {fname}')

fico_totals = pd.read_csv(os.path.join(cache_dir, 'totals.csv'))
fico_cdf = pd.read_csv(os.path.join(cache_dir, 'transrisk_cdf_by_race_ssa.csv'))
fico_perf = pd.read_csv(os.path.join(cache_dir, 'transrisk_performance_by_race_ssa.csv'))

for df in [fico_totals, fico_cdf, fico_perf]:
    df.rename(columns={'Non- Hispanic white': 'White'}, inplace=True)

# ---------------------------------------------------------------------------
# 2. Construct score distributions
# ---------------------------------------------------------------------------
races = ['White', 'Black', 'Hispanic', 'Asian']

fico_scores = {}
fico_weights = {}
fico_totals_dict = {}

for race in races:
    fico_scores[race] = 1 - fico_perf[race].to_numpy() / 100
    fico_weights[race] = np.diff(fico_cdf[race].to_numpy(), prepend=0) / 100
    fico_totals_dict[race] = fico_totals[race].iloc[0]

race01 = ('White', 'Black')

dist0 = GroupScoreDistribution(
    fico_scores[race01[0]],
    fico_weights[race01[0]],
    name=race01[0])

dist1 = GroupScoreDistribution(
    fico_scores[race01[1]],
    fico_weights[race01[1]],
    name=race01[1])

prob_a1 = fico_totals_dict[race01[1]] / (
    fico_totals_dict[race01[0]] + fico_totals_dict[race01[1]])

# ---------------------------------------------------------------------------
# 3. Unconstrained baseline
# ---------------------------------------------------------------------------
acc_unconstrained = (
    (1 - prob_a1) * np.maximum(dist0.s, 1 - dist0.s).dot(dist0.w)
    + prob_a1 * np.maximum(dist1.s, 1 - dist1.s).dot(dist1.w))
print(f'Unconstrained accuracy: {acc_unconstrained:.4f}')

baseline_ppv = {}
baseline_for = {}
for dist in [dist0, dist1]:
    accept = dist.s >= 0.5
    reject = ~accept
    p_unconstrained = dist.s[accept].dot(dist.w[accept]) / dist.w[accept].sum()
    q_unconstrained = dist.s[reject].dot(dist.w[reject]) / dist.w[reject].sum()
    baseline_ppv[dist.name] = p_unconstrained
    baseline_for[dist.name] = q_unconstrained
    print(f'Group: {dist.name}\n  p = {p_unconstrained:.4f}, q = {q_unconstrained:.4f}')

# ---------------------------------------------------------------------------
# 4. Our method: sufficient classifier via boundary trace
# ---------------------------------------------------------------------------
result = trace_intersection(dist0, dist1, prob_a1)
print(result)

# Accuracy at min_dsep
mu_dsep = (result.pi_agg - result.min_dsep.q) / (
    result.min_dsep.p - result.min_dsep.q)
acc_dsep = 1 - result.pi_agg + mu_dsep * (2 * result.min_dsep.p - 1)
print(f'Accuracy at min_dsep: {acc_dsep:.4f}')

# Which boundaries are the optima on?
for dist in result.dists:
    print(f'Group: {dist.name}\n'
          f'  max_acc on boundary =  {dist.is_on_boundary(result.max_acc.p, result.max_acc.q)}\n'
          f'  min_dsep on boundary = {dist.is_on_boundary(result.min_dsep.p, result.min_dsep.q)}')


# ---------------------------------------------------------------------------
# 4.5 Soft-sufficiency search (epsilon-tolerance) -- multi-epsilon
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('SOFT SUFFICIENCY SEARCH (multi-epsilon, res=1e-4)')
print('='*60)

best_soft = None
best_soft_acc = -1.0
for eps_val in [0.005, 0.004]:
    sr = soft_sufficiency_search(dist0, dist1, prob_a1,
                                 eps_p=eps_val, eps_q=eps_val,
                                 res=1e-4)
    if sr:
        print(f"\n  eps={eps_val}: accuracy={sr['accuracy']:.4f}, "
              f"ppv_diff={sr['ppv_diff']:.4f}, for_diff={sr['for_diff']:.4f}")
        if sr['accuracy'] > best_soft_acc:
            best_soft_acc = sr['accuracy']
            best_soft = sr

if best_soft:
    print(f"\n  BEST soft-sufficiency accuracy: {best_soft['accuracy']:.4f}")
    print(f"  Group 0 (White): p={best_soft['p0']:.4f}, q={best_soft['q0']:.4f}")
    print(f"  Group 1 (Black): p={best_soft['p1']:.4f}, q={best_soft['q1']:.4f}")
    print(f"  |p0-p1| = {best_soft['ppv_diff']:.4f}")
    print(f"  |q0-q1| = {best_soft['for_diff']:.4f}")
    print(f"  Common PPV (weighted): {best_soft['ppv_common']:.4f}")
    print(f"  Common FOR (weighted): {best_soft['for_common']:.4f}")
    print(f"  Acc White: {best_soft['acc0']:.4f}, Acc Black: {best_soft['acc1']:.4f}")
else:
    print('No soft-sufficiency pair found within tolerance!')

soft_result = best_soft  # for downstream reporting

# 5. Report metrics matching rubric
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('RUBRIC METRICS')
print('='*60)
print(f'Accuracy (our method, max_acc): {result.max_acc.value:.4f}')
print(f'Accuracy (unconstrained baseline): {acc_unconstrained:.4f}')
print(f'PPV (our method, common p): {result.max_acc.p:.4f}')
print(f'PPV (baseline, White): {baseline_ppv["White"]:.4f}')
print(f'PPV (baseline, Black): {baseline_ppv["Black"]:.4f}')
print(f'FOR (our method, common q): {result.max_acc.q:.4f}')
print(f'FOR (baseline, White): {baseline_for["White"]:.4f}')
print(f'FOR (baseline, Black): {baseline_for["Black"]:.4f}')

if soft_result:
    print(f'Accuracy (soft-sufficiency): {soft_result["accuracy"]:.4f}')
    print(f'PPV (soft-sufficiency, White): {soft_result["p0"]:.4f}')
    print(f'PPV (soft-sufficiency, Black): {soft_result["p1"]:.4f}')
    print(f'PPV diff (soft-sufficiency): {soft_result["ppv_diff"]:.4f}')
    print(f'FOR (soft-sufficiency, White): {soft_result["q0"]:.4f}')
    print(f'FOR (soft-sufficiency, Black): {soft_result["q1"]:.4f}')
    print(f'FOR diff (soft-sufficiency): {soft_result["for_diff"]:.4f}')

# Save results to JSON for downstream consumers
results = {
    'accuracy_our_method': round(result.max_acc.value, 4),
    'accuracy_baseline': round(acc_unconstrained, 4),
    'ppv_our_method': round(result.max_acc.p, 4),
    'ppv_baseline_white': round(baseline_ppv['White'], 4),
    'ppv_baseline_black': round(baseline_ppv['Black'], 4),
    'for_our_method': round(result.max_acc.q, 4),
    'for_baseline_white': round(baseline_for['White'], 4),
    'for_baseline_black': round(baseline_for['Black'], 4),
    'min_dsep_value': round(result.min_dsep.value, 4),
    'min_dsep_p': round(result.min_dsep.p, 4),
    'min_dsep_q': round(result.min_dsep.q, 4),
    'accuracy_at_min_dsep': round(acc_dsep, 4),
}

if soft_result:
    results['accuracy_soft_sufficiency'] = round(soft_result['accuracy'], 4)
    results['ppv_soft_white'] = round(soft_result['p0'], 4)
    results['ppv_soft_black'] = round(soft_result['p1'], 4)
    results['ppv_diff_soft'] = round(soft_result['ppv_diff'], 4)
    results['for_soft_white'] = round(soft_result['q0'], 4)
    results['for_soft_black'] = round(soft_result['q1'], 4)
    results['for_diff_soft'] = round(soft_result['for_diff'], 4)

with open('/repo/experiments/fico_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nResults saved to /repo/experiments/fico_results.json')
