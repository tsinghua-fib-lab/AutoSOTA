"""
Clean script to run SIGS search for Compressible Euler with enforced same-series constraint.

This script:
1. Loads the SIGS model and sampler
2. Samples spatial 2D series S(x,y)
3. Filters for pure trigonometric expressions (sin/cos with pi*)
4. Searches for best 4-tuple where ALL fields use THE SAME S(x,y)
5. Reports best system found
"""

import os
import sys
from pathlib import Path

# Set up paths
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# Import from euler_search
from euler_search import (
    create_compressible_euler_problem,
    build_mesh,
    euler_system_loss,
    model,
    device
)
from sampler import FlexibleVectorSampler

def search_same_series(
    sampler,
    model,
    n_series_samples=1000,
    n_candidates=50,
    n_combinations=500,
    seed=42,
    sin_sin_only=False,
    use_different_subclusters=False,
):
    """
    Search for best Euler system where ALL fields use the SAME spatial series S(x,y).

    Args:
        sampler: FlexibleVectorSampler instance
        model: VAE model
        n_series_samples: Number of series to sample from SIGS
        n_candidates: Number of top candidates to keep
        n_combinations: Number of random combinations to evaluate
        seed: Random seed
        sin_sin_only: If True, filter to sin(pi*i*x)*sin(pi*j*y) patterns
        use_different_subclusters: If True, each sum term uses a different subcluster
    """

    np.random.seed(seed)
    random.seed(seed)

    print(f"\n{'='*80}")
    print("COMPRESSIBLE EULER SEARCH - SAME SERIES FOR ALL FIELDS")
    print(f"{'='*80}\n")

    # Create problem
    ce_problem = create_compressible_euler_problem(seed=seed, K=3)
    X, Y = build_mesh(ce_problem)
    meshes = [X, Y]

    print(f"Manufactured solution uses K={ce_problem['parameters']['K']} Fourier modes")
    print(f"Problem domain: [{ce_problem['mesh']['x']['start']}, {ce_problem['mesh']['x']['end']}] x [{ce_problem['mesh']['y']['start']}, {ce_problem['mesh']['y']['end']}]")
    print(f"Mesh resolution: {ce_problem['mesh']['x']['points']} x {ce_problem['mesh']['y']['points']}\n")

    # Sample spatial series
    print(f"Sampling {n_series_samples} spatial 2D series from SIGS...")

    # Determine appropriate number of subclusters based on data size
    n_spatial_exprs = len(sampler.clusters_data['SPATIAL_2D']['expressions'])
    n_subclusters = min(100, max(1, n_spatial_exprs // 4))  # At least 4 exprs per cluster
    print(f"Using {n_subclusters} subclusters for {n_spatial_exprs} SPATIAL_2D expressions")

    # Enable different subclusters per term if requested
    if use_different_subclusters:
        sampler._use_different_subclusters_per_term = True
        print("Each sum term will use a DIFFERENT subcluster (maximum diversity)")
    else:
        sampler._use_different_subclusters_per_term = False
        print("All sum terms will use the SAME subcluster (maximum coherence)")

    series_sample_id = sampler.sample_coherent_sum_expressions(
        expression_template="A*B",
        role_categories={"A": "SPATIAL_2D", "B": "CONSTANT"},
        role_subclusters={"SPATIAL_2D": n_subclusters, "CONSTANT": 1},
        n_sum_terms=5,
        sum_operator="+",
        n_samples=n_series_samples,
        seed=seed,
        model=model,
    )

    S_exprs, _, _, _ = sampler.get_sampling_results(series_sample_id)
    print(f"Sampled {len(S_exprs)} expressions\n")

    # Filter for pure trig
    print("Filtering for pure trigonometric expressions...")
    pure_trig_exprs = []
    for expr in S_exprs:
        has_trig = ('sin' in expr or 'cos' in expr) and 'pi*' in expr
        has_bad = any(bad in expr for bad in ['exp(-(', 'log', 'sqrt', 'x^2*', 'x*y', '/x', '/y', 'sinh', 'cosh'])
        if has_trig and not has_bad:
            pure_trig_exprs.append(expr)

    print(f"Found {len(pure_trig_exprs)} pure trig expressions ({100*len(pure_trig_exprs)/len(S_exprs):.1f}%)\n")

    # Optional: filter for sin*sin only
    if sin_sin_only:
        print("Applying sin*sin filter...")

        def is_sin_sin_expr(expr: str) -> bool:
            """Check if expression contains only sin(pi*i*x)*sin(pi*j*y) terms"""
            if 'cos' in expr or 'tan' in expr:
                return False
            toks = expr.replace('-', '+-').split('+')
            toks = [t.strip() for t in toks if t.strip()]
            for t in toks:
                if t.count('sin(') != 2:
                    return False
                if 'pi*' not in t or '*x' not in t or '*y' not in t:
                    return False
            return True

        sin_sin_filtered = [s for s in pure_trig_exprs if is_sin_sin_expr(s)]
        print(f"Found {len(sin_sin_filtered)} sin*sin expressions\n")

        if len(sin_sin_filtered) >= n_candidates:
            S_exprs = sin_sin_filtered
        else:
            print(f"Not enough sin*sin expressions, using all {len(pure_trig_exprs)} pure trig\n")
            S_exprs = pure_trig_exprs
    else:
        S_exprs = pure_trig_exprs

    if len(S_exprs) < n_candidates:
        n_candidates = len(S_exprs)
        print(f"Reducing n_candidates to {n_candidates}\n")

    # Select candidate series
    candidates = random.sample(S_exprs, n_candidates)

    print(f"Selected {len(candidates)} candidates")
    print(f"First 3 examples:")
    for i, c in enumerate(candidates[:3]):
        print(f"  {i+1}. {c}")
    print()

    # Build ansatz for each candidate (same S for all fields)
    # rho = exp(S), u = S, v = S, p = exp(S)
    candidate_systems = []
    for s in candidates:
        candidate_systems.append({
            'S': s,
            'rho': f"exp({s})",
            'u': s,
            'v': s,
            'p': f"exp({s})",
        })

    # Random search: evaluate n_combinations
    print(f"Evaluating {n_combinations} random candidates (parallel)...\n")

    best_loss = float('inf')
    best_idx = None
    best_system = None

    def eval_candidate(it):
        idx = random.randrange(len(candidate_systems))
        sys = candidate_systems[idx]

        try:
            loss = euler_system_loss(
                rho_expr_str=sys['rho'],
                u_expr_str=sys['u'],
                v_expr_str=sys['v'],
                p_expr_str=sys['p'],
                problem=ce_problem,
                meshes=meshes,
            )
        except Exception as e:
            # If evaluation fails, return infinite loss
            loss = float('inf')

        return (it, idx, loss, sys)

    # Use all available CPUs for maximum parallelism
    n_workers = os.cpu_count() or 1
    print(f"Using {n_workers} parallel workers\n")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks at once for maximum parallelism
        futures = [executor.submit(eval_candidate, it) for it in range(n_combinations)]

        for fut in tqdm(as_completed(futures), total=n_combinations, desc="Searching"):
            it, idx, loss, sys = fut.result()

            if loss < best_loss:
                best_loss = loss
                best_idx = idx
                best_system = sys
                print(f"\n[iter {it}] New best loss: {loss:.6e}")
                print(f"  S = {sys['S'][:100]}...")
                print(f"  (rho=exp(S), u=S, v=S, p=exp(S))\n")

    # Report final best
    print(f"\n{'='*80}")
    print("BEST SYSTEM FOUND")
    print(f"{'='*80}\n")
    print(f"Loss: {best_loss:.6e}")
    print(f"Candidate index: {best_idx}\n")
    print(f"S(x,y) = {best_system['S']}\n")
    print(f"Ansatz:")
    print(f"  rho(x,y) = exp(S(x,y))")
    print(f"  u(x,y)   = S(x,y)")
    print(f"  v(x,y)   = S(x,y)")
    print(f"  p(x,y)   = exp(S(x,y))\n")
    print(f"Full expressions:")
    print(f"  rho = {best_system['rho']}")
    print(f"  u   = {best_system['u']}")
    print(f"  v   = {best_system['v']}")
    print(f"  p   = {best_system['p']}\n")

    return {
        'loss': best_loss,
        'idx': best_idx,
        'S': best_system['S'],
        'rho_expr': best_system['rho'],
        'u_expr': best_system['u'],
        'v_expr': best_system['v'],
        'p_expr': best_system['p'],
    }


if __name__ == "__main__":
    # Load sampler
    sampler = FlexibleVectorSampler(
        cluster_file="data/clusters.pkl",
        model=model,
        device=device,
    )

    # Run search
    result = search_same_series(
        sampler=sampler,
        model=model,
        n_series_samples=10000,
        n_candidates=200,
        n_combinations=2000,
        seed=42,
        sin_sin_only=True,
    )

    print("Done!")
