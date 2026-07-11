"""
Search for compressible Euler system with 4 independent ansatz functions.
Each field (rho, u, v, p) gets its own S function, then optimize all together with JAX.
"""
import os
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from euler_search import (
    create_compressible_euler_problem,
    euler_system_loss,
    device,
    model,
)


def search_four_independent_fields(
    sampler,
    model,
    n_series_samples=5000,
    n_candidates=200,
    n_combinations=1000,
    seed=42,
    sin_sin_only=False,
    use_different_subclusters=False,
):
    """
    Search for 4 independent S functions: S_rho, S_u, S_v, S_p
    Then construct: rho=exp(S_rho), u=S_u, v=S_v, p=exp(S_p)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create problem
    ce_problem = create_compressible_euler_problem()
    x_mesh = np.linspace(
        ce_problem['mesh']['x']['start'],
        ce_problem['mesh']['x']['end'],
        ce_problem['mesh']['x']['points']
    )
    y_mesh = np.linspace(
        ce_problem['mesh']['y']['start'],
        ce_problem['mesh']['y']['end'],
        ce_problem['mesh']['y']['points']
    )
    X, Y = np.meshgrid(x_mesh, y_mesh)
    meshes = [X, Y]

    print(f"Manufactured solution uses K={ce_problem['parameters']['K']} Fourier modes")
    print(f"Problem domain: [{ce_problem['mesh']['x']['start']}, {ce_problem['mesh']['x']['end']}] x [{ce_problem['mesh']['y']['start']}, {ce_problem['mesh']['y']['end']}]")
    print(f"Mesh resolution: {ce_problem['mesh']['x']['points']} x {ce_problem['mesh']['y']['points']}\n")

    # Sample spatial series FOR EACH FIELD
    print(f"Sampling {n_series_samples} spatial 2D series for EACH of 4 fields from SIGS...")

    # Determine appropriate number of subclusters based on data size
    n_spatial_exprs = len(sampler.clusters_data['SPATIAL_2D']['expressions'])
    n_subclusters = min(100, max(1, n_spatial_exprs // 4))
    print(f"Using {n_subclusters} subclusters for {n_spatial_exprs} SPATIAL_2D expressions")

    # Enable different subclusters per term if requested
    if use_different_subclusters:
        sampler._use_different_subclusters_per_term = True
        print("Each sum term will use a DIFFERENT subcluster (maximum diversity)")
    else:
        sampler._use_different_subclusters_per_term = False
        print("All sum terms will use the SAME subcluster (maximum coherence)")

    # Sample expressions for each field independently
    field_expressions = {}
    for field_name in ['rho', 'u', 'v', 'p']:
        print(f"\nSampling for field: {field_name}")
        series_sample_id = sampler.sample_coherent_sum_expressions(
            expression_template="A*B",
            role_categories={"A": "SPATIAL_2D", "B": "CONSTANT"},
            role_subclusters={"SPATIAL_2D": n_subclusters, "CONSTANT": 1},
            n_sum_terms=5,
            sum_operator="+",
            n_samples=n_series_samples,
            seed=seed + hash(field_name) % 1000,  # Different seed per field
            model=model,
        )

        exprs, _, _, _ = sampler.get_sampling_results(series_sample_id)
        print(f"Sampled {len(exprs)} expressions for {field_name}")

        # Filter for pure trig
        print(f"Filtering for pure trigonometric expressions...")
        pure_trig_exprs = []
        for expr in exprs:
            has_trig = ('sin' in expr or 'cos' in expr) and 'pi*' in expr
            has_bad = any(bad in expr for bad in ['exp(-(', 'log', 'sqrt', 'x^2*', 'x*y', '/x', '/y', 'sinh', 'cosh'])
            if has_trig and not has_bad:
                pure_trig_exprs.append(expr)

        print(f"Found {len(pure_trig_exprs)} pure trig expressions ({100*len(pure_trig_exprs)/len(exprs):.1f}%)")

        # Optional: filter for sin*sin only
        if sin_sin_only:
            print("Applying sin*sin filter...")

            def is_sin_sin_expr(s):
                # Check if expression contains only sin*sin patterns (no cos)
                import re
                # Find all trig terms
                trig_terms = re.findall(r'(sin|cos)\([^)]+\)\s*\*\s*(sin|cos)\([^)]+\)', s)
                if not trig_terms:
                    return False
                # All terms must be sin*sin
                for t in trig_terms:
                    if t[0] != 'sin' or t[1] != 'sin':
                        return False
                # Also check individual trig functions contain pi*
                trig_funcs = re.findall(r'(sin|cos)\(([^)]+)\)', s)
                for func, arg in trig_funcs:
                    if 'pi*' not in arg or '*x' not in arg or '*y' not in arg:
                        return False
                return True

            sin_sin_filtered = [s for s in pure_trig_exprs if is_sin_sin_expr(s)]
            print(f"Found {len(sin_sin_filtered)} sin*sin expressions\n")

            if len(sin_sin_filtered) >= n_candidates:
                field_expressions[field_name] = sin_sin_filtered
            else:
                print(f"Not enough sin*sin expressions, using all {len(pure_trig_exprs)} pure trig\n")
                field_expressions[field_name] = pure_trig_exprs
        else:
            field_expressions[field_name] = pure_trig_exprs

    # Select candidates for each field
    print(f"\nSelecting {n_candidates} candidates for each field...")
    field_candidates = {}
    for field_name in ['rho', 'u', 'v', 'p']:
        exprs = field_expressions[field_name]
        if len(exprs) < n_candidates:
            n_cand = len(exprs)
            print(f"Reducing n_candidates to {n_cand} for {field_name}")
        else:
            n_cand = n_candidates
        field_candidates[field_name] = random.sample(exprs, n_cand)
        print(f"  {field_name}: {len(field_candidates[field_name])} candidates")

    # Random search: evaluate n_combinations of (S_rho, S_u, S_v, S_p)
    print(f"\nEvaluating {n_combinations} random 4-field combinations (parallel)...\n")

    best_loss = float('inf')
    best_system = None

    def eval_candidate(it):
        # Randomly pick one expression for each field
        S_rho = random.choice(field_candidates['rho'])
        S_u = random.choice(field_candidates['u'])
        S_v = random.choice(field_candidates['v'])
        S_p = random.choice(field_candidates['p'])

        system = {
            'S_rho': S_rho,
            'S_u': S_u,
            'S_v': S_v,
            'S_p': S_p,
            'rho': f"exp({S_rho})",
            'u': S_u,
            'v': S_v,
            'p': f"exp({S_p})",
        }

        try:
            loss = euler_system_loss(
                rho_expr_str=system['rho'],
                u_expr_str=system['u'],
                v_expr_str=system['v'],
                p_expr_str=system['p'],
                problem=ce_problem,
                meshes=meshes,
            )
        except Exception as e:
            loss = float('inf')

        return (it, loss, system)

    # Use all available CPUs for maximum parallelism
    n_workers = os.cpu_count() or 1
    print(f"Using {n_workers} parallel workers\n")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks at once for maximum parallelism
        futures = [executor.submit(eval_candidate, it) for it in range(n_combinations)]

        for fut in tqdm(as_completed(futures), total=n_combinations, desc="Searching"):
            it, loss, system = fut.result()

            if loss < best_loss:
                best_loss = loss
                best_system = system
                print(f"\n[iter {it}] New best loss: {loss:.6e}")
                print(f"  S_rho = {system['S_rho'][:80]}...")
                print(f"  S_u   = {system['S_u'][:80]}...")
                print(f"  S_v   = {system['S_v'][:80]}...")
                print(f"  S_p   = {system['S_p'][:80]}...\n")

    # Report final best
    print(f"\n{'='*80}")
    print("BEST 4-FIELD SYSTEM FOUND")
    print(f"{'='*80}\n")
    print(f"Loss: {best_loss:.6e}\n")
    print(f"S_rho(x,y) = {best_system['S_rho']}\n")
    print(f"S_u(x,y)   = {best_system['S_u']}\n")
    print(f"S_v(x,y)   = {best_system['S_v']}\n")
    print(f"S_p(x,y)   = {best_system['S_p']}\n")
    print(f"Field definitions:")
    print(f"  rho(x,y) = exp(S_rho(x,y))")
    print(f"  u(x,y) = S_u(x,y)")
    print(f"  v(x,y) = S_v(x,y)")
    print(f"  p(x,y) = exp(S_p(x,y))\n")

    return {
        'loss': best_loss,
        'S_rho': best_system['S_rho'],
        'S_u': best_system['S_u'],
        'S_v': best_system['S_v'],
        'S_p': best_system['S_p'],
        'rho_expr': best_system['rho'],
        'u_expr': best_system['u'],
        'v_expr': best_system['v'],
        'p_expr': best_system['p'],
    }


if __name__ == "__main__":
    from sampler import FlexibleVectorSampler

    # Load sampler
    sampler = FlexibleVectorSampler(
        cluster_file="data/clusters.pkl",
        model=model,
        device=device,
    )

    # Run search
    result = search_four_independent_fields(
        sampler=sampler,
        model=model,
        n_series_samples=5000,
        n_candidates=200,
        n_combinations=1000,
        seed=42,
        sin_sin_only=False,
    )

    print(f"Best loss: {result['loss']:.6e}")
