from pathlib import Path
import sys

import numpy as np
import pandas as pd
import networkx as nx

PYCIPHOD_SRC = Path.home() / "code" / "pyCIPHOD" / "src"
if str(PYCIPHOD_SRC) not in sys.path:
    sys.path.insert(0, str(PYCIPHOD_SRC))

from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag
from pyciphod.utils.time_series.data_format import DTimeVar


# ============================================================
# Settings
# ============================================================

SETTINGS = {
    # General temporal FT-DAG with lags up to 2.
    # Changed edge is always instantaneous.
    "setting1_lag2": {
        "max_delay_for_graph": 2,
        "simulation_lag_max": 2,
        "keep_lagged": "all",
        "add_self_lag1": True,
        "iid": False,
        "description": "Temporal SEM with lagged, instantaneous, and mandatory self-lags, max lag 2.",
    },
    # General temporal FT-DAG with lags up to 1.
    # Changed edge is always instantaneous.
    "setting2_lag1": {
        "max_delay_for_graph": 1,
        "simulation_lag_max": 1,
        "keep_lagged": "all",
        "add_self_lag1": True,
        "iid": False,
        "description": "Temporal SEM with lagged, instantaneous, and mandatory self-lags, max lag 1.",
    },
    # Contemporaneous structural graph plus mandatory self-lags.
    "setting3_contemporaneous_with_self_lag": {
        "max_delay_for_graph": 1,
        "simulation_lag_max": 1,
        "keep_lagged": "none",
        "add_self_lag1": True,
        "iid": False,
        "description": "Contemporaneous SEM plus mandatory self-lags Xi(t-1)->Xi(t).",
    },
    # Pure iid contemporaneous SEM.
    "setting4_iid": {
        "max_delay_for_graph": 0,
        "simulation_lag_max": 0,
        "keep_lagged": "none",
        "add_self_lag1": False,
        "iid": True,
        "description": "IID contemporaneous SEM, no lagged edges and no self-lags.",
    },
}

SETTING_SEED_INDEX = {
    "setting1_lag2": 0,
    "setting2_lag1": 1,
    "setting3_contemporaneous_with_self_lag": 2,
    "setting4_iid": 3,
}

CHANGE_MODELS = {
    "single_edge",      # Change one instantaneous edge coefficient.
    "all_parents",      # Choose a target and change all its incoming coefficients.
    "all_parents_min2", # Same, but the target must have at least two incoming parents.
}

# ============================================================
# Generate benchmark runs in memory
# ============================================================

def generate_one_run(
    setting_name,
    p,
    n,
    rep,
    edge_prob,
    structure_seed,
    coef_seed,
    dataset_seed,
    change_seed,
    burn_in=200,
    min_abs_change=0.5,
    change_model="single_edge",
    min_incoming_parents=2,
):
    """
    :param setting_name: name of the benchmark setting to generate.
    :param p: number of variables in the generated graph.
    :param n: number of observations per regime.
    :param rep: replication index.
    :param edge_prob: edge probability used to sample the graph structure.
    :param structure_seed: random seed used for graph generation.
    :param coef_seed: random seed used for sampling structural coefficients.
    :param dataset_seed: random seed used for simulating the two regimes.
    :param change_seed: random seed used for sampling coefficient changes.
    :param burn_in: number of burn-in observations used in time-series simulations.
    :param min_abs_change: minimum absolute coefficient change imposed on shifted edges.
    :param change_model: type of mechanism change to generate.
    :param min_incoming_parents: minimum number of incoming parents required for the shifted target.
    :return: dictionary containing the two simulated regimes, the ground truth and metadata.
     This function generates one benchmark replication. It first samples a temporal causal graph, then simulates two regimes that differ by a coefficient change on one or several selected edges. 
     The output contains the simulated data, the changed edges, the shifted nodes, all graph edges, and metadata required by the evaluation scripts.
    """
    if change_model not in CHANGE_MODELS:
        raise ValueError(
            f"Unknown change_model={change_model!r}. Expected one of {sorted(CHANGE_MODELS)}."
        )

    setting = SETTINGS[setting_name]
    run_id = f"{setting_name}_p{p}_n{n}_rep{rep:03d}"

    ftdag, structure_seed_used = generate_valid_graph(
        p=p,
        edge_prob=edge_prob,
        setting_name=setting_name,
        structure_seed=structure_seed,
        change_model=change_model,
        min_incoming_parents=min_incoming_parents,
    )

    coefs_before = sample_edge_coefficients(ftdag, seed=coef_seed)
    coefs_after = dict(coefs_before)

    rng_change = np.random.default_rng(change_seed)
    changed_edges = choose_changed_edges(
        ftdag=ftdag,
        change_model=change_model,
        seed=structure_seed_used + 999,
        min_incoming_parents=min_incoming_parents,
    )

    coefficient_changes = []
    for changed_edge in changed_edges:
        old = float(coefs_before[changed_edge])
        new = sample_changed_coefficient(
            old=old,
            rng=rng_change,
            min_abs_change=min_abs_change,
        )
        coefs_after[changed_edge] = new
        coefficient_changes.append({
            **describe_edge(changed_edge),
            "old_coefficient": old,
            "new_coefficient": float(new),
        })

    if setting["iid"]:
        X1 = simulate_iid_contemporaneous_scm(
            ftdag,
            coefs_before,
            n=n,
            seed=dataset_seed,
        )
        X2 = simulate_iid_contemporaneous_scm(
            ftdag,
            coefs_after,
            n=n,
            seed=dataset_seed + 1,
        )
    else:
        X1 = simulate_linear_time_series_scm(
            ftdag=ftdag,
            coefs=coefs_before,
            lag_max=setting["simulation_lag_max"],
            n=n,
            burn_in=burn_in,
            seed=dataset_seed,
        )
        X2 = simulate_linear_time_series_scm(
            ftdag=ftdag,
            coefs=coefs_after,
            lag_max=setting["simulation_lag_max"],
            n=n,
            burn_in=burn_in,
            seed=dataset_seed + 1,
        )

    changed_edges_lagged = [edge_to_lagged_reference(edge) for edge in changed_edges]
    shifted_nodes = sorted({edge[1].name for edge in changed_edges_lagged})
    all_edges_ftdag = sorted(ftdag.get_directed_edges(), key=edge_mechanism_key)
    all_edges_lagged = [edge_to_lagged_reference(edge) for edge in all_edges_ftdag]

    metadata = {
        "simulation_lag_max": int(setting["simulation_lag_max"]),
        "iid": bool(setting["iid"]),
        "n_changed_edges": int(len(changed_edges)),
        "structure_seed_used": int(structure_seed_used),
        "dataset_seed": int(dataset_seed),
        "changed_edges_lagged": [describe_edge(edge) for edge in changed_edges_lagged],
    }

    truth = {
        "ftdag": ftdag,
        "changed_edges_ftdag": changed_edges,
        "changed_edges_lagged": changed_edges_lagged,
        "shifted_nodes": shifted_nodes,
    }

    return {
        "X1": X1,
        "X2": X2,
        "truth": truth,
        "metadata": metadata,
    }

def generate_many_runs(
    settings=None,
    p_list=None,
    n_list=None,
    n_reps=20,
    edge_prob=0.3,
    base_seed=12000,
    burn_in=200,
    min_abs_change=0.5,
    change_model="single_edge",
    min_incoming_parents=2,
):
    """
    :return: iterator over benchmark runs.
     This generator produces benchmark runs one by one.
    """
    
    if settings is None:
        settings = list(SETTINGS.keys())
    if p_list is None:
        p_list = [9]
    if n_list is None:
        n_list = [1000]

    for setting_name in settings:
        setting_idx = SETTING_SEED_INDEX[setting_name]
        for p in p_list:
            for n in n_list:
                for rep in range(1, n_reps + 1):
                    seed_offset = (
                        setting_idx * 10_000_000
                        + p * 100_000
                        + n
                        + rep
                    )
                    base = base_seed + seed_offset

                    yield generate_one_run(
                        setting_name=setting_name,
                        p=p,
                        n=n,
                        rep=rep,
                        edge_prob=edge_prob,
                        structure_seed=base,
                        coef_seed=base + 10_000,
                        dataset_seed=base + 100_000,
                        change_seed=base + 200_000,
                        burn_in=burn_in,
                        min_abs_change=min_abs_change,
                        change_model=change_model,
                        min_incoming_parents=min_incoming_parents,
                    )


# ============================================================
# Graph construction
# ============================================================

def generate_valid_graph(p,edge_prob,setting_name,structure_seed,change_model="single_edge",min_incoming_parents=2,max_graph_tries=500,):
    """
    :param max_graph_tries: maximum number of graph generation attempts.
    :return: a valid temporal causal graph and the random seed used to generate it.
    This function repeatedly samples temporal causal graphs until one satisfies the constraints imposed by the selected benchmark setting and change model.
    """
    setting = SETTINGS[setting_name]

    for attempt in range(max_graph_tries):
        candidate_seed = structure_seed + attempt
        candidate = create_random_ft_dag(
            num_ts=p,
            p_edge=edge_prob,
            causally_stationary=True,
            max_delay=setting["max_delay_for_graph"],
            seed=candidate_seed,
        )
        candidate = prepare_graph_for_setting(candidate, setting_name)

        if graph_is_valid_for_change_model(
            candidate,
            change_model=change_model,
            min_incoming_parents=min_incoming_parents,
        ):
            return candidate, candidate_seed

    raise RuntimeError(
        f"Could not generate a valid graph for {setting_name} after "
        f"{max_graph_tries} tries."
    )


def prepare_graph_for_setting(ftdag, setting_name):
    """
    :return: temporal causal graph satisfying the structural constraints of the selected setting.
     This function prepares a sampled FT-DAG for a specific benchmark setting. 
    """
    setting = SETTINGS[setting_name]

    ftdag = keep_one_edge_per_stationary_mechanism(ftdag)

    if setting["keep_lagged"] == "none":
        for u, v in list(ftdag.get_directed_edges()):
            if edge_lag((u, v)) > 0:
                ftdag.remove_directed_edge(u, v)

    if setting["add_self_lag1"]:
        variable_names = sorted({v.name for v in ftdag.get_vertices()})
        times = sorted({int(v.time) for v in ftdag.get_vertices()})
        nodes = {(v.name, int(v.time)): v for v in ftdag.get_vertices()}

        for name in variable_names:
            for t in times:
                if t - 1 not in times:
                    continue
                src = nodes[(name, t - 1)]
                tgt = nodes[(name, t)]
                if (src, tgt) not in ftdag.get_directed_edges():
                    ftdag.add_directed_edge(src, tgt)

    return keep_one_edge_per_stationary_mechanism(ftdag)


def keep_one_edge_per_stationary_mechanism(ftdag):
    """
    :return: temporal causal graph with one representative edge per stationary mechanism.
     This function removes duplicate copies of stationary causal mechanisms from the graph.
    """
    seen_mechanisms = set()
    edges_to_remove = []

    edges = sorted(
        list(ftdag.get_directed_edges()),
        key=lambda e: (edge_mechanism_key(e), int(e[1].time), int(e[0].time)),
    )

    for edge in edges:
        mechanism = edge_mechanism_key(edge)
        if mechanism in seen_mechanisms:
            edges_to_remove.append(edge)
        else:
            seen_mechanisms.add(mechanism)

    for u, v in edges_to_remove:
        ftdag.remove_directed_edge(u, v)

    return ftdag


def graph_is_valid_for_change_model(ftdag, change_model="single_edge", min_incoming_parents=2):
    """
    :return: True if the graph satisfies the constraints of the selected change model, False otherwise.
    """
    try:
        _ = get_instantaneous_order(ftdag)
    except Exception:
        return False

    if change_model in {"single_edge", "all_parents"}:
        return len(instantaneous_edges(ftdag)) > 0

    if change_model == "all_parents_min2":
        return len(find_shifted_target_candidates(ftdag, min_incoming_parents)) > 0 # verify that at least one o the instantaneous edges has 2 parents

    raise ValueError(
        f"Unknown change_model={change_model!r}. Expected one of {sorted(CHANGE_MODELS)}."
    )


# ============================================================
# Coefficients and changed mechanism
# ============================================================

def sample_edge_coefficients(
    ftdag,
    seed=0,
    low=0.2,
    high=0.8,
    max_abs_sum_per_target=0.95,
):
    """
    :param low: minimum absolute value of sampled coefficients.
    :param high: maximum absolute value of sampled coefficients.
    :param max_abs_sum_per_target: maximum allowed sum of absolute incoming coefficients for each target node.
    :return: dictionary mapping each graph edge to its sampled structural coefficient.
    """
    rng = np.random.default_rng(seed)

    incoming_edges_by_target = {}
    for edge in sorted(ftdag.get_directed_edges(), key=edge_mechanism_key):
        _, v = edge
        target_key = (v.name, int(v.time))
        incoming_edges_by_target.setdefault(target_key, []).append(edge)

    coefs = {}
    for target_key in sorted(incoming_edges_by_target):
        edges = incoming_edges_by_target[target_key]
        values = [rng.choice([-1.0, 1.0]) * rng.uniform(low, high) for _ in edges]

        abs_sum = float(np.sum(np.abs(values)))
        if abs_sum > max_abs_sum_per_target:
            scale = max_abs_sum_per_target / abs_sum
            values = [scale * x for x in values]

        for edge, value in zip(edges, values):
            coefs[edge] = float(value)

    return coefs


def choose_changed_edges(ftdag, change_model="single_edge", seed=0, min_incoming_parents=2):
    """
    :return: list of graph edges whose coefficients are modified.
    """
    rng = np.random.default_rng(seed)

    if change_model == "single_edge":
        candidates = sorted(instantaneous_edges(ftdag), key=edge_mechanism_key)
        if len(candidates) == 0:
            raise RuntimeError("No instantaneous candidate edge available.")
        return [candidates[int(rng.integers(len(candidates)))]]

    if change_model == "all_parents":
        candidates = sorted(instantaneous_edges(ftdag), key=edge_mechanism_key)
        if len(candidates) == 0:
            raise RuntimeError("No instantaneous candidate edge available.")
        _, target = candidates[int(rng.integers(len(candidates)))]
        return sorted([(u, v) for u, v in ftdag.get_directed_edges() if v == target],key=edge_mechanism_key,)

    if change_model == "all_parents_min2":
        candidates = find_shifted_target_candidates(ftdag, min_incoming_parents)
        if len(candidates) == 0:
            raise RuntimeError(
                "No shifted target available with at least "
                f"{min_incoming_parents} incoming parents."
            )
        target = candidates[int(rng.integers(len(candidates)))]
        return sorted([(u, v) for u, v in ftdag.get_directed_edges() if v == target],key=edge_mechanism_key,)

    raise ValueError(
        f"Unknown change_model={change_model!r}. Expected one of {sorted(CHANGE_MODELS)}."
    )


def find_shifted_target_candidates(ftdag, min_incoming_parents=2):
    """
    :return: sorted list of candidate shifted target nodes.
    This function identifies all target nodes that are incident to at least one instantaneous edge and have at least the specified number of incoming parents.
    """
    instantaneous_targets = {v for _, v in instantaneous_edges(ftdag)}
    candidates = []

    for target in instantaneous_targets:
        incoming = sorted([(u, v) for u, v in ftdag.get_directed_edges() if v == target],key=edge_mechanism_key,)
        if len(incoming) >= min_incoming_parents:
            candidates.append(target)

    return sorted(candidates, key=lambda node: (node.name, int(node.time)))


def sample_changed_coefficient(old, rng, low=0.2, high=1.0, min_abs_change=0.5):
    """
    :param old: original coefficient value.
    :param rng: NumPy random number generator.
    :param low: minimum absolute value of the sampled coefficient.
    :param high: maximum absolute value of the sampled coefficient.
    :param min_abs_change: minimum required absolute difference between the old and new coefficients.
    :return: sampled coefficient satisfying the minimum change constraint.
    """
    for _ in range(1000):
        new = float(rng.choice([-1.0, 1.0]) * rng.uniform(low, high))
        if abs(new - old) >= min_abs_change:
            return new
    raise RuntimeError("Could not sample a sufficiently different coefficient.")


# ============================================================
# Simulation
# ============================================================

def simulate_linear_time_series_scm(
    ftdag,
    coefs,
    lag_max,
    n=1000,
    burn_in=200,
    noise_scale=1.0,
    seed=0,
):
    """
    :return: dataframe containing the simulated multivariate time series.
    This function simulates a linear time-series structural causal model (SCM). At each time step, lagged effects are evaluated using values from previous time points, while contemporaneous effects are evaluated according to the instantaneous topological order of the causal graph.
    """
    rng = np.random.default_rng(seed)
    variable_names = sorted({v.name for v in ftdag.get_vertices()})
    instant_order = get_instantaneous_order(ftdag)
    edge_templates = build_edge_templates(ftdag, coefs)

    total_n = n + burn_in + lag_max + 1
    data = {name: np.zeros(total_n) for name in variable_names}

    for t in range(lag_max + 1, total_n):
        current = {name: rng.normal(scale=noise_scale) for name in variable_names}

        for src, tgt, lag, beta in edge_templates:
            if lag > 0:
                current[tgt] += beta * data[src][t - lag]

        for tgt in instant_order:
            for src, tgt2, lag, beta in edge_templates:
                if lag == 0 and tgt2 == tgt:
                    current[tgt] += beta * current[src]

        for name in variable_names:
            data[name][t] = current[name]

    start = burn_in + lag_max + 1
    return pd.DataFrame({name: values[start:start + n] for name, values in data.items()})


def simulate_iid_contemporaneous_scm(ftdag, coefs, n=1000, noise_scale=1.0, seed=0):
    """
    :return: dataframe containing the simulated observations.
    This function simulates independent and identically distributed (i.i.d.) observations from a linear structural equation model (SEM). Each observation is generated by first sampling independent Gaussian noise terms and then evaluating the contemporaneous structural equations according to the instantaneous topological order of the causal graph.
    """
    rng = np.random.default_rng(seed)
    variable_names = sorted({v.name for v in ftdag.get_vertices()})
    instant_order = get_instantaneous_order(ftdag)
    edge_templates = [
        (src, tgt, beta)
        for src, tgt, lag, beta in build_edge_templates(ftdag, coefs)
        if lag == 0
    ]

    rows = []
    for _ in range(n):
        current = {name: rng.normal(scale=noise_scale) for name in variable_names}

        for tgt in instant_order:
            for src, tgt2, beta in edge_templates:
                if tgt2 == tgt:
                    current[tgt] += beta * current[src]

        rows.append([current[name] for name in variable_names])

    return pd.DataFrame(rows, columns=variable_names)


def build_edge_templates(ftdag, coefs):
    """
    :return: list of edge templates of the form (source, target, lag, coefficient).
    """
    templates = []
    for edge in sorted(ftdag.get_directed_edges(), key=edge_mechanism_key):
        u, v = edge
        templates.append((u.name, v.name, edge_lag(edge), float(coefs[edge])))
    return templates


# ============================================================
# Edge conventions and helpers
# ============================================================

def edge_lag(edge):
    u, v = edge
    return int(v.time - u.time)

def edge_mechanism_key(edge):
    u, v = edge
    return (u.name, v.name, edge_lag(edge))

def instantaneous_edges(ftdag):
    return sorted([(u, v) for u, v in ftdag.get_directed_edges() if edge_lag((u, v)) == 0], key=edge_mechanism_key,)

def get_instantaneous_order(ftdag):
    variable_names = sorted({v.name for v in ftdag.get_vertices()})
    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)
    for u, v in instantaneous_edges(ftdag):
        graph.add_edge(u.name, v.name)
    try:
        return list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible as exc:
        raise RuntimeError("Instantaneous graph has a cycle.") from exc

def edge_to_lagged_reference(edge):
    u, v = edge
    lag = edge_lag(edge)
    return DTimeVar(u.name, -lag), DTimeVar(v.name, 0)

def describe_edge(edge):
    u, v = edge
    return {"source": u.name,"source_time": int(u.time), "target": v.name, "target_time": int(v.time), "lag": int(edge_lag(edge)),}