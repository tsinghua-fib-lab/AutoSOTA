from itertools import product
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pyciphod.causal_estimation.outcome_regression import GComputation


VARIABLES = ["Z1", "Z2","X1", "X2", "Y1"]

ALL_CELLS = pd.MultiIndex.from_product(
    [[0, 1]] * len(VARIABLES),
    names=VARIABLES,
)

METHOD_LABELS = {
    "basic_mcar_recovery": "MCAR recovery",
    "mar_ipw_recovery": "MAR recovery",
    "mnar_recovery_mp_dag": "MNAR-DAG recovery",
    "mnar_recovery_sva_cmcdmg": "MNAR-cm-C-DMG recovery",
}


def expit(value):
    return 1.0 / (1.0 + np.exp(-value))


def bernoulli_probability(value, probability_one):
    return (
        probability_one
        if value == 1
        else 1.0 - probability_one
    )


def simulate_binary_graph1(n=200000, seed=1):
    rng = np.random.default_rng(seed)

    # Latent variables
    u_z = rng.binomial(1, 0.75, size=n)
    u_zr = rng.binomial(1, 0.5, size=n)

    # Substantive variables
    z2 = rng.binomial(1, expit(-1.0 + 2.0 * u_z),)
    x1 = rng.binomial(1, expit(-2.0 + 4.0 * z2),)
    z1 = rng.binomial(1, expit(-2.0 * u_zr + 3.0 * x1),)
    x2 = rng.binomial(1, expit(-1.2 + 2.4 * x1),)
    y1 = rng.binomial(1, expit(-0.9 + 1.8 * x1 + 0.3 * x2),)

    # Z -> R_X1
    r_x1 = rng.binomial(1,expit(1.4 - 1.2 * z1),)

    # R_X2
    r_x2 = rng.binomial(1, 0.78, size=n)

    # U_ZR -> R_Y1, producing Z <-> R_Y1
    r_y1 = rng.binomial(1,expit(2.0 - 8.0 * u_zr),)

    data = pd.DataFrame({
        "Z1": z1,
        "Z2": z2,

        # Full
        "X1_full": x1,
        "X2_full": x2,
        "Y1_full": y1,

        # R indicators
        "RX1": r_x1,
        "RX2": r_x2,
        "RY1": r_y1,
    })

    # Proxies
    data["X1_star"] = np.where(r_x1 == 1, x1, np.nan)
    data["X2_star"] = np.where(r_x2 == 1, x2, np.nan)
    data["Y1_star"] = np.where(r_y1 == 1, y1, np.nan)
    return data


def simulate_binary_graph2(n=200000, seed=1):
    rng = np.random.default_rng(seed)

    # Latent variables
    u_x = rng.binomial(1, 0.75, size=n)
    u_zr = rng.binomial(1, 0.5, size=n)

    # Substantive variables
    x1 = rng.binomial(1, expit(-2.0 + 4.0 * u_x),)
    z2 = rng.binomial(1, expit(-1.0 + 5.0 * u_zr),)
    x2 = rng.binomial(1, expit(-1.2 + 2.4 * x1 - 1.3 * z2),)
    z1 = rng.binomial(1, expit(-0.2 - 6.0 * u_zr + 2.0 * x1 + 0.8 * x2 - 0.2 * z2),)
    y1 = rng.binomial(1, expit(-0.9 + 1.8 * x1 + 0.3 * x2),)

    # Z -> R_X1
    r_x1 = rng.binomial(1,expit(1.4 - 1.2 * z1 + 0.2 * z2),)

    # R_X2
    r_x2 = rng.binomial(1,expit(0.74 - 2.2 * z1 - 1.2 * z2),)


    # U_ZR -> R_Y1, producing Z <-> R_Y1
    r_y1 = rng.binomial(1,expit(2.0 - 8.0 * u_zr),)

    data = pd.DataFrame({
        "Z1": z1,
        "Z2": z2,

        # Full
        "X1_full": x1,
        "X2_full": x2,
        "Y1_full": y1,

        # R indicators
        "RX1": r_x1,
        "RX2": r_x2,
        "RY1": r_y1,
    })

    # Proxies
    data["X1_star"] = np.where(r_x1 == 1, x1, np.nan)
    data["X2_star"] = np.where(r_x2 == 1, x2, np.nan)
    data["Y1_star"] = np.where(r_y1 == 1, y1, np.nan)
    return data


def exact_full_data_distribution1():
    probabilities = {}

    for values in product([0, 1], repeat=5):
        z1, z2, x1, x2, y1 = values

        cell_probability = 0.0

        for u_z, u_zr in product([0, 1], repeat=2):
            p_latent = (
                bernoulli_probability(u_z, 0.75)
                * bernoulli_probability(u_zr, 0.50)
            )

            p_z2 = expit(-1.0 + 2.0 * u_z)
            p_x1 = expit(-2.0 + 4.0 * z2)
            p_z1 = expit(-2.0 * u_zr + 3.0 * x1)
            p_x2 = expit(-1.2 + 2.4 * x1)
            p_y1 = expit(-0.9 + 1.8 * x1 + 0.3 * x2)

            cell_probability += (
                p_latent
                * bernoulli_probability(z1, p_z1)
                * bernoulli_probability(z2, p_z2)
                * bernoulli_probability(x1, p_x1)
                * bernoulli_probability(x2, p_x2)
                * bernoulli_probability(y1, p_y1)
            )

        probabilities[values] = cell_probability

    result = pd.Series(
        probabilities,
        name="truth",
    )

    result.index = pd.MultiIndex.from_tuples(
        result.index,
        names=VARIABLES,
    )

    result = result.reindex(
        ALL_CELLS,
        fill_value=0.0,
    )

    assert np.isclose(result.sum(), 1.0)
    assert result.notna().all()
    assert (result >= 0).all()

    return result


def exact_full_data_distribution2():
    probabilities = {}

    for values in product([0, 1], repeat=5):
        z1, z2, x1, x2, y1 = values

        cell_probability = 0.0

        for u_x, u_zr in product([0, 1], repeat=2):
            p_latent = (
                bernoulli_probability(u_x, 0.75)
                * bernoulli_probability(u_zr, 0.50)
            )

            # Substantive variables
            p_x1 = expit(-2.0 + 4.0 * u_x)
            p_z2 = expit(-1.0 + 5.0 * u_zr)
            p_x2 = expit(-1.2 + 2.4 * x1 - 1.3 * z2)
            p_z1 = expit(-0.2 - 6.0 * u_zr + 2.0 * x1 + 0.8 * x2 - 0.2 * z2)
            p_y1 = expit(-0.9 + 1.8 * x1 + 0.3 * x2)

            cell_probability += (
                p_latent
                * bernoulli_probability(z1, p_z1)
                * bernoulli_probability(z2, p_z2)
                * bernoulli_probability(x1, p_x1)
                * bernoulli_probability(x2, p_x2)
                * bernoulli_probability(y1, p_y1)
            )

        probabilities[values] = cell_probability

    result = pd.Series(
        probabilities,
        name="truth",
    )

    result.index = pd.MultiIndex.from_tuples(
        result.index,
        names=VARIABLES,
    )

    result = result.reindex(
        ALL_CELLS,
        fill_value=0.0,
    )

    assert np.isclose(result.sum(), 1.0)
    assert result.notna().all()
    assert (result >= 0).all()

    return result


def recover_joint_distribution_dag1(data):
    # Markov blanket of RX1 is {Z1}
    p_rx1_given_z1 = (
        data.groupby("Z1")["RX1"]
        .mean()
        .reindex([0, 1])
    )

    # Markov blanket of RY1 is {Z1, X1}.
    # X1 is observable in the RX1=1 stratum.
    p_ry1_given_z1_x1 = (
        data.loc[data["RX1"] == 1]
        .groupby(["Z1", "X1_star"])["RY1"]
        .mean()
    )

    # RX2 has an empty substantive Markov blanket
    p_rx2 = data["RX2"].mean()

    complete_mask = (
        (data["RX1"] == 1)
        & (data["RX2"] == 1)
        & (data["RY1"] == 1)
    )

    complete = data.loc[
        complete_mask,
        ["Z1", "Z2", "X1_star", "X2_star", "Y1_star"],
    ].copy()

    complete = complete.rename(
        columns={
            "X1_star": "X1",
            "X2_star": "X2",
            "Y1_star": "Y1",
        }
    )

    complete[VARIABLES] = complete[VARIABLES].astype(int)

    p1 = complete["Z1"].map(p_rx1_given_z1).to_numpy()

    z1_x1_index = pd.MultiIndex.from_frame(
        complete[["Z1", "X1"]]
    )

    p2 = (
        p_ry1_given_z1_x1
        .reindex(z1_x1_index)
        .to_numpy()
    )

    denominator = p1 * p_rx2 * p2

    if np.any(~np.isfinite(denominator)):
        raise ValueError(
            "Error."
        )

    if np.any(denominator <= 0):
        raise ValueError(
            "Denominator is zero."
        )

    complete["_weight"] = 1.0 / denominator

    recovered = (
        complete.groupby(VARIABLES)["_weight"]
        .sum()
        .div(len(data))
        .reindex(ALL_CELLS, fill_value=0.0)
        .rename("recovered")
    )

    return recovered


def recover_joint_distribution_dag2(data):
    z_cells = pd.MultiIndex.from_product(
        [[0, 1], [0, 1]],
        names=["Z1", "Z2"],
    )

    # P(RX1=1 | Z1, Z2)
    p_rx1_given_z = (
        data.groupby(["Z1", "Z2"])["RX1"]
        .mean()
        .reindex(z_cells)
    )

    # P(RX2=1 | Z1, Z2)
    p_rx2_given_z = (
        data.groupby(["Z1", "Z2"])["RX2"]
        .mean()
        .reindex(z_cells)
    )

    # X1 and X2 are both observable when RX1=RX2=1.
    ry1_data = data.loc[
        (data["RX1"] == 1) & (data["RX2"] == 1),
        ["Z1", "Z2", "X1_star", "X2_star", "RY1"],
    ].copy()

    ry1_data = ry1_data.rename(
        columns={
            "X1_star": "X1",
            "X2_star": "X2",
        }
    )

    ry1_data[["Z1", "Z2", "X1", "X2"]] = (
        ry1_data[["Z1", "Z2", "X1", "X2"]]
        .astype(int)
    )

    substantive_cells = pd.MultiIndex.from_product(
        [[0, 1]] * 4,
        names=["Z1", "Z2", "X1", "X2"],
    )

    # P(RY1=1 | Z1, Z2, X1, X2, RX1=RX2=1)
    p_ry1_given_blanket = (
        ry1_data
        .groupby(["Z1", "Z2", "X1", "X2"])["RY1"]
        .mean()
        .reindex(substantive_cells)
    )

    complete_mask = (
        (data["RX1"] == 1)
        & (data["RX2"] == 1)
        & (data["RY1"] == 1)
    )

    complete = data.loc[
        complete_mask,
        ["Z1", "Z2", "X1_star", "X2_star", "Y1_star"],
    ].copy()

    if complete.empty:
        raise ValueError("There are no complete cases.")

    complete = complete.rename(
        columns={
            "X1_star": "X1",
            "X2_star": "X2",
            "Y1_star": "Y1",
        }
    )

    complete[VARIABLES] = complete[VARIABLES].astype(int)

    z_index = pd.MultiIndex.from_frame(
        complete[["Z1", "Z2"]]
    )

    blanket_index = pd.MultiIndex.from_frame(
        complete[["Z1", "Z2", "X1", "X2"]]
    )

    p1 = p_rx1_given_z.reindex(z_index).to_numpy()
    p2 = p_rx2_given_z.reindex(z_index).to_numpy()

    p3 = (
        p_ry1_given_blanket
        .reindex(blanket_index)
        .to_numpy()
    )

    denominator = p1 * p2 * p3

    if np.any(~np.isfinite(denominator)):
        raise ValueError(
            "A required Markov-blanket probability "
            "could not be estimated."
        )

    if np.any(denominator <= 0):
        raise ValueError(
            "A recovery denominator is zero."
        )

    complete["_weight"] = 1.0 / denominator

    recovered = (
        complete.groupby(VARIABLES)["_weight"]
        .sum()
        .div(len(data))
        .reindex(ALL_CELLS, fill_value=0.0)
        .rename("recovered")
    )

    return recovered


def recover_joint_distribution_cmcdmg(data):
    observed_x = (
        (data["RX1"] == 1)
        & (data["RX2"] == 1)
    )

    complete = observed_x & (data["RY1"] == 1)

    # P(RX1=RX2=1 | Z1,Z2)
    x_observation_data = data[["Z1", "Z2"]].copy()
    x_observation_data["observed_x"] = observed_x.astype(int)

    p_x_observed_given_z = (
        x_observation_data
        .groupby(["Z1", "Z2"])["observed_x"]
        .mean()
    )

    # P(RY1=1 | Z1,Z2,X1,X2,RX1=RX2=1)
    y_observation_data = data.loc[
        observed_x,
        ["Z1", "Z2", "X1_star", "X2_star", "RY1"],
    ].copy()

    y_observation_data = y_observation_data.rename(
        columns={
            "X1_star": "X1",
            "X2_star": "X2",
        }
    )

    y_observation_data[
        ["Z1", "Z2", "X1", "X2"]
    ] = y_observation_data[
        ["Z1", "Z2", "X1", "X2"]
    ].astype(int)

    p_y_observed_given_zx = (
        y_observation_data
        .groupby(["Z1", "Z2", "X1", "X2"])["RY1"]
        .mean()
    )

    complete_data = data.loc[
        complete,
        ["Z1", "Z2", "X1_star", "X2_star", "Y1_star"],
    ].copy()

    complete_data = complete_data.rename(
        columns={
            "X1_star": "X1",
            "X2_star": "X2",
            "Y1_star": "Y1",
        }
    )

    complete_data[VARIABLES] = (
        complete_data[VARIABLES].astype(int)
    )

    z_index = pd.MultiIndex.from_frame(
        complete_data[["Z1", "Z2"]]
    )

    zx_index = pd.MultiIndex.from_frame(
        complete_data[["Z1", "Z2", "X1", "X2"]]
    )

    pi_x = (
        p_x_observed_given_z
        .reindex(z_index)
        .to_numpy()
    )

    pi_y = (
        p_y_observed_given_zx
        .reindex(zx_index)
        .to_numpy()
    )

    denominator = pi_x * pi_y

    if np.any(~np.isfinite(denominator)):
        raise ValueError(
            "A required cluster-level probability "
            "could not be estimated."
        )

    if np.any(denominator <= 0):
        raise ValueError(
            "Cluster-level positivity failure."
        )

    complete_data["_weight"] = 1.0 / denominator

    recovered = (
        complete_data
        .groupby(VARIABLES)["_weight"]
        .sum()
        .div(len(data))
        .reindex(ALL_CELLS, fill_value=0.0)
        .rename("cluster_recovery")
    )

    return recovered


def recover_joint_distribution_mcar(data):
    complete_mask = (
        (data["RX1"] == 1)
        & (data["RX2"] == 1)
        & (data["RY1"] == 1)
    )

    complete = data.loc[
        complete_mask,
        ["Z1", "Z2", "X1_star", "X2_star", "Y1_star"],
    ].copy()

    if complete.empty:
        raise ValueError("There are no complete cases.")

    complete = complete.rename(
        columns={
            "X1_star": "X1",
            "X2_star": "X2",
            "Y1_star": "Y1",
        }
    )

    complete[VARIABLES] = complete[VARIABLES].astype(int)

    # Naive complete-case distribution
    recovered = (
        complete.value_counts(sort=False)
        .div(len(complete))
        .reindex(ALL_CELLS, fill_value=0.0)
        .rename("basic_mcar_recovery")
    )

    return recovered


def recover_joint_distribution_mar_ipw(data):
    complete_mask = (
        (data["RX1"] == 1)
        & (data["RX2"] == 1)
        & (data["RY1"] == 1)
    )

    estimation_data = data[["Z1", "Z2"]].copy()
    estimation_data["complete"] = complete_mask.astype(int)

    z_cells = pd.MultiIndex.from_product(
        [[0, 1], [0, 1]],
        names=["Z1", "Z2"],
    )

    # Wrong MAR assumption:
    # P(complete=1 | full data) = P(complete=1 | Z1, Z2)
    # WIHTOUT SMOOTHING
    # p_complete_given_z = (
    #     estimation_data
    #     .groupby(["Z1", "Z2"])["complete"]
    #     .mean()
    #     .reindex(z_cells)
    # )

    # WIHT SMOOTHING: avoid A MAR complete-case probability is zero
    counts = (
        estimation_data
        .groupby(["Z1", "Z2"])["complete"]
        .agg(["sum", "size"])
        .reindex(z_cells)
    )
    p_complete_given_z = (counts["sum"] + 0.5) / (counts["size"] + 2.0 * 0.5)

    if p_complete_given_z.isna().any():
        raise ValueError(
            "Some Z1,Z2 strata are absent from the sample."
        )

    if (p_complete_given_z <= 0).any():
        raise ValueError(
            "A MAR complete-case probability is zero."
        )

    complete = data.loc[
        complete_mask,
        ["Z1", "Z2", "X1_star", "X2_star", "Y1_star"],
    ].copy()

    complete = complete.rename(
        columns={
            "X1_star": "X1",
            "X2_star": "X2",
            "Y1_star": "Y1",
        }
    )

    complete[VARIABLES] = complete[VARIABLES].astype(int)

    z_index = pd.MultiIndex.from_frame(
        complete[["Z1", "Z2"]]
    )

    probabilities = (
        p_complete_given_z
        .reindex(z_index)
        .to_numpy()
    )

    complete["_weight"] = 1.0 / probabilities

    recovered = (
        complete.groupby(VARIABLES)["_weight"]
        .sum()
        .div(len(data))
        .reindex(ALL_CELLS, fill_value=0.0)
        .rename("mar_recovery")
    )
    return recovered


def total_variation(p, q):
    return 0.5 * np.abs(p - q).sum()


def compute_ground_truth(setting=1):
    def p_y_do(x1, x2):
        if setting == 1:
            return expit(-0.9 + 1.8 * x1 + 0.3 * x2)
        else:
            return expit(-0.9 + 1.8 * x1 + 0.3 * x2)

    joint_effect = p_y_do(1, 1) - p_y_do(0, 0)
    return float(joint_effect)

def effect_from_joint_distribution(joint):
    """
    joint is a Series indexed by:
    Z1, Z2, X1, X2, Y1
    containing probabilities.
    """

    joint = joint / joint.sum()

    def p_y1_given_x(x1, x2):
        subset = joint.xs((x1, x2), level=("X1", "X2"))

        numerator = subset.xs(1, level="Y1").sum()
        denominator = subset.sum()

        if denominator <= 0:
            raise ValueError(f"No mass for X1={x1}, X2={x2}")

        return numerator / denominator

    return p_y1_given_x(1, 1) - p_y1_given_x(0, 0)

def compare_recovery_methods(data, truth, setting=1):
    if setting == 1:
        recover_joint_distribution_mnar_dag = recover_joint_distribution_dag1(data)
    elif setting == 2:
        recover_joint_distribution_mnar_dag = recover_joint_distribution_dag2(data)
    else:
        raise ValueError("setting must be 1 or 2")

    recover_joint_distribution_mnar_cmcdmg = recover_joint_distribution_cmcdmg(data)

    estimates = {
        "basic_mcar_recovery":
            recover_joint_distribution_mcar(data),

        "mar_ipw_recovery":
            recover_joint_distribution_mar_ipw(data),

        "mnar_recovery_mp_dag":
            recover_joint_distribution_mnar_dag,

        "mnar_recovery_sva_cmcdmg":
            recover_joint_distribution_mnar_cmcdmg,
    }

    rows = []

    ground_truth_effect = compute_ground_truth(setting=setting)

    for method, estimate in estimates.items():
        mass = estimate.sum()
        normalized_estimate = estimate / mass

        causal_effect_estimate = effect_from_joint_distribution(normalized_estimate)

        rows.append({
            "method": method,
            "recovered_mass": mass,
            "mass_error": abs(mass - 1.0),
            "normalized_tv": 0.5 * np.abs(normalized_estimate - truth).sum(),
            "raw_l1_error": np.abs(estimate - truth).sum(),
            "maximum_normalized_cell_error": np.abs(normalized_estimate - truth).max(),
            "causal_effect_estimate": causal_effect_estimate,
            "causal_effect_error": abs(causal_effect_estimate - ground_truth_effect),
        })


    # for method, estimate in estimates.items():
    #     mass = estimate.sum()
    #
    #     if not np.isfinite(mass) or mass <= 0:
    #         raise ValueError(
    #             f"Invalid recovered mass for {method}."
    #         )
    #
    #     normalized_estimate = estimate / mass
    #
    #     rows.append({
    #         "method": method,
    #
    #         # Original, unnormalized recovered mass
    #         "recovered_mass": mass,
    #
    #         "mass_error": abs(mass - 1.0),
    #
    #         # Fair comparison of distributional shape
    #         "normalized_tv": (
    #             0.5
    #             * np.abs(normalized_estimate - truth).sum()
    #         ),
    #
    #         # Includes both shape and mass discrepancies
    #         "raw_l1_error": (
    #             np.abs(estimate - truth).sum()
    #         ),
    #
    #         "maximum_normalized_cell_error": (
    #             np.abs(normalized_estimate - truth).max()
    #         ),
    #     })

    return pd.DataFrame(rows)


def convergence_experiment(setting=1, sample_sizes=(1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000),repetitions=30,):
    if setting == 1:
        population_truth = exact_full_data_distribution1()
    else:
        population_truth = exact_full_data_distribution2()
    rows = []

    for n in sample_sizes:
        for repetition in range(repetitions):
            if setting == 1:
                data = simulate_binary_graph1(n=n, seed=100000 + 100 * repetition + n,)
            else:
                data = simulate_binary_graph2(n=n, seed=100000 + 100 * repetition + n,)

            r_vars = [ "RX1", "RX2", "RY1"]
            print(data)
            for r_var in r_vars:
                print(f"Proportion of {r_var}=0: {1-data[r_var].mean():.4f}")

            method_results = compare_recovery_methods(
                data,
                population_truth,
                setting=setting
            )

            method_results["n"] = n
            method_results["repetition"] = repetition

            rows.append(method_results)

    results = pd.concat(
        rows,
        ignore_index=True,
    )

    summary = (
        results
        .groupby(["n", "method"])
        .agg(
            mean_recovered_mass=(
                "recovered_mass",
                "mean",
            ),
            sd_recovered_mass=(
                "recovered_mass",
                "std",
            ),
            mean_mass_error=(
                "mass_error",
                "mean",
            ),
            mean_normalized_tv=(
                "normalized_tv",
                "mean",
            ),
            sd_normalized_tv=(
                "normalized_tv",
                "std",
            ),
            mean_raw_l1_error=(
                "raw_l1_error",
                "mean",
            ),
            mean_maximum_cell_error=(
                "maximum_normalized_cell_error",
                "mean",
            ),

            # New: causal-effect estimation
            mean_causal_effect_estimate=(
                "causal_effect_estimate",
                "mean",
            ),
            sd_causal_effect_estimate=(
                "causal_effect_estimate",
                "std",
            ),
            mean_causal_effect_error=(
                "causal_effect_error",
                "mean",
            ),
            sd_causal_effect_error=(
                "causal_effect_error",
                "std",
            ),
        )
        .reset_index()
    )

    # summary = (
    #     results
    #     .groupby(["n", "method"])
    #     .agg(
    #         mean_recovered_mass=(
    #             "recovered_mass",
    #             "mean",
    #         ),
    #         sd_recovered_mass=(
    #             "recovered_mass",
    #             "std",
    #         ),
    #         mean_mass_error=(
    #             "mass_error",
    #             "mean",
    #         ),
    #         mean_normalized_tv=(
    #             "normalized_tv",
    #             "mean",
    #         ),
    #         sd_normalized_tv=(
    #             "normalized_tv",
    #             "std",
    #         ),
    #         mean_raw_l1_error=(
    #             "raw_l1_error",
    #             "mean",
    #         ),
    #         mean_maximum_cell_error=(
    #             "maximum_normalized_cell_error",
    #             "mean",
    #         ),
    #     )
    #     .reset_index()
    # )

    return results, summary

def plot_tv_convergence(summary, setting_label=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, method_data in summary.groupby("method"):
        method_data = method_data.sort_values("n")

        ax.errorbar(
            method_data["n"],
            method_data["mean_normalized_tv"],
            yerr=method_data["sd_normalized_tv"],
            marker="o",
            capsize=3,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Total variation distance")
    # ax.set_title(
    #     "Recovery of the joint distribution"
    #     # if setting_label is None
    #     # else f"Recovery of the full joint distribution — {setting_label}"
    # )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_causal_effect_error_convergence(summary, setting_label=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, method_data in summary.groupby("method"):
        method_data = method_data.sort_values("n")

        ax.errorbar(
            method_data["n"],
            method_data["mean_causal_effect_error"],
            yerr=method_data["sd_causal_effect_error"],
            marker="o",
            capsize=3,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Erreur absolue sur l'effet causal")
    ax.set_title(
        "Estimation de l'effet causal"
        if setting_label is None
        else f"Estimation de l'effet causal — {setting_label}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_causal_effect_estimates(summary, setting_label=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    ground_truth_effect = compute_ground_truth(setting=setting)

    for method, method_data in summary.groupby("method"):
        method_data = method_data.sort_values("n")

        ax.errorbar(
            method_data["n"],
            method_data["mean_causal_effect_estimate"],
            yerr=method_data["sd_causal_effect_estimate"],
            marker="o",
            capsize=3,
            label=METHOD_LABELS.get(method, method),
        )

    ax.axhline(
        ground_truth_effect,
        linestyle="--",
        label="Ground truth"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Estimated causal effect")
    # ax.set_title(
    #     "Estimated causal effect"
    #     if setting_label is None
    #     else f"Estimated causal effect — {setting_label}"
    # )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    setting = 2  # 1 is a DAG with MCAR, MAR, and MNAR missingness; 2 is a DAG with MAR and MNAR missingness

    if setting == 1:
        results, summary = convergence_experiment(setting=1)
    else:
        results, summary = convergence_experiment(setting=2)
    print(summary)

    plot_tv_convergence(summary, setting_label=f"Setting {setting}")
    plot_causal_effect_estimates(summary, setting_label=f"Setting {setting}")

