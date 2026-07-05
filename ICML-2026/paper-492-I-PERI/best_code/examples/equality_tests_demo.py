"""
Démonstration des tests d'égalité de dépendance (PartialCorrelationEqualityTest,
CMIhEqualityTest, GsqEqualityTest, LinearRegressionCoefficientEqualityTest) via permutation.

Les générateurs créent désormais un DAG Z -> X, Z -> Y et W -> X, W -> Y
(aucune arête directe X -> Y). Ainsi X et Y sont dépendants marginalement mais
indépendants conditionnellement à {Z,W}.

Usage (depuis la racine du dépôt):
    python3 examples/equality_tests_demo.py

Remarques:
- Le script ajoute `src/` au sys.path pour importer le package local `pyciphod`.
- Les appels utilisent un petit nombre de permutations (100) pour une démonstration rapide.
  Pour des résultats stables, augmentez `n_permutations` (p.ex. 1000+).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

# Ajouter `src/` au path pour importer le package local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pyciphod.utils.stat_tests.equality_tests import (
    PartialCorrelationEqualityTest,
    CMIhEqualityTest,
    GsqEqualityTest,
    LinearRegressionCoefficientEqualityTest,
    KernelPartialCorrelationEqualityTest,
    SLearnerEqualityTest,
    GComputationEqualityTest,
)


def make_continuous_data(n=2000, seed=0):
    """Génère deux populations continues où Z and W causent X and Y.

    DataFrame columns: x, y, z_cont, w_cont
    """
    rng = np.random.default_rng(seed)

    # Parents for population 1 (strong positive marginal covariance)
    z1 = rng.normal(size=n)
    w1 = rng.normal(size=n)
    # X and Y are functions of Z and W plus independent noise (no direct X->Y)
    x1 = 1.5 * z1 + 1.0 * w1 + rng.normal(scale=0.4, size=n)
    y1 = 1.2 * z1 + 0.9 * w1 + rng.normal(scale=0.4, size=n)
    df1 = pd.DataFrame({"x": x1, "y": y1, "z_cont": z1, "w_cont": w1})

    # Parents for population 2 (change coefficients and noise slightly)
    # For population 2 flip signs on Y coefficients to create strong
    # negative marginal covariance (large contrast with population 1)
    z2 = rng.normal(size=n)
    w2 = rng.normal(size=n)
    x2 = 1.5 * z2 + 1.0 * w2 + rng.normal(scale=0.4, size=n)
    y2 = -1.2 * z2 - 0.9 * w2 + rng.normal(scale=0.4, size=n)
    df2 = pd.DataFrame({"x": x2, "y": y2, "z_cont": z2, "w_cont": w2})

    return df1, df2


def make_discrete_data(n_per_group=500, groups=3, seed=1):
    """Génère deux populations discrètes où Z_disc et W_disc causent X and Y.

    DataFrame columns: x (binary), y (binary), z_disc, w_disc
    """
    rng = np.random.default_rng(seed)

    def make_population(coeff_z, coeff_w, flip_noise=0.05):
        rows = []
        for gz in range(groups):
            for gw in range(groups):
                # for each cell (z,gz) x and y samples
                z_vals = np.full(n_per_group, gz, dtype=int)
                w_vals = np.full(n_per_group, gw, dtype=int)
                # probability depends on z and w via logits
                logit_x = coeff_z[0] * gz + coeff_w[0] * gw
                logit_y = coeff_z[1] * gz + coeff_w[1] * gw
                p_x = expit(logit_x)
                p_y = expit(logit_y)
                x = rng.binomial(1, p_x, size=n_per_group)
                y = rng.binomial(1, p_y, size=n_per_group)
                rows.append(pd.DataFrame({"x": x, "y": y, "z_disc": z_vals, "w_disc": w_vals}))
        return pd.concat(rows, ignore_index=True)

    # population 1 coefficients (for logits) -> same sign to induce positive
    # marginal association between x and y
    coeff_z1 = (1.5, 1.2)
    coeff_w1 = (1.0, 0.8)
    df1 = make_population(coeff_z1, coeff_w1, flip_noise=0.05)

    # population 2 flips sign on y coefficients to change marginal relation
    coeff_z2 = (1.5, -1.2)
    coeff_w2 = (1.0, -0.8)
    df2 = make_population(coeff_z2, coeff_w2, flip_noise=0.05)

    return df1, df2


def make_mixed_data(n_per_group=500, groups=3, seed=2):
    """Génère deux populations mixtes: z_disc (discret) et w_cont (continu)
    where both parents cause X and Y. X and Y are binary but depend on both.

    DataFrame columns: x (binary), y (binary), z_disc, w_cont
    """
    rng = np.random.default_rng(seed)

    def make_population(coeff_z, coeff_w, noise_scale=0.5):
        rows = []
        for gz in range(groups):
            w_vals = rng.normal(size=n_per_group * 1)
            z_vals = np.full(n_per_group, gz, dtype=int)
            logit_x = coeff_z[0] * gz + coeff_w[0] * w_vals
            logit_y = coeff_z[1] * gz + coeff_w[1] * w_vals
            p_x = expit(logit_x)
            p_y = expit(logit_y)
            x = rng.binomial(1, p_x, size=n_per_group)
            y = rng.binomial(1, p_y, size=n_per_group)
            rows.append(pd.DataFrame({"x": x, "y": y, "z_disc": z_vals, "w_cont": w_vals}))
        return pd.concat(rows, ignore_index=True)

    coeff_z1 = (0.6, 0.6)
    coeff_w1 = (0.8, 0.9)
    df1 = make_population(coeff_z1, coeff_w1, noise_scale=0.5)

    coeff_z2 = (0.6, -0.6)
    coeff_w2 = (0.8, -0.9)
    df2 = make_population(coeff_z2, coeff_w2, noise_scale=0.6)

    return df1, df2


def run_demo(n_permutations=1000, seed=42):
    print("Demonstration des tests d'égalité (permutation)\n")

    # For each scenario we run equality tests twice: marginal (no conditioning)
    # and conditional (conditioning on the parents Z and W). This shows that
    # X and Y are marginally dependent but (approximately) independent when
    # conditioning on Z and W.

    # ------------------ Continuous scenario ------------------
    print("1) Scénario continu: test d'égalité marginale et conditionnelle")
    df1_c, df2_c = make_continuous_data(n=2000, seed=seed)

    tests = [
        # ("PartialCorrelationEqualityTest", PartialCorrelationEqualityTest),
        ("LinearRegressionCoefficientEqualityTest", LinearRegressionCoefficientEqualityTest),
        # ("KernelPartialCorrelation", KernelPartialCorrelationEqualityTest),
        # ("CMIhEqualityTest", CMIhEqualityTest),
        # ("SlearnerEqualityTest", SLearnerEqualityTest),
        # ("GComputationEqualityTest", GComputationEqualityTest),
    ]

    for name, cls in tests:
        # marginal equality (no conditioning)
        inst_marg = cls(x="x", y="y", cond_list=[], drop_na=True)
        # If using CMIhEqualityTest here (not typical for continuous), allow
        # setting k/discrete_vars later; nothing to set for continuous tests
        try:
            # compute observed dependence values for diagnosis
            obs1 = inst_marg.get_dependence(df1_c)
            obs2 = inst_marg.get_dependence(df2_c)
            print(f"{name} observed (marginal): stat1={obs1}, stat2={obs2}, diff={None if obs1 is None or obs2 is None else abs(obs1-obs2)}")
            # p_marg = inst_marg.get_pvalue_by_permutation(df1_c, df2_c, n_permutations=n_permutations, seed=seed)
            p_marg = inst_marg.get_pvalue(df1_c, df2_c)
            print(f"{name} p-value (marginal): {p_marg}")
        except Exception as e:
            print(f"Erreur {name} marginal: {e}")

        # conditional equality (condition on z_cont and w_cont)
        inst_cond = cls(x="x", y="y", cond_list=["z_cont", "w_cont"], drop_na=True)
        try:
            obs1 = inst_cond.get_dependence(df1_c)
            obs2 = inst_cond.get_dependence(df2_c)
            print(f"{name} observed (cond): stat1={obs1}, stat2={obs2}, diff={None if obs1 is None or obs2 is None else abs(obs1-obs2)}")
            # p_cond = inst_cond.get_pvalue_by_permutation(df1_c, df2_c, n_permutations=n_permutations, seed=seed)
            p_cond = inst_cond.get_pvalue(df1_c, df2_c)
            print(f"{name} p-value (cond z_cont,w_cont): {p_cond}")
        except Exception as e:
            print(f"Erreur {name} conditionnel: {e}")

    # ------------------ Discrete scenario ------------------
    print("\n2) Scénario discret: test d'égalité marginale et conditionnelle")
    df1_d, df2_d = make_discrete_data(n_per_group=500, groups=3, seed=seed)

    tests_disc = [
        ("GsqEqualityTest", GsqEqualityTest),
        # ("CMIhEqualityTest", CMIhEqualityTest),
        # ("SlearnerEqualityTest", SLearnerEqualityTest),
        ("GComputationEqualityTest", GComputationEqualityTest),
    ]

    for name, cls in tests_disc:
        inst_marg = cls(x="x", y="y", cond_list=[], drop_na=True)
        # For CMIh-based tests, inform the estimator which variables are discrete
        if name == "CMIhEqualityTest":
            # mark x,y,z_disc,w_disc discrete for the estimator
            inst_marg.discrete_vars = ["x", "y", "z_disc", "w_disc"]
            inst_marg.k = 5
            # use light Laplace smoothing to stabilise discrete entropy estimates
            inst_marg.discrete_alpha = 0.1

        try:
            obs1 = inst_marg.get_dependence(df1_d)
            obs2 = inst_marg.get_dependence(df2_d)
            print(f"{name} observed (marginal): stat1={obs1}, stat2={obs2}, diff={None if obs1 is None or obs2 is None else abs(obs1-obs2)}")
            # If CMIh, show detailed components for diagnosis
            if name == "CMIhEqualityTest":
                comps1 = inst_marg.get_cmi_components(df1_d)
                comps2 = inst_marg.get_cmi_components(df2_d)
                print("CMIh components (marginal) pop1:", comps1)
                print("CMIh components (marginal) pop2:", comps2)
            p_marg = inst_marg.get_pvalue_by_permutation(df1_d, df2_d, n_permutations=n_permutations, seed=seed)
            print(f"{name} p-value (marginal): {p_marg}")
        except Exception as e:
            print(f"Erreur {name} marginal: {e}")

        inst_cond = cls(x="x", y="y", cond_list=["z_disc", "w_disc"], drop_na=True)
        if name == "CMIhEqualityTest":
            inst_cond.discrete_vars = ["x", "y", "z_disc", "w_disc"]
            inst_cond.k = 5
            inst_cond.discrete_alpha = 0.1
        try:
            obs1 = inst_cond.get_dependence(df1_d)
            obs2 = inst_cond.get_dependence(df2_d)
            print(f"{name} observed (cond): stat1={obs1}, stat2={obs2}, diff={None if obs1 is None or obs2 is None else abs(obs1-obs2)}")
            if name == "CMIhEqualityTest":
                comps1 = inst_cond.get_cmi_components(df1_d)
                comps2 = inst_cond.get_cmi_components(df2_d)
                print("CMIh components (cond z_disc,w_disc) pop1:", comps1)
                print("CMIh components (cond z_disc,w_disc) pop2:", comps2)
            p_cond = inst_cond.get_pvalue_by_permutation(df1_d, df2_d, n_permutations=n_permutations, seed=seed)
            print(f"{name} p-value (cond z_disc,w_disc): {p_cond}")
        except Exception as e:
            print(f"Erreur {name} conditionnel: {e}")

    # ------------------ Mixed scenario ------------------
    print("\n3) Scénario mixte: test d'égalité marginale et conditionnelle")
    df1_m, df2_m = make_mixed_data(n_per_group=1000, groups=3, seed=seed)

    # For mixed we focus on CMIh which supports mixed vars
    tests_disc = [
        # ("CMIhEqualityTest", CMIhEqualityTest),
        # ("SlearnerEqualityTest", SLearnerEqualityTest),
        ("GComputationEqualityTest", GComputationEqualityTest),
    ]

    for name, cls in tests_disc:
        inst_marg = cls(x="x", y="y", cond_list=[], drop_na=True)
        # mark discrete variables (x and z_disc) for the hybrid estimator
        inst_marg.discrete_vars = ["x", "y", "z_disc"]
        inst_marg.k = 5
        inst_marg.discrete_alpha = 0.1
        try:
            obs1 = inst_marg.get_dependence(df1_m)
            obs2 = inst_marg.get_dependence(df2_m)
            print(f"{name} observed (marginal): stat1={obs1}, stat2={obs2}, diff={None if obs1 is None or obs2 is None else abs(obs1-obs2)}")
            # Only attempt to get CMI components if the estimator provides the method
            if hasattr(inst_marg, "get_cmi_components"):
                try:
                    comps1 = inst_marg.get_cmi_components(df1_m)
                    comps2 = inst_marg.get_cmi_components(df2_m)
                    print("CMIh components (marginal mixed) pop1:", comps1)
                    print("CMIh components (marginal mixed) pop2:", comps2)
                except Exception as e:
                    print(f"Erreur computing components for {name} marginal: {e}")
            p_marg = inst_marg.get_pvalue_by_permutation(df1_m, df2_m, n_permutations=n_permutations, seed=seed)
            print(f"{name} p-value (marginal): {p_marg}")
        except Exception as e:
            print(f"Erreur {name} marginal: {e}")

        inst_cond = cls(x="x", y="y", cond_list=["z_disc", "w_cont"], drop_na=True)
        inst_cond.discrete_vars = ["x", "y", "z_disc"]
        inst_cond.k = 5
        inst_cond.discrete_alpha = 0.1
        try:
            obs1 = inst_cond.get_dependence(df1_m)
            obs2 = inst_cond.get_dependence(df2_m)
            print(f"{name} observed (cond): stat1={obs1}, stat2={obs2}, diff={None if obs1 is None or obs2 is None else abs(obs1-obs2)}")
            if hasattr(inst_cond, "get_cmi_components"):
                try:
                    comps1 = inst_cond.get_cmi_components(df1_m)
                    comps2 = inst_cond.get_cmi_components(df2_m)
                    print("CMIh components (cond mixed) pop1:", comps1)
                    print("CMIh components (cond mixed) pop2:", comps2)
                except Exception as e:
                    print(f"Erreur computing components for {name} conditional: {e}")
            p_cond = inst_cond.get_pvalue_by_permutation(df1_m, df2_m, n_permutations=n_permutations, seed=seed)
            print(f"{name} p-value (cond z_disc,w_cont): {p_cond}")
        except Exception as e:
            print(f"Erreur {name} conditionnel: {e}")


if __name__ == "__main__":
    run_demo(n_permutations=1000, seed=42)