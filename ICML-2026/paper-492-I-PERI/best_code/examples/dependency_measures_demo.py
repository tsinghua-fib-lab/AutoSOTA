"""Demo: usage examples for dependency measures in pyciphod

This script creates small synthetic datasets and prints results for three scenarios:
 - continuous scenario: PartialCorrelation, LinearRegressionCoefficient, CMIh
 - discrete scenario: Gsq, CMIh (discrete_vars)
 - mixed scenario: CMIh only

"""

import numpy as np
import pandas as pd

from pyciphod.utils import (
    PartialCorrelation,
    LinearRegressionCoefficient,
    Gsq,
    CMIh,
)

np.random.seed(42)


def demo_continuous_scenario():
    """Continuous scenario: use PartialCorrelation, LinearRegressionCoefficient, and CMIh."""
    print('\n=== Continuous scenario ===')
    n = 600
    x = np.random.normal(size=n)
    # z is a noisy function of x (mediator)
    z = 0.8 * x + 0.2 * np.random.normal(size=n)
    y = 2.0 * z + 0.1 * np.random.normal(size=n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})

    # Partial correlations
    pc_uncond = PartialCorrelation("x", "y")
    r_xy = pc_uncond.get_dependence(df)
    print(f"PartialCorrelation r(x,y) (unconditional): {r_xy}")

    pc_cond = PartialCorrelation("x", "y", cond_list=["z"])
    r_xy_given_z = pc_cond.get_dependence(df)
    print(f"PartialCorrelation r(x,y|z): {r_xy_given_z}")

    # Regression coefficient of x when controlling for z
    lrcoef = LinearRegressionCoefficient("x", "y", cond_list=["z"])
    beta_x = lrcoef.get_dependence(df)
    print(f"LinearRegressionCoefficient beta_x controlling for z: {beta_x}")

    # CMI estimates (continuous)
    cmi_xy = CMIh("x", "y").get_dependence(df)
    cmi_xy_given_z = CMIh("x", "y", cond_list=["z"]).get_dependence(df)
    print(f"CMI I(x;y) (continuous): {cmi_xy}")
    print(f"CMI I(x;y|z) (continuous): {cmi_xy_given_z}")


def demo_discrete_scenario():
    """Discrete scenario: use Gsq and CMIh with discrete_vars."""
    print('\n=== Discrete scenario ===')
    n = 800
    # X is categorical, Z a noisy copy of X (mediator), Y either independent or equal to X
    x = np.random.randint(0, 3, size=n)
    z = x.copy()  # perfect mediator for demonstration
    y_ind = np.random.randint(0, 3, size=n)
    y_dep = x.copy()

    df_ind = pd.DataFrame({"x": x, "y": y_ind, "z": z})
    df_dep = pd.DataFrame({"x": x, "y": y_dep, "z": z})

    # G-squared
    g_ind = Gsq("x", "y").get_dependence(df_ind)
    g_dep = Gsq("x", "y").get_dependence(df_dep)
    print(f"G² independent case: {g_ind}")
    print(f"G² dependent case (y==x): {g_dep}")

    # G-squared conditioned on z (should reduce dependence when z explains it)
    g_cond = Gsq("x", "y", cond_list=["z"]).get_dependence(df_dep)
    print(f"G² conditioned on z (z==x): {g_cond}")

    # CMI using discrete plugin (inform cmih of discrete variables)
    cmi_ind = CMIh("x", "y", discrete_vars=["x", "y"]).get_dependence(df_ind)
    cmi_dep = CMIh("x", "y", discrete_vars=["x", "y"]).get_dependence(df_dep)
    cmi_cond = CMIh("x", "y", cond_list=["z"], discrete_vars=["x", "y", "z"]).get_dependence(df_dep)
    print(f"CMI (discrete) independent: {cmi_ind}")
    print(f"CMI (discrete) dependent: {cmi_dep}")
    print(f"CMI (discrete) conditioned on z: {cmi_cond}")


def demo_mixed_scenario():
    """Mixed scenario: only CMIh (treat some vars as discrete)"""
    print('\n=== Mixed scenario ===')
    n = 700
    # X discrete, Y continuous dependent on X
    x = np.random.randint(0, 4, size=n)
    y = 1.5 * x + 0.5 * np.random.normal(size=n)
    # Z continuous (independent)
    z = np.random.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z})

    # Use CMIh treating x as discrete and y as continuous
    cmi_mixed = CMIh("x", "y", discrete_vars=["x"]).get_dependence(df)
    cmi_mixed_cond_z = CMIh("x", "y", cond_list=["z"], discrete_vars=["x"]).get_dependence(df)
    print(f"CMI (mixed) I(x;y) treating x as discrete: {cmi_mixed}")
    print(f"CMI (mixed) I(x;y|z) treating x as discrete: {cmi_mixed_cond_z}")


if __name__ == '__main__':
    demo_continuous_scenario()
    demo_discrete_scenario()
    demo_mixed_scenario()
    print('\nDemo finished.')
