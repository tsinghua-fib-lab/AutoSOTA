from itertools import product
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

VARIABLES = ["Z","X", "Y"]

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
    u_a = rng.binomial(1, 0.5, size=n)

    # Substantive variables
    a = rng.binomial(1, expit(-1.0 + 2.0 * u_a),)
    b = rng.binomial(1, expit(-2.0 + 4.0 * a),)
    c = rng.binomial(1, expit(-2.0 * a + 3.0 * b),)

    # B -> R_C
    r_c = rng.binomial(1,expit(1.4 - 1.2 * a),)

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
