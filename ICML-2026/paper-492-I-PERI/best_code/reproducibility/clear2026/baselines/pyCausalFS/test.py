#%%

from LSL.MBs.MBbyMB import MBbyMB 
from LSL.MBs.CMB.CMB import CMB

# Required packages
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import time
import sys
import os
from itertools import combinations

# Add parent folder to access locPC/
local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(local_path)


# Simulate a linear SCM from a given DAG
def simulate_linearSCM_from_dag(dag, nb_obs=1, coef_range=(-1, 1), sigma_range=(0.5, 1)):
    """
    Simulates data from a linear Structural Causal Model (SCM) defined by a DAG with random coefficients.
    """
    v = list(nx.topological_sort(dag))
    p = len(v)

    coef = np.zeros((p, p))
    low, high = coef_range
    for j in range(p):
        pa_v = list(dag.predecessors(v[j]))
        for k in range(p):
            if v[k] in pa_v:
                while np.abs(coef[j, k]) < 0.2:
                    coef[j, k] = np.random.uniform(low, high)

    coef_inv = np.linalg.inv(np.identity(p) - coef)

    sigma_low, sigma_high = sigma_range
    sigma_vec = np.random.uniform(sigma_low, sigma_high, size=p)

    data = np.zeros((nb_obs, p))
    for i in range(nb_obs):
        eps_i = np.random.normal(loc=0, scale=sigma_vec, size=p)
        data[i, :] = np.dot(coef_inv, eps_i)

    coef_df = pd.DataFrame(coef, index=v, columns=v)
    data_df = pd.DataFrame(data, columns=v)

    return coef_df, data_df

# Graphs

# Test LocPC
g = nx.DiGraph()
g.add_edges_from([
    ("Y", "N1"), ("N2", "Y"), ("D1", "N1"), ("D2", "N1"), ("D3", "N2"), ("D2", "D3"),
    ("D1", "A1"), ("D1", "A2"), ("A2", "D2"), ("W", "A1"), ("W", "A2")
])




coef, data = simulate_linearSCM_from_dag(g, 1000)


# MBbyMB(data = data, target = 1, alpha = 0.05, is_discrete=False)
res = CMB(data, 1,0.05, is_discrete=False)



# %%
