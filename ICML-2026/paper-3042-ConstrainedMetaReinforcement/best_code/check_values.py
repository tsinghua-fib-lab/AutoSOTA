import sys, pickle, numpy as np
sys.path.insert(0, ".")
from examples.safe_PCE import *

# Check optimal values for different noise levels
for noise in [0.1, 0.2, 0.3, 0.4, 0.5]:
    env = make_env(noise)
    policy, u_opt, v_opt = LP(env)
    print(f"Noise={noise}: u_opt={u_opt}, v_opt={v_opt}")

# Check pre-computed hat_Pi values
print("\nPre-computed hat_Pi:")
with open("examples/data/U_hatPi.pkl", "rb") as f:
    data = pickle.load(f)
for i, item in enumerate(data["hat_Pi"]):
    pi, u, v, u_s, v_s = item
    print(f"  hat_Pi[{i}]: u={u}, v={v}")
