import sys, numpy as np
sys.path.insert(0, ".")
from examples.safe_PCE import *

# Compare GDA vs exact LP for noise=0.3
env = make_env(0.3)

# GDA (entropy regularized)
p_gda, u_gda, v_gda = LP(env)
print(f"GDA (beta=0.1): u={u_gda:.6f}, v={v_gda}")

# Exact LP (scipy linprog)
occ, sol = env.lp_solve()
p_lp = env.occ2policy(occ)
u_lp = abs(sol.fun)
r_lp, c_lp = value_function_r_c(env, p_lp)
print(f"Exact LP: u={u_lp:.6f} (sol.fun={abs(sol.fun):.6f}), v={c_lp}")
print(f"Exact LP (value_function): u={r_lp:.6f}, v={c_lp}")

# Compare policies
print(f"\nPolicy comparison (first 5 states):")
for s in range(5):
    print(f"  State {s}: GDA={p_gda[s,:]}, LP={p_lp[s,:]}")

# Check if GDA policy satisfies the constraint
_, c_gda = value_function_r_c(env, p_gda)
print(f"\nGDA constraint value: {c_gda}")
print(f"LP constraint value: {c_lp}")

# Compare with pre-computed hat_Pi values
import pickle
with open("examples/data/U_hatPi.pkl", "rb") as f:
    data = pickle.load(f)
print(f"\nPre-computed hat_Pi policies for similar noise:")
for i, item in enumerate(data["hat_Pi"]):
    pi, u, v, u_s, v_s = item
    noise = data["U"][i]
    if abs(noise - 0.3) < 0.02:
        print(f"  noise={float(noise):.6f}: u={u}, v={v}")
