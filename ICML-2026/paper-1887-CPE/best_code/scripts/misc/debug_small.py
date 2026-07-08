import numpy as np
from synthetic_hitl_causal_dpo import run_demo

# Tiny setup: 3 nodes, small number of particles and rounds
out = run_demo(
    D=3,
    S=50,
    T=10,
    beta_edge=8.0,
    beta_dir=-1.5,
    lam=0.0,
    flip_prob=0.3,
    add_remove_prob=0.3,
    weight_noise=0.5,
    edge_prob_true=0.9,
    seed=42
)

A_true = out["A_star"]
marginals_final = out["posterior_marginals"]
logs = out["logs"]

print("Ground truth adjacency:\n", A_true)

for row in logs:
    r = row["round"]
    acc = row["exist_acc@0.5"]
    print(f"\n--- Round {r} ---")
    print("Accuracy @0.5:", acc)
    # Show posterior marginals matrix for this round
    marg = np.array(row["marginals"])
    print("Posterior marginals:\n", np.round(marg, 2))


