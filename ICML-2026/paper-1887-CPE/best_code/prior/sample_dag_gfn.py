# load gfn
import argparse
import numpy as np
import torch
from prior.train_dag_gfn import EdgePolicy, init_state, list_actions, add_edge

def sample_dag_from_gfn(policy: EdgePolicy, D: int, rng: np.random.Generator, max_edges: int) -> np.ndarray:
    st = init_state(D)
    with torch.no_grad():
        while True:
            acts = list_actions(st)
            A_flat = torch.tensor(st.A.reshape(1, D * D), dtype=torch.float32)
            stop_logit, node = policy(A_flat)
            stop_logit = float(stop_logit.item())
            node = node[0]

            edge_logits = []
            for (i, j) in acts:
                li = node[i].unsqueeze(0)
                lj = node[j].unsqueeze(0)
                edge_logits.append(float(policy.edge_bilinear(li, lj).item()))

            logits = np.array([stop_logit] + edge_logits, dtype=np.float32)
            # sample
            m = float(np.max(logits))
            probs = np.exp(logits - m)
            probs = probs / probs.sum()
            aidx = int(rng.choice(len(probs), p=probs))

            if aidx == 0:
                break
            (i, j) = acts[aidx - 1]
            st = add_edge(st, i, j)
            if st.edges >= max_edges:
                break
    return st.A.astype(float)  # ParticlePosterior expects weighted matrices; 0/1 ok




