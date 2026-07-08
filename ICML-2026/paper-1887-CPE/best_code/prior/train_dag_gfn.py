#!/usr/bin/env python3
"""
train_dag_gfn.py

Train a DAG-generating GFlowNet (Trajectory Balance) to sample DAGs
proportional to exp(BIC(G; X) / tau).

Outputs:
  - sachs_gfn_ckpt_std.pt  (policy params)
  - optionally a cached set of sampled graphs

Notes:
- This is a baseline-grade implementation (clarity > speed).
- Acyclicity enforced via reachability matrix updates.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------- Scoring: Linear-Gaussian BIC -----------------------

def bic_score_linear_gaussian(X: np.ndarray, A: np.ndarray, ridge: float = 1e-3) -> float:
    """
    BIC score for linear Gaussian BN with parent sets defined by adjacency A (D,D).
    Returns a scalar (higher is better).
    """
    n, D = X.shape
    ll = 0.0
    k_params = 0

    for j in range(D):
        pa = np.where(A[:, j] == 1)[0]
        y = X[:, j]
        if pa.size == 0:
            # model: y ~ N(mu, sigma^2) with mu=0 after standardization (or include intercept if desired)
            resid = y - y.mean()
            sigma2 = float(np.mean(resid ** 2) + 1e-12)
            ll += -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
            k_params += 1  # sigma
        else:
            Xp = X[:, pa]
            # ridge regression
            XtX = Xp.T @ Xp
            XtX.flat[:: XtX.shape[0] + 1] += ridge
            Xty = Xp.T @ y
            beta = np.linalg.solve(XtX, Xty)
            resid = y - Xp @ beta
            sigma2 = float(np.mean(resid ** 2) + 1e-12)
            ll += -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
            k_params += pa.size + 1  # betas + sigma

    bic = ll - 0.5 * k_params * math.log(n)
    return float(bic)


# ----------------------- DAG environment (edge-add) -------------------------

@dataclass
class DagState:
    A: np.ndarray          # (D,D) adjacency, 0/1
    reach: np.ndarray      # (D,D) reachability (transitive closure), bool/int
    edges: int

def init_state(D: int) -> DagState:
    A = np.zeros((D, D), dtype=np.int8)
    reach = np.zeros((D, D), dtype=np.int8)
    return DagState(A=A, reach=reach, edges=0)

def can_add_edge(state: DagState, i: int, j: int) -> bool:
    if i == j:
        return False
    if state.A[i, j] == 1:
        return False
    # adding i->j creates a cycle if j can reach i already
    if state.reach[j, i] == 1:
        return False
    return True

def add_edge(state: DagState, i: int, j: int) -> DagState:
    """
    Return a NEW state after adding edge i->j, updating reachability in O(D^2).
    """
    D = state.A.shape[0]
    A = state.A.copy()
    reach = state.reach.copy()

    A[i, j] = 1

    # Update reachability: any u that can reach i can now reach any v reachable from j; plus i reaches j, etc.
    # Standard incremental transitive closure update:
    # Let U = {u | reach[u,i]=1 or u==i}, V = {v | reach[j,v]=1 or v==j}
    U = np.where(reach[:, i] == 1)[0].tolist()
    if i not in U:
        U.append(i)
    V = np.where(reach[j, :] == 1)[0].tolist()
    if j not in V:
        V.append(j)

    for u in U:
        for v in V:
            reach[u, v] = 1

    return DagState(A=A, reach=reach, edges=state.edges + 1)

def list_actions(state: DagState) -> List[Tuple[int, int]]:
    """
    All valid edge additions.
    """
    D = state.A.shape[0]
    acts = []
    for i in range(D):
        for j in range(D):
            if can_add_edge(state, i, j):
                acts.append((i, j))
    return acts


# ----------------------- GFlowNet policy network ---------------------------

class EdgePolicy(nn.Module):
    """
    Scores actions given a state.

    We embed the state by flattening adjacency (D^2) and passing through an MLP.
    Then score each possible action (i,j) by a learned bilinear form from node embeddings.
    """
    def __init__(self, D: int, hidden: int = 256):
        super().__init__()
        self.D = D
        self.state_mlp = nn.Sequential(
            nn.Linear(D * D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.node_proj = nn.Linear(hidden, D * 32)  # produce node embeddings (D,32)
        self.stop_head = nn.Linear(hidden, 1)

        self.edge_bilinear = nn.Bilinear(32, 32, 1, bias=False)

    def forward(self, A_flat: torch.Tensor):
        """
        A_flat: (B, D*D)
        Returns:
          stop_logit: (B,1)
          node_emb: (B, D, 32)
        """
        h = self.state_mlp(A_flat)
        stop_logit = self.stop_head(h)
        node = self.node_proj(h).view(-1, self.D, 32)
        return stop_logit, node

    def action_logits(self, A: np.ndarray) -> Tuple[float, List[Tuple[int,int]], np.ndarray]:
        """
        Compute logits for STOP and each valid edge action.
        Returns: stop_logit (float), actions (list), edge_logits (np array len(actions))
        """
        D = self.D
        A_flat = torch.tensor(A.reshape(1, D * D), dtype=torch.float32)
        stop_logit, node = self.forward(A_flat)
        stop_logit = float(stop_logit.item())
        node = node[0]  # (D,32)

        # enumerate valid actions using numpy state
        st = DagState(A=A, reach=transitive_closure(A), edges=int(A.sum()))
        acts = list_actions(st)

        if len(acts) == 0:
            return stop_logit, [], np.zeros((0,), dtype=np.float32)

        # score each (i,j) via bilinear of node embeddings
        logits = []
        for (i, j) in acts:
            li = node[i].unsqueeze(0)
            lj = node[j].unsqueeze(0)
            logits.append(float(self.edge_bilinear(li, lj).item()))
        return stop_logit, acts, np.array(logits, dtype=np.float32)


def transitive_closure(A: np.ndarray) -> np.ndarray:
    """
    Compute reachability (Floyd-Warshall-ish using repeated squaring).
    For D<=50, this is fine at init; incremental update used after actions.
    """
    D = A.shape[0]
    reach = (A > 0).astype(np.int8)
    # Warshall
    for k in range(D):
        reach = np.maximum(reach, (reach[:, [k]] * reach[[k], :]))
    return reach


# ----------------------- Training (Trajectory Balance) ----------------------

def logsumexp_np(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m)) + 1e-12))

def sample_categorical(logits: np.ndarray, rng: np.random.Generator) -> int:
    probs = np.exp(logits - logsumexp_np(logits))
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))

def train(
    X: np.ndarray,
    D: int,
    steps: int,
    batch_size: int,
    tau: float,
    lr: float,
    hidden: int,
    max_edges: int,
    ridge: float,
    seed: int,
    out_ckpt: str,
):
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")

    policy = EdgePolicy(D=D, hidden=hidden).to(device)
    logZ = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    opt = optim.Adam(list(policy.parameters()) + [logZ], lr=lr)

    for it in range(1, steps + 1):
        opt.zero_grad()
        loss_acc = 0.0

        for _ in range(batch_size):
            # roll out a trajectory
            state = init_state(D)
            log_pf = 0.0
            log_pb = 0.0  # we’ll use uniform backward over number of parents actions (simple)

            traj_states = []
            traj_actions = []

            while True:
                traj_states.append(state.A.copy())

                # build logits for STOP + valid edges
                acts = list_actions(state)
                stop_logit, node = policy(torch.tensor(state.A.reshape(1, D * D), dtype=torch.float32))
                stop_logit = float(stop_logit.item())
                node = node[0]  # (D,32)

                edge_logits = []
                for (i, j) in acts:
                    li = node[i].unsqueeze(0)
                    lj = node[j].unsqueeze(0)
                    edge_logits.append(float(policy.edge_bilinear(li, lj).item()))

                # logits: [STOP] + edges
                logits = np.array([stop_logit] + edge_logits, dtype=np.float32)
                aidx = sample_categorical(logits, rng)
                logp = float(logits[aidx] - logsumexp_np(logits))
                log_pf += logp

                if aidx == 0:
                    # STOP
                    traj_actions.append(("stop", -1, -1))
                    break
                else:
                    (i, j) = acts[aidx - 1]
                    traj_actions.append(("add", i, j))
                    state = add_edge(state, i, j)

                if state.edges >= max_edges:
                    traj_actions.append(("stop_forced", -1, -1))
                    break

            # terminal reward
            bic = bic_score_linear_gaussian(X, state.A, ridge=ridge)
            logR = bic / tau  # log reward

            # backward prob (simple): uniform over number of edges to remove along the constructed order
            # This is a crude but acceptable baseline: log_pb = -sum_t log(out_degree_backward)
            # Here we assume each forward add has exactly 1 inverse action (remove last edge), so pb=1.
            log_pb = 0.0

            # TB loss: (logZ + log_pf - logR - log_pb)^2
            tb = (logZ + torch.tensor(log_pf) - torch.tensor(logR) - torch.tensor(log_pb)) ** 2
            loss_acc = loss_acc + tb

        loss = loss_acc / float(batch_size)
        loss.backward()
        opt.step()

        if it % 200 == 0 or it == 1:
            print(f"[train] it={it:05d} loss={float(loss.item()):.4f} logZ={float(logZ.item()):.3f}")

    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({"policy": policy.state_dict(), "logZ": float(logZ.item()), "D": D}, out_ckpt)
    print(f"[train] wrote checkpoint: {out_ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npz", required=True, help="NPZ containing X (n,D)")
    ap.add_argument("--out_ckpt", default="sachs_gfn_ckpt_std.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tau", type=float, default=10.0, help="reward temperature: logR = BIC/tau")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_edges", type=int, default=200)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--standardize_data", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dat = np.load(args.data_npz)
    X = dat["X"]
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    if args.standardize_data:
        # Standardize for linear-Gaussian scoring stability
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    D = X.shape[1]

    train(
        X=X,
        D=D,
        steps=args.steps,
        batch_size=args.batch_size,
        tau=args.tau,
        lr=args.lr,
        hidden=args.hidden,
        max_edges=args.max_edges,
        ridge=args.ridge,
        seed=args.seed,
        out_ckpt=args.out_ckpt,
    )


if __name__ == "__main__":
    main()