"""PUCT-guided trajectory collection for relation posteriors (training phase)."""

import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx


def puct_calib_collect_experience(
    core,
    G_sub: nx.DiGraph,
    masked_q: str,
    qents: List[str],
    aents: List[str],
):
    """Lightweight PUCT rollouts on a single question subgraph."""
    if (not core.puct_calib) or (core.puct_calib_num_sims <= 0):
        return
    if (G_sub is None) or (len(G_sub) == 0):
        return

    starts = [s for s in (qents or []) if s in G_sub]
    targets = [t for t in (aents or []) if t in G_sub]
    if (not starts) or (not targets):
        return

    dist = {}
    try:
        rev = G_sub.reverse(copy=False)
        cutoff = int(core.max_hop)
        for tgt in targets:
            try:
                dmap = nx.single_source_shortest_path_length(rev, tgt, cutoff=cutoff)
            except Exception:
                continue
            for n, d in dmap.items():
                if (n not in dist) or (d < dist[n]):
                    dist[n] = int(d)
    except Exception:
        dist = {}

    sim_cache: Dict[str, float] = {}

    def _get_loc(rel: str) -> float:
        if rel in sim_cache:
            return sim_cache[rel]
        v = float(core._sim_score(masked_q + "?", rel))
        sim_cache[rel] = v
        return v

    def _expand_actions(node: str) -> List[Tuple[str, str]]:
        out = []
        try:
            for nb in G_sub.neighbors(node):
                edge_rel = G_sub[node][nb].get("relation", None)
                if edge_rel is None:
                    continue
                rel_options = edge_rel if isinstance(edge_rel, list) else [edge_rel]
                for rel in rel_options:
                    if not isinstance(rel, str):
                        continue
                    a, b = core.relation_posteriors.get(rel, [1.0, 1.0])
                    prior_mean = a / (a + b) if (a + b) > 0 else 0.5
                    if prior_mean < core.prior_threshold:
                        continue
                    out.append((nb, rel))
        except Exception:
            return []
        return out

    N_s: Dict[Tuple[str, int], int] = defaultdict(int)
    N_sa: Dict[Tuple[Tuple[str, int], Tuple[str, str]], int] = defaultdict(int)
    W_sa: Dict[Tuple[Tuple[str, int], Tuple[str, str]], float] = defaultdict(float)
    P_sa: Dict[Tuple[Tuple[str, int], Tuple[str, str]], float] = {}
    children: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}

    def _softmax(x: List[float]) -> List[float]:
        if not x:
            return []
        m = max(x)
        exps = [math.exp(v - m) for v in x]
        s = sum(exps) + 1e-12
        return [e / s for e in exps]

    def _ensure_expanded(state: Tuple[str, int]):
        if state in children:
            return
        node, depth = state
        acts = _expand_actions(node)
        if not acts:
            children[state] = []
            return

        logits = []
        for nb, rel in acts:
            loc = _get_loc(rel)
            sem = -loc
            a, b = core.relation_posteriors.get(rel, [1.0, 1.0])
            prior_mean = a / (a + b) if (a + b) > 0 else 0.5
            logit = (sem + float(core.puct_calib_prior_w) * float(prior_mean)) / max(
                1e-6, float(core.puct_calib_temp)
            )
            logits.append(float(logit))

        probs = _softmax(logits)
        K = max(1, int(core.treeg_branch_size))
        idxs = list(range(len(acts)))
        idxs.sort(key=lambda i: probs[i], reverse=True)
        idxs = idxs[: min(K, len(idxs))]
        acts_k = [acts[i] for i in idxs]
        probs_k = [probs[i] for i in idxs]
        s = sum(probs_k) + 1e-12
        probs_k = [p / s for p in probs_k]
        children[state] = acts_k
        for a_t, p in zip(acts_k, probs_k):
            P_sa[(state, a_t)] = float(p)

    def _evaluate(node: str, depth: int) -> float:
        if node in targets:
            return 1.0
        if depth >= int(core.max_hop):
            return 0.0
        rem = int(core.max_hop) - depth
        d = dist.get(node, 10**9)
        return 0.2 if (d <= rem) else 0.0

    sims = int(core.puct_calib_num_sims)
    cpuct = float(core.puct_calib_cpuct)

    for _ in range(sims):
        start = random.choice(starts)
        node = start
        depth = 0
        traj: List[Tuple[Tuple[str, int], Tuple[str, str]]] = []
        rels_used: List[str] = []

        while True:
            state = (node, depth)
            if (node in targets) or (depth >= int(core.max_hop)):
                reward = _evaluate(node, depth)
                break

            _ensure_expanded(state)
            acts = children.get(state, [])
            if not acts:
                reward = _evaluate(node, depth)
                break

            Ns = N_s[state]
            best = None
            best_score = -1e9
            for a_t in acts:
                nsa = N_sa[(state, a_t)]
                q = (W_sa[(state, a_t)] / nsa) if nsa > 0 else 0.0
                p = P_sa.get((state, a_t), 0.0)
                u = cpuct * p * math.sqrt(Ns + 1.0) / (1.0 + nsa)
                s_val = q + u
                if s_val > best_score:
                    best_score = s_val
                    best = a_t

            if best is None:
                reward = _evaluate(node, depth)
                break

            traj.append((state, best))
            rels_used.append(best[1])
            node = best[0]
            depth += 1

        for (state, a_t) in traj:
            N_s[state] += 1
            N_sa[(state, a_t)] += 1
            W_sa[(state, a_t)] += float(reward)

        if rels_used:
            denom = float(max(1, len(rels_used)))
            if reward >= 0.9:
                inc = float(core.puct_calib_update_scale) / denom
                for r in rels_used:
                    core.relation_posteriors[r][0] += inc
            else:
                dec = float(core.puct_calib_fail_beta) / denom
                if dec > 0:
                    for r in rels_used:
                        core.relation_posteriors[r][1] += dec
