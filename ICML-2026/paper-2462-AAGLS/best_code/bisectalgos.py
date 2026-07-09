import networkx as nx
import metis
from scipy.sparse.linalg import eigsh
import math
import numpy as np
import random

def spectral_bisect(G, balancedfactor=0):
        if nx.is_connected(G):
            n = G.number_of_nodes()
            if n <= 1:
                nodes = list(G.nodes())
                return nx.subgraph(G, nodes), nx.subgraph(G, [])
            if n == 2:
                nodes = list(G.nodes())
                return nx.subgraph(G, [nodes[0]]), nx.subgraph(G, [nodes[1]])

            nodes = list(G.nodes())
            idx = {u: i for i, u in enumerate(nodes)}

            # Unnormalized Laplacian (normalized=False)
            L = nx.laplacian_matrix(G, nodelist=nodes, weight="weight")

            # Adaptive eigensolver: dense eigh for small subgraphs, shift-invert for large
            if n < 200:
                # Dense eigh is faster for small matrices (no sparse overhead)
                L_dense = L.toarray()
                vals, vecs = np.linalg.eigh(L_dense)
                fiedler = vecs[:, 1]
            else:
                vals, vecs = eigsh(L, k=2, sigma=-1.0, which="LM")
                fiedler = vecs[:, 1]

            # --- Sweep cut on sorted Fiedler values ---
            sweep_order = np.argsort(fiedler)  # prefixes define candidate S
            inS = np.zeros(n, dtype=bool)

            cut = 0.0
            best_score = float("inf")
            best_cut = float("inf")
            best_k = None

            # adjacency list with weights (fast incremental updates)
            nbrs = [[] for _ in range(n)]
            if G.is_multigraph():
                # aggregate parallel edges
                for u, v, data in G.edges(data=True):
                    w = data.get("weight", 1.0)
                    iu, iv = idx[u], idx[v]
                    nbrs[iu].append((iv, w))
                    nbrs[iv].append((iu, w))
            else:
                for u, v, data in G.edges(data=True):
                    w = data.get("weight", 1.0)
                    iu, iv = idx[u], idx[v]
                    nbrs[iu].append((iv, w))
                    nbrs[iv].append((iu, w))

            # try S = first k nodes in sweep_order, for k=1..n-1
            for kpos in range(n - 1):
                v = sweep_order[kpos]
                # add v to S, update cut incrementally
                for u, w in nbrs[v]:
                    if inS[u]:
                        # edge was crossing (v outside, u inside) -> now internal
                        cut -= w
                    else:
                        # edge was internal to outside -> now crossing
                        cut += w
                inS[v] = True

                k = kpos + 1
                denom = min(k, n - k)
                if denom <= balancedfactor*n:
                    continue

                score = cut / denom

                # tie-break: prefer smaller score, then smaller cut
                if (score < best_score) or (score == best_score and cut < best_cut):
                    best_score = score
                    best_cut = cut
                    best_k = k

            left_set = [nodes[i] for i in sweep_order[:best_k]]
            right_set = [u for u in nodes if u not in left_set]

            return nx.subgraph(G, left_set), nx.subgraph(G, right_set)

        else:
            left = max(nx.connected_components(G), key=len)
            right = [i for i in G.nodes() if i not in left]
            return nx.subgraph(G, left), nx.subgraph(G, right)


def metismulti_bisect(G, m=None):
    import math
    import numpy as np
    import networkx as nx
    import metis

    n = G.number_of_nodes()
    nodes = list(G.nodes())

    # Trivial cases
    if n <= 1:
        return nx.subgraph(G, nodes), nx.subgraph(G, [])

    # --- how many target sizes to try ---
    # smallest side ~ log(n), number of sizes ~ sqrt(n)
    min_k = max(1, int(round(math.log(max(n, 2)))))  # natural log; ~log(n)
    min_k = 1
    max_k = max(1, n // 2)
    if m == None: m = max(2, int(round(math.sqrt(n))))

    # candidate target sizes on [min_k, max_k]
    ks = np.geomspace(min_k, max_k, m)
    ks = np.unique(np.clip(np.rint(ks).astype(int), 1, max_k))

    def boundary_weight(S, T):
        # Sum weights of edges crossing (S,T). Works for weighted + multigraph.
        wsum = 0.0
        for u, v in nx.edge_boundary(G, S, T):
            # For MultiGraph, edge_boundary yields (u,v) without key; weight lookup via get_edge_data.
            data = G.get_edge_data(u, v)
            if data is None:
                continue
            if G.is_multigraph():
                for _, attr in data.items():
                    wsum += float(attr.get("weight", 1.0))
            else:
                wsum += float(data.get("weight", 1.0))
        return wsum

    best_score = float("inf")
    best_cut = float("inf")
    best_left = None
    best_right = None

    for k in ks:
        # target partition weights for METIS (fraction of total vertex weight)
        w = float(k) / float(n)
        tpwgts = [w, 1.0 - w]

        try:
            # ufactor controls allowed imbalance; keep your original very-permissive value
            edgecuts, parts = metis.part_graph(G, 2, tpwgts=tpwgts, ufactor=10*1000)
        except Exception:
            parts = None

        if parts is not None:
            left_ids = [i for i, part in enumerate(parts) if part == 0]
            right_ids = [i for i, part in enumerate(parts) if part == 1]
            left = [nodes[i] for i in left_ids]
            right = [nodes[i] for i in right_ids]
        else:
            left, right = [], []

        # safety net if metis didn't produce a real split
        if not left or not right:
            left = nodes[:k]
            right = nodes[k:]
            if not left or not right:
                half = n // 2
                left = nodes[:half]
                right = nodes[half:]

        denom = min(len(left), len(right))
        if denom <= 0:
            continue

        cut = boundary_weight(left, right)
        score = cut / float(denom)

        # choose best sparsity; tie-break on smaller cut
        if (score < best_score) or (score == best_score and cut < best_cut):
            best_score = score
            best_cut = cut
            best_left = left
            best_right = right

    # Final fallback (should rarely trigger)
    if best_left is None or best_right is None:
        half = n // 2
        best_left = nodes[:half]
        best_right = nodes[half:]

    return nx.subgraph(G, best_left), nx.subgraph(G, best_right)