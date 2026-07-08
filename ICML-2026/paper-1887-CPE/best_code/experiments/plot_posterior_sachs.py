# plot_posterior_sacks.py
import argparse, json, os
import numpy as np
from graphviz import Digraph
from PyPDF2 import PdfMerger

def digraph_from_adj(A, nodes, title, edge_label_fn=None):
    g = Digraph(format="pdf")
    g.attr(rankdir="LR", labelloc="t", label=title, fontsize="18")
    g.attr("node", shape="ellipse", fontsize="12")
    for n in nodes:
        g.node(n)
    D = len(nodes)
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            if A[i, j] == 1:
                attrs = {}
                if edge_label_fn is not None:
                    attrs["label"] = edge_label_fn(i, j)
                    attrs["fontsize"] = "10"
                g.edge(nodes[i], nodes[j], **attrs)
    return g

def threshold_by_topk(P, k):
    D = P.shape[0]
    edges = []
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            edges.append((P[i, j], i, j))
    edges.sort(reverse=True, key=lambda x: x[0])
    A = np.zeros((D, D), dtype=int)
    for p, i, j in edges[:k]:
        A[i, j] = 1
    return A

def threshold_by_tau(P, tau):
    A = (P >= tau).astype(int)
    np.fill_diagonal(A, 0)
    return A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_json", required=True, help="A single sachs_<policy>_seed*.json")
    ap.add_argument("--meta_json", required=True, help="results_sachs/sachs_meta.json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["topk", "tau"], default="topk")
    ap.add_argument("--k", type=int, default=17)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--which", choices=["final", "init"], default="final")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    meta = json.load(open(args.meta_json, "r"))
    nodes = meta["nodes"]
    ref_edges = meta["ref_edges"]
    D = len(nodes)
    idx = {n:i for i,n in enumerate(nodes)}
    A_ref = np.zeros((D, D), dtype=int)
    for a,b in ref_edges:
        A_ref[idx[a], idx[b]] = 1

    run = json.load(open(args.run_json, "r"))
    key = "posterior_marginals_final" if args.which == "final" else "posterior_marginals_init"
    P = np.asarray(run[key], dtype=float)
    P = np.clip(P, 0.0, 1.0)

    if args.mode == "topk":
        A_post = threshold_by_topk(P, k=args.k)
        title_post = f"Posterior graph (top-{args.k} edges)"
    else:
        A_post = threshold_by_tau(P, tau=args.tau)
        title_post = f"Posterior graph (p ≥ {args.tau:.2f})"

    # Label posterior edges with probabilities (optional but nice)
    def lab(i,j):
        return f"{P[i,j]:.2f}"

    g_ref = digraph_from_adj(A_ref, nodes, "Reference (17 arcs)")
    g_post = digraph_from_adj(A_post, nodes, title_post, edge_label_fn=lab)

    ref_path = os.path.join(args.outdir, "ref_graph")
    post_path = os.path.join(args.outdir, "post_graph")
    g_ref.render(ref_path, cleanup=True)
    g_post.render(post_path, cleanup=True)

    # merge into one pdf for convenience
    merger = PdfMerger()
    merger.append(ref_path + ".pdf")
    merger.append(post_path + ".pdf")
    merged = os.path.join(args.outdir, "ref_vs_post.pdf")
    with open(merged, "wb") as f:
        merger.write(f)

    print("Wrote:")
    print(" -", ref_path + ".pdf")
    print(" -", post_path + ".pdf")
    print(" -", merged)

if __name__ == "__main__":
    main()