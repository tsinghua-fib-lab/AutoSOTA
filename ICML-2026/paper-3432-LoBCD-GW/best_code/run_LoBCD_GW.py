import time
import pickle
import argparse
from collections import defaultdict
import numpy as np
import networkx as nx
import torch
import torch.backends.cudnn as cudnn
from torch import amp

cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import LoBCD_GW as dyn
from LoBCD_GW import create_dynamic_vi_gw, DynamicVI_GW_Simple


def add_noisy_edges(graph: nx.Graph, noisy_level: float) -> nx.Graph:
    if noisy_level <= 0.0:
        return graph
    g2 = graph.copy()
    nodes = np.array(list(g2.nodes()))
    n = len(nodes)
    m = g2.number_of_edges()
    need = int(noisy_level * max(m, 1))
    if need <= 0:
        return g2
    rng = np.random.default_rng()
    added, per_batch, max_batches = 0, max(need * 4, 256), 8
    for _ in range(max_batches):
        if added >= need:
            break
        u_idx = rng.integers(0, n, size=per_batch)
        v_idx = rng.integers(0, n, size=per_batch)
        for uu, vv in zip(nodes[u_idx], nodes[v_idx]):
            if added >= need:
                break
            if uu != vv and not g2.has_edge(uu, vv):
                g2.add_edge(uu, vv)
                added += 1
    return g2


@torch.no_grad()
def node_correctness_gpu(X: torch.Tensor) -> float:
    n = X.size(0)
    pred = X.argmax(dim=1)
    acc = (pred == torch.arange(n, device=X.device)).float().mean()
    return float(acc.item())


@torch.no_grad()
def gap_gpu(X: torch.Tensor) -> float:
    n = X.size(0)
    inv = 1.0 / n
    rs = X.sum(dim=1) - inv
    cs = X.sum(dim=0) - inv
    g = torch.linalg.norm(rs) + torch.linalg.norm(cs)
    return float(g.item())


def nx_to_arrays(G_or_pair, noise_level: float):
    if isinstance(G_or_pair, (tuple, list)) and len(G_or_pair) == 2:
        A, B = G_or_pair
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        return A, B
    if isinstance(G_or_pair, nx.Graph):
        A = nx.to_numpy_array(G_or_pair, dtype=np.float32)
    else:
        A = np.asarray(G_or_pair, dtype=np.float32)
    if noise_level > 0.0 and isinstance(G_or_pair, nx.Graph):
        G2 = add_noisy_edges(G_or_pair, noise_level)
        B = nx.to_numpy_array(G2, dtype=np.float32)
    else:
        B = A
    return A, B


def edge_list_to_adj(E: np.ndarray, n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float32)
    A[E[0], E[1]] = 1.0
    A[E[1], E[0]] = 1.0
    return A


def load_edge_pairs(obj):
    E_list = obj["E_list"]
    N_list = obj["N_list"]
    H = obj.get("H", None)
    test = obj.get("test", None)
    adjs = [edge_list_to_adj(np.asarray(E), int(N)) for E, N in zip(E_list, N_list)]
    if isinstance(test, np.ndarray) and test.ndim == 2 and test.shape[1] == 2:
        pairs_idx = [(int(i), int(j)) for i, j in test]
    else:
        k = (len(adjs) // 2) * 2
        pairs_idx = [(i, i + 1) for i in range(0, k, 2)]
    perms = [None] * len(pairs_idx)
    if isinstance(H, np.ndarray):
        if H.ndim == 3 and H.shape[1] == H.shape[2] and len(H) == len(pairs_idx):
            perms = [H[p] for p in range(len(pairs_idx))]
        elif H.ndim == 2 and H.shape[0] == len(pairs_idx) and (H.shape[1] == N_list[0]):
            for p, vec in enumerate(H):
                n = int(vec.shape[0])
                P = np.zeros((n, n), dtype=np.float32)
                P[np.arange(n), vec.astype(int)] = 1.0
                perms[p] = P
        elif H.ndim == 2 and H.shape[0] == H.shape[1]:
            P = H.astype(np.float32)
            perms = [P for _ in pairs_idx]
    pairs = [(adjs[i], adjs[j]) for (i, j) in pairs_idx]
    return pairs, perms


def load_single_graphs_from_edge_pack(obj):
    E_list = obj["E_list"]
    N_list = obj["N_list"]
    graphs = []
    for E, N in zip(E_list, N_list):
        A = edge_list_to_adj(np.asarray(E), int(N))
        G = nx.from_numpy_array(A)
        graphs.append(G)
    return graphs


load_douban_pairs = load_edge_pairs


def to_device_async(A_np, B_np, device, prefetch_stream=None):
    A_cpu = torch.from_numpy(A_np)
    B_cpu = torch.from_numpy(B_np)
    if device == "cuda":
        A_cpu = A_cpu.pin_memory()
        B_cpu = B_cpu.pin_memory()
        with torch.cuda.stream(prefetch_stream):
            A_dev = A_cpu.to(device, non_blocking=True)
            B_dev = B_cpu.to(device, non_blocking=True)
        evt = torch.cuda.Event()
        prefetch_stream.record_event(evt)
        return A_dev, B_dev, evt
    else:
        return A_cpu.to(device), B_cpu.to(device), None


def process_one(
    idx,
    A_t,
    B_t,
    rho_list,
    max_iter,
    eps,
    rho_decay,
    rho_min,
    use_amp,
    r_ratio_cut,
    sinkhorn_iters,
    check_every,
    device,
    print_every,
):
    t0 = time.time()
    n = A_t.size(0)
    a = torch.full((n,), 1.0 / n, device=device, dtype=A_t.dtype)
    b = torch.full((n,), 1.0 / n, device=device, dtype=A_t.dtype)
    results = []
    X_prev = None
    amp_ctx = amp.autocast(device_type="cuda") if (use_amp and device == "cuda") else torch.no_grad()
    for rho in rho_list:
        start1 = time.time()
        with torch.inference_mode(), amp_ctx:
            X_hat, _ = create_dynamic_vi_gw(
                A=A_t, B=B_t, a=a, b=b,
                X_init=X_prev,
                rho0=rho, min_rho=rho_min,
                eps=eps, max_iter=max_iter, rho_decay=rho_decay,
                r_ratio_cut=r_ratio_cut,
                sinkhorn_iters=sinkhorn_iters,
                check_every=check_every,
                print_every=print_every,
            )
        t1 = time.time() - start1
        P_rounded1 = dyn.round_hungarian_from_affinity(A_t, B_t, X_hat)
        acc1 = node_correctness_gpu(P_rounded1)
        gap1 = gap_gpu(X_hat)
        start2 = time.time()
        try:
            X_ref, _ = DynamicVI_GW_Simple(
                A=A_t, B=B_t, a=a, b=b, X_init=X_hat,
                rho0=100.0, min_rho=100.0, eps=eps, max_iter=100, rho_decay=rho_decay,
                r_ratio_cut=r_ratio_cut,
                sinkhorn_iters=sinkhorn_iters,
                check_every=check_every,
                print_every=50,
            )
        except TypeError:
            X_ref, _ = DynamicVI_GW_Simple(
                A=A_t, B=B_t, a=a, b=b, X_init=X_hat,
                rho0=100, min_rho=100, eps=eps, max_iter=100, rho_decay=rho_decay
            )
        t2 = time.time() - start2
        P_rounded = dyn.round_hungarian_from_affinity(A_t, B_t, X_ref)
        acc2 = node_correctness_gpu(P_rounded)
        gap2 = gap_gpu(X_ref)
        results.append((rho, gap1, acc1 * 100, t1, gap2, acc2 * 100, t2))
        print(f"[G{idx + 1}] rho={rho:g} | acc1={acc1:.4f} ({t1:.2f}s) -> acc2={acc2:.4f} ({t2:.2f}s)")
        X_prev = X_ref
    return idx, results, time.time() - t0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='reddit', choices=['proteins', 'reddit', 'enzymes', 'synthetic', 'douban', 'acm500', 'acm1000', 'dblp500', 'dblp1000'])
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--rho', type=float, nargs='+', default=[0.1])
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--rho_decay', type=float, default=1.0)
    parser.add_argument('--rho_min', type=float, default=1e-1)
    parser.add_argument('--output_file', type=str, default='result_optimized.txt')
    parser.add_argument('--compile', type=int, default=0)
    parser.add_argument('--amp', type=int, default=0)
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--r_ratio_cut', type=float, default=1.0)
    parser.add_argument('--sinkhorn_iters', type=int, default=4)
    parser.add_argument('--check_every', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=50)
    args = parser.parse_args()
    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    print("=" * 48)
    print("PyTorch / Device Info:")
    print(f"  use_gpu: {args.use_gpu}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  Selected device: {device}")
    if device == "cuda":
        print(f"  CUDA devices: {torch.cuda.device_count()}")
        print(f"  Current: {torch.cuda.current_device()} -> {torch.cuda.get_device_name()}")
    print("=" * 48)
    if args.compile:
        try:
            dyn.DynamicVI_GW = torch.compile(dyn.DynamicVI_GW, fullgraph=False)
            dyn.DynamicVI_GW_Simple = torch.compile(dyn.DynamicVI_GW_Simple, fullgraph=False)
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[compile] skipped ({e})")
    if args.dataset == 'proteins':
        with open('data/PROTEINS/matching.pk', 'rb') as f:
            graphs, _ = pickle.load(f)
    elif args.dataset == 'reddit':
        with open('data/REDDIT-BINARY/matching.pk', 'rb') as f:
            graphs = pickle.load(f)[:500]
    elif args.dataset == 'enzymes':
        with open('data/ENZYMES/matching.pk', 'rb') as f:
            graphs = pickle.load(f)
    elif args.dataset == 'douban':
        with open('data/DOUBAN/douban.pkl', 'rb') as f:
            D = pickle.load(f)
        graphs = load_single_graphs_from_edge_pack(D)
    elif args.dataset == 'acm500':
        with open('data/ACM/K3n500.pkl', 'rb') as f:
            D = pickle.load(f)
        graphs = load_single_graphs_from_edge_pack(D)
    elif args.dataset == 'acm1000':
        with open('data/ACM/K3n1000.pkl', 'rb') as f:
            D = pickle.load(f)
        graphs = load_single_graphs_from_edge_pack(D)
    elif args.dataset == 'dblp500':
        with open('data/DBLP/K3n500.pkl', 'rb') as f:
            D = pickle.load(f)
        graphs = load_single_graphs_from_edge_pack(D)
    elif args.dataset == 'dblp1000':
        with open('data/DBLP/K3n1000.pkl', 'rb') as f:
            D = pickle.load(f)
        graphs = load_single_graphs_from_edge_pack(D)
    else:
        graphs, noise_graphs = [], []
        print('------------------Node Matching on Synthetic Database---------------')
        with open('data/Random/graph1.pk', 'rb') as f:
            graph_pairs = pickle.load(f)
            print(graph_pairs)
            for num_node in [500, 1000, 1500, 2000, 2500]:
                for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                    for G, G_noise in graph_pairs[(num_node, noise_level)]:
                        graphs.append(G)
                        noise_graphs.append(G_noise)
    total = len(graphs)
    results = [None] * total
    print(f"Dataset={args.dataset} | #Graphs={total} | device={device}")
    print(f"rho={args.rho} | max_iter={args.max_iter} | eps={args.eps} | amp={args.amp} | compile={args.compile} | r_ratio_cut={args.r_ratio_cut} | sinkhorn_iters={args.sinkhorn_iters} | check_every={args.check_every}")
    prefetch_stream = torch.cuda.Stream() if device == "cuda" else None
    A0, B0 = nx_to_arrays(graphs[0], args.noise_level)
    A_dev, B_dev, evt = to_device_async(A0, B0, device, prefetch_stream)
    t_all = time.time()
    for i in range(total):
        if device == "cuda" and evt is not None:
            torch.cuda.current_stream().wait_event(evt)
        idx, res_list, cost = process_one(
            i, A_dev, B_dev, args.rho, args.max_iter, args.eps, args.rho_decay, args.rho_min,
            use_amp=bool(args.amp),
            r_ratio_cut=args.r_ratio_cut,
            sinkhorn_iters=args.sinkhorn_iters,
            check_every=args.check_every,
            device=device,
            print_every=args.print_every,
        )
        results[idx] = res_list
        print(f"[G{idx + 1}/{total}] done in {cost:.2f}s")
        if i + 1 < total:
            A_np, B_np = nx_to_arrays(graphs[i + 1], args.noise_level)
            A_dev, B_dev, evt = to_device_async(A_np, B_np, device, prefetch_stream)
    total_time = time.time() - t_all
    print(f"Total processing time: {total_time:.2f}s")
    summary = defaultdict(lambda: {'gap1': [], 'acc1': [], 't1': [], 'gap2': [], 'acc2': [], 't2': []})
    for res_list in results:
        for rho, gap1, acc1, t1, gap2, acc2, t2 in res_list:
            summary[rho]['gap1'].append(gap1)
            summary[rho]['acc1'].append(acc1)
            summary[rho]['t1'].append(t1)
            summary[rho]['gap2'].append(gap2)
            summary[rho]['acc2'].append(acc2)
            summary[rho]['t2'].append(t2)
    import json
    with open(args.output_file, 'a+', encoding='utf-8') as f:
        f.write("================================================\n")
        f.write(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n")
        f.write("------------------------------------------------\n")
        for rho in args.rho:
            gap1 = float(np.mean(summary[rho]['gap1']))
            acc1 = float(np.mean(summary[rho]['acc1']))
            time1 = float(np.sum(summary[rho]['t1']))
            gap2 = float(np.mean(summary[rho]['gap2']))
            acc2 = float(np.mean(summary[rho]['acc2']))
            time2 = float(np.sum(summary[rho]['t2']))
            line = (f"Data:{args.dataset},Noise:{args.noise_level},rho:{rho},gap:{gap1:.2e},acc:{acc1:.2f},time:{time1:.1f},gap2:{gap2:.2e},acc2:{acc2:.2f},time2:{time2:.1f}")
            print(line)
            f.write(line + "\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write("================================================\n\n")
    print("Results saved to", args.output_file)
