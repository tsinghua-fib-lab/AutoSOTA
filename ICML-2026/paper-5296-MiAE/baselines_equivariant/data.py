from pathlib import Path

import torch
from torch_geometric.data import Data
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_existing_path(path, script_dir=None):
    candidate = Path(path)
    if candidate.exists():
        return candidate

    if script_dir is None:
        script_dir = Path(__file__).resolve().parent

    for base in (script_dir, script_dir.parent):
        candidate = base / path
        if candidate.exists():
            return candidate

    return Path(path)


def infer_num_classes_from_graphs(graphs):
    labels = [int(graph.y.view(-1)[0].item()) for graph in graphs if hasattr(graph, "y")]
    if not labels:
        return None
    return max(labels) + 1


def build_graph_from_ted_batch_minimal(
    batch,
    cutoff: float = 8.0,
    neighbor_cap: int | None = None,
    feature_mode: str = "constant",
    device=None,
):
    ca_idx = 1

    coords = batch["coords"]
    mask = batch["mask"].bool()
    seq_ids = batch["seq_ids"]

    if device is None:
        device = coords.device

    coords = coords.to(device)
    mask = mask.to(device)
    seq_ids = seq_ids.to(device)

    batch_size, length = mask.shape

    pos_all = coords[:, :, ca_idx, :].contiguous()
    n_per_graph = mask.sum(dim=1)

    batch_idx_flat, res_idx_flat = mask.nonzero(as_tuple=True)
    n_nodes = batch_idx_flat.numel()

    pos = pos_all[batch_idx_flat, res_idx_flat, :]

    if feature_mode == "token":
        x = seq_ids[batch_idx_flat, res_idx_flat].unsqueeze(-1)
    else:
        x = torch.ones((n_nodes, 1), device=device)

    batch_idx = batch_idx_flat

    sq = (pos_all**2).sum(-1, keepdim=True)
    dot = torch.matmul(pos_all, pos_all.transpose(-2, -1))
    dist2 = (sq + sq.transpose(-2, -1) - 2.0 * dot).clamp_min_(0.0)
    dmat = dist2.sqrt()

    mask_i = mask.unsqueeze(2)
    mask_j = mask.unsqueeze(1)
    edge_mask = (dmat <= cutoff) & (dmat > 0) & mask_i & mask_j

    if neighbor_cap is not None:
        raise NotImplementedError("neighbor_cap is not implemented")

    b_idx, i_idx, j_idx = edge_mask.nonzero(as_tuple=True)

    global_index = torch.full((batch_size, length), -1, device=device, dtype=torch.long)
    global_index[batch_idx_flat, res_idx_flat] = torch.arange(n_nodes, device=device)

    src = global_index[b_idx, i_idx]
    dst = global_index[b_idx, j_idx]
    edge_index = torch.stack([src, dst], dim=0)

    edge_vec = pos[dst] - pos[src]
    dist = edge_vec.norm(dim=1)

    return {
        "pos": pos,
        "edge_index": edge_index,
        "edge_vec": edge_vec,
        "dist": dist,
        "batch_idx": batch_idx,
        "n_per_graph": n_per_graph,
        "x": x,
    }


def batches_to_protein_graphs(graph_batches):
    graphs = []
    for graph_cpu, y_cpu in graph_batches:
        pos = graph_cpu["pos"]
        x = graph_cpu["x"]
        edge_index = graph_cpu["edge_index"]
        batch_idx = graph_cpu["batch_idx"]
        labels = y_cpu

        for b in range(labels.shape[0]):
            node_mask = batch_idx == b
            if not torch.any(node_mask):
                continue

            node_idx = node_mask.nonzero(as_tuple=False).view(-1)
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            e_idx = edge_mask.nonzero(as_tuple=False).view(-1)
            ei = edge_index[:, e_idx]

            mapping = -torch.ones(batch_idx.size(0), dtype=torch.long)
            mapping[node_idx] = torch.arange(node_idx.size(0), dtype=torch.long)
            ei_local = mapping[ei]

            graphs.append(
                Data(
                    x=x[node_idx],
                    pos=pos[node_idx],
                    edge_index=ei_local,
                    y=labels[b].long(),
                )
            )

    return graphs


def build_graph_batches(loader, feature_mode="constant", cutoff=8.0):
    graph_batches = []
    for batch in tqdm(loader):
        batch_gpu = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        graph = build_graph_from_ted_batch_minimal(
            batch_gpu,
            feature_mode=feature_mode,
            cutoff=cutoff,
            device=device,
        )
        y = batch_gpu["label"].long().view(-1)
        graph_cpu = {
            k: (v.cpu() if torch.is_tensor(v) else v) for k, v in graph.items()
        }
        graph_batches.append((graph_cpu, y.cpu()))

    return graph_batches


def protein_to_graph(coords, aa_ids, label, cutoff=8.0, feature_mode="constant"):
    batch = {
        "coords": coords.unsqueeze(0),
        "mask": torch.ones(1, coords.shape[0], dtype=torch.bool),
        "seq_ids": aa_ids.unsqueeze(0),
    }
    graph = build_graph_from_ted_batch_minimal(
        batch,
        cutoff=cutoff,
        feature_mode=feature_mode,
        device=coords.device,
    )
    return Data(
        x=graph["x"],
        pos=graph["pos"],
        edge_index=graph["edge_index"],
        y=torch.tensor(int(label)),
    )
