# -----------------------------------------------------------------------------#
# Transfer Regimes
# -----------------------------------------------------------------------------#
from models import *
from train import *
from util import canonical_undirected_edges, ProtocolConfig

# -----------------------------------------------------------------------------#
# Embedding Transfer (E-Replace, E-Concat)
# -----------------------------------------------------------------------------#

"""
Embedding transfer using frozen encoder from source model.

mode in {"replace", "concat"}.
    - For F->H: source="F": embeddings -> LP target
    - For H->F: source="H": embeddings -> NC target
"""
def run_embedding_transfer(mode: str, source: str,  # "F" or "H"
                           F_model: NodeClassifier, H_model: LinkModel, data, edge_index_train: torch.Tensor,
                           train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor,
                           pos_train: np.ndarray, neg_train: np.ndarray, pos_val: np.ndarray,
                           neg_val: np.ndarray, pos_test: np.ndarray, neg_test: np.ndarray,
                           device: torch.device, num_features: int, num_classes: int, cfg: ProtocolConfig,
                           ) -> Dict[str, float]:
    assert mode in {"replace", "concat"}

    data = data.to(device)
    edge_index_train = edge_index_train.to(device)
    x = data.x

    F_model = F_model.to(device)
    H_model = H_model.to(device)

    F_model.eval()
    H_model.eval()

    with torch.no_grad():
        Z_F = F_model.encoder(x, edge_index_train).detach()
        Z_H = H_model.encoder(x, edge_index_train).detach()

    results: Dict[str, float] = {}

    if source == "F":
        # F -> H embedding transfer: LP target
        if mode == "replace":
            x_et = Z_F
        else: # concat
            x_et = torch.cat([x, Z_F], dim=-1)

        in_ch = x_et.size(1)

        H_ET_model, _, H_ET_test_auc = train_link_model(
            data=data,
            edge_index_train=edge_index_train,
            num_features=num_features,
            pos_train=pos_train,
            neg_train=neg_train,
            pos_val=pos_val,
            neg_val=neg_val,
            pos_test=pos_test,
            neg_test=neg_test,
            device=device,
            hidden=cfg.hidden,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            epochs=cfg.link_epochs,
            patience=cfg.patience,
            init_encoder_state=None,  # fresh GNN over embeddings
            x_override=x_et,
            in_channels_override=in_ch,
        )

        key = f"LP_ET_F2H_{mode}_test_auc"
        results[key] = float(H_ET_test_auc)

    elif source == "H":
        # H -> F embedding transfer: NC target
        if mode == "replace":
            x_et = Z_H
        else:
            # concat
            x_et = torch.cat([x, Z_H], dim=-1)

        in_ch = x_et.size(1)

        F_ET_model, _, F_ET_test_acc = train_node_model(
            data=data,
            edge_index_train=edge_index_train,
            num_features=num_features,
            num_classes=num_classes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            device=device,
            hidden=cfg.hidden,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            epochs=cfg.node_epochs,
            patience=cfg.patience,
            init_encoder_state=None,
            x_override=x_et,
            in_channels_override=in_ch,
        )

        key = f"NC_ET_H2F_{mode}_test_acc"
        results[key] = float(F_ET_test_acc)

    else:
        raise ValueError("source must be 'F' or 'H'.")

    return results

# -----------------------------------------------------------------------------#
# Prompt-transfer helpers (constants like homophily, clustering)
# -----------------------------------------------------------------------------#
def edge_homophily_ratio(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """
    Edge homophily: fraction of undirected edges whose endpoints share label.
    """
    edges = canonical_undirected_edges(edge_index)
    if edges.size == 0:
        return 0.0

    y_np = y.cpu().numpy()
    same = 0

    for u, v in edges:
        if y_np[u] == y_np[v]:
            same += 1

    return float(same / edges.shape[0])


def node_homophily_ratio(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """
    Node homophily as average over nodes of the fraction of neighbors that share
    the same label.
    """
    edges = canonical_undirected_edges(edge_index)
    num_nodes = y.size(0)

    if edges.size == 0 or num_nodes == 0:
        return 0.0

    y_np = y.cpu().numpy()
    same_cnt = np.zeros(num_nodes, dtype=np.float64)
    deg = np.zeros(num_nodes, dtype=np.float64)

    for u, v in edges:
        u = int(u)
        v = int(v)

        deg[u] += 1.0
        deg[v] += 1.0

        if y_np[u] == y_np[v]:
            same_cnt[u] += 1.0
            same_cnt[v] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.divide(same_cnt, deg, out=np.zeros_like(same_cnt), where=deg > 0)

    return float(frac.mean())


def compute_default_prompt_consts(data) -> List[float]:
    """
    Compute default prompt constants:
        [edge_homophily, node_homophily, global_clustering_coef]

    Ensures everything is computed on CPU for compatibility with networkx.
    """
    # Work on a CPU copy for safety
    data_cpu = data.cpu()

    H_e = edge_homophily_ratio(data_cpu.edge_index, data_cpu.y)
    H_n = node_homophily_ratio(data_cpu.edge_index, data_cpu.y)

    # Global clustering coefficient via NetworkX
    import networkx as nx

    G = to_networkx(data_cpu, to_undirected=True)
    global_cc = float(nx.transitivity(G))

    return [H_e, H_n, global_cc]
