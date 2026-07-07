# -----------------------------------------------------------------------------#
# Homotopy-style barrier (direct, via endpoint + cross losses)
# -----------------------------------------------------------------------------#

from models import *
from train import _make_edge_batch
from util import ProtocolConfig


"""
Given encoder_state, build node+link models with that encoder and F/H-head from scratch (for eval only) and compute:
    - node val loss (cross-entropy)
    - link val loss (BCE on val edges)
"""
def compute_val_losses_for_encoder(encoder_state: Dict[str, torch.Tensor], data,
                                   edge_index_train: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor,
                                   pos_val: np.ndarray, neg_val: np.ndarray, device: torch.device, num_features: int,
                                   num_classes: int, hidden: int, num_layers: int, dropout: float,
                                   ) -> Tuple[float, float]:
    data = data.to(device)
    edge_index_train = edge_index_train.to(device)
    x = data.x

    # Node model for loss only (new classifier)
    node_model = NodeClassifier(
        in_channels=num_features,
        hidden_channels=hidden,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    node_model.encoder.load_state_dict(encoder_state)
    node_model.eval()

    with torch.no_grad():
        logits = node_model(x, edge_index_train)
        node_loss = F.cross_entropy(logits[val_mask], data.y[val_mask]).item()

    # Link model (new lp_head)
    link_model = LinkModel(
        in_channels=num_features,
        hidden_channels=hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    link_model.encoder.load_state_dict(encoder_state)
    link_model.eval()

    val_pairs, val_labels = _make_edge_batch(pos_val, neg_val, device)

    with torch.no_grad():
        z = link_model.encoder(x, edge_index_train)
        scores = link_model.lp_head(z, val_pairs)
        link_loss = F.binary_cross_entropy(scores, val_labels).item()

    return node_loss, link_loss

"""
Direct barrier (no interpolation):

    - Baseline endpoint losses:
        L_node_F : node val loss at F_model
        L_link_H : link val loss at H_model

    - Cross losses using frozen encoders:
        L_node_Henc : node val loss with H.encoder in a fresh node model
        L_link_Fenc : link val loss with F.encoder in a fresh link model

Normalized sums:
    tilde_S_F = 1 + L_link_Fenc / L_link_H
    tilde_S_H = L_node_Henc / L_node_F + 1

Barrier_direct = max(tilde_S_F, tilde_S_H) - 2
"""
def compute_barrier(F_model: NodeClassifier, H_model: LinkModel, data, edge_index_train: torch.Tensor,
                    train_mask: torch.Tensor, val_mask: torch.Tensor, pos_val: np.ndarray, neg_val: np.ndarray,
                    device: torch.device, num_features: int, num_classes: int, cfg: ProtocolConfig,
                    ) -> float:
    data = data.to(device)
    edge_index_train = edge_index_train.to(device)
    x = data.x

    F_model = F_model.to(device)
    H_model = H_model.to(device)

    F_model.eval()
    H_model.eval()

    # Baseline endpoint val losses
    with torch.no_grad():
        logits_F = F_model(x, edge_index_train)
        L_node_F = F.cross_entropy(logits_F[val_mask], data.y[val_mask]).item()

        val_pairs, val_labels = _make_edge_batch(pos_val, neg_val, device)
        z_H = H_model.encoder(x, edge_index_train)
        scores_H = H_model.lp_head(z_H, val_pairs)
        L_link_H = F.binary_cross_entropy(scores_H, val_labels).item()

    eps = 1e-12

    # Encoder state dicts
    enc_F = F_model.encoder.state_dict()
    enc_H = H_model.encoder.state_dict()

    # Losses for F encoder in both tasks (we only need link)
    node_loss_Fenc, link_loss_Fenc = compute_val_losses_for_encoder(
        encoder_state=enc_F,
        data=data,
        edge_index_train=edge_index_train,
        train_mask=train_mask,
        val_mask=val_mask,
        pos_val=pos_val,
        neg_val=neg_val,
        device=device,
        num_features=num_features,
        num_classes=num_classes,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    # Losses for H encoder in both tasks (we only need node)
    node_loss_Henc, link_loss_Henc = compute_val_losses_for_encoder(
        encoder_state=enc_H,
        data=data,
        edge_index_train=edge_index_train,
        train_mask=train_mask,
        val_mask=val_mask,
        pos_val=pos_val,
        neg_val=neg_val,
        device=device,
        num_features=num_features,
        num_classes=num_classes,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    # We only need cross losses:
    L_link_Fenc = link_loss_Fenc  # F encoder in LP model
    L_node_Henc = node_loss_Henc  # H encoder in node model

    tilde_S_F = 1.0 + (L_link_Fenc / (L_link_H + eps))
    tilde_S_H = (L_node_Henc / (L_node_F + eps)) + 1.0

    barrier_direct = max(tilde_S_F, tilde_S_H) - 2.0
    return float(barrier_direct)


# -----------------------------------------------------------------------------#
# ProbeCompat (warm-start-delta based)
# -----------------------------------------------------------------------------#
"""Probe head over frozen embeddings (no encoder)."""
class LinkProbe(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.pred = LinkPredictor(embed_dim, hidden)

    def forward(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        return self.pred(z, edge_pairs)

"""
Train a linear probe on frozen embeddings z for node classification.
Returns (best_val_acc, test_acc_at_best).
"""
def train_node_probe(z: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor,
                     val_mask: torch.Tensor, test_mask: torch.Tensor, num_classes: int,
                     device: torch.device, epochs: int = 200,
                     ) -> Tuple[float, float]:
    z = z.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    model = nn.Linear(z.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)

    best_val_acc = 0.0
    best_state = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(z)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(z)
            pred = logits.argmax(dim=-1)
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(z)
        pred = logits.argmax(dim=-1)
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

    return best_val_acc, test_acc

"""
Train a link probe on frozen embeddings z.
Returns (best_val_auc, test_auc_at_best).
"""
def train_link_probe(z: torch.Tensor, pos_train: np.ndarray, neg_train: np.ndarray, pos_val: np.ndarray,
                     neg_val: np.ndarray, pos_test: np.ndarray, neg_test: np.ndarray,
                     device: torch.device, hidden: int = 64, epochs: int = 200
                     ) -> Tuple[float, float]:
    z = z.to(device)

    train_pairs, train_labels = _make_edge_batch(pos_train, neg_train, device)
    val_pairs, val_labels = _make_edge_batch(pos_val, neg_val, device)
    test_pairs, test_labels = _make_edge_batch(pos_test, neg_test, device)

    model = LinkProbe(embed_dim=z.size(1), hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)

    best_val_auc = 0.0
    best_state = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(z, train_pairs)
        loss = F.binary_cross_entropy(preds, train_labels)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = model(z, val_pairs).detach().cpu().numpy()
            labels_np = val_labels.detach().cpu().numpy()
            val_auc = roc_auc_score(labels_np, scores)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        scores = model(z, test_pairs).detach().cpu().numpy()
        labels_np = test_labels.detach().cpu().numpy()
        test_auc = roc_auc_score(labels_np, scores)

    return best_val_auc, test_auc

"""
ProbeCompat based on warm-start transfer differences:

    - F embeddings -> link probe (n -> ℓ)
    - H embeddings -> node probe (ℓ -> n)

Let:
    Δ_ws_n2l    = LP_F2H_test_auc - LP_base_test_auc
    Δ_ws_l2n    = NC_H2F_test_acc - NC_base_test_acc
    Δ_probe_n2l = Probe_T_n2l_test_auc - LP_base_test_auc
    Δ_probe_l2n = Probe_T_l2n_test_acc - NC_base_test_acc

Then:
    C_n2l = Δ_probe_n2l / Δ_ws_n2l
    C_l2n = Δ_probe_l2n / Δ_ws_l2n
    ProbeCompat = 0.5 * (C_n2l + C_l2n)
"""
def compute_probecompat(F_model: NodeClassifier, H_model: LinkModel, data, edge_index_train: torch.Tensor,
                        train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor,
                        pos_train: np.ndarray, neg_train: np.ndarray, pos_val: np.ndarray,
                        neg_val: np.ndarray, pos_test: np.ndarray, neg_test: np.ndarray,
                        device: torch.device, num_classes: int,
                        NC_base_test_acc: float, LP_base_test_auc: float, NC_H2F_test_acc: float, LP_F2H_test_auc: float,
                        cfg: ProtocolConfig,
                        ) -> Dict[str, float]:
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

    # Node probe using Z_H (l -> n)
    _, T_l2n_test_acc = train_node_probe(
        z=Z_H,
        y=data.y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
        device=device,
        epochs=cfg.probe_epochs,
    )

    # Link probe using Z_F (n -> l)
    _, T_n2l_test_auc = train_link_probe(
        z=Z_F,
        pos_train=pos_train,
        neg_train=neg_train,
        pos_val=pos_val,
        neg_val=neg_val,
        pos_test=pos_test,
        neg_test=neg_test,
        device=device,
        hidden=cfg.hidden,
        epochs=cfg.probe_epochs,
    )

    eps = 1e-8

    # Warm-start improvements (upper bounds)
    delta_ws_n2l = LP_F2H_test_auc - LP_base_test_auc
    delta_ws_l2n = NC_H2F_test_acc - NC_base_test_acc

    # Probe improvements
    delta_probe_n2l = T_n2l_test_auc - LP_base_test_auc
    delta_probe_l2n = T_l2n_test_acc - NC_base_test_acc

    # Normalize by warm-start deltas (handle tiny / zero with signed eps)
    U_n2l = np.sign(delta_ws_n2l) * max(abs(delta_ws_n2l), eps)
    U_l2n = np.sign(delta_ws_l2n) * max(abs(delta_ws_l2n), eps)

    C_n2l = delta_probe_n2l / U_n2l
    C_l2n = delta_probe_l2n / U_l2n

    probecompat = 0.5 * (C_n2l + C_l2n)

    return {
        "Probe_T_n2l_test_auc": float(T_n2l_test_auc),
        "Probe_T_l2n_test_acc": float(T_l2n_test_acc),
        "Probe_Delta_WS_n2l": float(delta_ws_n2l),
        "Probe_Delta_WS_l2n": float(delta_ws_l2n),
        "Probe_Delta_Probe_n2l": float(delta_probe_n2l),
        "Probe_Delta_Probe_l2n": float(delta_probe_l2n),
        "Probe_C_n2l": float(C_n2l),
        "Probe_C_l2n": float(C_l2n),
        "ProbeCompat": float(probecompat),
    }
