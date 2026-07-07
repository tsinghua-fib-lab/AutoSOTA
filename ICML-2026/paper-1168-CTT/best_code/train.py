# -----------------------------------------------------------------------------#
# Training:
# -----------------------------------------------------------------------------#

from models import *

"""
Selection score for early stopping / lambda sweep.
Default: simple convex combination of val metrics.
"""
def _val_joint_score(val_acc: float, val_auc: float, alpha: float = 0.5,) -> float:
    return float(alpha * val_acc + (1.0 - alpha) * val_auc)


def _make_edge_batch(pos_edges: np.ndarray, neg_edges: np.ndarray, device: torch.device) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    pos = torch.from_numpy(pos_edges.T).long().to(device)  # (2, P)
    neg = torch.from_numpy(neg_edges.T).long().to(device)  # (2, N)

    edge_pairs = torch.cat([pos, neg], dim=1)  # (2, P+N)

    labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))], dim=0,).float().to(device)

    return edge_pairs, labels


# -----------------------------------------------------------------------------#
# Training loops (with optional x_override & in_channels_override)
# -----------------------------------------------------------------------------#

"""
Train node classifier F on A_train.

Returns:
    model, best val acc, final test acc.

Notes:
    - If init_encoder_state is provided, encoder weights are initialized from it (for H->F warm start).
    - If x_override is provided, it is used instead of data.x.
    - If in_channels_override is provided, it is used to build the encoder.
"""
def train_node_model(data, edge_index_train: torch.Tensor, num_features: int, num_classes: int,
                     train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor,
                     device: torch.device, hidden: int = 64, num_layers: int = 2, dropout: float = 0.5,
                     epochs: int = 200, patience: int = 50,
                     init_encoder_state: Optional[Dict[str, torch.Tensor]] = None,
                     x_override: Optional[torch.Tensor] = None,
                     in_channels_override: Optional[int] = None,
                     ) -> Tuple[NodeClassifier, float, float]:
    in_ch = in_channels_override if in_channels_override is not None else num_features

    model = NodeClassifier(
        in_channels=in_ch,
        hidden_channels=hidden,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    if init_encoder_state is not None:
        model.encoder.load_state_dict(init_encoder_state)

    data = data.to(device)
    edge_index_train = edge_index_train.to(device)

    if x_override is not None:
        x_override = x_override.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        if x_override is not None:
            x_in = x_override
        else:
            x_in = data.x

        out = model(x_in, edge_index_train)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_in, edge_index_train)
            pred = logits.argmax(dim=-1)
            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        if x_override is not None:
            x_in = x_override
        else:
            x_in = data.x

        logits = model(x_in, edge_index_train)
        pred = logits.argmax(dim=-1)
        test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

    return model, best_val_acc, test_acc

"""
Train link predictor H on A_train.

Returns:
    model, best val AUC, final test AUC.

Notes:
    - If init_encoder_state is provided, encoder weights are initialized from it (for F->H warm start).
    - If x_override is provided, it is used instead of data.x.
    - If in_channels_override is provided, it is used to build the encoder.
"""
def train_link_model(data, edge_index_train: torch.Tensor, num_features: int,
                     pos_train: np.ndarray, neg_train: np.ndarray, pos_val: np.ndarray,
                     neg_val: np.ndarray, pos_test: np.ndarray, neg_test: np.ndarray,
                     device: torch.device, hidden: int = 64, num_layers: int = 2, dropout: float = 0.5,
                     epochs: int = 200, patience: int = 50,
                     init_encoder_state: Optional[Dict[str, torch.Tensor]] = None,
                     x_override: Optional[torch.Tensor] = None,
                     in_channels_override: Optional[int] = None,
                     ) -> Tuple[LinkModel, float, float]:
    in_ch = in_channels_override if in_channels_override is not None else num_features

    model = LinkModel(
        in_channels=in_ch,
        hidden_channels=hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    if init_encoder_state is not None:
        model.encoder.load_state_dict(init_encoder_state)

    data = data.to(device)
    edge_index_train = edge_index_train.to(device)

    if x_override is not None:
        x_override = x_override.to(device)

    train_pairs, train_labels = _make_edge_batch(pos_train, neg_train, device)
    val_pairs, val_labels = _make_edge_batch(pos_val, neg_val, device)
    test_pairs, test_labels = _make_edge_batch(pos_test, neg_test, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        if x_override is not None:
            x_in = x_override
        else:
            x_in = data.x

        z = model.encoder(x_in, edge_index_train)
        preds = model.lp_head(z, train_pairs)

        loss = F.binary_cross_entropy(preds, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encoder(x_in, edge_index_train)
            val_scores = model.lp_head(z, val_pairs).detach().cpu().numpy()
            val_labels_np = val_labels.detach().cpu().numpy()
            val_auc = roc_auc_score(val_labels_np, val_scores)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        if x_override is not None:
            x_in = x_override
        else:
            x_in = data.x

        z = model.encoder(x_in, edge_index_train)
        test_scores = model.lp_head(z, test_pairs).detach().cpu().numpy()
        test_labels_np = test_labels.detach().cpu().numpy()
        test_auc = roc_auc_score(test_labels_np, test_scores)

    return model, best_val_auc, test_auc

# -----------------------------------------------------------------------------#
# Joint Head: shared encoder + NC head + LP head
# -----------------------------------------------------------------------------#
"""
Joint training baseline on shared encoder.

mode="lambda":
    minimize lam * L_NC + (1-lam) * L_LP

mode="uncertainty":
    Kendall-style weighting:
        L = (1/(2 sigma_N^2)) L_NC + (1/(2 sigma_L^2)) L_LP + log sigma_N + log sigma_L

Early stopping uses a validation selection score combining NC val acc and LP val AUC.

Returns a dict with best-val-selected test metrics and the chosen settings.
"""
def train_joint_model(data, edge_index_train: torch.Tensor, num_features: int, num_classes: int,
                      train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor,
                      pos_train: np.ndarray, neg_train: np.ndarray, pos_val: np.ndarray,
                      neg_val: np.ndarray, pos_test: np.ndarray, neg_test: np.ndarray,
                      device: torch.device, hidden: int = 64, num_layers: int = 2, dropout: float = 0.5,
                      epochs: int = 200, patience: int = 50,
                      mode: str = "lambda",  # {"lambda", "uncertainty"}
                      lam: float = 0.5,      # used if mode=="lambda"
                      alpha_val_select: float = 0.5,  # weight for val selection score
                      ) -> Dict[str, float]:
    assert mode in {"lambda", "uncertainty"}
    if mode == "lambda":
        assert 0.0 <= lam <= 1.0

    data = data.to(device)
    edge_index_train = edge_index_train.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Fixed edge batches
    train_pairs, train_labels = _make_edge_batch(pos_train, neg_train, device)
    val_pairs, val_labels = _make_edge_batch(pos_val, neg_val, device)
    test_pairs, test_labels = _make_edge_batch(pos_test, neg_test, device)

    model = JointNC_LP(
        in_channels=num_features,
        hidden_channels=hidden,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        lp_hidden=hidden,
    ).to(device)

    # Optional uncertainty parameters
    if mode == "uncertainty":
        # Use log_sigmas for numerical stability; sigma = exp(log_sigma)
        log_sigma_nc = nn.Parameter(torch.zeros((), device=device))
        log_sigma_lp = nn.Parameter(torch.zeros((), device=device))
        params = list(model.parameters()) + [log_sigma_nc, log_sigma_lp]
    else:
        log_sigma_nc = None
        log_sigma_lp = None
        params = list(model.parameters())

    optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=5e-4)

    best_score = -1e18
    best_state = None
    best_extra = {}
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        x_in = data.x

        # Forward once; reuse z for both heads
        _, nc_logits, lp_scores = model(x_in, edge_index_train, train_pairs)

        # NC loss (only on train nodes)
        loss_nc = F.cross_entropy(nc_logits[train_mask], data.y[train_mask])

        # LP loss (fixed positives+negatives)
        loss_lp = F.binary_cross_entropy(lp_scores, train_labels)

        if mode == "lambda":
            loss = lam * loss_nc + (1.0 - lam) * loss_lp
        else:
            # sigma = exp(log_sigma), sigma^2 = exp(2 log_sigma)
            sigma2_nc = torch.exp(2.0 * log_sigma_nc)
            sigma2_lp = torch.exp(2.0 * log_sigma_lp)
            loss = (
                (0.5 / sigma2_nc) * loss_nc
                + (0.5 / sigma2_lp) * loss_lp
                + log_sigma_nc
                + log_sigma_lp
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
        optimizer.step()

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            z, nc_logits_val, _ = model(data.x, edge_index_train, None)
            pred = nc_logits_val.argmax(dim=-1)
            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()

            val_scores = model.lp_head(z, val_pairs).detach().cpu().numpy()
            val_labels_np = val_labels.detach().cpu().numpy()
            val_auc = roc_auc_score(val_labels_np, val_scores)

            score = _val_joint_score(val_acc, val_auc, alpha=alpha_val_select)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0

            if mode == "uncertainty":
                best_extra = {
                    "log_sigma_nc": float(log_sigma_nc.detach().cpu().item()),
                    "log_sigma_lp": float(log_sigma_lp.detach().cpu().item()),
                }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test ----
    model.eval()
    with torch.no_grad():
        _, nc_logits_test, _ = model(data.x, edge_index_train, None)
        pred = nc_logits_test.argmax(dim=-1)
        test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

        z, _, _ = model(data.x, edge_index_train, None)
        test_scores = model.lp_head(z, test_pairs).detach().cpu().numpy()
        test_labels_np = test_labels.detach().cpu().numpy()
        test_auc = roc_auc_score(test_labels_np, test_scores)

    out = {
        "NC_joint_test_acc": float(test_acc),
        "LP_joint_test_auc": float(test_auc),
        "Joint_val_select_score": float(best_score),
        "Joint_mode": 0.0 if mode == "lambda" else 1.0,  # numeric for aggregation if you want
    }

    if mode == "lambda":
        out["Joint_lambda"] = float(lam)
    else:
        out.update(best_extra)

    return out

"""
Grid-search lambda and return the best by the same val selection score.
"""
def train_joint_lambda_sweep(data, edge_index_train: torch.Tensor, num_features: int, num_classes: int,
                             train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor,
                             pos_train: np.ndarray, neg_train: np.ndarray, pos_val: np.ndarray,
                             neg_val: np.ndarray, pos_test: np.ndarray, neg_test: np.ndarray,
                             device: torch.device, hidden: int, num_layers: int, dropout: float,
                             epochs: int, patience: int, lambdas: List[float], alpha_val_select: float = 0.5
                             ) -> Dict[str, float]:

    best = None
    best_score = -1e18
    best_lam = None

    for lam in lambdas:
        res = train_joint_model(
            data=data,
            edge_index_train=edge_index_train,
            num_features=num_features,
            num_classes=num_classes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            pos_train=pos_train,
            neg_train=neg_train,
            pos_val=pos_val,
            neg_val=neg_val,
            pos_test=pos_test,
            neg_test=neg_test,
            device=device,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            patience=patience,
            mode="lambda",
            lam=float(lam),
            alpha_val_select=alpha_val_select,
        )

        if res["Joint_val_select_score"] > best_score:
            best_score = res["Joint_val_select_score"]
            best = res
            best_lam = lam

    assert best is not None
    best["Joint_best_lambda"] = float(best_lam)
    return best