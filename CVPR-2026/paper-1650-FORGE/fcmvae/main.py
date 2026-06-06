import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split

try:
    from .data import FCMDataset, load_mat_fcm_dataset
    from .fcmvae import VAEGraphGate
    from .train import evaluate, set_seed, train_one_epoch
except ImportError:
    from data import FCMDataset, load_mat_fcm_dataset
    from fcmvae import VAEGraphGate
    from train import evaluate, set_seed, train_one_epoch


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    mat_path_graph: str = "./data/graph.mat"
    start_idx: int = 0
    count: int = -1
    n_node: int = 116
    latent_dim: int = 116
    K: int = 16
    enc_d_model: int = 256
    enc_heads: int = 4
    enc_layers: int = 2
    enc_k_pe: int = 16
    enc_d_ff: int = 512
    enc_dropout: float = 0.1
    enc_out_dim: int = 256
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta_kl: float = 2.0
    epochs: int = 100
    train_ratio: float = 0.8
    checkpoint_path: str = ""
    export_npz_path: str = ""
    export_num_samples: int = -1
    export_temperature: float = 1.0
    export_adj_threshold: float = 0.4
    export_use_conditional: bool = True


def main(cfg: Config = None):
    cfg = cfg or Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    x_roi, y, a = load_mat_fcm_dataset(cfg.mat_path_graph, cfg.start_idx, cfg.count)
    dataset = FCMDataset(x_roi, y, a)
    label_values, _ = torch.sort(torch.unique(y))
    n_total = len(dataset)
    n_train = max(1, int(cfg.train_ratio * n_total))
    n_train = min(n_train, n_total - 1) if n_total > 1 else 1
    n_val = n_total - n_train
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator) if n_val > 0 else (dataset, dataset)
    if hasattr(train_set, "indices"):
        train_indices = torch.as_tensor(train_set.indices, dtype=torch.long)
        x_train = dataset.X_roi[train_indices]
        y_train = dataset.y[train_indices]
        a_train = dataset.A[train_indices]
    else:
        x_train, y_train, a_train = dataset.X_roi, dataset.y, dataset.A
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    model = VAEGraphGate(
        n_node=cfg.n_node,
        latent_dim=cfg.latent_dim,
        k_rank=cfg.K,
        enc_d_model=cfg.enc_d_model,
        enc_heads=cfg.enc_heads,
        enc_layers=cfg.enc_layers,
        enc_k_pe=cfg.enc_k_pe,
        enc_d_ff=cfg.enc_d_ff,
        enc_dropout=cfg.enc_dropout,
        enc_out_dim=cfg.enc_out_dim,
        num_classes=int(label_values.numel()),
    ).to(device)
    model.set_label_values(label_values)
    fisher_stats = model.fit_fisher_statistics(x_train)
    print(f"Fitted Fisher statistics: mu={fisher_stats['fisher_mu']:.6f}, sigma={fisher_stats['fisher_sigma']:.6f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, beta_kl=cfg.beta_kl)
        val_loss, trait_mse = evaluate(model, val_loader, device, beta_kl=cfg.beta_kl)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | trait_MSE={trait_mse:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    stats = model.fit_conditional_statistics(y_train, a_train)
    print(f"Fitted conditional statistics for labels: {stats['labels'].tolist()}")
    if cfg.checkpoint_path:
        os.makedirs(os.path.dirname(cfg.checkpoint_path) or ".", exist_ok=True)
        torch.save({"model_state": model.state_dict(), "config": vars(cfg), "best_val": best_val}, cfg.checkpoint_path)
        print(f"Saved checkpoint to {cfg.checkpoint_path}")
    if cfg.export_npz_path:
        num_samples = len(x_train) if cfg.export_num_samples is None or cfg.export_num_samples < 0 else cfg.export_num_samples
        info = model.export_npz(
            cfg.export_npz_path,
            num_samples=num_samples,
            temperature=cfg.export_temperature,
            conditional=cfg.export_use_conditional,
            adj_threshold=cfg.export_adj_threshold,
            seed=cfg.seed,
        )
        print(f"Exported synthetic samples to {info['path']}")
        print(f"Export summary: {info['label_hist']}")
    return model


if __name__ == "__main__":
    main()
