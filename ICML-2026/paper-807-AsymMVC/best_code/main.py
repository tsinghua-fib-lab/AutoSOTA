import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from model import HAMC_Model, mse_loss, contrastive_loss_euclidean, sinkhorn_knopp
from utils import data_load, cluster_acc, set_seed, calculate_entropy

def run_warmup(model, dataloader, args, ckpt_path):
    """ Phase 1: Warmup Training or Loading """
    if os.path.exists(ckpt_path) and not args.force_warmup:
        print(f">>> Found cached warmup checkpoint: {ckpt_path}")
        print(">>> Loading weights and SKIPPING Phase 1...")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print(">>> Phase 1: Warmup (Euclidean + Reconstruction)")
        optimizer_warm = optim.Adam(model.parameters(), lr=args.warmup_lr)

        for epoch in range(args.warmup_epochs):
            loss_warm_c = 0
            loss_warm_r = 0
            for x0, x1 in dataloader:
                optimizer_warm.zero_grad()
                vs, _, xs_rec = model([x0, x1])
                c_loss = contrastive_loss_euclidean(vs[0], vs[1])
                r_loss = mse_loss(xs_rec[0], x0) + mse_loss(xs_rec[1], x1)
                loss = args.warm_c_weight * c_loss + r_loss
                loss.backward()
                optimizer_warm.step()
                loss_warm_c += args.warm_c_weight * c_loss.item()
                loss_warm_r += r_loss.item()

            print(f"Warmup Epoch {epoch+1}: Loss_c {loss_warm_c:.4f}/Loss_r {loss_warm_r:.4f}")

        torch.save(model.state_dict(), ckpt_path)
        print(f">>> Warmup finished. Weights saved to {ckpt_path}")

def run_init_prototypes(model, full_dataloader, Y, args):
    """ Phase 2: Prototypes Initialization """
    print(">>> Phase 2: Init Prototypes")
    with torch.no_grad():
        all_v0 = []
        all_v1 = []
        for x0, x1 in full_dataloader:
            vs, _, _ = model([x0, x1])
            all_v0.append(vs[0].cpu().numpy())
            all_v1.append(vs[1].cpu().numpy())

        all_v0 = np.concatenate(all_v0, axis=0)
        all_v1 = np.concatenate(all_v1, axis=0)

        v0_norm = normalize(all_v0, axis=1)
        v1_norm = normalize(all_v1, axis=1)

        feat = (v0_norm + v1_norm) / 2
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(feat)
        yp = kmeans.predict(feat)

        acc = cluster_acc(Y, yp)
        print(f"Init K-Means - ACC: {acc:.4f}")
        init_centers = torch.tensor(kmeans.cluster_centers_[:, :args.latent_dim]).float().to(args.device)
        model.prototypes.data = F.normalize(init_centers, dim=1)
        print(f"Prototypes initialized.")

def run_evaluation(model, full_dataloader, args, epoch=0):
    """ Independent Evaluation Function """
    model.eval()
    all_preds = []
    with torch.no_grad():
        proto_eval = model.get_hyp_prototypes()
        for x0, x1 in full_dataloader:
            x0, x1 = x0.to(args.device), x1.to(args.device)
            _, zs, _ = model([x0, x1])
            d = (torch.stack([model.hyp.dist(z, proto_eval) for z in zs[0]]) +
                 torch.stack([model.hyp.dist(z, proto_eval) for z in zs[1]])) / 2
            all_preds.append(torch.argmin(d, dim=1).cpu().numpy())

    y_pred = np.concatenate(all_preds)
    return y_pred

def run_training(model, dataloader, full_dataloader, args, Y_true, best_ckpt_path):
    """ Phase 3: Main Training Loop """
    print(">>> Phase 3: Training ")
    print(f">>> ALGO-04: Quality-weighted alignment (α0·α1 weighting)")
    if args.grad_clip > 0:
        print(f">>> CODE-02: Gradient clipping enabled (max_norm={args.grad_clip})")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    hyp = model.hyp
    best_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        meters = {'loss': 0, 'clu': 0, 'rec': 0, 'align': 0}

        for x0, x1 in dataloader:
            optimizer.zero_grad()
            vs, zs, xs_rec = model([x0, x1])
            proto_hyp = model.get_hyp_prototypes()

            # 1. Distances & Sinkhorn
            dist_v0 = torch.stack([hyp.dist(z, proto_hyp) for z in zs[0]], dim=0)
            dist_v1 = torch.stack([hyp.dist(z, proto_hyp) for z in zs[1]], dim=0)
            Q0 = sinkhorn_knopp(-dist_v0)
            Q1 = sinkhorn_knopp(-dist_v1)
            H0 = calculate_entropy(Q0)
            H1 = calculate_entropy(Q1)

            # 2. Gating
            entropies = torch.stack([H0, H1], dim=1)
            alphas = F.softmax(-entropies / args.temp_gate, dim=1)
            alpha0, alpha1 = alphas[:, 0].unsqueeze(1), alphas[:, 1].unsqueeze(1)

            # 3. ALGO-04: Quality-weighted alignment
            dist_01 = hyp.dist(zs[0], zs[1].detach())
            dist_10 = hyp.dist(zs[1], zs[0].detach())
            w10 = torch.clamp(H1 - H0, min=0).detach()
            w01 = torch.clamp(H0 - H1, min=0).detach()
            align_per_sample = 0.5 * (w10 * dist_10.pow(2) + w01 * dist_01.pow(2))
            quality_weight = (alpha0 * alpha1).detach()  # down-weight uncertain views
            loss_align = (quality_weight * align_per_sample).mean()
            # 4. Clustering Target & Mask
            avg_logits = alpha0 * (-dist_v0) + alpha1 * (-dist_v1)
            Q_final = sinkhorn_knopp(avg_logits)
            y_target = torch.argmax(Q_final, dim=1)

            max_probs, _ = torch.max(Q_final, dim=1)
            k_idx = int(max_probs.shape[0] * 0.5)
            k_idx = max(1, k_idx)
            top_k_val, _ = torch.topk(max_probs, k=k_idx)
            mask = (max_probs >= top_k_val[-1]).float()

            # 5. Losses
            loss_cluster = 0
            loss_rec = 0
            for i, (dist, z, x_rec) in enumerate(zip([dist_v0, dist_v1], zs, xs_rec)):
                pos_dist = torch.gather(dist, 1, y_target.unsqueeze(1))
                log_prob = -pos_dist/args.tau - torch.log(torch.sum(torch.exp(-dist/args.tau), dim=1, keepdim=True) + 1e-8)
                loss_cluster += -(log_prob * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-8)
                loss_rec += mse_loss(x_rec, [x0, x1][i])

            total_loss = loss_rec + args.loss_clu_weight * loss_cluster + args.loss_align_weight * loss_align
            total_loss.backward()

            # CODE-02: Gradient clipping for hyperbolic training stability
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            # Meters
            meters['loss'] += total_loss.item()
            meters['clu'] += args.loss_clu_weight * loss_cluster.item()
            meters['rec'] += loss_rec.item()
            meters['align'] += args.loss_align_weight * loss_align.item()

            # 6. Momentum Update
            with torch.no_grad():
                for k in range(args.n_clusters):
                    valid_mask = ((y_target == k).float() * mask).unsqueeze(1)
                    if valid_mask.sum() > 0:
                        w0, w1 = valid_mask * alpha0, valid_mask * alpha1
                        feat_sum = (w0 * vs[0]).sum(0) + (w1 * vs[1]).sum(0)
                        weight_sum = w0.sum() + w1.sum()
                        if weight_sum > 0:
                            target = F.normalize(feat_sum / (weight_sum + 1e-8), dim=0)
                            model.prototypes[k] = args.momentum * model.prototypes[k] + (1-args.momentum) * target
                model.prototypes.data = torch.clamp(model.prototypes.data, min=-10, max=10)

        # Evaluation Period
        if epoch == 0 or (epoch + 1) % 5 == 0:
            y_pred = run_evaluation(model, full_dataloader, args, epoch)
            acc = cluster_acc(Y_true, y_pred)
            nmi = nmi_score(Y_true, y_pred)
            ari = ari_score(Y_true, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_ckpt_path)

            print(f"Training Epoch {epoch+1}: Loss: {meters['loss']:.2f}")

    print(f">>> Training Finished. Saved to {best_ckpt_path}")

def main():
    global args
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    if args.cuda:
        args.device_use = "cuda:" + str(args.device_num)
    args.device = torch.device(args.device_use if args.cuda else "cpu")
    print("USE {}".format(args.device))

    set_seed(args.seed)

    # Data Load
    X, Y = data_load(args)
    args.n_clusters = len(np.unique(Y))
    args.n_views = len(X)
    args.n_input = [X[0].shape[1], X[1].shape[1]]
    print(f"Data Loaded: {args.dataset}, Samples: {X[0].shape[0]}, Clusters: {args.n_clusters}")

    args.basis_path = "./SaveWeight/" + args.dataset + "/"

    # Paths
    if not os.path.exists(args.basis_path): os.makedirs(args.basis_path)

    warmup_ckpt = os.path.join(args.basis_path,
        f"warmup_{args.dataset}_sd{args.seed}"
        f"_dim{args.latent_dim}_bs{args.batch_size}_ep{args.warmup_epochs}_lr{args.warmup_lr}.pth")

    best_model_ckpt = os.path.join(args.basis_path,
        f"final_{args.dataset}_sd{args.seed}_tau{args.tau}_mom{args.momentum}_lr{args.lr}"
        f"_clu{args.loss_clu_weight}_dis{args.loss_align_weight}_gate{args.temp_gate}.pth")

    # Preprocessing
    scaler = MinMaxScaler() if args.dataset in ['Reuters', 'Wiki'] else StandardScaler()
    X0 = scaler.fit_transform(X[0])
    X1 = scaler.fit_transform(X[1])

    # Dataloaders
    X0_t = torch.from_numpy(np.nan_to_num(X0)).float().to(args.device)
    X1_t = torch.from_numpy(np.nan_to_num(X1)).float().to(args.device)
    dataset = TensorDataset(X0_t, X1_t)

    # Generator for reproducibility
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False
    )
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Model Init
    model = HAMC_Model(args.n_input, args.latent_dim, args.n_clusters).to(args.device)

    # --- Logic Branch ---
    if args.train:
        # [Training Mode]
        run_warmup(model, train_loader, args, warmup_ckpt)    # Phase 1
        run_init_prototypes(model, full_loader, Y, args)      # Phase 2
        run_training(model, train_loader, full_loader, args, Y, best_model_ckpt) # Phase 3

    # [Inference Mode]
    if os.path.exists(best_model_ckpt):
        print(f">>> Loading Best Model from: {best_model_ckpt}")
        model.load_state_dict(torch.load(best_model_ckpt))
        y_pred = run_evaluation(model, full_loader, args)
        acc = cluster_acc(Y, y_pred)
        nmi = nmi_score(Y, y_pred)
        ari = ari_score(Y, y_pred)
        print(f"--- Inference Result ---")
        print(f"ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f}")
    else:
        print(f"Error: Checkpoint not found at {best_model_ckpt}")
        print("Please run with --train 1 first.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HAMC Modular')
    # Environment
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CUB')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--force_warmup', default=False, type=bool)
    # Architecture
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=64)
    # Warmup
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_lr', type=float, default=5e-4)
    parser.add_argument('--warm_c_weight', type=float, default=10)
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.98)
    parser.add_argument('--tau', type=float, default=0.3)
    parser.add_argument('--temp_gate', type=float, default=0.7)
    parser.add_argument('--loss_clu_weight', type=float, default=0.1)
    parser.add_argument('--loss_align_weight', type=float, default=1.0)
    # CODE-02: Gradient clipping
    parser.add_argument('--grad_clip', type=float, default=10.0,
                        help='Gradient clipping max_norm (0 to disable)')
    main()
