import math
import time

import numpy as np
import torch

from dataprocessing import get_multiview_data
from loss import MultiPathConsensusLoss


def pre_train(network_model, mv_data, batch_size, epochs, optimizer):
    """Pretrain view-specific autoencoders with reconstruction loss only."""
    t = time.time()
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        network_model.train()
        network_model.set_impute(False)

        for _, (sub_data_views, _) in enumerate(mv_data_loader):
            dvs, _ = network_model.reconstruct_only(sub_data_views)

            loss = 0.0
            for v in range(num_views):
                mask = (
                    (sub_data_views[v].abs().sum(dim=1) > 1e-6)
                    & (~torch.isnan(sub_data_views[v]).any(dim=1))
                ).unsqueeze(1).float()

                loss = loss + criterion(sub_data_views[v] * mask, dvs[v] * mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Pre-training, epoch {epoch + 1}, loss={total_loss / num_samples:.7f}")

    print("Pre-training finished. Time: {:.2f}s".format(time.time() - t))
    return None


def contrastive_train(
    network_model,
    mv_data,
    mvc_loss,
    batch_size,
    beta,
    gamma,
    temperature_l,
    normalized,
    epoch,
    optimizer,
    epoch_open_ot=30,
    epoch_open_knn=50,
    impute_th_start=0.55,
    impute_th_end=0.55,
    inf_temp=0.2,
    inf_lambda_fuse=1.0,
    inf_lambda_uni=1.0,
    inf_lambda_mask=1.0,
    inf_lambda_cons=0.05,
):
    """Train MAGIC with multi-path consensus, semantic alignment, and reconstruction."""
    network_model.train()
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()

    if epoch < epoch_open_ot:
        network_model.set_impute(False)
    else:
        network_model.set_impute(True)
        network_model.set_impute_knn(epoch >= epoch_open_knn)

        progress = min(
            1.0,
            max(0.0, (epoch - epoch_open_ot) / max(1, epoch_open_knn - epoch_open_ot)),
        )
        threshold = impute_th_start * (1.0 - progress) + impute_th_end * progress
        network_model.set_impute_conf(threshold)

    consensus_loss = MultiPathConsensusLoss(
        temperature=inf_temp,
        lambda_fuse=inf_lambda_fuse,
        lambda_uni=inf_lambda_uni,
        lambda_mask=inf_lambda_mask,
        lambda_cons=inf_lambda_cons,
    )

    total_loss = 0.0

    for _, (sub_data_views, _) in enumerate(mv_data_loader):
        lbps, dvs, _, _, view_sims, aug = network_model(
            sub_data_views,
            return_aug=True,
        )

        use_consensus = not (
            inf_lambda_fuse == 0
            and inf_lambda_uni == 0
            and inf_lambda_mask == 0
            and inf_lambda_cons == 0
        )

        if use_consensus:
            loss_items = consensus_loss(aug)
            loss = gamma * loss_items["loss_total"]
        else:
            loss = torch.tensor(0.0, device=sub_data_views[0].device)

        view_sim_idx = 0
        for i in range(num_views):
            for j in range(i + 1, num_views):
                current_view_sim = view_sims[view_sim_idx]

                loss = loss + current_view_sim * beta * mvc_loss.forward_label(
                    lbps[i],
                    lbps[j],
                    temperature_l,
                    normalized,
                )

                loss = loss + current_view_sim * beta * mvc_loss.forward_prob(
                    lbps[i],
                    lbps[j],
                )

                view_sim_idx += 1

        for v in range(num_views):
            mask = (
                (sub_data_views[v].abs().sum(dim=1) > 1e-6)
                & (~torch.isnan(sub_data_views[v]).any(dim=1))
            ).unsqueeze(1).float()

            loss = loss + criterion(sub_data_views[v] * mask, dvs[v] * mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss_per_sample = total_loss / num_samples
    print(
        f"Contrastive training, epoch {epoch + 1}, "
        f"loss={loss_per_sample:.7f}, "
        f"impute={network_model.impute_enabled}, "
        f"knn={network_model.impute_enable_knn}"
    )

    if (epoch + 1) % 10 == 0:
        acc, nmi, pur, ari = valid(network_model, mv_data, batch_size)
        print(
            f"[Eval] epoch {epoch + 1}: "
            f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f}"
        )

    return total_loss


def inference(network_model, mv_data, batch_size):
    network_model.eval()
    mv_data_loader, num_views, _, _ = get_multiview_data(mv_data, batch_size)

    pred_vectors = [[] for _ in range(num_views)]
    soft_vector = []
    labels_vector = []

    for _, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
        with torch.no_grad():
            lbps, _, _, _, _ = network_model(sub_data_views)

            def confidence(prob):
                entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=1).mean()
                num_clusters = prob.size(1)
                return float(1.0 - entropy.item() / math.log(num_clusters + 1e-12))

            weights = torch.tensor(
                [confidence(prob) for prob in lbps],
                device=lbps[0].device,
                dtype=lbps[0].dtype,
            )
            weights = weights / (weights.sum() + 1e-12)

            log_probs = torch.stack([torch.log(prob + 1e-12) for prob in lbps], dim=0)
            combined = (weights.view(-1, 1, 1) * log_probs).sum(dim=0)
            fused_prob = torch.softmax(combined, dim=1)

        for v in range(num_views):
            pred_label = torch.argmax(lbps[v], dim=1)
            pred_vectors[v].extend(pred_label.detach().cpu().numpy())

        soft_vector.extend(fused_prob.detach().cpu().numpy())
        labels_vector.extend(sub_labels)

    for v in range(num_views):
        pred_vectors[v] = np.array(pred_vectors[v])

    labels_vector = np.array(labels_vector).reshape(len(soft_vector))
    total_pred = np.argmax(np.array(soft_vector), axis=1)

    return total_pred, pred_vectors, labels_vector


def valid(network_model, mv_data, batch_size):
    from metrics import calculate_metrics

    total_pred, _, labels_vector = inference(network_model, mv_data, batch_size)

    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
    print(f"ACC = {acc:.4f} NMI = {nmi:.4f} PUR = {pur:.4f} ARI = {ari:.4f}")

    return acc, nmi, pur, ari