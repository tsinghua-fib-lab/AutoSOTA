import torch
import ot

import linkage


def optim_linkage(
    X_source, X_target, y_source, thresh, theta0=None, method="single", soft=False
):
    dim = X_source.shape[-1]
    if theta0 is not None:
        theta = torch.clone(theta0).detach()
    else:
        theta = torch.zeros(dim)
    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    loss = torch.nn.MSELoss()
    prev_loss = 0.0
    total_loss = 100.0

    while abs(total_loss - prev_loss) > thresh:
        prev_loss = total_loss
        pred_source = X_source @ theta
        pred_target = X_target @ theta

        pred1_source = pred_source[: num_source // 2]
        pred2_source = pred_source[num_source // 2 :]
        pred1_target = pred_target[: num_target // 2]
        pred2_target = pred_target[num_target // 2 :]
        source_feat_cond = [pred1_source[:, None], pred2_source[:, None]]
        target_feat = torch.hstack((pred1_target, pred2_target))[:, None]
        Z = linkage.compute_cluster(source_feat_cond, target_feat, method)
        # linkage.plot_cluster(Z, num_targets=target_feat.shape[0], num_classes=2)
        y_pseudo = linkage.compute_pseudolabels(
            Z, num_targets=target_feat.shape[0], num_classes=2, soft=soft
        )[:, -1]
        # Scale 0,1 to 1, -1
        y_pseudo = -2 * y_pseudo + 1

        X_full = torch.vstack((X_source, X_target))
        y_full = torch.hstack((y_source, y_pseudo))
        theta, res, _, _ = torch.linalg.lstsq(X_full, y_full)
        total_loss = loss(X_full @ theta, y_full)
        # theta, res, _, _ = torch.linalg.lstsq(X_target, y_pseudo)
        # total_loss = loss(X_target @ theta, y_pseudo)
        print(f"Total loss: {total_loss}")
    return (theta, X_target @ theta)


# This is just for debugging
def optim_input_linkage(X_source, X_target, y_source, method="single", soft=False):
    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    loss = torch.nn.MSELoss()
    x1_source = X_source[: num_source // 2, 1:]
    x2_source = X_source[num_source // 2 :, 1:]
    x1_target = X_target[: num_target // 2, 1:]
    x2_target = X_target[num_target // 2 :, 1:]
    source_cond = [x1_source, x2_source]
    target_inputs = torch.vstack((x1_target, x2_target))
    Z = linkage.compute_cluster(source_cond, target_inputs, method)
    # linkage.plot_cluster(Z, num_targets=target_feat.shape[0], num_classes=2)
    y_pseudo = linkage.compute_pseudolabels(Z, num_target, num_classes=2, soft=soft)[
        :, -1
    ]
    # Scale 0,1 to 1, -1
    y_pseudo = -2 * y_pseudo + 1

    X_full = torch.vstack((X_source, X_target))
    y_full = torch.hstack((y_source, y_pseudo))
    theta, res, _, _ = torch.linalg.lstsq(X_full, y_full)
    total_loss = loss(X_full @ theta, y_full)
    # theta, res, _, _ = torch.linalg.lstsq(X_target, y_pseudo)
    # total_loss = loss(X_target @ theta, y_pseudo)
    print(f"Total loss: {total_loss}")
    return (theta, X_target @ theta)


def optim_wrr_sgd(X_source, X_target, y_source, thresh, theta0=None):
    # Optimize the Wasserstein-regularized Risk (WRR)
    dim = X_source.shape[-1]
    if theta0 is not None:
        theta = theta0.clone().detach()
    else:
        theta = torch.zeros(dim)
    theta.requires_grad = True
    opt = torch.optim.Adam([theta], lr=1e-3)
    loss = torch.nn.MSELoss()
    # num_source = X_source.shape[0]
    # num_target = X_target.shape[0]
    # w_source = torch.ones(num_source) / num_source
    # w_target = torch.ones(num_target) / num_target

    prev_loss = 0.0
    total_loss = 100.0
    while abs(total_loss - prev_loss) > thresh:
        prev_loss = total_loss
        pred_source = X_source @ theta
        pred_target = X_target @ theta
        Gamma = ot.emd_1d(pred_source, pred_target)
        # ot_dist = ot_loss(w_source, pred_source, w_target, pred_target)
        source_loss = loss(pred_source, y_source)

        # Compute the WRR bound for squared distance
        cost_mat = ot.utils.euclidean_distances(
            pred_source[:, torch.newaxis], pred_target[:, torch.newaxis], squared=True
        )
        ot_dist = torch.sum(Gamma * cost_mat)
        total_loss = source_loss + 0.5 * ot_dist
        total_loss.backward()

        print(
            f"Total loss: {total_loss}, source loss: {source_loss}, ot dist: {ot_dist}, theta: {theta.detach().numpy()}"
        )
        opt.step()
        opt.zero_grad()
    theta.requires_grad = False
    return (theta, X_target @ theta)


def optim_wrr_iter_ls(X_source, X_target, y_source, thresh, theta0=None):
    dim = X_source.shape[-1]
    if theta0 is not None:
        theta = torch.clone(theta0).detach()
    else:
        theta = torch.zeros(dim)
    X_pq = torch.cat((X_source, X_target))
    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    loss = torch.nn.MSELoss()
    prev_loss = 0.0
    total_loss = 100.0
    w_source = torch.ones(num_source) / num_source
    w_target = torch.ones(num_target) / num_target
    scale_bound = 0.5

    while abs(total_loss - prev_loss) > thresh:
        prev_loss = total_loss
        pred_source = X_source @ theta
        pred_target = X_target @ theta
        Gamma = ot.emd_1d(pred_source, pred_target)
        M_pq = torch.cat(
            (
                torch.cat((torch.diag(w_source), -Gamma), dim=1),
                torch.cat((-Gamma.T, torch.diag(w_target)), dim=1),
            ),
            dim=0,
        )
        R = num_source * (X_pq.T @ M_pq @ X_pq) / scale_bound
        theta = torch.linalg.solve(X_source.T @ X_source + R, X_source.T @ y_source)
        source_loss = loss(pred_source, y_source)

        # Compute the WRR bound for squared distance
        cost_mat = ot.utils.euclidean_distances(
            pred_source[:, torch.newaxis], pred_target[:, torch.newaxis], squared=True
        )
        ot_cost = torch.sum(Gamma * cost_mat)
        total_loss = source_loss + scale_bound * ot_cost

        print(
            f"Total loss: {total_loss}, source loss: {source_loss}, ot dist: {ot_cost}, theta: {theta.detach().numpy()}"
        )
    return (theta, X_target @ theta)
