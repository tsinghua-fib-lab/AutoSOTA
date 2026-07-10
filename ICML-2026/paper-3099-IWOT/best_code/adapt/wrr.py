import torch
import geomloss
import ot

import linkage


# Wasserstein Marginal Distance regularized source risk minimization using model outputs
class WRR:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = loss_fun
        self.name = "WRR"
        self.opt = opt
        self.scale = config["scale"]
        self.p = config["norm"]
        self.reg = config["entropy_reg"]
        self.print_info = config["print_info"]
        self.propagate_labels = config["propagate_labels"]
        self.compute_ultrametric = config["compute_ultrametric"]
        self.est_entanglement = config["estimate_entanglement"]
        self.softmax_temperature = config["softmax_temperature"]
        model, self.opt = fabric.setup(model, self.opt)

    def calc_ot_loss(self, f_source, f_target):
        # dist_source = torch.cdist(f_source, f_source, p=2)
        # dist_target = torch.cdist(f_target, f_target, p=2)
        # gromov_cost = torch.sqrt(ot.gromov.gromov_wasserstein2(dist_source, dist_target, symmetric=True))

        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        cost = ot_loss(f_source, f_target)
        return cost

    def calc_ot_map(self, f_source, f_target, device):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]
        w_source = torch.ones(num_source, device=device) / num_source
        w_target = torch.ones(num_target, device=device) / num_target
        cost_mat = torch.cdist(f_source, f_target, p=2)
        ot_mat = ot.emd(w_source, w_target, cost_mat, numItermax=5000)
        return ot_mat, torch.sum(ot_mat * cost_mat)

    # For now storing this code here (which doesn't work well anyway) to simplify adapt()
    def propagate_label(self, pred_source, pred_target, y_source, fabric):
        ot_map, _ = self.calc_ot_map(pred_source, pred_target, fabric.device)
        num_target = pred_target.shape[0]
        y_hat_target = (
            num_target * ot_map.T @ torch.argmax(y_source, dim=1).float()
        ).to(torch.int64)
        return self.loss_fun.forward_int(pred_target, y_hat_target)

    def estimate_entanglement(self, model, cost_mat, y_source):
        dist_to_source = torch.zeros(cost_mat.shape[-1], model.num_classes)
        for i in range(model.num_classes):
            idx = torch.argmax(y_source, dim=1) == i
            dist_to_source[:, i] = torch.sum(cost_mat[idx], dim=0)
        tau = self.softmax_temperature
        y_pseudo = torch.softmax(-dist_to_source / tau, dim=1)

        # Compute expected distances given pseudo-probabilities
        # exp_dist_mat = torch.zeros(y_source.shape[0], y_pseudo.shape[0])
        # for i in range(model.num_classes):
        #     y_label = torch.zeros_like(y_pseudo)
        #     y_label[:, i] = 1
        #     exp_dist_mat += y_pseudo[:, i] * torch.cdist(y_source, y_label)
        # return exp_dist_mat

        # Use probabilities directly to compute soft-distance
        return torch.cdist(y_source, y_pseudo)

    def compute_wrr(self, pred_source, pred_target, fabric):
        try:
            ot_cost = self.calc_ot_loss(pred_source, pred_target)
        except Exception:
            print("For some reason geomloss failed. Reverting to ot.emd routine")
            _, ot_cost = self.calc_ot_map(pred_source, pred_target, fabric.device)
        return ot_cost

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        pred_source = model(X_source)
        pred_target = model(X_target)
        num_source = pred_source.shape[0]
        num_target = pred_target.shape[0]
        source_loss = self.loss_fun(pred_source, y_source)

        if self.propagate_labels is True:
            reg_loss = self.propagate_label(pred_source, pred_target, y_source, fabric)
        elif self.compute_ultrametric is True:
            breakpoint()
            ultra_dist_mat = linkage.compute_soft_cluster(pred_source, pred_target)
            w_source = torch.ones(num_source, device=fabric.device) / num_source
            w_target = torch.ones(num_target, device=fabric.device) / num_target
            ot_mat = ot.emd(w_source, w_target, ultra_dist_mat, numItermax=5000)
            reg_loss = torch.sum(ot_mat * ultra_dist_mat)
        elif self.est_entanglement is True:
            w_source = torch.ones(num_source, device=fabric.device) / num_source
            w_target = torch.ones(num_target, device=fabric.device) / num_target
            cost_mat = torch.cdist(pred_source, pred_target, p=2)
            full_mat = cost_mat + self.estimate_entanglement(model, cost_mat, y_source)
            # ultra_dist_mat = linkage.compute_soft_cluster(pred_source, pred_target)
            # full_mat = cost_mat + self.estimate_entanglement(model, ultra_dist_mat, y_source)
            ot_mat = ot.emd(w_source, w_target, full_mat, numItermax=5000)
            reg_loss = torch.sum(ot_mat * full_mat)

            # Add regularization based on source labels
            # ot_trans = ot.da.SinkhornLpl1Transport(reg_e=0.1, reg_cl=0.1, max_iter=10, max_inner_iter=200, tol=1e-8, verbose=True, log=False, metric="euclidean")
            # ot_trans.fit(Xs=pred_source, ys=torch.argmax(y_source, dim=-1), Xt=pred_target)
            # reg_loss = torch.sum(ot_trans.coupling_ * cost_mat)

            # Add selective filtering based on distance
            # std_cost, mean_cost = torch.std_mean(full_mat)
            # thresh = mean_cost - std_cost
            # filter_mat = full_mat < thresh
            # reg_loss = torch.sum(ot_mat * filter_mat * full_mat)

            # Add selective filtering based on source margin
            # std_margin, mean_margin, margin, correct = utils.calc_margin(pred_source, y_source)
            # thresh = mean_margin + std_margin
            # filter_vec_margin = margin > thresh
            # mat = ot_mat * filter_mat * full_mat
            # reg_loss = torch.sum(mat[correct][filter_vec_margin])

        else:
            reg_loss = self.compute_wrr(pred_source, pred_target, fabric)

        total_loss = self.scale * source_loss + reg_loss
        fabric.backward(total_loss)
        self.opt.step()
        if self.print_info:
            print(
                f"WRR: {total_loss.item()}, reg_loss: {reg_loss.item()}, source_loss: {source_loss.item()}"
            )
