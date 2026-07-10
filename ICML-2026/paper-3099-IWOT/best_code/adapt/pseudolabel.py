import torch

import linkage


# Hierarchical linkage clustering based pseudolabeler
class Pseudolabel:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = loss_fun
        self.name = "PL"
        self.opt = opt
        self.linkage = config["linkage"]
        model, self.opt = fabric.setup(model, self.opt)

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)

        num_targets = pred_target.shape[0]
        pred_source_cond = []
        for i in range(model.num_classes):
            cond = pred_source[torch.argmax(y_source, dim=1) == i]
            pred_source_cond.append(cond)
        Z = linkage.compute_cluster(pred_source_cond, pred_target, method=self.linkage)
        y_pseudo = linkage.compute_pseudolabels(
            Z, num_targets, model.num_classes, soft=False
        )
        pseudo_target_loss = self.loss_fun(pred_target, y_pseudo)

        loss = source_loss + pseudo_target_loss
        fabric.backward(loss)
        self.opt.step()
