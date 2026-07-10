import torch
import geomloss


class OracleLJE:
    def __init__(self, fabric, model, loss_fun, opt):
        self.name = "LJE oracle"  # low-joint-error
        print(f"Initializing Oracle as {self.name}")
        self.loss_fun = loss_fun
        self.opt = opt
        model, self.opt = fabric.setup(model, self.opt)

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        pred_source = model(X_source)
        loss = self.loss_fun(pred_source, y_source)
        if len(y_target) > 0:
            loss += self.loss_fun(model(X_target), y_target)
        else:
            print("No target labels provided to LJE oracle!")
        fabric.backward(loss)
        self.opt.step()
        self.opt.zero_grad()


class OracleCC:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.name = "CC oracle"  # close-conditionals
        print(f"Initializing Oracle as {self.name}")
        self.loss_fun = loss_fun
        self.opt = opt
        self.reg = config["entropy_reg"]
        self.p = config["norm"]
        self.mode = config["mode"]  # joint vs. weighted-joint vs. conditional
        self.add_source_loss = config["add_source_loss"]
        model, self.opt = fabric.setup(model, self.opt)

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        pred_source = model(X_source)
        loss = self.loss_fun(pred_source, y_source)
        if len(y_target) == 0:
            raise Exception("No target labels provided to CC oracle")
        pred_target = model(X_target)
        if self.mode == "conditional":
            loss += self.calc_max_cond_wrr_dist(
                model, fabric, pred_source, y_source, pred_target, y_target
            )
        elif self.mode == "joint":
            loss += self.calc_joint_wrr_dist(
                model, fabric, pred_source, y_source, pred_target, y_target
            )
        elif self.mode == "weighted_joint":
            dist = self.calc_weighted_joint_wrr_dist(
                model, fabric, pred_source, y_source, pred_target, y_target
            )
            if self.add_source_loss:
                loss += dist
            else:
                loss = dist
        fabric.backward(loss)
        self.opt.step()
        self.opt.zero_grad()

    def calc_weighted_joint_wrr_dist(
        self, model, fabric, pred_source, y_source, pred_target, y_target
    ):
        full_source = torch.cat((pred_source, y_source), dim=1)
        full_target = torch.cat((pred_target, y_target), dim=1)
        cost_mat = torch.cdist(full_source, full_target, 2)
        source_losses = self.loss_fun(pred_source, y_source, reduction="none")
        cost_mat = cost_mat + source_losses[:, None]

        num_target = y_target.shape[0]
        w_target = torch.ones(num_target, device=fabric.device) / num_target
        ot_mat = torch.softmax(-cost_mat / self.reg, dim=0) * w_target[None, :]

        return torch.sum(ot_mat * cost_mat)

    def calc_joint_wrr_dist(
        self, model, fabric, pred_source, y_source, pred_target, y_target
    ):
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        full_source = torch.cat((pred_source, y_source), dim=1)
        full_target = torch.cat((pred_target, y_target), dim=1)
        return ot_loss(full_source, full_target)

    def calc_max_cond_wrr_dist(
        self, model, fabric, pred_source, y_source, pred_target, y_target
    ):
        w_dist = torch.zeros(model.num_classes, device=fabric.device)
        num_min_per_class = 5
        for i in range(model.num_classes):
            x_class_source = pred_source[y_source.argmax(dim=1) == i]
            x_class_target = pred_target[y_target.argmax(dim=1) == i]
            if (
                x_class_source.shape[0] > num_min_per_class
                and x_class_target.shape[0] > num_min_per_class
            ):
                w_dist[i] = self.calc_w_distance(x_class_source, x_class_target)
        return torch.max(w_dist)

    def calc_w_distance(self, x_source, x_target):
        ot_loss = geomloss.SamplesLoss("sinkhorn", p=self.p, blur=self.reg)
        w_dist = ot_loss(x_source, x_target)
        return w_dist
