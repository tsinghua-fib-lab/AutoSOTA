import torch
import copy
import geomloss


class ConstrainedWRR:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = copy.deepcopy(loss_fun)
        self.name = "ConsWRR"
        self.opt = opt
        self.p = config["norm"]
        self.reg = config["entropy_reg"]
        self.thresh = config["thresh"]
        self.scale = config["scale"]
        self.mode = 0
        model, self.opt = fabric.setup(model, self.opt)

    def calc_ot(self, f_source, f_target):
        ### Python crashes regularly with POT so switching to GeomLoss
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        return ot_loss(f_source, f_target)

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):

        self.opt.zero_grad()
        pred_source = model(X_source)
        pred_target = model(X_target)
        ot_cost = self.calc_ot(pred_source, pred_target)

        if self.mode == 0:
            loss = ot_cost
            fabric.backward(loss)
        else:
            source_loss = self.loss_fun(pred_source, y_source)
            loss = source_loss + self.scale * ot_cost
            fabric.backward(loss)
        self.opt.step()

    @torch.no_grad()
    def validate(self, model, fabric, X_source, y_source, X_target):
        pred_source = model(X_source)
        pred_target = model(X_target)
        ot_cost = self.calc_ot(pred_source, pred_target)

        if ot_cost > self.thresh:
            self.mode = 0  # minimize OT
            print("Constraint threshold exceeded. Minimizing constraint")
        else:
            self.mode = 1  # minimize scaled WRR
            print("Constraint threshold satisfied. Minimizing WRR.")
