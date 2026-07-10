import torch
import ot


class JDOT:
    def __init__(
        self,
        config,
        fabric,
        model,
        loss_fun,
        opt,
    ):
        self.loss_fun = loss_fun
        self.name = "JDOT"
        self.opt = opt
        model, self.opt = fabric.setup(model, self.opt)
        self.alpha = config["alpha"]
        self.lamb = config["lambda"]
        self.add_source_loss = config["add_source_loss"]
        self.use_squared_dist = config["use_squared_dist"]
        model.track_features(config["track_layer"])

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        prob, w_dist = self.transport(
            model, X_source, X_target, y_source, fabric.device
        )
        loss = self.calc_loss(model, X_source, y_source, X_target, prob)
        fabric.backward(loss)
        self.opt.step()
        self.opt.zero_grad()

    def transport(self, model, X_source, X_target, y_source, device):
        num_source = X_source.shape[0]
        num_target = X_target.shape[0]
        # Weights of the points
        w_source = torch.ones(num_source, device=device) / num_source
        w_target = torch.ones(num_target, device=device) / num_target

        model(X_source)
        source_activations = torch.clone(model.features)
        pred_target = model(X_target)
        target_activations = torch.clone(model.features)
        cost_mat = ot.utils.euclidean_distances(
            source_activations, target_activations, squared=self.use_squared_dist
        )
        cost_mat = self.alpha * cost_mat + self.lamb * self.calc_loss_mat(
            y_source, pred_target
        )
        prob_mat = ot.emd(a=w_source, b=w_target, M=cost_mat).type(torch.float)
        if self.use_squared_dist is True:
            return prob_mat, torch.sqrt(torch.sum(prob_mat * cost_mat))
        return prob_mat, torch.sum(prob_mat * cost_mat)

    def calc_loss(self, model, X_train, y_train, X_shift, prob_mat):
        source_loss = 0.0
        val = self.loss_fun(model(X_train), y_train)
        if self.add_source_loss is True:
            source_loss += val
        source_activations = torch.clone(model.features)

        idx_source, idx_target = torch.where(prob_mat)
        probs = prob_mat[torch.where(prob_mat)]
        y_pred = model(X_shift)
        layer_loss = torch.sum(
            probs
            * torch.dist(model.features[idx_target], source_activations[idx_source])
        )
        losses = self.loss_fun(
            y_pred[idx_target], y_train[idx_source], reduction="none"
        )
        if len(losses.shape) == 2:
            losses = torch.mean(losses, dim=1)
        target_loss = torch.sum(probs * losses)
        return source_loss + self.alpha * layer_loss + self.lamb * target_loss

    def calc_loss_mat(self, y_source, y_pred):
        num_source = y_source.shape[0]
        num_target = y_pred.shape[0]
        ys = torch.repeat_interleave(y_source, num_target, dim=0)
        yt = y_pred.repeat(num_source, 1)
        loss_mat = self.loss_fun(yt, ys, reduction="none").reshape(
            num_source, num_target
        )
        return loss_mat
