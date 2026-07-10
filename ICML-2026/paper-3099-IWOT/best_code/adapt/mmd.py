import torch


# TODO: For now we only use a Gaussian kernel
class MMD:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = loss_fun
        self.name = "MMD"
        self.opt = opt
        self.alpha = config["alpha"]
        self.gammas = config["gammas"]
        self.p = 1
        if config["use_squared_dist"] is True:
            self.p = 2
        self.use_features = False
        if config["use_features"] is True:
            self.use_features = True
            model.track_features(config["track_layer"])
        model, self.opt = fabric.setup(model, self.opt)

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        pred_source = model(X_source)
        err = self.loss_fun(pred_source, y_source)
        if self.use_features is True:
            source_activations = torch.clone(model.features)
            model(X_target)
            target_activations = torch.clone(model.features)
            err += self.alpha * self.calc_mmd(source_activations, target_activations)
        else:
            pred_target = model(X_target)
            err += self.alpha * self.calc_mmd(pred_source, pred_target)
        fabric.backward(err)
        self.opt.step()
        self.opt.zero_grad()

    def calc_mmd(self, pred_source, pred_target):
        K_source = self.calc_kernel(pred_source, pred_source).mean()
        K_target = self.calc_kernel(pred_target, pred_target).mean()
        K_source_target = self.calc_kernel(pred_source, pred_target).mean()
        return K_source + K_target - 2 * K_source_target

    def calc_kernel(self, X_source, X_target):
        mat_dist = torch.cdist(X_source, X_target) ** self.p
        kernel_mat = torch.zeros_like(mat_dist)
        for gamma in self.gammas:
            kernel_mat += torch.exp(-gamma * mat_dist)
        return kernel_mat
