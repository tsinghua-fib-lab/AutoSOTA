import torch
from torch import nn
from torch.autograd import Function
import numpy as np

import utils
from models.conv import ConvDomainClassifier
from models.mlp import MultiLayerPerceptron as MLP
from load_model import init_lazy_discriminator


# TODO: Replace with WarmGRL code from fdal repo
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    def __init__(self, config, fabric, model, loss_fun, scenario):
        super(DANN, self).__init__()
        if config["discriminator"] == "mlp":
            discr = MLP([100, 10, 2], nn.ReLU())
        elif config["discriminator"] == "conv":
            discr = ConvDomainClassifier()
        else:
            raise Exception("Unknown domain classifier!")

        if "conv" in model.name:
            model.track_features(config["conv_feat_layer"])
        elif model.name == "MLP":
            model.track_features(config["mlp_feat_layer"])
        elif "RESNET" in model.name:
            model.track_features(-1)
        else:
            raise Exception("Model not found!")
        init_lazy_discriminator(discr, model, scenario, use_features=True)
        self.discriminator = discr
        self.model = model
        self.name = "DANN"
        self.opt = torch.optim.Adam(
            self.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self, self.opt = fabric.setup(self, self.opt)
        self.loss_class = loss_fun
        self.loss_domain = nn.CrossEntropyLoss()
        self.num_epochs = config["num_epochs"]
        self.num_batches = config["num_batches"]
        self.idx = 0

    def forward_adversarial(self, model, X_data):
        epoch_idx = self.idx // self.num_batches
        p = (self.idx + epoch_idx * self.num_batches) / (
            self.num_epochs * self.num_batches
        )
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        class_output = model(X_data)
        reverse_feature = ReverseLayerF.apply(model.features, alpha)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        source_batch_size = X_source.shape[0]
        # Feeding in source inputs
        domain_label = utils.one_hot(
            torch.zeros(source_batch_size, device=fabric.device, dtype=torch.long), 2
        )
        class_output, domain_output = self.forward_adversarial(model, X_source)
        err_s_label = self.loss_class(class_output, y_source)
        err_s_domain = self.loss_domain(domain_output, domain_label)

        # Feeding in target labels
        target_batch_size = X_target.shape[0]
        domain_label = utils.one_hot(
            torch.ones(target_batch_size, device=fabric.device, dtype=torch.long), 2
        )
        _, domain_output = self.forward_adversarial(model, X_target)

        err_t_domain = self.loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        # print(f"errors: {err_t_domain}, {err_s_domain}, {err_s_label}")

        fabric.backward(err)
        self.opt.step()
        self.opt.zero_grad()
        self.idx += 1
