import torch

from fDAL import fDALLearner
from models.conv import ConvDomainClassifier
from load_model import init_lazy_discriminator


class FDAL:
    def __init__(self, config, fabric, model, loss_fun, scenario):
        self.name = "FDAL"
        print(f"Initializing {self.name}")
        juncture = config["juncture"]
        if 'RESNET' in model.name:
            model.net = torch.nn.Sequential(*list(model.children()))
        backbone = model.net[:juncture]
        taskhead = model.net[juncture:]
        if config["auxhead"] == "none":
            aux_head = None
        elif config["auxhead"] == "conv":
            aux_head = ConvDomainClassifier(model.num_classes)
            aux_head = init_lazy_discriminator(
                aux_head, backbone, scenario, use_features=False
            )
        else:
            raise Exception("Unknown auxiliary head / domain classifier!")
        self.clip_grad_val = config["clip_grad_val"]
        self.learner = fDALLearner(
            backbone,
            taskhead,
            loss_fun,
            divergence=config["divergence"],
            reg_coef=config["reg_coef"],
            n_classes=model.num_classes,
            aux_head=aux_head,
            grl_params=config["grl"],
        )

        # FDAL is different from other algorithms in that it keeps its own modules internally, unlike
        # others which take a fixed model from outside!
        self.opt = torch.optim.Adam(
            self.learner.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.learner, self.opt = fabric.setup(self.learner, self.opt)

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        loss, stats = self.learner((X_source, X_target), y_source)
        fabric.backward(loss)
        torch.nn.utils.clip_grad_norm(self.learner.parameters(), self.clip_grad_val)
        self.opt.step()
        # print(f"Task loss: {stats["taskloss"]}, fdal loss: {stats["fdal_loss"]}")
