import os
import copy
import types
import torch
import torchvision
from torch import nn

import utils
from models.conv import ConvNet, ConvNet2, LeNet, SmallCNN
from models.mlp import MultiLayerPerceptron as MLP


def load_model(config, fabric, scenario):
    model = init_model(config, scenario)
    folder_path = "save_files/" + scenario.name + "/"
    save_path = folder_path + model.name + ".pth"
    if config["pretrain"]:
        try:
            # Load parameters from a file
            model.load_state_dict(torch.load(save_path, weights_only=True))
            print(f"Saved model found! Loading parameters from file: {save_path}")
        except FileNotFoundError:
            print(f"Saved model {model.name} NOT found!")
    model.train()
    return model


def init_model(config, scenario):
    if config["model"] == "MLP":
        model = MLP(
            layer_sizes=[scenario.input_size, 200, 100, scenario.num_classes],
            f_nonlinear=nn.ReLU(),
            use_batchnorm=config.get("use_batchnorm", False),
        )
    elif config["model"] == "ConvNet":
        model = ConvNet(num_classes=scenario.num_classes)
    elif config["model"] == "ConvNet2":
        model = ConvNet2(num_classes=scenario.num_classes)
    elif config["model"] == "LeNet":
        model = LeNet(num_classes=scenario.num_classes)
    elif config["model"] == "SmallCNN":
        model = SmallCNN(num_classes=scenario.num_classes)
    elif config["model"] == "ResNet":
        model = init_resnet(
            config["resnet_size"], config["resnet_load_imagenet_weights"], scenario.num_channels, scenario.num_classes
        )
    else:
        raise Exception("Model not found")
    init_lazy_modules(model, scenario)
    print(f"Initialized model {model.name}")
    return model


def init_resnet(size, load_imagenet_weights, num_inp_channels, num_classes):
    weights = None
    if load_imagenet_weights is True:
        weights = "IMAGENET1K_V1"
    if size == 18:
        model = torchvision.models.resnet18(weights)
        model.name = "RESNET18"
    elif size == 50:
        model = torchvision.models.resnet50(weights)
        model.name = "RESNET50"
    else:
        raise Exception("Resnet size not allowed!")
    model.num_classes = num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if num_inp_channels == 1:
        # Average the pretrained RGB filters to
        # get a single-channel equivalent
        avg_weight = model.conv1.weight.sum(dim=1, keepdim=True) / 3
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = avg_weight

    def track_features(model, layer_id):
        # Ignores the layer_id!
        # Register hooks for the layers you're interested in
        def fun(module, inputs, outputs):
            model.features = inputs[0]

        model.fc.register_forward_hook(fun)

    @torch.no_grad()
    def save_params(model):
        model.state = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def restore_params(model):
        model.load_state_dict(model.state)
        return dict(model.named_parameters())

    model.track_features = types.MethodType(track_features, model)
    model.save_params = types.MethodType(save_params, model)
    model.restore_params = types.MethodType(restore_params, model)
    return model


def init_lazy_modules(model, scenario):
    model = model.to("cpu")
    for X_source, y_source in scenario.source_dataloader:
        model(X_source.to("cpu"))
        break
    return model


def init_lazy_discriminator(discr, model, scenario, use_features):
    model = model.to("cpu")
    discr = discr.to("cpu")
    for X_source, y_source in scenario.source_dataloader:
        if use_features is True:
            model(X_source.to("cpu"))
            discr(model.features)
        else:
            discr(model(X_source.to("cpu")))
        break
    return discr


def pretrain_model(model, config, fabric, scenario, loss_fun, opt):
    folder_path = "save_files/" + scenario.name + "/"
    save_path = folder_path + model.name + ".pth"

    if os.path.exists(save_path):
        # Load parameters from a file
        state = torch.load(save_path, map_location=fabric.device, weights_only=True)
        model.load_state_dict(state)
        print(f"Saved model found! Loaded parameters from file: {save_path}")
        model, opt = fabric.setup(model, opt)
        return model

    print(f"Saved model {model.name} NOT found!")
    model, opt = fabric.setup(model, opt)
    print(f"Pretraining {config['num_pretrain_epochs']} epochs...")
    if config["pretrain_on_both"] is True:
        print(
            "========= DEBUG MODE ON: USING TARGET LABELS TO PRETRAIN LJE ORACLE MODEL ======="
        )
        model = utils.train_model_on_source_and_target(
            config, model, loss_fun, scenario, opt, fabric
        )
    else:
        model = utils.train_model_on_source(
            config, model, loss_fun, scenario, opt, fabric
        )
    return model
