import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(1)


def replace_bn_with_gn(module, num_groups=32, transfer_weights=True):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features

            groups = num_groups
            if num_channels < groups:
                groups = num_channels // 2

            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels)

            if transfer_weights:
                if child.weight is not None:
                    gn.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    gn.bias.data.copy_(child.bias.data)

            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups, transfer_weights)


def get_model(config):
    model_name = config.model.model_name
    num_classes = 10 if config.data.dataset == "CIFAR10" else 100

    if model_name in ["ResNet18", "ResNet18_GN"]:
        weights = None
        if config.model.use_pretrained:
            print("Loading ImageNet pretrained weights...")
            weights = ResNet18_Weights.IMAGENET1K_V1
        raw_resnet = resnet18(weights=weights)

        if model_name == "ResNet18_GN":
            print("Converting BN to GN (with weight transfer)...")
            replace_bn_with_gn(raw_resnet, num_groups=32, transfer_weights=True)
        if config.model.freeze_bn:
            print("Freezing BatchNorm layers...")
            for module in raw_resnet.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

        in_features = raw_resnet.fc.in_features
        raw_resnet.fc = torch.nn.Linear(in_features, num_classes)

        layers = [
            raw_resnet.conv1,
            raw_resnet.bn1,
            raw_resnet.relu,
            raw_resnet.maxpool,
            raw_resnet.layer1,
            raw_resnet.layer2,
            raw_resnet.layer3,
            raw_resnet.layer4,
            raw_resnet.avgpool,
            Flatten(),
            raw_resnet.fc,
        ]

        return nn.Sequential(*layers)

    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
