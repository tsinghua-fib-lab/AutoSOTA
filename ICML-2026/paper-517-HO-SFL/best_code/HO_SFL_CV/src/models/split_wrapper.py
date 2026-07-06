import torch.nn as nn


def freeze_bn_layers(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    for module in model.modules():
        if isinstance(module, bn_types):

            for param in module.parameters():
                param.requires_grad = False


def split_model(model, split_point):
    if not isinstance(model, nn.Sequential):
        raise ValueError("Model must be nn.Sequential for easy splitting.")

    layers = list(model.children())

    if split_point >= len(layers):
        raise ValueError(f"Split point {split_point} too large. Max is {len(layers)}.")

    client_part = nn.Sequential(*layers[:split_point])
    freeze_bn_layers(client_part)
    server_part = nn.Sequential(*layers[split_point:])

    return client_part, server_part
