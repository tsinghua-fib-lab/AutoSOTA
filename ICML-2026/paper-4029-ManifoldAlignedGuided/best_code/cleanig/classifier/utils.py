import os

import torch
import torch.nn as nn
import torchvision.models as models


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dims=[4, 4],
        output_dim=2,
        activation=nn.ReLU(),
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.layers.append(nn.Linear(input_dim, hidden_dims[i]))
            self.layers.append(nn.ReLU())
            input_dim = hidden_dims[i]
        self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNetWrapper(nn.Module):
    def __init__(self, model_name):
        super(ResNetWrapper, self).__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", model_name, pretrained=True
        )

    def forward(self, x):
        return self.model(x)


def get_classifier(model_name, dataset_name, image_size, num_classes, pretrained=True):
    model_classes = dict(
        inception=models.googlenet,
        vgg16=models.vgg16,
        resnet18=models.resnet18,
        resnet34=models.resnet34,
        resnet50=models.resnet50,
    )

    # NOTE:
    # - This repo maps "inception" to torchvision's GoogLeNet.
    # - GoogLeNet has optional auxiliary classifier heads (aux1/aux2).
    # - The fine-tuned Oxford checkpoints we ship were trained with aux heads
    #   disabled, so we must instantiate GoogLeNet with aux_logits=False
    #   when loading those checkpoints (i.e., when not using the ImageNet
    #   pretrained weights as-is).
    if (
        model_name == "inception"
        and dataset_name in ["oxfordflower", "oxfordpet"]
        and not pretrained
    ):
        model = models.googlenet(pretrained=False, aux_logits=False)
    else:
        model = model_classes[model_name](pretrained=pretrained)

    if dataset_name in ["oxfordflower", "oxfordpet"]:
        if "resnet" in model_name:
            model.fc = nn.Identity()
            dummy_input = torch.randn(1, 3, image_size, image_size)
            dummy_features = model(dummy_input)
            feature_size = dummy_features.view(-1).shape[0]
            model.fc = nn.Linear(feature_size, num_classes)
        elif "vgg" in model_name:
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-7])

            dummy_input = torch.randn(1, 3, image_size, image_size)
            dummy_features = model(dummy_input)
            model.feature_size = dummy_features.view(-1).shape[0]

            model.classifier = nn.Sequential(
                nn.Linear(model.feature_size, 150),
                nn.BatchNorm1d(150),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(150, num_classes),
            )
        elif "inception" in model_name:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported")
    elif dataset_name in ["imagenet"]:
        # Use pretrained model as-is (ImageNet has 1000 classes)
        pass
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return model
