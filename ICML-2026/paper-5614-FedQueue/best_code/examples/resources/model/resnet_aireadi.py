import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights


def ResNet50(num_classes, pretrained=False):
    if pretrained:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
