from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from .wide_resnet import WideResNet28, WideResNet34
from .senet import SENet18

def get_model(model_name: str, num_classes: int = 10, img_size: int = 32, in_channels: int = 3):
    """
    Get the model for the given model name.
    
    Args:
        model_name (str): Name of the model to get.
        num_classes (int): Number of classes in the dataset.
        img_size (int): Size of the image.
    """
    
    match model_name:
        case "ResNet18":
            return ResNet18(num_classes=num_classes, in_channels=in_channels)
        case "ResNet34":
            return ResNet34(num_classes=num_classes, in_channels=in_channels)
        case "ResNet50":
            return ResNet50(num_classes=num_classes, in_channels=in_channels)
        case "ResNet101":
            return ResNet101(num_classes=num_classes, in_channels=in_channels)
        case "ResNet152":
            return ResNet152(num_classes=num_classes, in_channels=in_channels)
        case "PreActResNet18":
            return PreActResNet18(num_classes=num_classes, in_channels=in_channels)
        case "PreActResNet34":
            return PreActResNet34(num_classes=num_classes, in_channels=in_channels)
        case "PreActResNet50":
            return PreActResNet50(num_classes=num_classes, in_channels=in_channels)
        case "PreActResNet101":
            return PreActResNet101(num_classes=num_classes, in_channels=in_channels)
        case "PreActResNet152":
            return PreActResNet152(num_classes=num_classes, in_channels=in_channels)
        case "WideResNet28":
            return WideResNet28(num_classes=num_classes, in_channels=in_channels)
        case "WideResNet34":
            return WideResNet34(num_classes=num_classes, in_channels=in_channels)
        case "SENet18":
            return SENet18(num_classes=num_classes, in_channels=in_channels)
        case _:
            raise ValueError("Invalid Model!")
