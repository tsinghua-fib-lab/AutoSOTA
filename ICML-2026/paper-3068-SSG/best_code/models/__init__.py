from .resnet import *
from .swin_transformer import *


from torchvision.models._api import get_model, get_model_builder, get_model_weights, get_weight, list_models, Weights, WeightsEnum
from torchvision.models import detection, optical_flow, quantization, segmentation, video