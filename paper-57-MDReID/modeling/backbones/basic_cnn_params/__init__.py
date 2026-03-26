from __future__ import absolute_import
import torch
from modeling.backbones.basic_cnn_params.cal import CAL
from modeling.backbones.basic_cnn_params.pcb import *
from modeling.backbones.basic_cnn_params.mlfn import *
from modeling.backbones.basic_cnn_params.hacnn import *
from modeling.backbones.basic_cnn_params.osnet import *
from modeling.backbones.basic_cnn_params.senet import *
from modeling.backbones.basic_cnn_params.mudeep import *
from modeling.backbones.basic_cnn_params.nasnet import *
from modeling.backbones.basic_cnn_params.resnet import *
from modeling.backbones.basic_cnn_params.densenet import *
from modeling.backbones.basic_cnn_params.xception import *
from modeling.backbones.basic_cnn_params.osnet_ain import *
from modeling.backbones.basic_cnn_params.resnetmid import *
from modeling.backbones.basic_cnn_params.shufflenet import *
from modeling.backbones.basic_cnn_params.squeezenet import *
from modeling.backbones.basic_cnn_params.inceptionv4 import *
from modeling.backbones.basic_cnn_params.mobilenetv2 import *
from modeling.backbones.basic_cnn_params.resnet_ibn_a import *
from modeling.backbones.basic_cnn_params.resnet_ibn_b import *
from modeling.backbones.basic_cnn_params.shufflenetv2 import *
from modeling.backbones.basic_cnn_params.inceptionresnetv2 import *

__model_factory = {
    # image classification models
    'cal':CAL,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnet50_fc512': resnet50_fc512,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,
    'densenet121_fc512': densenet121_fc512,
    'inceptionresnetv2': inceptionresnetv2,
    'inceptionv4': inceptionv4,
    'xception': xception,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet50_ibn_b': resnet50_ibn_b,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2_x1_0': mobilenetv2_x1_0,
    'mobilenetv2_x1_4': mobilenetv2_x1_4,
    'shufflenet': shufflenet,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,
    # reid-specific models
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'mlfn': mlfn,
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'osnet_ain_x1_0': osnet_ain_x1_0,
    'osnet_ain_x0_75': osnet_ain_x0_75,
    'osnet_ain_x0_5': osnet_ain_x0_5,
    'osnet_ain_x0_25': osnet_ain_x0_25
}


def show_avai_models():

    print(list(__model_factory.keys()))


def build_model(
    name, num_classes, loss='softmax', pretrained=False, use_gpu=True
):

    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )