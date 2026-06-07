import timm
import os

import torch
import torch.nn as nn

from tllib.alignment.dann import ImageClassifier
from safetensors.torch import load_file


def get_model(config):
    """
    vitb16
    """

    model = timm.create_model('vit_base_patch16_224', pretrained=False)

    # Load pretrained weights from local file
    pretrained_path = '/repo/pretrained/model.safetensors'
    if os.path.exists(pretrained_path):
        state_dict = load_file(pretrained_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")

    model.out_features = model.head.in_features
    model.head = nn.Identity()

    pool_layer = torch.nn.Identity()
    classifier = ImageClassifier(model, config['num_classes'], pool_layer=pool_layer, bottleneck_dim=config['bottleneck']).cuda()


    return classifier