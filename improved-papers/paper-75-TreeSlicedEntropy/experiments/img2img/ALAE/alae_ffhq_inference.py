# import torch.utils.data
# from net import *
from model import Model
# from launcher import run
from checkpointer import Checkpointer
# from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
import numpy as np

import argparse

# from PIL import Image
# import bimpy
import logging
# from defaults import get_cfg_defaults

from matplotlib import pyplot as plt
# %matplotlib inline
import torch
import os
from tqdm import tqdm

# torch.set_default_device("cuda")

def load_config(default_config, training_artifacts_dir):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c", "--config-file",
        default=default_config,
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args(args=["OUTPUT_DIR", training_artifacts_dir])
    defaults = get_cfg_defaults()

    cfg = defaults
    config_file = args.config_file
    if len(os.path.splitext(config_file)[1]) == 0:
        config_file += '.yaml'
    if not os.path.exists(config_file) and os.path.exists(os.path.join('configs', config_file)):
        config_file = os.path.join('configs', config_file)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg

def load_model(default_config, training_artifacts_dir):
    lreq.use_implicit_lreq.set(True)

    indices = [0, 1, 2, 3, 4, 10, 11, 17, 19]

    labels = ["gender",
              "smile",
              "attractive",
              "wavy-hair",
              "young",
              "big lips",
              "big nose",
              "chubby",
              "glasses",
              ]

#     default_config='configs/ffhq.yaml'

    cfg = load_config(default_config, training_artifacts_dir)

#     torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)
#     model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_d
    mapping_fl = model.mapping_f
    dlatent_avg = model.dlatent_avg

    logger = logging.getLogger("logger")

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()

    model.eval()
    
    return model

def encode(model, x):
    layer_count = 9

    zlist = []
    for i in range(x.shape[0]):
        Z, _ = model.encode(x[i][None, ...], layer_count - 1, 1)
        zlist.append(Z)
    Z = torch.cat(zlist)
    Z = Z.repeat(1, model.mapping_f.num_layers, 1)
    return Z

def decode(model, x):
    x = x[:, None, :].repeat(1, model.mapping_f.num_layers, 1)
    layer_count = 9
    decoded = []
    for i in range(x.shape[0]):
        r = model.decoder(x[i][None, ...], layer_count - 1, 1, noise=True)
        decoded.append(r)
    return torch.cat(decoded)
