"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
"""

import math
import os
import pickle
from argparse import ArgumentParser

import data_utils as dtul
import datasets as dt
import grid_predictor as grid
import losses as lo
import matplotlib.pyplot as plt
import models
import numpy as np
import torch
import utils as ut
from dataloader import *
from eval_helper import *
from paths import get_paths
from torch import optim
from trainer import *
from trainer_helper import *

parser = make_args_parser()
args = parser.parse_args()

trainer = Trainer(args, console=True)


trainer.run_train()
trainer.run_eval_final()
val_preds = trainer.run_eval_spa_enc_only(
    eval_flag_str="LocEnc ", load_model=True
)
