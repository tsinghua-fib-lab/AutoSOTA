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

import matplotlib.pyplot as plt
import numpy as np
import torch


class LocationDataLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        loc_feats,
        labels,
        users,
        num_classes,
        is_train,
        cnn_features=None,
        device="cpu",
    ):
        self.loc_feats = loc_feats
        self.labels = labels
        self.users = users
        self.is_train = is_train
        self.num_classes = num_classes
        # cnn_features: torch.tensor, shape (num_sample, 2048), features from trained image classifier
        self.cnn_features = cnn_features
        self.device = device

    def __len__(self):
        return len(self.loc_feats)

    def __getitem__(self, index):
        loc_feat = self.loc_feats[index, :].to(self.device)
        loc_class = self.labels[index].to(self.device)
        user = self.users[index].to(self.device)

        if self.cnn_features is None:
            if self.is_train:
                return loc_feat, loc_class, user
            else:
                return loc_feat, loc_class
        else:
            # cnn_features: (2048)
            cnn_features = self.cnn_features[index, :].to(self.device)
            if self.is_train:
                return loc_feat, loc_class, user, cnn_features
            else:
                return loc_feat, loc_class, cnn_features
