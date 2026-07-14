"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, num_users=1):
        """
        Args:
            num_inputs: input embedding dimention
            num_classes: number of categories we want to classify
            num_filts: hidden embedding dimention
        """
        super().__init__()
        self.inc_bias = False
        self.num_filts = num_filts
        self.num_classes = num_classes
        self.num_users = num_users
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)

        self.feats = nn.Sequential(
            nn.Linear(num_inputs, num_filts),
            nn.ReLU(inplace=True),
            ResLayer(num_filts),
            ResLayer(num_filts),
            ResLayer(num_filts),
            ResLayer(num_filts),
        )

    def forward(self, x, class_of_interest=None, return_feats=False):
        """
        Args:
            x: torch.FloatTensor(), input location features (batch_size, input_loc_dim = 2 or 3 or ...)
            class_of_interest: the class id we want to extract
            return_feats: whether or not just return location embedding
        """
        loc_emb = self.feats(x)
        if return_feats:
            # loc_emb: (batch_size, num_filts)
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        # return (batch_size, num_classes)
        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        """
        Return:
            shape (batch_size)
        """
        # note: self.class_emb.weight shape (num_classes, num_filts)
        if self.inc_bias:
            return (
                torch.matmul(x, self.class_emb.weight[class_of_interest, :])
                + self.class_emb.bias[class_of_interest]
            )
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])


class TangNet(nn.Module):
    def __init__(self, ip_loc_dim, feats_dim, loc_dim, num_classes, use_loc):
        super().__init__()
        """
        ip_loc_dim: the number of grids to be consider
        feats_dim: the image embedding dimention
        loc_dim: the dimention of location featire
        num_classes: the number of image category
        use_loc: True/False
        """
        self.use_loc = use_loc
        # the location embedding matrix
        self.fc_loc = nn.Linear(ip_loc_dim, loc_dim)
        if self.use_loc:
            self.fc_class = nn.Linear(feats_dim + loc_dim, num_classes)
        else:
            self.fc_class = nn.Linear(feats_dim, num_classes)

    def forward(self, loc, net_feat):
        """
        Args:
            locs: one hot vector of one location
            net_feat: the image features
        """
        if self.use_loc:
            x = torch.sigmoid(self.fc_loc(loc))
            x = self.fc_class(torch.cat((x, net_feat), 1))
        else:
            x = self.fc_class(net_feat)
        return F.log_softmax(x, dim=1)


class LocationEncoder(nn.Module):
    def __init__(
        self, spa_enc, num_inputs, num_classes, num_filts, num_users=1
    ):
        """
        Args:
            spa_enc: the spatial encoder
            num_inputs: input embedding dimention
            num_classes: number of categories we want to classify
            num_filts: hidden embedding dimention
        """
        super().__init__()
        self.spa_enc = spa_enc
        self.inc_bias = False
        self.num_filts = num_filts
        self.num_classes = num_classes
        self.num_users = num_users
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)

    def forward(self, x, class_of_interest=None, return_feats=False):
        """
        Args:
            x: torch.FloatTensor(), input location features (batch_size, input_loc_dim = 2)
            class_of_interest: the class id we want to extract
            return_feats: whether or not just return location embedding
        """
        # loc_feat: (batch_size, 1, input_loc_dim = 2)
        loc_feat = torch.unsqueeze(x, dim=1)
        loc_feat = loc_feat.cpu().data.numpy()

        # loc_embed: torch.Tensor(), (batch_size, 1, spa_embed_dim = num_filts)
        loc_embed = self.spa_enc(loc_feat)
        # loc_emb: torch.Tensor(), (batch_size, spa_embed_dim = num_filts)
        loc_emb = loc_embed.squeeze(1)
        if return_feats:
            # loc_emb: (batch_size, num_filts)
            return loc_emb
        if class_of_interest is None:
            # class_pred: (batch_size, num_classes)
            class_pred = self.class_emb(loc_emb)
        else:
            # class_pred: shape (batch_size)
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        """
        Args:
            x: (batch_size, num_filts)
        Return:
            shape (batch_size)
        """
        # note: self.class_emb.weight shape (num_classes, num_filts)
        if self.inc_bias:
            return (
                torch.matmul(x, self.class_emb.weight[class_of_interest, :])
                + self.class_emb.bias[class_of_interest]
            )
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])


class LocationImageEncoder(nn.Module):
    def __init__(
        self,
        loc_enc,
        train_loss,
        unsuper_loss="none",
        cnn_feat_dim=2048,
        spa_enc_type="gridcell",
    ):
        """
        Args:
            loc_enc: LocationEncoder() or FCNet()
        """
        super().__init__()
        self.loc_enc = loc_enc
        if spa_enc_type in ["geo_net"]:
            self.spa_enc = loc_enc
        else:
            self.spa_enc = loc_enc.spa_enc
        self.inc_bias = loc_enc.inc_bias
        self.class_emb = loc_enc.class_emb
        self.user_emb = loc_enc.user_emb

        self.cnn_feat_dim = cnn_feat_dim
        self.loc_emb_dim = loc_enc.num_filts

        if unsuper_loss == "none":
            return
        elif unsuper_loss == "l2regress":
            self.loc_dec = nn.Linear(
                in_features=self.loc_emb_dim,
                out_features=self.cnn_feat_dim,
                bias=True,
            )
        elif "imgcontloss" in unsuper_loss or "contsoftmax" in unsuper_loss:
            self.img_dec = nn.Linear(
                in_features=self.cnn_feat_dim,
                out_features=self.loc_emb_dim,
                bias=True,
            )
        else:
            raise Exception(f"Unknown unsuper_loss={unsuper_loss}")

    def forward(self, x, class_of_interest=None, return_feats=False):
        """
        Args:
            x: torch.FloatTensor(), input location features (batch_size, input_loc_dim = 2)
            class_of_interest: the class id we want to extract
            return_feats: whether or not just return location embedding
        """
        return self.loc_enc.forward(x, class_of_interest, return_feats)

    def eval_single_class(self, x, class_of_interest):
        """
        Args:
            x: (batch_size, num_filts)
        Return:
            shape (batch_size)
        """
        return self.loc_enc.eval_single_class(x, class_of_interest)
