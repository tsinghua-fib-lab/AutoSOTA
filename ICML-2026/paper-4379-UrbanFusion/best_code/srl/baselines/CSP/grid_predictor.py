"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.

Class for making dense predictions on a 2D grid.
"""

import math
import sys

import numpy as np
import torch

sys.path.append("../")
import geo_prior.utils as ut


class GridPredictor:

    def __init__(self, mask, params, mask_only_pred=False):
        """
        Args:
            mask: (1002, 2004) mask for the earth,
                  (lat,  lon ), so that when you plot it, it will be naturally the whole globe

        """
        # set up coordinates to make dense prediction on grid
        self.device = params["device"]
        self.params = params
        self.mask = mask
        self.use_date_feats = params["use_date_feats"]
        self.spa_enc_type = params["spa_enc_type"]

        # np.gradient compute the x (np.gradient(mask)[0]) and y (np.gradient(mask)[1]) partial drevative
        self.mask_lines = np.gradient(mask)[0] ** 2 + np.gradient(mask)[1] ** 2
        self.mask_lines[self.mask_lines > 0.0] = 1.0

        # set up feature grid this will be height X width X num feats
        # gird_lon: (2004), a list of lon, but from [-1, 1]
        grid_lon = torch.linspace(-1, 1, mask.shape[1]).to(self.device)
        # torch.tensor.repeat() like numpy.tile()
        # grid_lon: (1002, 2004, 1)
        # 1002 -> mask longitude dimention (90, -90)
        # 2004 -> mask latitude dimention (-180, 180)
        grid_lon = grid_lon.repeat(mask.shape[0], 1).unsqueeze(2)
        # grid_lat: (1002), a list of lat, but from [-1, 1]
        grid_lat = torch.linspace(1, -1, mask.shape[0]).to(self.device)
        # grid_lat: (1002, 2004, 1)
        grid_lat = (
            grid_lat.repeat(mask.shape[1], 1).transpose(0, 1).unsqueeze(2)
        )
        dates = torch.zeros(
            mask.shape[0], mask.shape[1], 1, device=self.device
        )

        if self.spa_enc_type == "geo_net":
            if self.use_date_feats:
                # loc_time_feats: (1002, 2004, 3), 3 means (lon, lat, date)
                loc_time_feats = torch.cat((grid_lon, grid_lat, dates), 2)
                loc_time_feats = ut.encode_loc_time(
                    loc_time_feats[:, :, :2],
                    loc_time_feats[:, :, 2],
                    concat_dim=2,
                    params=params,
                )
            else:
                loc_time_feats = torch.cat((grid_lon, grid_lat), 2)
                loc_time_feats = ut.encode_loc_time(
                    loc_time_feats[:, :, :2], None, concat_dim=2, params=params
                )
        elif (
            self.spa_enc_type in ut.get_spa_enc_list()
            or self.spa_enc_type in ut.get_spa_enc_baseline_list()
        ):
            assert self.use_date_feats == False
            grid_lon = grid_lon * 180
            grid_lat = grid_lat * 90
            # loc_time_feats: (1002, 2004, 2)
            loc_time_feats = torch.cat((grid_lon, grid_lat), 2)
        else:
            raise Exception("spa_enc not defined!!!")

        self.feats = loc_time_feats

        # for mask only prediction
        if mask_only_pred:
            # get 1-d index in mask where value == 1
            self.mask_inds = np.where(self.mask.ravel() == 1)[0]
            # feats_local: all (lon, lat, date) feature where mask value == 1
            self.feats_local = self.feats.reshape(
                self.feats.shape[0] * self.feats.shape[1], self.feats.shape[2]
            )[self.mask_inds, :].clone()

    def dense_prediction(
        self, model, class_of_interest, time_step=0, mask_op=True
    ):
        """
        Given model, we show the probability distribution over the world of the class_of_interest
        Return:
            grid_pred: (1002, 2004)
        """
        # make prediction for entire grid at different time steps - by looping over columns
        # time_step should be in range [0, 1]

        # feats_change_time: (1002, 2004, 3) or (1002, 2004, 2)
        feats_change_time = self.feats.clone()
        assert self.use_date_feats == False
        if self.use_date_feats:
            feats_change_time = self.update_date_feats(
                feats_change_time, time_step
            )

        grid_pred = np.zeros(self.mask.shape, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            # loop throough each longitude
            for col in range(feats_change_time.shape[1]):
                # pred: (batch_size)
                pred = model(
                    feats_change_time[:, col, :],
                    class_of_interest=class_of_interest,
                )
                grid_pred[:, col] = pred.cpu().numpy()

        if mask_op:
            return grid_pred * self.mask + self.mask_lines
        else:
            return grid_pred

    def dense_prediction_sum(self, model, time_step=0, mask_op=True):
        # make prediction for entire grid at different time steps - by looping over columns
        # takes the mean prediction for each class

        feats_change_time = self.feats.clone()
        if self.use_date_feats:
            feats_change_time = self.update_date_feats(
                feats_change_time, time_step
            )

        grid_pred = np.zeros(self.mask.shape, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for col in range(feats_change_time.shape[1]):
                pred = model(feats_change_time[:, col, :]).sum(1)
                grid_pred[:, col] = pred.cpu().numpy()

        max_val = grid_pred.max()
        if mask_op:
            return grid_pred * self.mask + self.mask_lines, max_val
        else:
            return grid_pred, max_val

    def dense_prediction_masked(self, model, class_of_interest, time_step):
        # only masks predictions for valid datapoints
        if self.use_date_feats:
            self.feats_local = self.update_date_feats(
                self.feats_local, time_step
            )

        model.eval()
        with torch.no_grad():
            pred = model(self.feats_local, class_of_interest=class_of_interest)

        grid_pred = self.create_full_output(self, pred.cpu().numpy())
        return grid_pred

    def dense_prediction_masked_feats(self, model, time_step):
        # only masks predictions for valid datapoints
        if self.use_date_feats:
            feats_local = self.update_date_feats(self.feats_local, time_step)

        model.eval()
        with torch.no_grad():
            feats = model(self.feats_local, return_feats=True)

        return feats

    def create_full_output(self, pred):
        """
        Given a global prediction matrix by using  prediction of all valid data points
        Fill out the mask
        Args:
            pred: (..., len(self.mask_inds)), prediction of all valid data points
        """
        grid_pred = np.zeros(
            self.mask.shape[0] * self.mask.shape[1], dtype=np.float32
        )
        grid_pred[self.mask_inds] = pred
        return grid_pred.reshape((self.mask.shape[0], self.mask.shape[1]))

    def update_date_feats(self, feats, time_step):
        # helper function - for visualization we want to vary the date
        offset = 0
        if self.params["loc_encode"] == "encode_cos_sin":
            offset = 4
        elif self.params["loc_encode"] == "encode_3D":
            offset = 3

        if len(feats.shape) == 2:
            if self.params["date_encode"] == "encode_cos_sin":
                feats[:, offset] = math.sin(math.pi * (2 * time_step - 1))
                feats[:, offset + 1] = math.cos(math.pi * (2 * time_step - 1))
            elif self.params["date_encode"] == "encode_none":
                feats[:, offset] = 2 * time_step - 1
        else:
            if self.params["date_encode"] == "encode_cos_sin":
                feats[:, :, offset] = math.sin(math.pi * (2 * time_step - 1))
                feats[:, :, offset + 1] = math.cos(
                    math.pi * (2 * time_step - 1)
                )
            elif self.params["date_encode"] == "encode_none":
                feats[:, :, offset] = 2 * time_step - 1

        return feats
