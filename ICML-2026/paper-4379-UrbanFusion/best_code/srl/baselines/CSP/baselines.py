"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
"""

import math

import numpy as np
import utils as ut


def compute_neighbor_prior(
    train_classes, num_classes, eval_loc, nn_tree, hyper_params, ptype
):
    # If we have a valid eval location compute the distance to all examples in
    # the train set and keep those within a specified distance or number. If the
    # location is not valid, the prior will be uniform.

    geo_prior = np.ones(num_classes)  # adding a uniform prior
    if (eval_loc[0] is not None) and (not np.isnan(eval_loc[0])):
        # Note that sklearn haversine expects distance to be [lat, lon] not [lon, lat]
        if ptype == "distance":
            neighbor_inds = nn_tree.query_radius(
                eval_loc[::-1][np.newaxis, ...], r=hyper_params["dist_thresh"]
            )[0]
        elif ptype == "knn":
            neighbor_inds = nn_tree.query(
                eval_loc[::-1][np.newaxis, ...],
                k=hyper_params["num_neighbors"],
            )[1][0]

        cls_id, cls_cnt = np.unique(
            train_classes[neighbor_inds], return_counts=True
        )
        geo_prior[cls_id] += cls_cnt

    geo_prior /= geo_prior.sum()

    return geo_prior


class GridPrior:
    def __init__(self, locs, classes, num_classes, hyper_params):

        self.lon_bins = hyper_params["gp_size"][0]
        self.lat_bins = hyper_params["gp_size"][1]
        self.grid = np.zeros((self.lat_bins, self.lon_bins, num_classes))
        self.uniform_prior = np.ones(num_classes)
        self.uniform_prior /= self.uniform_prior.sum()

        locs_scaled = locs.copy()
        locs_scaled[:, 0] = (locs_scaled[:, 0] + 180) / 360.0
        locs_scaled[:, 0] *= self.lon_bins
        locs_scaled[:, 1] = (locs_scaled[:, 1] + 90) / 180.0
        locs_scaled[:, 1] *= self.lat_bins

        bins = [np.arange(self.lat_bins + 1), np.arange(self.lon_bins + 1)]
        denom, _, _ = np.histogram2d(
            locs_scaled[:, 1], locs_scaled[:, 0], bins
        )
        denom = (
            denom + (num_classes * hyper_params["pseudo_count"]) - num_classes
        )

        for ss in range(num_classes):
            inds = np.where(classes == ss)[0]
            cnt, _, _ = np.histogram2d(
                locs_scaled[inds, 1], locs_scaled[inds, 0], bins
            )

            # hallucinate some counts with the beta prior
            cnt += hyper_params["pseudo_count"]
            cnt -= 1

            self.grid[:, :, ss] = cnt / denom

    def eval(self, loc):

        if (loc[0] is not None) and (not np.isnan(loc[0])):
            loc_scaled = np.zeros(2)
            loc_scaled[0] = (loc[0] + 180) / 360.0
            loc_scaled[0] *= self.lon_bins
            loc_scaled[1] = (loc[1] + 90) / 180.0
            loc_scaled[1] *= self.lat_bins
            return self.grid[int(loc_scaled[1]), int(loc_scaled[0]), :]
        else:
            return self.uniform_prior


def hashable_loc(loc, q):
    # helper function make locations hashable
    return (int(np.floor(loc[0] / q)), int(np.floor(loc[1] / q)))


def create_kde_grid(train_classes, train_locs, hyper_params):
    # quantize locations:
    assert hyper_params["kde_quant"] > 0
    quantized_train_locs = (
        np.floor(train_locs / hyper_params["kde_quant"])
        * hyper_params["kde_quant"]
    )
    # reduce to unique locations, counting repetitions:
    binned_train_classes = []
    binned_train_locs = []
    counts = []
    idx_dict = {}  # {class: {loc: idx}}
    cur_idx = 0
    for ii in range(len(quantized_train_locs)):
        loc_key = hashable_loc(
            quantized_train_locs[ii], hyper_params["kde_quant"]
        )
        current_class = train_classes[ii]
        if current_class not in idx_dict:
            idx_dict[current_class] = {}
        if loc_key not in idx_dict[current_class]:
            binned_train_classes.append(current_class)
            binned_train_locs.append(quantized_train_locs[ii])
            counts.append(1)
            idx_dict[current_class][loc_key] = cur_idx
            cur_idx += 1
        else:
            counts[idx_dict[current_class][loc_key]] += 1
    return (
        np.array(binned_train_classes),
        np.array(binned_train_locs),
        np.array(counts),
    )


def kde_prior(
    train_classes, train_locs, num_classes, eval_loc, kde_params, hyper_params
):
    # Implementation of the KDE technique from Berg et al. 2014, excluding time.
    # Additionally, supports haversine distance.

    if np.isnan(eval_loc[0]) or np.isnan(
        eval_loc[1]
    ):  # For invalid coordinates, default to uniform prior.
        return np.ones(num_classes) / float(num_classes)

    # compute adaptive kernel bandwidth and induced neighbors:
    dist_to_neighbors = kde_params["nn_tree_kde"].query(
        eval_loc[::-1][np.newaxis, ...], k=hyper_params["kde_nb"]
    )[0]
    kernel_bandwidth = 0.5 * np.max(
        dist_to_neighbors
    )  # kernel bandwidth for this eval_loc
    if kernel_bandwidth == 0:
        raise ValueError(
            "All data points are at the same location - try reducing quantization."
        )
    neighbor_inds = kde_params["nn_tree_kde"].query_radius(
        eval_loc[::-1][np.newaxis, ...], r=2 * kernel_bandwidth + 1e-9
    )[0]
    assert len(neighbor_inds) > 0
    if hyper_params["kde_dist_type"] == "haversine":
        distances = (
            ut.distance_pw_haversine(
                train_locs[neighbor_inds, :], eval_loc[np.newaxis, ...], 1
            )[:, 0]
            ** 2
        )
    else:
        distances = (
            (train_locs[neighbor_inds, ::-1] - eval_loc[::-1]) ** 2
        ).sum(1)

    # restrict to subset of interest
    kde_params_counts_r = kde_params["counts"][neighbor_inds]
    train_classes_r = train_classes[neighbor_inds]

    # calculate kde
    kernel_dim = 2
    Q = ((2 * math.pi * kernel_bandwidth) ** (-kernel_dim * 0.5)) * np.exp(
        -distances / (2 * kernel_bandwidth**2)
    )
    kde_den = np.dot(kde_params_counts_r, Q)
    assert kde_den > 0

    # sum over each class
    Q_prod = kde_params_counts_r * Q
    kde_num = np.zeros(num_classes)
    bin_cnt = np.bincount(train_classes_r, Q_prod)
    kde_num[: bin_cnt.shape[0]] = bin_cnt

    kde_num = kde_num + np.min(
        kde_num[np.nonzero(kde_num)]
    )  # add small value for all species
    kde = kde_num / kde_den
    geo_prior = kde / np.sum(kde)
    return geo_prior
