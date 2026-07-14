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
from copy import deepcopy

import data_utils as dtul
import datasets as dt
import grid_predictor as grid
import losses as lo
import matplotlib.pyplot as plt
import models
import numpy as np
import torch
import utils as ut
from analysis import *
from paths import get_paths
from torch import optim


def compute_mrr_by_lat(
    locs, d_mrr, do_positive=True, num_interval=18, intervals=None
):

    interval = 180.0 / num_interval
    if intervals is None:
        intervals = np.arange(-90.0, 90.0, interval)
        cluster_labels = np.floor((locs[:, 1] - (-90)) / interval)
    else:
        cluster_labels = np.zeros(locs.shape[0]) - 1
        for idx in range(len(intervals) - 1):
            lowerbound = intervals[idx]
            upperbound = intervals[idx + 1]
            inds = (locs[:, 1] >= lowerbound) * (locs[:, 1] < upperbound)
            cluster_labels[inds] = idx

    cluster2mrr = dict()
    for idx in range(cluster_labels.shape[0]):
        if cluster_labels[idx] == -1:
            continue
        if cluster_labels[idx] not in cluster2mrr:
            cluster2mrr[cluster_labels[idx]] = []
        if not np.isnan(cluster_labels[idx]):
            cluster2mrr[cluster_labels[idx]].append(d_mrr[idx])

    mrr_list = []
    propotion_list = []
    len_list = []
    total = 0

    cluster_list = list(range(len(intervals) - 1))

    for cc in cluster_list:
        #     print("({}, {})".format(intervals[cc], intervals[cc+1]))
        if cc in cluster2mrr:
            mrr_list.append(np.mean(cluster2mrr[cc]))
            a = np.asarray(cluster2mrr[cc])
            if do_positive:
                propotion_list.append(
                    1.0 * np.sum(a > 0) / len(cluster2mrr[cc])
                )
            else:
                propotion_list.append(
                    1.0 * np.sum(a < 0) / len(cluster2mrr[cc])
                )
            len_list.append(len(cluster2mrr[cc]))
            total += len(cluster2mrr[cc])
        else:
            mrr_list.append(0)
            propotion_list.append(0)
            len_list.append(0)
    centers = []
    for idx in range(len(intervals) - 1):
        lowerbound = intervals[idx]
        upperbound = intervals[idx + 1]
        centers.append((lowerbound + upperbound) / 2)

    # len_list = [l*1.0/total for l in len_list]
    return mrr_list, propotion_list, len_list, centers


def compute_mrr_by_latlongrid(
    locs, d_mrr, lat_intervals=None, lon_intervals=None
):

    cluster_labels = np.zeros(locs.shape[0]) - 1

    for idx in range(len(lat_intervals) - 2, -1, -1):
        lower = lat_intervals[idx]
        upper = lat_intervals[idx + 1]
        for x_idx in range(len(lon_intervals) - 1):
            left = lon_intervals[x_idx]
            right = lon_intervals[x_idx + 1]

            inds = (
                (locs[:, 1] >= lower)
                * (locs[:, 1] < upper)
                * (locs[:, 0] >= left)
                * (locs[:, 0] < right)
            )
            cluster_labels[inds] = idx * (len(lon_intervals) - 1) + x_idx

    cluster2mrr = dict()
    cluster2mrr_list = dict()
    for cc in np.unique(cluster_labels):
        if cc != -1:
            cluster2mrr[cc] = np.mean(d_mrr[cluster_labels == cc])
            cluster2mrr_list[cc] = d_mrr[cluster_labels == cc]

    boxes = []
    box_d_mmr_list = []
    box_num_samples = []

    for idx in range(len(lat_intervals) - 2, -1, -1):
        lower = lat_intervals[idx]
        upper = lat_intervals[idx + 1]
        for x_idx in range(len(lon_intervals) - 1):
            left = lon_intervals[x_idx]
            right = lon_intervals[x_idx + 1]

            cc = idx * (len(lon_intervals) - 1) + x_idx
            extent = (left, right, lower, upper)
            boxes.append(extent)

            if cc in cluster2mrr:
                box_d_mmr_list.append(cluster2mrr[cc])
                box_num_samples.append(cluster2mrr_list[cc].shape[0])
            else:
                box_d_mmr_list.append(0)
                box_num_samples.append(0)

    return boxes, box_d_mmr_list, box_num_samples


# def plot_locs(mask_lines, locs = None, colors = "red", cmap='magma', intervals = None,
#               map_range = (-180, 180, -90, 90), categories = None, size = 2):
#     # locs = val_locs
#     # the color of the dot indicates the date
#     # colors = np.sin(np.pi*train_dates[inds])
#     # colors = "red"

#     # plot GT locations
#     plt.close('all')
#     im_width  = mask_lines.shape[1]
#     im_height = mask_lines.shape[0]
#     fig = plt.figure(num=0, figsize=[im_width/250, im_height/250], dpi=100)
#     ax = plt.axes([0, 0.05, 0.9, 0.9 ]) #left, bottom, width, height
#     ax.imshow(1-mask_lines, extent=map_range, cmap='gray')
#     # fig, ax = plt.subplots(figsize=(im_width/250, im_height/250), dpi=100)
#     # ax.imshow(1-mask_lines, extent=map_range, cmap='gray')


#     if locs is not None:
#         #inds = np.where(train_classes==class_of_interest)[0]
#         #print('{} instances of: '.format(len(inds)) + classes[class_of_interest])
#         inds = np.arange(locs.shape[0])
#         if categories is None:
#             colors = 'r'
#             ax.scatter(locs[inds, 0], locs[inds, 1], c=colors, s=size, cmap=cmap, vmin=0, vmax=1)
#         else:
#             cat = categories.astype(int)
#             N = int(np.max(cat)+1)
#             # define the colormap
#             cmap = plt.cm.jet
#             # extract all colors from the .jet map
#             cmaplist = [cmap(i) for i in range(cmap.N)]
#             # create the new map
#             cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

#             # define the bins and normalize
#             bounds = np.linspace(0,N,N+1)
#             norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#             # make the scatter
#             ax.scatter(locs[inds, 0], locs[inds, 1], c=cat, s=size, cmap=cmap,  norm=norm)

#     if intervals is not None:
#         for i in list(intervals[1:-1]):
#             if i == 0:
#                 ax.hlines(y=i, xmin=-180, xmax=180, color='black')
#             else:
#                 ax.hlines(y=i, xmin=-180, xmax=180, color='black', linestyles='dotted')
#             ax.text(x = -180, y = i, s = "{}".format(int(i)), ha='right', va='center')


#     plt.gca().axes.get_xaxis().set_visible(False)
#     plt.gca().axes.get_yaxis().set_visible(False)
#     plt.gca().set_frame_on(False)
#     plt.tight_layout()
#     return fig, ax


def plot_locs(
    mask_lines,
    locs=None,
    colors="red",
    cmap="magma",
    intervals=None,
    map_range=(-180, 180, -90, 90),
    categories=None,
    fig=None,
    ax=None,
    pt_size=2,
):
    # locs = val_locs
    # the color of the dot indicates the date
    # colors = np.sin(np.pi*train_dates[inds])
    # colors = "red"
    if fig is None or ax is None:
        # plot GT locations
        plt.close("all")
        im_width = mask_lines.shape[1]
        im_height = mask_lines.shape[0]
        # fig = plt.figure(num=0, figsize=[im_width/250, im_height/250], dpi=100)
        # plt.imshow(1-mask_lines, extent=map_range, cmap='gray')

        fig, ax = plt.subplots(
            figsize=(im_width / 250, im_height / 250), dpi=100
        )
    img = ax.imshow(1 - mask_lines, extent=map_range, cmap="gray")

    if locs is not None:
        # inds = np.where(train_classes==class_of_interest)[0]
        # print('{} instances of: '.format(len(inds)) + classes[class_of_interest])
        inds = np.arange(locs.shape[0])
        if categories is None:
            colors = "r"
            plt.scatter(
                locs[inds, 0],
                locs[inds, 1],
                c=colors,
                s=pt_size,
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
        else:
            cat = categories.astype(int)
            N = int(np.max(cat) + 1)
            # define the colormap
            cmap = plt.cm.jet
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # create the new map
            cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(0, N, N + 1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            # make the scatter
            plt.scatter(
                locs[inds, 0],
                locs[inds, 1],
                c=cat,
                s=pt_size,
                cmap=cmap,
                norm=norm,
            )

    if intervals is not None:
        for i in list(intervals[1:-1]):
            if i == 0:
                plt.hlines(y=i, xmin=-180, xmax=180, color="black")
            else:
                plt.hlines(
                    y=i,
                    xmin=-180,
                    xmax=180,
                    color="black",
                    linestyles="dotted",
                )
            plt.text(
                x=-180, y=i, s=f"{int(i)}", ha="right", va="center"
            )

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    return fig, ax


import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def make_op_dir(params):
    param_args = "{lr:.4f}_{freq:d}_{min_radius:.7f}_{num_hidden_layer:d}_{hidden_dim:d}".format(
        lr=params["lr"],
        freq=params["frequency_num"],
        min_radius=params["min_radius"],
        num_hidden_layer=params["num_hidden_layer"],
        hidden_dim=params["hidden_dim"],
    )
    if params["meta_type"] == "":
        img_folder = "ims_{}_{}_{}".format(
            params["dataset"], params["spa_enc_type"], param_args
        )
    else:
        img_folder = "ims_{}_{}_{}_{}".format(
            params["dataset"],
            params["meta_type"],
            params["spa_enc_type"],
            param_args,
        )

    op_dir = f"image/{img_folder}/"
    return op_dir


def plot_prediction(
    model,
    params,
    mask,
    train_classes,
    class_of_interest,
    classes,
    train_locs,
    train_dates,
    op_dir,
):
    op_dir = make_op_dir(params)
    if not os.path.isdir(op_dir):
        os.makedirs(op_dir)

    gp = grid.GridPredictor(mask, params)

    plot_gt_locations(
        params,
        mask,
        train_classes,
        class_of_interest,
        classes,
        train_locs,
        train_dates,
        op_dir,
    )

    grid_pred = gp.dense_prediction(model, class_of_interest)
    op_file_name = "{}gt_{}_{}_{}_predict.jpg".format(
        op_dir,
        str(class_of_interest).zfill(4),
        classes[class_of_interest].replace(" ", "-"),
        params["spa_enc_type"],
    )

    plt.imshow(1 - grid_pred, cmap="afmhot", vmin=0, vmax=1)
    plt.imsave(op_file_name, 1 - grid_pred, cmap="afmhot", vmin=0, vmax=1)
    return grid_pred
