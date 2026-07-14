"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
"""

import json
import math
import os
import pickle
from argparse import ArgumentParser
from collections import Counter

import baselines as bl
import datasets as dt
import grid_predictor as grid
import matplotlib as mpl
import matplotlib.pylab as plab
import matplotlib.pyplot as plt
import models
import numpy as np
import pandas as pd
import torch
import utils as ut
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from paths import get_paths
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import BallTree, DistanceMetric

# import skimage





def plot_locs(
    mask_lines,
    locs=None,
    colors="red",
    cmap="magma",
    intervals=None,
    map_range=(-180, 180, -90, 90),
    categories=None,
    size=2,
):
    # locs = val_locs
    # the color of the dot indicates the date
    # colors = np.sin(np.pi*train_dates[inds])
    # colors = "red"

    # plot GT locations
    plt.close("all")
    im_width = mask_lines.shape[1]
    im_height = mask_lines.shape[0]
    plt.figure(num=0, figsize=[im_width / 250, im_height / 250], dpi=100)
    plt.imshow(1 - mask_lines, extent=map_range, cmap="gray")

    if locs is not None:
        # inds = np.where(train_classes==class_of_interest)[0]
        # print('{} instances of: '.format(len(inds)) + classes[class_of_interest])
        inds = np.arange(locs.shape[0])
        if categories is None:
            # colors = 'r'
            plt.scatter(
                locs[inds, 0],
                locs[inds, 1],
                c=colors,
                s=size,
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
                s=size,
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


def load_mask(mask_dir="../data/", mask_filename="ocean_mask.npy"):
    mask = np.load(mask_dir + mask_filename, allow_pickle=True)
    mask_lines = np.gradient(mask)[0] ** 2 + np.gradient(mask)[1] ** 2
    mask_lines[mask_lines > 0.0] = 1.0
    return mask_lines, mask


def make_baseline_model_file(eval_params):
    eval_params["algs"] = ["no_prior", eval_params["spa_enc"]]

    meta_str = ""
    if eval_params["dataset"] in ["birdsnap", "nabirds"]:
        meta_str = "_" + eval_params["meta_type"]

    nn_model_path = "{}model_{}{}_{}.pth.tar".format(
        eval_params["trained_models_root"],
        eval_params["dataset"],
        meta_str,
        eval_params["spa_enc"],
    )
    #     nn_model_path_tang = "{}bl_tang_{}{}_gps.pth.tar".format(
    #         eval_params['trained_models_root'],
    #         eval_params['dataset'],
    #         meta_str
    #         )
    return nn_model_path


def make_sphere_model_file(params, eval_params):
    eval_params["algs"] = ["no_prior", eval_params["spa_enc"]]

    meta_str = ""
    if eval_params["dataset"] in ["birdsnap", "nabirds"]:
        meta_str = "_" + eval_params["meta_type"]

    param_args = "{lr:.4f}_{freq:d}_{min_radius:.7f}_{num_hidden_layer:d}_{hidden_dim:d}".format(
        lr=params["lr"],
        freq=params["frequency_num"],
        min_radius=params["min_radius"],
        num_hidden_layer=params["num_hidden_layer"],
        hidden_dim=params["hidden_dim"],
    )

    nn_model_path = "{}model_{}{}_{}_{}.pth.tar".format(
        eval_params["trained_models_root"],
        eval_params["dataset"],
        meta_str,
        eval_params["spa_enc"],
        param_args,
    )
    #     nn_model_path_tang = "{}bl_tang_{}{}_gps.pth.tar".format(
    #         eval_params['trained_models_root'],
    #         eval_params['dataset'],
    #         meta_str
    #         )
    return nn_model_path


def load_feat_model(
    params, eval_params, nn_model_path, val_locs, val_dates, train_locs
):

    net_params = torch.load(nn_model_path, map_location="cpu")
    params = net_params["params"]

    # construct features
    # val_feats_net: shape [batch_size, 2], torch.tensor
    val_feats_net = ut.generate_model_input_feats(
        spa_enc_type=params["spa_enc_type"],
        locs=val_locs,
        dates=val_dates,
        params=params,
        device=eval_params["device"],
    )

    model = ut.get_model(
        train_locs=train_locs,
        params=params,
        spa_enc_type=params["spa_enc_type"],
        num_inputs=params["num_feats"],
        num_classes=params["num_classes"],
        num_filts=params["num_filts"],
        num_users=params["num_users"],
        device=eval_params["device"],
    )

    model.load_state_dict(net_params["state_dict"])
    return params, val_feats_net, model


def compute_acc(
    val_preds,
    val_classes,
    val_split,
    val_feats=None,
    train_classes=None,
    train_feats=None,
    prior_type="no_prior",
    prior=None,
    hyper_params=None,
):
    """
    Computes accuracy on held out set with a specified prior. Not very efficient
    as it loops though each example one at a time.
    Args:
        val_preds: CNN pretrained model's image prediction of class
        val_classes: [batch_size, 1], the list of image category id
        val_split: for bridsnap, np.ones() (batch_size)
        val_feats: the inpit location features, shape [batch_size, x]
        train_classes:
        train_feats:
        prior_type: 'geo_net'
        prior: the model itself
    Return:
        pred_classes: (batch_size), the list of joint predicted image category
    """

    top_k_acc = {}
    for kk in [1, 3, 5, 10]:
        top_k_acc[kk] = np.zeros(len(val_classes))
    max_class = np.max(list(top_k_acc.keys()))
    pred_classes = []  # the list of joint predicted image category

    pred_list = []

    for ind in range(len(val_classes)):

        # select the type of prior to be used
        if prior_type == "no_prior":
            pred = val_preds[ind, :]

        elif prior_type == "train_freq":
            pred = val_preds[ind, :] * prior

        elif prior_type == "nn_dist":
            geo_prior = bl.compute_neighbor_prior(
                train_classes,
                val_preds.shape[1],
                val_feats[ind, :],
                prior,
                hyper_params,
                ptype="distance",
            )
            pred = val_preds[ind, :] * geo_prior

        elif prior_type == "nn_knn":
            geo_prior = bl.compute_neighbor_prior(
                train_classes,
                val_preds.shape[1],
                val_feats[ind, :],
                prior,
                hyper_params,
                ptype="knn",
            )
            pred = val_preds[ind, :] * geo_prior

        elif prior_type == "kde":
            geo_prior = bl.kde_prior(
                train_classes,
                train_feats,
                val_preds.shape[1],
                val_locs[ind, :],
                prior,
                hyper_params,
            )
            pred = val_preds[ind, :] * geo_prior

        elif prior_type == "grid":
            geo_prior = prior.eval(val_feats[ind, :])
            pred = val_preds[ind, :] * geo_prior

        elif prior_type in ["geo_net"] + ut.get_spa_enc_list():
            # if there is no location info won't use prior
            # pred: the pretrained CNN image class prediction distribution
            pred = val_preds[ind, :]
            with torch.no_grad():
                # if all image have location infor
                if torch.isnan(val_feats[ind, 0]).item() == 0:
                    # net_prior: (1, num_classes), the spa_enc model image class prediction distribution
                    net_prior = prior(val_feats[ind, :].unsqueeze(0))
                    net_prior = (
                        net_prior.cpu().data.numpy()[0, :].astype(np.float64)
                    )
                    # net_prior /= net_prior.sum()  # does not matter for argmax
                    pred = pred * net_prior

        elif prior_type == "tang_et_al":
            # if there is no location info won't use prior
            pred = val_preds[ind, :]
            with torch.no_grad():
                if torch.isnan(val_feats["val_locs"][ind, 0]).item() == 0:
                    # takes location and network features as input
                    pred = prior(
                        val_feats["val_locs"][ind, :].unsqueeze(0),
                        val_feats["val_feats"][ind, :].unsqueeze(0),
                    )
                    pred = pred.cpu().data.numpy()[0, :].astype(np.float64)

        # store accuracy of prediction
        pred_classes.append(np.argmax(pred))
        top_N = np.argsort(pred)[-max_class:]
        for kk in top_k_acc.keys():
            if val_classes[ind] in top_N[-kk:]:
                top_k_acc[kk][ind] = 1

        pred_list.append(np.expand_dims(pred, axis=0))

    # print final accuracy
    # some datasets have mutiple splits. These are represented by integers for each example in val_split
    for ii, split in enumerate(np.unique(val_split)):
        print(f" Split ID: {ii}")
        inds = np.where(val_split == split)[0]
        for kk in np.sort(list(top_k_acc.keys())):
            print(
                " Top {}\tacc (%):   {}".format(
                    kk, round(top_k_acc[kk][inds].mean() * 100, 2)
                )
            )

    # preds: (num_sample, num_classes)
    preds = np.concatenate(pred_list, axis=0)

    # ranks: np.array(), [batch_size], the rank of the correct class label for each sample, start from 1
    ranks = get_label_rank(loc_pred=preds, loc_class=val_classes)

    top_k_acc = {}
    for kk in [1, 3, 5, 10]:
        top_k_acc[kk] = (ranks <= kk).astype(int)

    # print final accuracy
    # some datasets have mutiple splits. These are represented by integers for each example in val_split
    for ii, split in enumerate(np.unique(val_split)):
        print(f" Split ID: {ii}")
        inds = np.where(val_split == split)[0]
        for kk in np.sort(list(top_k_acc.keys())):
            print(
                " Top {}\tacc (%):   {}".format(
                    kk, round(top_k_acc[kk][inds].mean() * 100, 2)
                )
            )

    return pred_classes, ranks


def get_label_rank(loc_pred, loc_class):
    """
    Args:
        loc_pred: np matrix, [batch_size, num_classes], the prediction probability distribution of each sample over all classes
        loc_class: np matrix, [batch_size], the ground truth class
    """
    loc_pred_ = loc_pred
    # loc_pred_idx: [batch_size, num_classes], the reverse rank (large->small) of all classes based on the probability
    loc_pred_idx = np.argsort(loc_pred_, axis=-1)[:, ::-1]

    # the rank for each class in each sample
    ranks_ = np.argsort(loc_pred_idx, axis=-1) + 1

    loc_class_ = loc_class

    nids = np.arange(loc_pred_.shape[0])

    # rank_list: np.array(), [batch_size], the rank of the correct class label for each sample, start from 1
    rank_list = ranks_[nids, loc_class_]

    # num_classes = loc_pred_.shape[1]

    # loc_class_ = loc_class.cpu().data.numpy()

    # # loc_class_: [batch_size, num_classes], the correct class label for each sample
    # loc_class_ = np.repeat(np.expand_dims(loc_class_, axis = 1), num_classes, axis = 1)

    # # rank_list: np.array(), [batch_size], the rank of the correct class label for each sample, start from 1
    # rank_list = np.argmax(loc_pred_idx == loc_class_, axis = 1) + 1
    return rank_list


def compute_mrr_by_lat(locs, d_mrr, do_positive=True, num_interval=18):

    interval = 180.0 / num_interval
    cluster_labels = np.floor((locs[:, 1] - (-90)) / interval)
    cluster2mrr = dict()
    for idx in range(cluster_labels.shape[0]):
        if cluster_labels[idx] not in cluster2mrr:
            cluster2mrr[cluster_labels[idx]] = []
        if not np.isnan(cluster_labels[idx]):
            cluster2mrr[cluster_labels[idx]].append(d_mrr[idx])

    intervals = np.arange(-90.0, 90.0, interval)

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

    centers = intervals + interval / 2
    centers = list(centers[:-1])
    # len_list = [l*1.0/total for l in len_list]
    return mrr_list, propotion_list, len_list, centers


def json_load(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data


def json_dump(data, filepath, pretty_format=True):
    with open(filepath, "w") as fw:
        if pretty_format:
            json.dump(data, fw, indent=2, sort_keys=True)
        else:
            json.dump(data, fw)


def pickle_dump(obj, pickle_filepath):
    with open(pickle_filepath, "wb") as f:
        pickle.dump(obj, f, protocol=2)


def pickle_load(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def spa_enc_embed_clustering(
    enc_dec,
    num_cluster,
    coords,
    mask=None,
    model_type=None,
    params=None,
    tsne_comp=4,
):

    if model_type == "geo_net":
        # (num_y, num_x, 2)
        locs = np.asarray(coords)
        num_y, num_x, _ = locs.shape
        locs = np.reshape(locs, (num_y * num_x, -1))

        dates = np.random.rand(num_y * num_x, 2)

        train_feats = ut.generate_model_input_feats(
            spa_enc_type="geo_net",
            locs=locs,
            dates=dates,
            params=params,
            device="cpu",
        )

        res = enc_dec.forward(train_feats, return_feats=True)
        res_data = res.data.tolist()
        res_np = np.asarray(res_data)
        _, embed_dim = res_np.shape
        res_np = np.reshape(res_np, (num_y, num_x, embed_dim))
    else:
        # shape: (num_y*num_x, embed_dim)
        res = enc_dec.spa_enc.forward(coords)

        res_data = res.data.tolist()
        res_np = np.asarray(res_data)
        num_y, num_x, embed_dim = res_np.shape

    if mask is not None:
        # make all ocean embedding => [0, 0, ...0]
        mask_ = np.expand_dims(mask, axis=-1)
        mask_ = np.repeat(mask_, embed_dim, axis=-1)
        res_np = res_np * mask_
        inds = np.where(res_np == 0)
        res_np[inds] = res_np[inds] + math.sqrt(1.0 / embed_dim)
    embeds = np.reshape(res_np, (num_y * num_x, -1))
    embeds_norm = np.linalg.norm(embeds, ord=None, axis=-1, keepdims=True)
    embeds = embeds / embeds_norm

    #     embeds = TSNE(n_components=tsne_comp).fit_transform(embeds)

    embed_clusters = AgglomerativeClustering(
        n_clusters=num_cluster, affinity="cosine", linkage="complete"
    ).fit(embeds)
    cluster_labels = np.reshape(embed_clusters.labels_, (num_y, num_x))

    return embeds, cluster_labels


def make_enc_map(
    mask,
    mask_lines,
    cluster_labels,
    num_cluster,
    extent,
    margin=0,
    coords_mat=None,
    coords_color="red",
    colorbar=False,
    img_path=None,
    xlabel=None,
    ylabel=None,
):
    cmap = plt.cm.terrain
    bounds = np.arange(-0.5, num_cluster + 0.5, 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.close("all")
    im_width = mask_lines.shape[1]
    im_height = mask_lines.shape[0]
    fig = plt.figure(num=0, figsize=[im_width / 250, im_height / 250], dpi=100)

    #     pt_x_list, pt_y_list = plot_poi_by_type(enc_dec, type2pts, tid)

    plt.imshow(cluster_labels, extent=extent, cmap=cmap, norm=norm)
    #     mask_  = mask + mask_lines
    mask_data = np.ma.masked_where(1 - mask == 0, 1 - mask)
    mask_cmap = mpl.cm.get_cmap("hot")
    mask_cmap.set_bad(color="w", alpha=0)
    mask_cmap.set_under(color="w")
    plt.imshow(mask_data, extent=extent, cmap=mask_cmap, vmin=0.001)

    mask_lines_data = np.ma.masked_where(mask_lines == 0, mask_lines)
    mask_cmap = mpl.cm.get_cmap("gray")
    mask_cmap.set_bad(color="w", alpha=0)
    mask_cmap.set_under(color="w")
    plt.imshow(
        mask_lines_data, extent=extent, interpolation="nearest", cmap="gray"
    )

    #     mask_lines_data = np.ma.masked_where(mask_lines==0, mask_lines)
    #     plt.imshow(mask_lines_data, extent=extent, cmap='binary')
    #     plt.imshow(mask_lines, extent=extent, cmap='binary')
    #     mask_cmap = mpl.cm.get_cmap('hot')
    #     mask_cmap.set_bad(color = 'w', alpha=0)
    #     mask_cmap.set_under(color='red')
    #     plt.imshow(mask_lines_data, extent=extent, cmap= mask_cmap, vmin = 0.001)
    # plt.colorbar()

    # We must be sure to specify the ticks matching our target names
    if colorbar:
        plt.colorbar(ticks=bounds - 0.5)
    if coords_mat is not None:
        if coords_mat.shape:
            plt.scatter(
                coords_mat[:, 0],
                coords_mat[:, 1],
                s=1.5,
                c=coords_color,
                alpha=0.5,
            )

    #     plt.scatter(pt_x_list, pt_y_list, s=1.5, c="red", alpha=0.5)
    #     plt.xlim(extent[0]-margin, extent[1]+margin)
    #     plt.ylim(extent[2]-margin, extent[3]+margin)
    #     if xlabel is not None:
    #         plt.xlabel(xlabel)
    #     if ylabel is not None:
    #         plt.ylabel(ylabel)

    #     if model_type == "global":
    #         plt.xticks(np.arange(extent[0]-margin, extent[1]+margin, 10000))
    #     fig = plt.gcf()
    # fig.suptitle(tid2type[tid])
    #     plt.xlabel(poi_type, fontsize=10)

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    plt.show()
    if img_path:
        fig.savefig(img_path, dpi=100)


def visualize_encoder(
    module, layername, coords, extent, num_ch=8, img_path=None
):
    if layername == "input_emb":
        res = module.make_input_embeds(coords)
        if type(res) == torch.Tensor:
            res = res.data.numpy()
        elif type(res) == np.ndarray:
            res = res
        print(res.shape)
        res_np = res
    elif layername == "output_emb":
        res = module.forward(coords)
        embed_dim = res.size()[2]
        res_data = res.data.tolist()
        res_np = np.asarray(res_data)

    num_rows = num_ch / 8

    plt.figure(figsize=(28, 50))
    #     for i in range(embed_dim):
    for i in range(num_ch):
        if num_ch <= 8:
            ax = plt.subplot(1, num_ch, i + 1)
        else:
            ax = plt.subplot(num_rows, 8, i + 1)
        ax.imshow(res_np[:, :, i][::-1, :], extent=extent)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        #         plt.tight_layout()
        #         plt.title(i, fontsize=160)
        plt.title(i, fontsize=40)
    fig = plt.gcf()
    plt.show()
    plt.draw()
    if img_path:
        fig.savefig(img_path, dpi=300, bbox_inches="tight")


def plot_gt_locations(
    params,
    mask,
    train_classes,
    class_of_interest,
    classes,
    train_locs,
    train_dates,
    op_dir,
):
    """
    plot GT locations for the class of interest, with mask in the backgrpund
    Args:
        params:
        mask: (1002, 2004) mask for the earth,
              (lat,  lon ), so that when you plot it, it will be naturally the whole globe
        train_classes: [batch_size, 1], the list of image category id
        class_of_interest: 0
        classes: a dict(), class id => class name
        train_locs: [batch_size, 2], location data
        train_dates: [batch_size, 1], the list of date
        op_dir:
    """

    im_width = (params["map_range"][1] - params["map_range"][0]) // 45  # 8
    im_height = (params["map_range"][3] - params["map_range"][2]) // 45  # 4
    plt.figure(num=0, figsize=[im_width, im_height])
    plt.imshow(mask, extent=params["map_range"], cmap="tab20")

    """
    when np.where(condition, x, y) with no x,y, it like np.asarray(condition).nonzero()
    np.where(train_classes==class_of_interest) return a tuple,
    a tuple of arrays, one for each dimension of a,
    containing the indices of the non-zero elements in that dimension
    """
    # inds: the indices in train_classes 1st dim where the class id == class_of_interest
    inds = np.where(train_classes == class_of_interest)[0]
    print(f"{len(inds)} instances of: " + classes[class_of_interest])

    # the color of the dot indicates the date
    colors = np.sin(np.pi * train_dates[inds])
    plt.scatter(
        train_locs[inds, 0],
        train_locs[inds, 1],
        c=colors,
        s=2,
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_frame_on(False)

    #     op_file_name = op_dir + 'gt_' + str(class_of_interest).zfill(4) + '.jpg'
    op_file_name = "{}gt_{}_{}.jpg".format(
        op_dir,
        str(class_of_interest).zfill(4),
        classes[class_of_interest].replace(" ", "-"),
    )
    plt.savefig(op_file_name, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.show()


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
    op_file_name = "{}gt_{}_{}_{}_{}_predict.jpg".format(
        op_dir,
        params["dataset"],
        str(class_of_interest).zfill(4),
        classes[class_of_interest].replace(" ", "-"),
        params["spa_enc_type"],
    )

    plt.imshow(1 - grid_pred, cmap="afmhot", vmin=0, vmax=1)
    plt.imsave(op_file_name, 1 - grid_pred, cmap="afmhot", vmin=0, vmax=1)
    return grid_pred


def select_locs_give_extent(locs, region_extent=[-180, 180, -90, 90]):
    # select locs in a region_extent
    lat_inds = (locs[:, 1] > region_extent[2]) * (
        locs[:, 1] < region_extent[3]
    )
    lon_inds = (locs[:, 0] > region_extent[0]) * (
        locs[:, 0] < region_extent[1]
    )
    inds = lon_inds * lat_inds
    return inds


def get_dmrr_per_class_in_region(
    d_mrr, locs, classes_np, region_extent=[-180, 180, -90, 90], topn=None
):
    """
    Args:
        locs: [N, 2], the traimg/val locations
        region_extent: the region we want to see, default [-180, 180, -90, 90]
        classes_np: [N], the golden label for each location
        d_mrr: [N], delta MRR for each locs
    """
    # select locs in a region_extent
    inds = select_locs_give_extent(locs, region_extent)
    # give sorted class by their number of samples in this region
    ccs, cnts = np.unique(classes_np[inds], return_counts=True)
    sort_inds = np.argsort(cnts)

    d_mrr_0[inds]
    val_classes[inds]
    if topn == None:
        # topn = np.sum(cnts >= 5)
        topn = cnts.shape[0]
    d_mrr_per_cls = []
    for cls_ind in range(topn):
        # the top ith class id
        cls = ccs[sort_inds][cls_ind]
        # the list of d_mrr for the ith class id
        d_mrr_cls = d_mrr[inds][classes_np[inds] == cls]
        # mean mrr
        mean_d_mrr = np.mean(d_mrr_cls)
        d_mrr_per_cls.append(mean_d_mrr)
    return (
        ccs[sort_inds][:topn],
        cnts[sort_inds][:topn],
        np.asarray(d_mrr_per_cls),
    )


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

    for idx in range(len(lat_intervals) - 1, 0, -1):
        lower = lat_intervals[idx - 1]
        upper = lat_intervals[idx]
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
    for cc in np.unique(cluster_labels):
        if cc != -1:
            cluster2mrr[cc] = np.mean(d_mrr[cluster_labels == cc])

    box_list = []
    for idx in range(len(lat_intervals) - 1, 0, -1):
        lower = lat_intervals[idx - 1]
        upper = lat_intervals[idx]
        for x_idx in range(len(lon_intervals) - 1):
            left = lon_intervals[x_idx]
            right = lon_intervals[x_idx + 1]

            cc = idx * (len(lon_intervals) - 1) + x_idx
            extent = (left, right, lower, upper)
            if cc in cluster2mrr:
                box_list.append((extent, cluster2mrr[cc]))
            else:
                box_list.append((extent, 0))

    return box_list
