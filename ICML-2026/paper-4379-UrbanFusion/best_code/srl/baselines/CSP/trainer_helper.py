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

import datasets as dt
import grid_predictor as grid
import losses as lo
import matplotlib.pyplot as plt
import models
import numpy as np
import torch
import utils as ut
from dataloader import *
from paths import get_paths
from torch import optim


def unsupervise_train(
    model,
    data_loader,
    optimizer,
    epoch,
    params,
    logger=None,
    neg_rand_type="spherical",
):
    model.train()

    assert params["unsuper_loss"] != "none"
    # assert params['load_cnn_features_train']
    # assert data_loader.cnn_features is not None

    # adjust the learning rate
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = params['lr'] * (params['lr_decay'] ** epoch)

    loss_avg = ut.AverageMeter()
    inds = torch.arange(params["batch_size"]).to(params["device"])

    for batch_idx, batch_data in enumerate(data_loader):
        # if params['load_cnn_features_train']:
        loc_feat, loc_class, user_ids, cnn_features = batch_data
        # else:
        #     loc_feat, loc_class, user_ids = batch_data
        """
        loc_feat: (batch_size, input_feat_dim)
        loc_class: (batch_size)
        user_ids: (batch_size)
        cnn_features: (batch_size, cnn_feat_dim = 2048)
        """
        optimizer.zero_grad()
        if params["unsuper_loss"] == "l2regress":
            loss = lo.l2regress_loss(
                model, params, loc_feat, cnn_features, inds
            )
        elif "imgcontloss" in params["unsuper_loss"]:
            loss = lo.imgcontloss_loss(
                model, params, loc_feat, cnn_features, inds
            )
        elif "contsoftmax" in params["unsuper_loss"]:
            loss = lo.contsoftmax_loss(
                model, params, loc_feat, cnn_features, inds
            )

        # loss = lo.embedding_loss(model, params, loc_feat, loc_class, user_ids, inds, neg_rand_type = neg_rand_type)

        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item(), len(loc_feat))

        if (batch_idx % params["log_frequency"] == 0 and batch_idx != 0) or (
            batch_idx == (len(data_loader) - 1)
        ):
            logger.info(
                "[{}/{}]\tUnsupervised {} Loss  : {:.4f}".format(
                    batch_idx * params["batch_size"],
                    len(data_loader.dataset),
                    params["unsuper_loss"],
                    loss_avg.avg,
                )
            )


def unsupervise_eval(model, data_loader, params, logger=None):
    model.eval()

    assert params["unsuper_loss"] != "none"
    # assert params['load_cnn_features']

    # adjust the learning rate
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = params['lr'] * (params['lr_decay'] ** epoch)

    loss_avg = ut.AverageMeter()
    inds = torch.arange(params["batch_size"]).to(params["device"])

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # if params['load_cnn_features_train']:
            loc_feat, loc_class, cnn_features = batch_data
            # else:
            #     loc_feat, loc_class, user_ids = batch_data
            """
            loc_feat: (batch_size, input_feat_dim)
            loc_class: (batch_size)
            user_ids: (batch_size)
            cnn_features: (batch_size, cnn_feat_dim = 2048)
            """

            if params["unsuper_loss"] == "l2regress":
                loss = lo.l2regress_loss(
                    model, params, loc_feat, cnn_features, inds
                )
            elif "imgcontloss" in params["unsuper_loss"]:
                loss = lo.imgcontloss_eval(
                    model, params, loc_feat, cnn_features, inds
                )
            elif "contsoftmax" in params["unsuper_loss"]:
                loss = lo.contsoftmax_loss(
                    model, params, loc_feat, cnn_features, inds
                )

            loss_avg.update(loss.item(), len(loc_feat))

    logger.info(
        "Unsupervised {} Test loss   : {:.4f}".format(
            params["unsuper_loss"], loss_avg.avg
        )
    )


def train(
    model,
    data_loader,
    optimizer,
    epoch,
    params,
    logger=None,
    neg_rand_type="spherical",
):
    model.train()

    # adjust the learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = params["lr"] * (params["lr_decay"] ** epoch)

    loss_avg = ut.AverageMeter()
    inds = torch.arange(params["batch_size"]).to(params["device"])

    for batch_idx, batch_data in enumerate(data_loader):
        if params["load_cnn_features_train"]:
            loc_feat, loc_class, user_ids, cnn_features = batch_data
        else:
            loc_feat, loc_class, user_ids = batch_data
        """
        loc_feat: (batch_size, input_feat_dim)
        loc_class: (batch_size)
        user_ids: (batch_size)
        cnn_features: (batch_size, cnn_feat_dim = 2048)
        """
        optimizer.zero_grad()

        loss = lo.embedding_loss(
            model,
            params,
            loc_feat,
            loc_class,
            user_ids,
            inds,
            neg_rand_type=neg_rand_type,
        )

        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item(), len(loc_feat))

        if (batch_idx % params["log_frequency"] == 0 and batch_idx != 0) or (
            batch_idx == (len(data_loader) - 1)
        ):
            logger.info(
                "[{}/{}]\tLoss  : {:.4f}".format(
                    batch_idx * params["batch_size"],
                    len(data_loader.dataset),
                    loss_avg.avg,
                )
            )


def test(model, data_loader, params, logger=None):
    # NOTE the test loss only tracks the BCE it is not the full loss used during training
    # the test loss is the -log() of the correct class prediction probability, the lower is better
    model.eval()
    loss_avg = ut.AverageMeter()

    inds = torch.arange(params["batch_size"]).to(params["device"])
    with torch.no_grad():

        for batch_data in data_loader:
            if params["load_cnn_features"]:
                loc_feat, loc_class, cnn_features = batch_data
            else:
                loc_feat, loc_class = batch_data
            """
            loc_feat: (batch_size, input_feat_dim)
            loc_class: (batch_size)
            cnn_features: (batch_size, cnn_feat_dim = 2048)
            """
            # loc_pred: (batch_size, num_classes)
            loc_pred = model(loc_feat)
            # pos_loss: (batch_size)
            pos_loss = lo.bce_loss(
                loc_pred[inds[: loc_feat.shape[0]], loc_class]
            )
            loss = pos_loss.mean()

            loss_avg.update(loss.item(), loc_feat.shape[0])

    logger.info(f"Test loss   : {loss_avg.avg:.4f}")


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

    op_file_name = op_dir + "gt_" + str(class_of_interest).zfill(4) + ".jpg"
    plt.savefig(op_file_name, dpi=400, bbox_inches="tight", pad_inches=0)
