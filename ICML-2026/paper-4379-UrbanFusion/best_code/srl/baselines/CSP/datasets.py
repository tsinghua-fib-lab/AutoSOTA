"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from data_utils import *
from paths import get_paths
from scipy import sparse


def load_dataset(
    params,
    eval_split,
    train_remove_invalid,
    eval_remove_invalid,
    load_cnn_predictions=False,
    load_cnn_features=False,
    load_cnn_features_train=False,
):
    """
    Args:
        params: the input paramters
        eval_split: 'val', 'test'
        train_remove_invalid: True/False, whether or not remove invalid images data sample from train/val dataset
        load_cnn_predictions: whether or not load CNN pretrained model's image prediction of class
        load_cnn_features: whether or not load the CNN features of valid dataset image
        load_cnn_features_train: whether or not load the CNN features of training dataset image

    """
    train_imgs, val_imgs = None, None
    if params["dataset"] == "inat_2017":

        data_dir = get_paths("inat_2017_data_dir")
        num_classes = 5089
        class_of_interest = 3731

        # load observations
        train_locs, train_classes, train_users, train_dates, train_inds = (
            load_inat_data(
                data_dir,
                "train2017_locations.json",
                "train2017.json",
                train_remove_invalid,
            )
        )
        if eval_split == "val":
            val_locs, val_classes, val_users, val_dates, val_inds = (
                load_inat_data(
                    data_dir,
                    eval_split + "2017_locations.json",
                    eval_split + "2017.json",
                    eval_remove_invalid,
                )
            )
        elif eval_split == "test":
            val_locs, val_classes, val_users, val_dates, val_inds = (
                load_inat_data(
                    data_dir,
                    eval_split + "2017_locations.json",
                    eval_split + "2017_DO_NOT_SHARE.json",
                    eval_remove_invalid,
                )
            )
            val_split = (
                pd.read_csv(
                    data_dir + "kaggle_solution_2017_DO_NOT_SHARE.csv"
                )["usage"].values
                == "Private"
            )

        # load class names
        with open(data_dir + "categories2017.json") as da:
            cls_data = json.load(da)
        class_names = [cc["name"] for cc in cls_data]
        class_ids = [cc["id"] for cc in cls_data]
        classes = dict(zip(class_ids, class_names))

        if load_cnn_predictions:
            val_preds = load_sparse_feats(
                data_dir
                + "features_inception/inat2017_"
                + eval_split
                + "_preds_sparse.npz"
            )

        if load_cnn_features:
            if params["inat2018_resolution"] == "standard":
                val_feats = np.load(
                    data_dir
                    + "features_inception/inat2017_"
                    + eval_split
                    + "_net_feats.npy"
                )
            elif params["inat2018_resolution"] == "pretrain":
                val_feats = np.load(
                    data_dir
                    + "features_inception_pretrain/inat2017_"
                    + eval_split
                    + "_pretrain_net_feats.npy"
                )
                print(
                    f"Load Pretrained inception_v3 feature of inat_2017 {eval_split} data -> {val_feats.shape}"
                )

        if load_cnn_features_train:
            if params["inat2018_resolution"] == "standard":
                train_feats = np.load(
                    data_dir
                    + "features_inception/inat2017_train_net_feats.npy"
                )
            elif params["inat2018_resolution"] == "pretrain":
                train_feats = np.load(
                    data_dir
                    + "features_inception_pretrain/inat2017_train_pretrain_net_feats.npy"
                )
                print(
                    f"Load Pretrained inception_v3 feature inat_2017 of train data -> {train_feats.shape}"
                )

    elif params["dataset"] == "inat_2018":

        data_dir = get_paths("inat_2018_data_dir")
        num_classes = 8142
        class_of_interest = 3731  # wood thrush

        # load observations
        train_data = load_inat_data(
            data_dir,
            "train2018_locations.json",
            "train2018.json",
            train_remove_invalid,
            params["load_img"],
        )
        if params["load_img"]:
            (
                train_locs,
                train_classes,
                train_users,
                train_dates,
                train_inds,
                train_imgs,
            ) = train_data
        else:
            train_locs, train_classes, train_users, train_dates, train_inds = (
                train_data
            )
        if eval_split == "val":
            val_data = load_inat_data(
                data_dir,
                eval_split + "2018_locations.json",
                eval_split + "2018.json",
                eval_remove_invalid,
                params["load_img"],
            )
        elif eval_split == "test":
            val_data = load_inat_data(
                data_dir,
                eval_split + "2018_locations.json",
                eval_split + "2018_DO_NOT_SHARE.json",
                eval_remove_invalid,
                params["load_img"],
            )
            val_split = (
                pd.read_csv(
                    data_dir + "kaggle_solution_2018_DO_NOT_SHARE.csv"
                )["usage"].values
                == "Private"
            )
        if params["load_img"]:
            val_locs, val_classes, val_users, val_dates, val_inds, val_imgs = (
                val_data
            )
        else:
            val_locs, val_classes, val_users, val_dates, val_inds = val_data

        # load class names
        with open(data_dir + "categories2018.json") as da:
            cls_data = json.load(da)
        class_names = [cc["name"] for cc in cls_data]
        class_ids = [cc["id"] for cc in cls_data]
        classes = dict(zip(class_ids, class_names))

        if load_cnn_predictions:
            if params["cnn_pred_type"] == "full":
                if params["inat2018_resolution"] == "high_res":
                    val_preds = load_sparse_feats(
                        data_dir
                        + "features_inception_hr/inat2018_"
                        + eval_split
                        + "_preds_sparse.npz"
                    )
                else:
                    val_preds = load_sparse_feats(
                        data_dir
                        + "features_inception/inat2018_"
                        + eval_split
                        + "_preds_sparse.npz"
                    )
            elif params["cnn_pred_type"] == "fewshot":
                if params["cnn_model"] == "inception_v3":
                    fewshot_folder = "fewshot/"
                else:
                    fewshot_folder = f"fewshot-{params['cnn_model']}/"
                val_preds_file = make_model_res_file(
                    data_dir=data_dir + f"{fewshot_folder}/",
                    dataset="inat2018",
                    eval_split=eval_split,
                    res_type="preds_sparse",
                    sample_ratio=params["train_sample_ratio"],
                )
                val_preds = load_sparse_feats(val_preds_file)
                print(
                    f"Load Few-Shot Pretrained {params['cnn_model']} prediction of inat_2018 {eval_split} data from {val_preds_file}"
                )
                print(f"Preidction Matrix shape -> {val_preds.shape}")
            else:
                raise Exception(
                    f"Unrecognized cnn_pred_type -> {params['cnn_pred_type']}"
                )
        if load_cnn_features:
            if params["inat2018_resolution"] == "high_res":
                val_feats = np.load(
                    data_dir
                    + "features_inception_hr/inat2018_"
                    + eval_split
                    + "_net_feats.npy"
                )
            elif params["inat2018_resolution"] == "pretrain":
                if params["cnn_model"] == "inception_v3":
                    pretrain_folder = "features_inception_pretrain/"
                else:
                    pretrain_folder = (
                        f"features_inception_pretrain-{params['cnn_model']}/"
                    )
                val_feats = np.load(
                    data_dir
                    + f"{pretrain_folder}/inat2018_"
                    + eval_split
                    + "_pretrain_net_feats.npy"
                )
                print(
                    f"Load Pretrained {params['cnn_model']} feature of inat_2018 {eval_split} data -> {val_feats.shape}"
                )
            else:
                val_feats = np.load(
                    data_dir
                    + "features_inception/inat2018_"
                    + eval_split
                    + "_net_feats.npy"
                )

        if load_cnn_features_train:
            if params["inat2018_resolution"] == "high_res":
                train_feats = np.load(
                    data_dir
                    + "features_inception_hr/inat2018_train_net_feats.npy"
                ).astype(np.float32)
            elif params["inat2018_resolution"] == "pretrain":
                if params["cnn_model"] == "inception_v3":
                    pretrain_folder = "features_inception_pretrain/"
                else:
                    pretrain_folder = (
                        f"features_inception_pretrain-{params['cnn_model']}/"
                    )
                train_feats_file = (
                    data_dir
                    + f"{pretrain_folder}/inat2018_train_pretrain_net_feats.npy"
                )
                train_feats = np.load(train_feats_file).astype(np.float32)
                print(
                    f"Load Pretrained {params['cnn_model']} feature inat_2018 of train data -> {train_feats.shape} from {train_feats_file}"
                )
            else:
                train_feats = np.load(
                    data_dir
                    + "features_inception/inat2018_train_net_feats.npy"
                ).astype(np.float32)

    elif params["dataset"] == "fmow":

        data_dir = get_paths("fmow_data_dir")
        num_classes = 62
        class_of_interest = 0  # wood thrush

        # load observations
        train_locs, train_classes, train_users, train_dates, train_inds = (
            load_inat_data(
                data_dir,
                "train_location.json",
                "train.json",
                train_remove_invalid,
            )
        )

        if eval_split == "val":
            val_locs, val_classes, val_users, val_dates, val_inds = (
                load_inat_data(
                    data_dir,
                    eval_split + "_location.json",
                    eval_split + ".json",
                    eval_remove_invalid,
                )
            )
        elif eval_split == "test":
            raise Exception("fMOW Test dataset not available")

        # load class names
        with open(data_dir + "category.json") as da:
            cls_data = json.load(da)
        classes = {v: k for k, v in cls_data.items()}

        # if load_cnn_predictions:
        #     val_preds = np.load(data_dir + 'feature_moco/fmow_' + eval_split + '_preds.npy')
        if load_cnn_predictions:
            if params["cnn_pred_type"] == "full":
                val_preds = np.load(
                    data_dir + "feature_moco/fmow_" + eval_split + "_preds.npy"
                )
            elif params["cnn_pred_type"] == "fewshot":
                val_preds_file = make_model_res_file(
                    data_dir=data_dir + "fewshot/",
                    dataset="fmow",
                    eval_split=eval_split,
                    res_type="preds",
                    sample_ratio=params["train_sample_ratio"],
                )
                val_preds = np.load(val_preds_file)
                print(
                    f"Load Few-Shot Pretrained MOCO-V3 prediction of fmow {eval_split} data from {val_preds_file}"
                )
                print(f"Preidction Matrix shape -> {val_preds.shape}")
            else:
                raise Exception(
                    f"Unrecognized cnn_pred_type -> {params['cnn_pred_type']}"
                )

        if load_cnn_features:
            if params["inat2018_resolution"] == "pretrain":
                val_feats = np.load(
                    data_dir
                    + "features_inception_pretrain/fmow_"
                    + eval_split
                    + "_pretrain_net_feats.npy"
                ).astype(np.float32)
                print(
                    f"Load Pretrained MOCO-V3 feature of {eval_split} data -> {val_feats.shape} from {data_dir + 'features_inception_pretrain/fmow_' + eval_split + '_pretrain_net_feats.npy'}"
                )
            else:
                raise Exception(f"Unknown inat2018_resolution flag")

        if load_cnn_features_train:
            if params["inat2018_resolution"] == "pretrain":
                train_feats = np.load(
                    data_dir
                    + "features_inception_pretrain/fmow_train_pretrain_net_feats.npy"
                ).astype(np.float32)
                print(
                    f"Load Pretrained MOCO-V3 feature of train data -> {train_feats.shape} from {data_dir + 'features_inception_pretrain/fmow_train_pretrain_net_feats.npy'}"
                )
            else:
                raise Exception(f"Unknown inat2018_resolution flag")

    elif params["dataset"] == "birdsnap":

        data_dir = get_paths("birdsnap_data_dir")
        ann_file_name = "birdsnap_with_loc_2019.json"
        num_classes = 500
        class_of_interest = 0

        # load observations
        train_locs, train_classes, train_users, train_dates, train_inds = (
            load_bird_data(
                data_dir,
                ann_file_name,
                "train",
                train_remove_invalid,
                params["meta_type"],
            )
        )
        val_locs, val_classes, val_users, val_dates, val_inds = load_bird_data(
            data_dir,
            ann_file_name,
            eval_split,
            eval_remove_invalid,
            params["meta_type"],
        )

        # load class names
        with open(data_dir + ann_file_name) as da:
            class_names = json.load(da)["classes"]
        # classes: a dict(), class id => class name
        classes = dict(zip(range(len(class_names)), class_names))

        if load_cnn_predictions:
            # load CNN pretrained model's image prediction of class
            val_preds = load_sparse_feats(
                data_dir
                + "features_inception/birdsnap_"
                + eval_split
                + "_preds_sparse.npz"
            )

        if load_cnn_features:
            val_feats = np.load(
                data_dir
                + "features_inception/birdsnap_"
                + eval_split
                + "_net_feats.npy"
            )

        if load_cnn_features_train:
            train_feats = np.load(
                data_dir + "features_inception/birdsnap_train_net_feats.npy"
            )

    elif params["dataset"] == "nabirds":

        data_dir = get_paths("nabirds_data_dir")
        ann_file_name = "nabirds_with_loc_2019.json"
        num_classes = 555
        class_of_interest = 0

        # load observations
        train_locs, train_classes, train_users, train_dates, train_inds = (
            load_bird_data(
                data_dir,
                ann_file_name,
                "train",
                train_remove_invalid,
                params["meta_type"],
            )
        )
        val_locs, val_classes, val_users, val_dates, val_inds = load_bird_data(
            data_dir,
            ann_file_name,
            eval_split,
            eval_remove_invalid,
            params["meta_type"],
        )

        # load class names
        with open(data_dir + ann_file_name) as da:
            class_names = json.load(da)["classes"]
        classes = dict(zip(range(len(class_names)), class_names))

        if load_cnn_predictions:
            val_preds = load_sparse_feats(
                data_dir
                + "features_inception/nabirds_"
                + eval_split
                + "_preds_sparse.npz"
            )

        if load_cnn_features:
            val_feats = np.load(
                data_dir
                + "features_inception/nabirds_"
                + eval_split
                + "_net_feats.npy"
            )

        if load_cnn_features_train:
            train_feats = np.load(
                data_dir + "features_inception/nabirds_train_net_feats.npy"
            )

    elif params["dataset"] == "yfcc":

        data_dir = get_paths("yfcc_data_dir")
        print("  No user or date features for yfcc.")
        params["use_date_feats"] = False
        params["balanced_train_loader"] = False
        num_classes = 100
        class_of_interest = 9  # beach

        # load observations
        train_locs, train_classes, train_users, train_dates = load_yfcc_data(
            data_dir, "train_test_split.csv", "train"
        )
        val_locs, val_classes, val_users, val_dates = load_yfcc_data(
            data_dir, "train_test_split.csv", eval_split
        )
        train_inds = np.arange(train_locs.shape[0])
        val_inds = np.arange(val_locs.shape[0])

        # load class names
        da = pd.read_csv(data_dir + "class_names.csv")
        classes = dict(zip(da["id"].values, da["name"].values))

        if load_cnn_predictions:
            val_preds = np.load(
                data_dir
                + "features_inception/YFCC_"
                + eval_split
                + "_preds.npy"
            )

        if load_cnn_features:
            val_feats = np.load(
                data_dir
                + "features_inception/YFCC_"
                + eval_split
                + "_net_feats.npy"
            )

        if load_cnn_features_train:
            train_feats = np.load(
                data_dir + "features_inception/YFCC_train_net_feats.npy"
            )

    elif params["dataset"] == "syntconsband":

        data_dir = get_paths("syntconsband_data_dir")
        num_classes = 10
        class_of_interest = 8  # wood thrush

        # load observations
        train_locs, train_classes, train_inds = pickle_load(
            data_dir + "/syntconsband_train.pkl"
        )
        val_locs, val_classes, val_inds = pickle_load(
            data_dir + f"/syntconsband_{eval_split}.pkl"
        )

        train_users = np.zeros(train_classes.shape[0]).astype(int)
        train_dates = np.zeros(train_classes.shape[0]).astype(float)
        val_users = np.zeros(val_classes.shape[0]).astype(int)
        val_dates = np.zeros(val_classes.shape[0]).astype(float)

        classes = dict(
            zip(
                list(range(num_classes)),
                [str(i) for i in list(range(num_classes))],
            )
        )

        if load_cnn_predictions:
            val_preds = np.ones((val_classes.shape[0], num_classes))
        if load_cnn_features:
            val_feats = None
    elif params["dataset"].startswith("vmf"):
        """
        path pattern: vmfC{num_classes}S{sample_size}L{kappa_low}H{kappa_high}/
        """
        data_dir = get_paths("vmf_data_dir")

        # load observations
        train_locs, train_classes, train_inds = pickle_load(
            data_dir + "/{}_train_locations.pkl".format(params["dataset"])
        )
        val_locs, val_classes, val_inds = pickle_load(
            data_dir
            + "/{}_{}_locations.pkl".format(params["dataset"], eval_split)
        )

        train_users = np.zeros(train_classes.shape[0]).astype(int)
        train_dates = np.zeros(train_classes.shape[0]).astype(float)
        val_users = np.zeros(val_classes.shape[0]).astype(int)
        val_dates = np.zeros(val_classes.shape[0]).astype(float)

        classes_list = list(np.unique(train_classes))

        classes = dict(zip(classes_list, classes_list))
        num_classes = len(classes_list)
        class_of_interest = 0

        if load_cnn_predictions:
            val_preds = np.ones((val_classes.shape[0], num_classes))
        if load_cnn_features:
            val_feats = None

    if load_cnn_features_train and train_remove_invalid:
        train_feats = train_feats[train_inds, :]

    if load_cnn_features and eval_remove_invalid:
        val_feats = val_feats[val_inds, :]

    if load_cnn_predictions and eval_remove_invalid:
        val_preds = val_preds[val_inds, :]

    # return data in dictionary
    op = {}
    op["train_locs"] = train_locs  # (num_train, 2), training image locations
    op["train_classes"] = (
        train_classes  # (num_train, ), training image class labels
    )
    op["train_users"] = train_users  # (num_train, ), training image user ids
    op["train_dates"] = train_dates  # (num_train, ), training dates
    op["train_inds"] = (
        train_inds  # (num_train, ), the indices training data keeps
    )
    op["train_imgs"] = train_imgs  # (num_train, ), the train image file path

    op["val_locs"] = val_locs  # (num_val, 2), val image locations
    op["val_classes"] = val_classes  # (num_val, ), val image class labels
    op["val_users"] = val_users  # (num_val, ), val image user ids
    op["val_dates"] = val_dates  # (num_val, ), float, val dates
    op["val_inds"] = val_inds  # (num_val, ), the indices val data keeps
    op["val_imgs"] = val_imgs  # (num_val, ), the val image file path

    op["class_of_interest"] = (
        class_of_interest  # (1), the class id of an interested class
    )
    op["classes"] = classes  # dict(), class id -> class labels
    op["num_classes"] = num_classes  # int, number of class

    if load_cnn_predictions:
        op["val_preds"] = (
            val_preds  # (num_val, num_classes) class predictions from trained image classifier
        )
    if load_cnn_features:
        op["val_feats"] = (
            val_feats  # (num_val, 2048), features from trained image classifier
        )
        assert val_feats.shape[0] == val_locs.shape[0]
    if load_cnn_features_train:
        op["train_feats"] = (
            train_feats  # (num_train, 2048), features from trained image classifier
        )
        assert train_feats.shape[0] == train_locs.shape[0]

    # if it exists add the data split
    try:
        op["val_split"] = val_split
    except:
        op["val_split"] = np.ones(val_locs.shape[0], dtype=np.int)

    return op


def load_sparse_feats(file_path, invert=False):
    feats = sparse.load_npz(file_path)
    feats = np.array(feats.todense(), dtype=np.float32)
    if invert:
        eps = 10e-5
        feats = np.clip(feats, eps, 1.0 - eps)
        feats = np.log(feats / (1.0 - feats))
    return feats


def load_bird_data(
    ip_dir,
    ann_file_name,
    split_name,
    remove_empty=False,
    meta_type="orig_meta",
):
    """
    Args:
        ip_dir: data file directory
        ann_file_name: the json file name
            data_orig: dict()
                key: train / valid / test
                value: a list of imageOBJ
                    each imageOBJ: dict()
                        {
                            "valid_image": True/False
                            "im_path": image data
                            "class_id": class label of image, int
                            "orig_meta":
                                {
                                    "user_id": phototgrapher id, int
                                    "lon":
                                    "lat":
                                }
                            "ebird_meta":
                                {
                                    "user_id": phototgrapher id, int
                                    "lon":
                                    "lat":
                                }
                        }

        split_name: train / valid / test
        remove_empty:
        meta_type:
            orig_meta: original metadata
            ebird_meta: the simulated metadata
    Return:
        locs: np.arrary, [batch_size, 2], location data
        classes: np.arrary, [batch_size], the list of image category id
        users: np.arrary, [batch_size], the list of user id
        dates: np.arrary, [batch_size], the list of date
        valid_inds: np.arrary, [batch_size], the list of data sample index which have valid data
    """
    print("Loading " + os.path.basename(ann_file_name) + " - " + split_name)
    print("   using meta data: " + meta_type)

    # load annotation info
    with open(ip_dir + ann_file_name) as da:
        data_orig = json.load(da)

    data = [dd for dd in data_orig[split_name] if dd["valid_image"]]
    imgs = np.array([dd["im_path"] for dd in data])
    classes = np.array([dd["class_id"] for dd in data]).astype(np.int)
    users = [dd[meta_type]["user_id"] for dd in data]
    users = np.array([-1 if uu is None else uu for uu in users]).astype(np.int)
    dates = np.array([dd[meta_type]["date"] for dd in data]).astype(np.float32)
    lon = [dd[meta_type]["lon"] for dd in data]
    lat = [dd[meta_type]["lat"] for dd in data]
    locs = (np.vstack((lon, lat)).T).astype(np.float32)

    print(f"\t {len(data_orig[split_name])} total entries")
    print(f"\t {len(data)} entries with images")

    # a valid data sample means: 1) a no None longitude; 2) a userID not 0; 3) a date not None
    valid_inds = (~np.isnan(locs[:, 0])) & (users >= 0) & (~np.isnan(dates))
    if remove_empty:
        locs = locs[valid_inds, :]
        users = users[valid_inds]
        dates = dates[valid_inds]
        classes = classes[valid_inds]

    print(f"\t {valid_inds.sum()} entries with meta data")
    if not remove_empty:
        print("\t keeping entries even without metadata")

    return locs, classes, users, dates, valid_inds


def load_inat_data(
    ip_dir, loc_file_name, ann_file_name, remove_empty=False, load_img=False
):
    """
    Args:
        ip_dir: data file directory
        loc_file_name: meta data file, contain location, date, user_id
            if '_large' in loc_file_name: also contain image label
        ann_file_name: contain image label
        load_img: whether or not load image file path
    """
    # TODO clean this up and remove loop
    print("Loading " + os.path.basename(loc_file_name))

    # load location info
    with open(ip_dir + loc_file_name) as da:
        loc_data = json.load(da)
    loc_data_dict = dict(zip([ll["id"] for ll in loc_data], loc_data))

    if "_large" in loc_file_name:
        # special case where the loc data also includes meta data such as class
        locs = [[ll["lon"], ll["lat"]] for ll in loc_data]
        dates = [ll["date_c"] for ll in loc_data]
        classes = [ll["class"] for ll in loc_data]
        users = [ll["user_id"] for ll in loc_data]
        keep_inds = np.arange(len(locs))
        print(f"\t {len(locs)} valid entries")

    else:
        # otherwise load regualar iNat data
        """
        ann_file_name: a list of dict(), each item:
        {
        "images": [{
                "id":
                }, ...],
        "annotations":[{
                "image_id": id of the image
                "category_id": class label of the image
                }, ...]
        }
        """

        # load annotation info
        with open(ip_dir + ann_file_name) as da:
            data = json.load(da)

        ids = [tt["id"] for tt in data["images"]]
        ids_all = [ii["image_id"] for ii in data["annotations"]]
        classes_all = [ii["category_id"] for ii in data["annotations"]]
        classes_mapping = dict(zip(ids_all, classes_all))
        if load_img:
            imgs_all = [tt["file_name"] for tt in data["images"]]

        # store locations and associated classes
        locs = []
        classes = []
        users = []
        dates = []
        miss_cnt = 0
        keep_inds = []
        imgs = []
        for ii, tt in enumerate(ids):

            if remove_empty and (
                (loc_data_dict[tt]["lon"] is None)
                or (loc_data_dict[tt]["user_id"] is None)
            ):
                miss_cnt += 1
            else:
                if loc_data_dict[tt]["lon"] is None:
                    loc = [np.nan, np.nan]
                else:
                    loc = [loc_data_dict[tt]["lon"], loc_data_dict[tt]["lat"]]

                if loc_data_dict[tt]["user_id"] is None:
                    u_id = -1
                else:
                    u_id = loc_data_dict[tt]["user_id"]

                locs.append(loc)
                classes.append(classes_mapping[int(tt)])
                users.append(u_id)
                dates.append(loc_data_dict[tt]["date_c"])
                keep_inds.append(ii)
                if load_img:
                    imgs.append(imgs_all[ii])

        print(f"\t {len(locs)} valid entries")
        if remove_empty:
            print(
                "\t {} entries excluded with missing meta data".format(
                    miss_cnt
                )
            )
    if load_img:
        return (
            np.array(locs).astype(np.float32),
            np.array(classes).astype(np.int),
            np.array(users).astype(np.int),
            np.array(dates).astype(np.float32),
            np.array(keep_inds),
            np.array(imgs),
        )

    else:
        return (
            np.array(locs).astype(np.float32),
            np.array(classes).astype(np.int),
            np.array(users).astype(np.int),
            np.array(dates).astype(np.float32),
            np.array(keep_inds),
        )


def load_yfcc_data(data_dir, ann_file_name, split_name):
    """
    Return:
        locs: [data_size, 2]  (lon, lat)
        classes: [data_size], class labels
        users: [data_size], all -1
        dates: [data_size], all 0
    """
    da = pd.read_csv(data_dir + ann_file_name)
    locs = da[da["split"] == split_name][["lon", "lat"]].values.astype(
        np.float32
    )
    classes = da[da["split"] == split_name]["class"].values
    users = np.ones(locs.shape[0], dtype=np.int) * -1
    dates = np.zeros(locs.shape[0], dtype=np.float32)
    return locs, classes, users, dates
