import os
import torch
import numpy as np
from appfl.misc.data import Dataset
from datasets.PreprocessedData.NREL_Preprocess import NRELDataDownloader


def get_nrel(args):
    ## sanity checks
    if args.dataset not in ["NRELCA", "NRELIL", "NRELNY"]:
        raise ValueError(
            "Currently only NREL data for CA, IL, NY are supported. Please modify download script if you want a different state."
        )

    state_idx = args.dataset[4:]
    if not os.path.isfile(args.dataset_dir + "/%sdataset.npz" % args.dataset):
        # building ID's whose data to download
        if state_idx == "CA":
            b_id = [
                15227,
                15233,
                15241,
                15222,
                15225,
                15228,
                15404,
                15429,
                15460,
                15445,
                15895,
                16281,
                16013,
                16126,
                16145,
                47395,
                15329,
                15504,
                15256,
                15292,
                15294,
                15240,
                15302,
                15352,
                15224,
                15231,
                15243,
                17175,
                17215,
                18596,
                15322,
                15403,
                15457,
                15284,
                15301,
                15319,
                15221,
                15226,
                15229,
                15234,
                15237,
                15239,
            ]
        if state_idx == "IL":
            b_id = [
                108872,
                109647,
                110111,
                108818,
                108819,
                108821,
                108836,
                108850,
                108924,
                108879,
                108930,
                108948,
                116259,
                108928,
                109130,
                113752,
                115702,
                118613,
                108816,
                108840,
                108865,
                108888,
                108913,
                108942,
                108825,
                108832,
                108837,
                109548,
                114596,
                115517,
                109174,
                109502,
                109827,
                108846,
                108881,
                108919,
                108820,
                108823,
                108828,
                108822,
                108864,
                108871,
            ]
        if state_idx == "NY":
            b_id = [
                205362,
                205863,
                205982,
                204847,
                204853,
                204854,
                204865,
                204870,
                204878,
                205068,
                205104,
                205124,
                205436,
                213733,
                213978,
                210920,
                204915,
                205045,
                204944,
                205129,
                205177,
                204910,
                205024,
                205091,
                204849,
                204860,
                204861,
                208090,
                210116,
                211569,
                204928,
                204945,
                205271,
                204863,
                204873,
                204884,
                204842,
                204843,
                204844,
                204867,
                204875,
                204880,
            ]
        downloader = NRELDataDownloader(dset_tags=[state_idx], b_id=[b_id])
        downloader.download_data()
        downloader.save_data(fname=args.dataset_dir + "/NREL%sdataset.npz" % state_idx)

    dset_raw = np.load(args.dataset_dir + "/NREL%sdataset.npz" % state_idx)["data"]
    if args.num_clients > dset_raw.shape[0]:
        raise ValueError("More clients requested than present in dataset.")
    if args.n_features != dset_raw.shape[-1]:
        raise ValueError(
            "Incorrect number of features passed as argument, the number of features present in dataset is %d."
            % dset_raw.shape[-1]
        )

    ## process dataset
    dset_reduced_clients = dset_raw[
        : np.minimum(dset_raw.shape[0], args.num_clients), :, :
    ].copy()
    dset_train = dset_reduced_clients[
        :, : int(args.train_test_boundary * dset_reduced_clients.shape[1]), :
    ].copy()
    dset_test = dset_reduced_clients[
        :, int(args.train_test_boundary * dset_reduced_clients.shape[1]) :, :
    ].copy()
    # scale to [0,1]
    for idx_f in range(dset_train.shape[-1]):
        feature_minval = dset_train[:, :, idx_f].min()
        feature_maxval = dset_train[:, :, idx_f].max()
        if (
            feature_maxval == feature_minval
        ):  # if min-max scaling isn't possible because min=max, just replace with 1
            dset_train[:, :, idx_f] = np.ones_like(dset_train[:, :, idx_f])
            dset_test[:, :, idx_f] = np.ones_like(dset_test[:, :, idx_f])
        else:  # min-max scaling
            dset_train[:, :, idx_f] = (dset_train[:, :, idx_f] - feature_minval) / (
                feature_maxval - feature_minval
            )
            dset_test[:, :, idx_f] = (dset_test[:, :, idx_f] - feature_minval) / (
                feature_maxval - feature_minval
            )
    # record as train and test datasets
    # TODO: ensure that there is enough temporal data which is greater than lookback
    train_datasets = []
    test_inputs = []
    test_labels = []
    for idx_cust in range(dset_train.shape[0]):
        # train dataset entries
        train_inputs = []
        train_labels = []
        for idx_time in range(dset_train.shape[1] - args.n_lookback):
            train_inputs.append(
                dset_train[idx_cust, idx_time : idx_time + args.n_lookback, :]
            )
            train_labels.append([dset_train[idx_cust, idx_time + args.n_lookback, 0]])
        dset = Dataset(
            torch.FloatTensor(np.array(train_inputs)),
            torch.FloatTensor(np.array(train_labels)),
        )
        train_datasets.append(dset)
        # test dataset entries
        for idx_time in range(dset_test.shape[1] - args.n_lookback):
            test_inputs.append(
                dset_test[idx_cust, idx_time : idx_time + args.n_lookback, :]
            )
            test_labels.append([dset_train[idx_cust, idx_time + args.n_lookback, 0]])
    test_dataset = Dataset(
        torch.FloatTensor(np.array(test_inputs)),
        torch.FloatTensor(np.array(test_labels)),
    )

    return train_datasets, test_dataset
