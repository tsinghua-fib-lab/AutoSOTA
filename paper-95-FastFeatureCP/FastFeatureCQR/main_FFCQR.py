import os
import time
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

from datasets import datasets
from conformal import helper
from conformal.icp import (
    IcpRegressor,
    RegressorNc,
    FeatRegressorNc,
    CqrIcpRegressor,
    CqrFeatRegressorNc,
    CQR_FFCPErrorErrFunc,
    CQR_FFCP_RegressorNc,
    AbsErrorErrFunc,
    FeatErrorErrFunc,
    QuantileRegErrFunc,
)
from conformal.utils import (
    compute_coverage,
    FFCP_compute_coverage,
    seed_torch,
)
from conformal.cqr_torch_models import all_q_model


def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")


def partition_bins(x_test, y_test, num_bins):
    percentails = [np.percentile(y_test, 100 * i / num_bins) for i in range(1, num_bins)]
    percentails = [-float("inf")] + percentails + [float("inf")]
    cut_bins = pd.cut(y_test, percentails, labels=False)
    output = [(x_test[cut_bins == i], y_test[cut_bins == i]) for i in range(num_bins)]
    return output


def partition_test(x_test, y_test):
    if sum(y_test == 0) / len(y_test) > 0.5:
        zero_sample_x = x_test[y_test == 0]
        zero_sample_y = y_test[y_test == 0]
        non_zero_x = x_test[y_test != 0]
        non_zero_y = y_test[y_test != 0]
        non_zero_parts = partition_bins(non_zero_x, non_zero_y, num_bins=2)
        return [(zero_sample_x, zero_sample_y)] + non_zero_parts
    else:
        return partition_bins(x_test, y_test, num_bins=3)


def main(x_train, y_train, x_test, y_test, idx_train, idx_cal, seed, args):

    quantiles_net = [alpha, 1 - alpha]
    dir = f"ckpt/cqr_{args.data}_{quantiles_net}"

    if os.path.exists(os.path.join(dir, f"model_seed{seed}.pt")) and not args.no_resume:
        model = all_q_model(quantiles=quantiles_net, in_shape=in_shape, hidden_size=args.hidden_size, dropout=args.dropout)
        print(f"==> Load model from {dir}")
        model.load_state_dict(torch.load(os.path.join(dir, f"model_seed{seed}.pt"), map_location=device))
    else:
        model = None

    quantile_estimator = helper.AllQNet_RegressorAdapter(model=model, fit_params=None, in_shape=in_shape,
                                                         hidden_size=args.hidden_size, quantiles=quantiles_net,
                                                         learn_func=nn_learn_func, epochs=args.epochs, device=device,
                                                         batch_size=args.batch_size, dropout=args.dropout, lr=args.lr, wd=args.wd,
                                                         test_ratio=cv_test_ratio, random_state=cv_random_state,
                                                         use_rearrangement=False)

    if float(args.feat_norm) <= 0 or args.feat_norm == "inf":
        args.feat_norm = "inf"
        print("Use inf as feature norm")
    else:
        args.feat_norm = float(args.feat_norm)

    nc = RegressorNc(quantile_estimator, QuantileRegErrFunc(), is_cqr=True)

    icp = IcpRegressor(nc)

    if os.path.exists(os.path.join(dir, f"model_seed{seed}.pt")) and not args.no_resume:
        pass
    else:
        icp.fit(x_train[idx_train, :], y_train[idx_train])
        makedirs(dir)
        print(f"==> Saving model at {dir}/model_seed{seed}.pt")
        torch.save(quantile_estimator.model.state_dict(), os.path.join(dir, f"model_seed{seed}.pt"))

    icp.calibrate(x_train[idx_cal, :], y_train[idx_cal])

    predictions = icp.predict(x_test, significance=alpha)

    y_lower, y_upper = predictions[..., 0], predictions[..., 1]
    cqr_coverage_cp_qnet, cqr_length_cp_qnet, cqr_weighted_length_cp_qnet = compute_coverage(y_test, y_lower, y_upper, alpha,
                                                                                 name="CQRNet RegressorNc")

    nc = CqrFeatRegressorNc(quantile_estimator, inv_lr=args.feat_lr, inv_step=args.feat_step,
                            feat_norm=args.feat_norm, certification_method=args.cert_method)
    icp2 = CqrIcpRegressor(nc)
    icp2.calibrate(x_train[idx_cal, :], y_train[idx_cal])
    predictions = icp2.predict(x_test, significance=alpha)

    y_lower, y_upper = predictions[..., 0], predictions[..., 1]

    FCP_coverage_cp_qnet, FCP_length_cp_qnet, FCP_weighted_length_cp_qnet = compute_coverage(y_test, y_lower, y_upper, alpha,
                                                                                 name="CQR FeatRegressorNc")
    in_coverage = icp2.if_in_coverage(x_test, y_test, significance=alpha)
    FCP_tighter_coverage = np.sum(in_coverage) * 100 / len(in_coverage)
    print('tighter coverage of FeatRegressorNc is {}'.format(FCP_tighter_coverage))

    nc_FFCQR = CQR_FFCP_RegressorNc(quantile_estimator)
    FFCQR_result_list_length = []
    FFCQR_result_list_coverage = []
    for layer_index in range(5):
        length_FFCQR_each_layer = []
        coverage_FFCQR_each_layer = []
        intervals_FFCQR = nc_FFCQR.predict(quantile_estimator.model, x_train[idx_cal, :], y_train[idx_cal], x_test, y_test, layer_index,significance=alpha)
        y_lower, y_upper = intervals_FFCQR[..., 0],intervals_FFCQR[..., 1]

        coverage_FFCQR, length_FFCQR = FFCP_compute_coverage(y_test, y_lower, y_upper, alpha, name="FFCP_RegressorNc")
        coverage_FFCQR_each_layer.append(coverage_FFCQR)
        length_FFCQR_each_layer.append(length_FFCQR)
        FFCQR_result_list_length.append(length_FFCQR_each_layer)
        FFCQR_result_list_coverage.append(coverage_FFCQR_each_layer)

    return cqr_coverage_cp_qnet, cqr_length_cp_qnet, FCP_tighter_coverage, FCP_length_cp_qnet, FFCQR_result_list_coverage, FFCQR_result_list_length


"""
python main.py --fo sgd --fl 1e-2 --fs 140 --fn -1 
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=-1, type=int)
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument("--data", type=str, default="meps19", help="meps20 fb1 fb2 blog")
    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage error")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", "--bs", type=int, default=64)
    parser.add_argument("--hidden_size", "--hs", type=int, default=64)
    parser.add_argument("--dropout", "--do", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument("--feat_opt", "--fo", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--feat_lr", "--fl", type=float, default=1e-2)
    parser.add_argument("--feat_step", "--fs", type=int, default=None)
    parser.add_argument("--feat_norm", "--fn", default=-1)
    parser.add_argument("--cert_method", "--cm", type=int, default=0, choices=[0, 1, 2, 3]) 
    args = parser.parse_args()

    cqr_coverage_list, cqr_length_list = [], []
    FCQR_coverage_list, FCQR_length_list = [], []
    FFCQR_coverage_list, FFCQR_length_list = [], []

    for seed in args.seed:
        seed_torch(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
        device = torch.device("cpu") if args.device < 0 else torch.device("cuda")

        nn_learn_func = torch.optim.Adam

        # Ask for a reduced coverage when tuning the network parameters by
        # cross-validataion to avoid too concervative initial estimation of the
        # prediction interval. This estimation will be conformalized by CQR.
        quantiles_net = [0.49, 0.51]

        # ratio of held-out data, used in cross-validation
        cv_test_ratio = 0.05
        # desired miscoverage error
        # alpha = 0.1
        alpha = args.alpha
        # desired quanitile levels
        quantiles = [0.05, 0.95]
        # used to determine the size of test set
        test_ratio = 0.2
        # seed for splitting the data in cross-validation.
        cv_random_state = 1

        dataset_base_path = "./datasets/"
        dataset_name = args.data
        X, y = datasets.GetDataset(dataset_name, dataset_base_path)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        n_train = x_train.shape[0]
        in_shape = x_train.shape[1]
        out_shape = y_train.shape[1] if len(y_train.shape) > 1 else 1

        print("Dataset: %s" % (dataset_name))
        print("Dimensions: train set (n=%d, p=%d) ; test set (n=%d, p=%d)" %
              (x_train.shape[0], x_train.shape[1], x_test.shape[0], x_test.shape[1]))

        # divide the data into proper training set and calibration set
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train / 2))
        idx_train, idx_cal = idx[:n_half], idx[n_half:2 * n_half]

        # zero mean and unit variance scaling
        scalerX = StandardScaler()
        scalerX = scalerX.fit(x_train[idx_train])
        x_train = scalerX.transform(x_train)
        x_test = scalerX.transform(x_test)

        # scale the labels by dividing each by the mean absolute response
        mean_y_train = np.mean(np.abs(y_train[idx_train]))
        y_train = np.squeeze(y_train) / mean_y_train
        y_test = np.squeeze(y_test) / mean_y_train

        cqr_covrage, cqr_length, FCQR_coverage, FCQR_length, FFCQR_coverage, FFCQR_length\
            = main(x_train, y_train, x_test, y_test, idx_train, idx_cal, seed, args)
        cqr_coverage_list.append(cqr_covrage)
        cqr_length_list.append(cqr_length)
        FCQR_coverage_list.append(FCQR_coverage)
        FCQR_length_list.append(FCQR_length)
        FFCQR_length_list.append(FFCQR_length)
        FFCQR_coverage_list.append(FFCQR_coverage)

    average_cqr_coverage = sum(cqr_coverage_list) / len(cqr_coverage_list)
    average_FCQR_coverage = sum(FCQR_coverage_list) / len(FCQR_coverage_list)
    average_cqr_length = sum(cqr_length_list) / len(cqr_length_list)
    average_FCQR_length = sum(FCQR_length_list) / len(FCQR_length_list)


    print(f'VanillaCQR coverage: {np.mean(average_cqr_coverage)} \\pm {np.std(cqr_coverage_list)}',
          f'VanillaCQR length: {np.mean(average_cqr_length)} \\pm {np.std(cqr_length_list)}')
    print(f'FeatureCQR coverage: {np.mean(average_FCQR_coverage)} \\pm {np.std(FCQR_coverage_list)}',
          f'FeatureCQR estimated length: {np.mean(average_FCQR_length)} \\pm {np.std(FCQR_length_list)}')

    coverage_each_layer_array = np.array(FFCQR_coverage_list)
    length_each_layer_array = np.array(FFCQR_length_list)

    coverage_each_layer_array = np.squeeze(coverage_each_layer_array)
    length_each_layer_array = np.squeeze(length_each_layer_array)

    coverage_each_layer_array = np.transpose(coverage_each_layer_array)
    length_each_layer_array = np.transpose(length_each_layer_array)

    for k in range(coverage_each_layer_array.shape[0]):
        print("layer: %d" % k)
        print(f'FFCP each layer coverage: {np.mean(coverage_each_layer_array[k])} \\pm {np.std(coverage_each_layer_array[k])}',
            f'FFCP each layer length: {np.mean(length_each_layer_array[k])} \\pm {np.std(length_each_layer_array[k])}')
