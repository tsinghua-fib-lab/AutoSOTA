# pylint: disable=E1101
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
from sklearn import metrics

from timesURL_src.collator import CLDataCollator


class TimeDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = []
        for instance in data:
            values, times, mask = instance
            if len(values) == len(times) and len(times) == len(mask) and len(values) >= 2:
                self.data.append(instance)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def evaluate(dim, rec, dec, test_loader, args, num_sample=10, device="cuda"):
    mse, test_n = 0.0, 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(args.device)
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, dim: 2 * dim],
                test_batch[:, :, -1],
            )
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean, qz0_logvar = (
                out[:, :, : args.latent_dim],
                out[:, :, args.latent_dim:],
            )
            epsilon = torch.randn(
                num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(args.device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            batch, seqlen = observed_tp.size()
            time_steps = (
                observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
            )
            pred_x = dec(z0, time_steps)
            pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            pred_x = pred_x.mean(0)
            mse += mean_squared_error(observed_data, pred_x, observed_mask) * batch
            test_n += batch
    return mse / test_n


def compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, args, device):
    observed_data, observed_mask \
        = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2 * dim]

    noise_std = args.std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if args.norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl


def evaluate_classifier(model, test_loader, dec=None, args=None, classifier=None,
                        dim=0, reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(args.device), label.to(args.device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                       :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(args.device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if args.classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    # pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x)
                else:
                    out = classifier(z0)
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    true = np.concatenate(true, 0)
    acc = np.mean(pred.argmax(1) == true)

    # print(true.shape)
    # print(pred.shape)
    # print(np.sum(pred, axis = 1))

    if args.dataset == 'physionet' or args.dataset == 'MIMIC-III':
        auc = metrics.roc_auc_score(true, pred[:, 1])
    elif args.dataset == 'PersonActivity':
        auc = 0.

    return test_loss / pred.shape[0], acc, auc


def evaluate_regressor(model, test_loader, dec=None, args=None, classifier=None, dim=0):
    total_len = 0
    test_mse_loss = 0
    test_mae_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(args.device), label.to(args.device)
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            batch_len = test_batch.shape[0]
            total_len += batch_len
            test_mse_loss += nn.MSELoss()(out[:, 0], label).item() * batch_len
            test_mae_loss += nn.L1Loss()(out[:, 0], label).item() * batch_len

    return test_mse_loss / total_len, test_mae_loss / total_len


def evaluate_interpolator(model, test_loader, dec=None, args=None, classifier=None, dim=0):
    total_values = 0
    total_mse_loss = 0
    total_mae_loss = 0

    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(args.device), label.to(args.device)
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)

            target_data, target_mask = label[:, :, :dim], label[:, :, dim:2 * dim].bool()
            num_values = torch.sum(target_mask).item()
            total_mse_loss += nn.MSELoss()(out[target_mask], target_data[target_mask]).item() * num_values
            total_mae_loss += nn.L1Loss()(out[target_mask], target_data[target_mask]).item() * num_values
            total_values += num_values

    return total_mse_loss / total_values, total_mae_loss / total_values


def subsample_timepoints(data, time_steps, mask, percentage_tp_to_sample=None):
    # Subsample percentage of points from each time series
    for i in range(data.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        subsampled_idx = sorted(np.random.choice(
            non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask


def generate_irregular_samples(data, input_dim):
    combined_data = []
    max_len = 0
    for i in range(data.shape[0]):
        zero_time_indices_list = torch.where(data[i, :, -1][1:] == 0)[0]
        curr_len = zero_time_indices_list[0].item() + 1 if len(zero_time_indices_list) else data.shape[1]
        max_len = max(max_len, curr_len)
        values = data[i, :curr_len, : input_dim]
        times = data[i, :curr_len, -1]
        mask = data[i, :curr_len, input_dim: 2 * input_dim]
        single_data = [values, times, mask]
        combined_data.append(single_data)
    return combined_data, max_len


def generate_batches(X_train, X_val, args):
    input_dim = (X_train.shape[2] - 1) // 2

    X_train, train_max_len = generate_irregular_samples(X_train, input_dim)
    # X_val, val_max_len = generate_irregular_samples(X_val, input_dim)

    # max_len = max(train_max_len, val_max_len)
    max_len = train_max_len

    pretrain_data = TimeDataset(X_train)
    # val_data = TimeDataset(X_val)

    train_cl_collator = CLDataCollator(max_len=max_len, args=args)

    # batch_size = min(min(len(val_data), args.batch_size), 256)
    batch_size = min(min(len(pretrain_data), args.batch_size), 256)
    train_dataloader = DataLoader(pretrain_data, batch_size=batch_size, shuffle=True, collate_fn=train_cl_collator,
                                  num_workers=0)
    # val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=train_cl_collator,
    #                             num_workers=8)

    data_objects = {"train_dataloader": train_dataloader,
                    # "val_dataloader": val_dataloader,
                    "input_dim": input_dim,
                    "max_len": max_len,
                    "n_train_batches": len(train_dataloader),
                    # "n_test_batches": len(val_dataloader),
                    }

    return data_objects


def get_unlabeled_pretrain_data(X_train, args):
    # X_train = torch.load(args.path + 'X_train.pt')
    # X_val = torch.load(args.path + 'X_val.pt')
    X_train = torch.from_numpy(X_train)
    print('X_train: ' + str(X_train.shape))
    # print('X_val: ' + str(X_val.shape))

    # data_objects = generate_batches(X_train, X_val, args)
    data_objects = generate_batches(X_train, None, args)

    return data_objects


def get_finetune_data(args):
    X_train, y_train = torch.load(args.path + 'X_train.pt'), torch.load(args.path + 'y_train.pt')
    X_val, y_val = torch.load(args.path + 'X_val.pt'), torch.load(args.path + 'y_val.pt')
    X_test, y_test = torch.load(args.path + 'X_test.pt'), torch.load(args.path + 'y_test.pt')
    input_dim = (X_train.shape[2] - 1) // 2

    print('X_train: ' + str(X_train.shape) + ' y_train: ' + str(y_train.shape))
    print('X_val: ' + str(X_val.shape) + ' y_val: ' + str(y_val.shape))
    print('X_test: ' + str(X_test.shape) + ' y_test: ' + str(y_test.shape))

    if args.task == 'classification':
        train_data_combined = TensorDataset(X_train, y_train.long().squeeze())
        val_data_combined = TensorDataset(X_val, y_val.long().squeeze())
        test_data_combined = TensorDataset(X_test, y_test.long().squeeze())
    elif args.task == 'regression' or args.task == 'interpolation':
        train_data_combined = TensorDataset(X_train, y_train.float())
        val_data_combined = TensorDataset(X_val, y_val.float())
        test_data_combined = TensorDataset(X_test, y_test.float())

    train_dataloader = DataLoader(train_data_combined, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data_combined, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data_combined, batch_size=args.batch_size, shuffle=False)

    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim}

    return data_objects
