import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# Classification data loading & normalization (P12 / P19 / PAM)
# ============================================================
def get_data_split(base_path='./data/P12', split_path='', dataset='P12'):
    if dataset in ('P12', 'PAM'):
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
    elif dataset == 'P19':
        Pdict_list = np.load(base_path + '/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)
    else:
        raise ValueError(f"Unknown classification dataset: {dataset}")

    idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    y = arr_outcomes[:, -1].reshape((-1, 1))
    return Ptrain, Pval, Ptest, y[idx_train], y[idx_val], y[idx_test]


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f) if vals_f.size > 0 else 0.0
        stdf[f] = np.std(vals_f)
    return mf, stdf


def getStats_static(P_tensor, dataset='P12'):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

    if dataset == 'P12':
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == 'P19':
        bool_categorical = [0, 1, 0, 0, 0, 0]
    else:
        bool_categorical = [0] * S

    for s in range(S):
        if bool_categorical[s] == 0:
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s)
            ss[s] = np.std(vals_s)
    return ms, ss


def mask_normalize(P_tensor, mf, stdf):
    """Normalize time series variables; missing values are zeroed out post-normalization."""
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    M = (P_tensor > 0).astype(P_tensor.dtype)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    return np.concatenate([Pnorm_tensor, M], axis=2)


def mask_normalize_static(P_tensor, ms, ss):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    for s in range(S):
        Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0
    return Ps.reshape((S, N)).transpose((1, 0))


def tensorize_normalize_extract_feature_pam_patch(P, y, mf, stdf, args):
    """PAM-specific patch tensorization (no static features, time = linspace)."""
    T, F = P[0].shape
    P_time = np.zeros((len(P), T, 1))
    for i in range(len(P)):
        P_time[i] = torch.linspace(0, T, T).reshape(-1, 1)
    P_tensor = mask_normalize(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)
    P_time = torch.Tensor(P_time) / 60.0

    y_tensor = torch.Tensor(y[:, 0]).type(torch.LongTensor)

    combined_vals, combined_mask, combined_tt = [], [], []
    for b in range(P_tensor.shape[0]):
        record = P_tensor[b]
        X_list, Mask_list, T_list = [], [], []

        cur_x = record[:, :F]
        cur_mask = record[:, F: 2 * F]
        cur_t = P_time[b][:, 0]
        if cur_t[0] == 0.0:
            cur_t[0] = 0.0001

        st, ed = 0, args.patch_size / 60
        step = args.stride / 60
        for _ in range(args.npatch):
            cur_ind = np.where((cur_t > st) & (cur_t <= ed))[0]
            if len(cur_ind) == 0:
                X_list.append(torch.zeros(size=[1, F]))
                Mask_list.append(torch.zeros(size=[1, F]))
                T_list.append(torch.zeros(size=[1]))
            else:
                X_list.append(torch.tensor(cur_x[cur_ind]))
                Mask_list.append(torch.tensor(cur_mask[cur_ind]))
                T_list.append(torch.tensor(cur_t[cur_ind]))
            st += step
            ed += step

        combined_vals.append(pad_sequence(X_list).float())
        combined_mask.append(pad_sequence(Mask_list).float())
        combined_tt.append(pad_sequence(T_list).float())

    observed_data = pad_sequence(combined_vals, batch_first=True).permute(0, 2, 1, 3)
    observed_tp = pad_sequence(combined_tt, batch_first=True).unsqueeze(-1).repeat(1, 1, 1, F).permute(0, 2, 1, 3)
    observed_mask = pad_sequence(combined_mask, batch_first=True).permute(0, 2, 1, 3)

    return observed_data, observed_mask, None, observed_tp, y_tensor, observed_data.shape[2]


def tensorize_normalize_extract_feature_patch(P, y, mf, stdf, ms, ss, args):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time_tensor = np.zeros((len(P), T, F))
    P_static_tensor = np.zeros((len(P), D))
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P[i]['time'] = P[i]['time'] / 60 if T == 60 else P[i]['time'] / 2880
        P_time_tensor[i] = P[i]['time']
        P_static_tensor[i] = P[i]['extended_static']

    P_tensor = mask_normalize(P_tensor, mf, stdf)
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)

    y_tensor = torch.Tensor(y[:, 0]).type(torch.LongTensor)

    combined_vals, combined_mask, combined_tt = [], [], []
    for b in range(P_tensor.shape[0]):
        record = P_tensor[b]
        X_list, Mask_list, T_list = [], [], []

        cur_x = record[:, :F]
        cur_mask = record[:, F: 2 * F]
        cur_t = P_time_tensor[b][:, 0]
        if cur_t[0] == 0.0:
            cur_t[0] = 0.0001

        # P12 uses 48h normalization (48-hour ICU window);
        # P19 / PAM use 60-unit normalization (60-hour ICU window for P19, 60-unit linspace for PAM).
        if args.dataset == 'P12':
            st, ed = 0, args.patch_size / 48
            step = args.stride / 48
        else:
            st, ed = 0, args.patch_size / 60
            step = args.stride / 60

        for i in range(args.npatch):
            cur_ind = np.where((cur_t > st) & (cur_t <= ed))[0]
            if len(cur_ind) == 0:
                X_list.append(torch.zeros(size=[1, F]))
                Mask_list.append(torch.zeros(size=[1, F]))
                T_list.append(torch.zeros(size=[1]))
            else:
                X_list.append(torch.tensor(cur_x[cur_ind]))
                Mask_list.append(torch.tensor(cur_mask[cur_ind]))
                T_list.append(torch.tensor(cur_t[cur_ind]))
            st += step
            ed += step

        combined_vals.append(pad_sequence(X_list).float())
        combined_mask.append(pad_sequence(Mask_list).float())
        combined_tt.append(pad_sequence(T_list).float())

    observed_data = pad_sequence(combined_vals, batch_first=True).permute(0, 2, 1, 3)
    observed_tp = pad_sequence(combined_tt, batch_first=True).unsqueeze(-1).repeat(1, 1, 1, F).permute(0, 2, 1, 3)
    observed_mask = pad_sequence(combined_mask, batch_first=True).permute(0, 2, 1, 3)

    return observed_data, observed_mask, torch.FloatTensor(P_static_tensor), observed_tp, y_tensor, observed_data.shape[2]


def evaluate_model_patch(model, P_tensor, P_mask_tensor, P_static_tensor, P_time_tensor,
                         batch_size=100, n_classes=2):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_mask_tensor = P_mask_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()

    if P_static_tensor is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.cuda()

    N = P_tensor.shape[0]
    n_batches, rem = N // batch_size, N % batch_size
    out = torch.zeros(N, n_classes)
    start = 0
    for _ in range(n_batches):
        P = P_tensor[start:start + batch_size]
        P_time = P_time_tensor[start:start + batch_size]
        P_mask = P_mask_tensor[start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        out[start:start + batch_size] = model.classification(P, P_time, P_mask, Pstatic).detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem]
        P_time = P_time_tensor[start:start + rem]
        P_mask = P_mask_tensor[start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + rem]
        out[start:start + rem] = model.classification(P, P_time, P_mask, Pstatic).detach().cpu()
    return out


# ============================================================
# t-SNE visualization (Figure 4 in the paper)
# ============================================================
def t_sne_visualize(model, y_test, P_tensor, P_mask_tensor, P_static_tensor, P_time_tensor,
                    batch_size=100, save_path='./logs/tsne_plot.png'):
    model.eval()
    unique_labels = np.unique(y_test)
    num_classes = len(unique_labels)
    print(f"Detected {num_classes} classes: {unique_labels}")

    P_tensor = P_tensor.cuda()
    P_mask_tensor = P_mask_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    if P_static_tensor is not None:
        P_static_tensor = P_static_tensor.cuda()

    N = P_tensor.shape[0]
    n_batches, rem = N // batch_size, N % batch_size
    all_features = []
    start = 0

    with torch.no_grad():
        for _ in range(n_batches):
            P = P_tensor[start:start + batch_size]
            P_mask = P_mask_tensor[start:start + batch_size]
            P_time = P_time_tensor[start:start + batch_size]
            Pstatic = P_static_tensor[start:start + batch_size] if P_static_tensor is not None else None
            h = model.classification(P, P_time, P_mask, Pstatic, feature=True)
            all_features.append(h.detach().cpu().numpy())
            start += batch_size
        if rem > 0:
            P = P_tensor[start:start + rem]
            P_mask = P_mask_tensor[start:start + rem]
            P_time = P_time_tensor[start:start + rem]
            Pstatic = P_static_tensor[start:start + rem] if P_static_tensor is not None else None
            h = model.classification(P, P_time, P_mask, Pstatic, feature=True)
            all_features.append(h.detach().cpu().numpy())

    features = np.concatenate(all_features, axis=0).reshape(N, -1)
    print(f"Running t-SNE for {features.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(12, 8))
    cmap = plt.cm.get_cmap('tab10' if num_classes <= 10 else 'Set3')
    for idx, label in enumerate(unique_labels):
        mask = (y_test.ravel() == label)
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                    label=f'Class {int(label)}', color=cmap(idx), alpha=0.6, s=20)
    plt.legend(loc='best', title="Labels")
    plt.title(f't-SNE Visualization ({num_classes} Classes)')
    plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved: {save_path}")


# ============================================================
# Forecasting batch utilities (used by collate_fn)
# ============================================================
def normalize_masked_data(data, mask, att_min, att_max):
    scale = att_max - att_min
    scale = scale + (scale == 0) * 1e-8
    data_norm = (data - att_min) / scale
    data_norm[mask == 0] = 0
    if torch.isnan(data_norm).any():
        raise Exception("nans!")
    return data_norm


def normalize_masked_tp(data, att_min, att_max):
    scale = att_max - att_min
    scale = scale + (scale == 0) * 1e-8
    data_norm = (data - att_min) / scale
    if torch.isnan(data_norm).any():
        raise Exception("nans!")
    return data_norm


def get_device(tensor):
    if tensor.is_cuda:
        return torch.device(tensor.get_device())
    return torch.device("cuda:0")


def add_mask(data_dict):
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]
    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))
    data_dict["observed_mask"] = mask
    return data_dict


def split_data_forecast(data_dict, dataset, n_observed_tp):
    split_dict = {
        "observed_data": data_dict["data"][:, :n_observed_tp, :].clone(),
        "observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
        "data_to_predict": data_dict["data"][:, n_observed_tp:, :].clone(),
        "tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone(),
    }
    split_dict["observed_mask"] = None
    split_dict["mask_predicted_data"] = None
    split_dict["labels"] = None
    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"][:, n_observed_tp:].clone()
    split_dict["mode"] = "forecast"
    return split_dict


def split_and_subsample_batch(data_dict, args, n_observed_tp):
    return add_mask(split_data_forecast(data_dict, args.dataset, n_observed_tp))


def split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices, max_len=None):
    """Pack observations into (B, npatch, max_patch_len, n_dim) tensors for patch-based models."""
    device = get_device(data_dict["data"])

    split_dict = {
        "tp_to_predict": data_dict["tp_to_predict"].clone(),
        "data_to_predict": data_dict["data_to_predict"].clone(),
        "mask_predicted_data": data_dict["mask_predicted_data"].clone(),
    }

    observed_tp = data_dict["time_steps"].clone()
    observed_data = data_dict["data"].clone()
    observed_mask = data_dict["mask"].clone()

    n_batch, n_tp, n_dim = observed_data.shape
    observed_tp_patches = observed_tp.view(1, 1, -1, 1).repeat(n_batch, args.npatch, 1, n_dim)
    observed_data_patches = observed_data.view(n_batch, 1, n_tp, n_dim).repeat(1, args.npatch, 1, 1)
    observed_mask_patches = observed_mask.view(n_batch, 1, n_tp, n_dim).repeat(1, args.npatch, 1, 1)

    max_patch_len = args.maxlen
    for i in range(args.npatch):
        indices = patch_indices[i]
        if len(indices) == 0:
            continue
        st_ind, ed_ind = indices[0], indices[-1]
        n_data_points = observed_mask[:, st_ind:ed_ind + 1].sum(dim=1).max().item()
        max_patch_len = max(max_patch_len, int(n_data_points))

    observed_mask_patches_fill = torch.zeros_like(observed_mask_patches, dtype=observed_mask.dtype)
    patch_indices_final = torch.full((n_batch, args.npatch, max_patch_len, n_dim), n_tp).to(device)
    observed_mask_patches_fill_reindex = torch.zeros_like(patch_indices_final, dtype=observed_mask.dtype)
    aux_tensor = torch.arange(max_patch_len).view(1, max_patch_len, 1).repeat(n_batch, 1, n_dim).to(device)

    for i in range(args.npatch):
        indices = patch_indices[i]
        if len(indices) == 0:
            continue
        st_ind, ed_ind = indices[0], indices[-1]
        observed_mask_patches_fill[:, i, st_ind:ed_ind + 1] = observed_mask[:, st_ind:ed_ind + 1, :]
        L = observed_mask[:, st_ind:ed_ind + 1, :].sum(dim=1, keepdim=True)
        observed_mask_patches_fill_reindex[:, i] = (aux_tensor < L)

    mask_inds = torch.nonzero(observed_mask_patches_fill_reindex.permute(0, 1, 3, 2), as_tuple=True)
    ind_values = torch.nonzero(observed_mask_patches_fill.permute(0, 1, 3, 2), as_tuple=True)[-1]
    patch_indices_final.index_put_((mask_inds[0], mask_inds[1], mask_inds[3], mask_inds[2]), ind_values)

    pad_zeros = torch.zeros([n_batch, args.npatch, 1, n_dim]).to(device)
    observed_tp_patches = torch.cat([observed_tp_patches, pad_zeros], dim=2).gather(2, patch_indices_final)
    observed_data_patches = torch.cat([observed_data_patches, pad_zeros], dim=2).gather(2, patch_indices_final)
    observed_mask_patches = torch.cat([observed_mask_patches, pad_zeros], dim=2).gather(2, patch_indices_final)

    split_dict["observed_tp"] = observed_tp_patches
    split_dict["observed_data"] = observed_data_patches
    split_dict["observed_mask"] = observed_mask_patches
    return split_dict


def get_max_patch_len(total_dataset, args):
    global_max_patch_len = 0
    for _, tt, _, _ in total_dataset:
        observed_tp = tt[torch.lt(tt, args.history)]
        st, ed = 0, args.patch_size
        for i in range(args.npatch):
            if i == args.npatch - 1:
                inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
            else:
                inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
            global_max_patch_len = max(global_max_patch_len, len(inds))
            st += args.stride
            ed += args.stride
    print(f"Global Max Patch Length Found: {global_max_patch_len}")
    return global_max_patch_len


# ============================================================
# Misc utilities
# ============================================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=(), displaying=True, saving=True, debug=False, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if saving:
        fh = logging.FileHandler(logpath, mode=mode)
        fh.setLevel(logger.level)
        logger.addHandler(fh)
    if displaying:
        ch = logging.StreamHandler()
        ch.setLevel(logger.level)
        logger.addHandler(ch)
    logger.info(filepath)
    for f in package_files:
        logger.info(f)
        with open(f, 'r') as pf:
            logger.info(pf.read())
    return logger


def inf_generator(iterable):
    """Loop forever over a DataLoader."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def get_next_batch(dataloader):
    return next(dataloader)
