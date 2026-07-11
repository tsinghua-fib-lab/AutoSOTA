import os

import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset


class MultiviewData(Dataset):
    def __init__(self, db, device, path="datasets/", missing_ratio=None):
        self.data_views = list()

        if db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, 'MSRC_v1-missing-0.1.mat'))
            X1 = mat['msr1'].astype(np.float32)
            X2 = mat['msr2'].astype(np.float32)
            X3 = mat['msr3'].astype(np.float32)
            X4 = mat['msr4'].astype(np.float32)
            X5 = mat['msr5'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.data_views.append(X4)
            self.data_views.append(X5)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['truth'])).astype(np.int32)

        elif db == "MNIST-USPS":
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS_missing_0.7.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1.reshape(X1.shape[0], -1))
            self.data_views.append(X2.reshape(X2.shape[0], -1))
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "BDGP":
            if missing_ratio is not None:
                fname = f'BDGP-missing-{missing_ratio}.mat'
            else:
                fname = 'BDGP-missing-0.3.mat'
            mat = sio.loadmat(os.path.join(path, fname))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "Fashion":
            mat = sio.loadmat(os.path.join(path, 'Fashion-missing-0.5.mat'))
            X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db in ["FMNIST", "mnist"]:
            mat = sio.loadmat(os.path.join(path, 'mnist4-missing-0.3.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            X3 = mat['X3'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "scene15":
            mat = sio.loadmat(os.path.join(path, '15scene-missing-0.7.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        else:
            raise NotImplementedError

        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_data_views = list()
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            sub_data_views.append(data_view[index])

        return sub_data_views, self.labels[index]


def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=False,
        drop_last=False,
    )

    return mv_data_loader, num_views, num_samples, num_clusters
