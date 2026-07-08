import os
import lmdb
import h5py
import numpy as np
import sigpy as sp
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.helper import parse_int_list
from pathlib import Path
from PIL import Image


class ImageFolder(Dataset):
    def __init__(self, root, 
                 id_list=None,           # string, e.g., '0-9,2-5'
                 resolution=256,
                 num_channels=3, 
                 img_ext='png'):
        super().__init__()
        self.root = root
        self.resolution = resolution
        self.num_channels = num_channels
        self.resizer = transforms.Resize((resolution, resolution))
        id_list = parse_int_list(id_list)
        if id_list is None:
            # search for all images in the folder
            # Define the file extensions to search for
            extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
            img_path_list = [file for ext in extensions for file in Path(root).rglob(ext)]
            img_path_list = sorted(img_path_list)
            self.id2path = {i: img_path for i, img_path in enumerate(img_path_list)}
            self.length = len(img_path_list)
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.id2path = {i: os.path.join(self.root, f'{str(id).zfill(5)}.{img_ext}') for i, id in enumerate(id_list)}
            self.length = len(id_list)
            self.id_list = id_list

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.id2path[idx]
        img = self.load_raw_image(img_path)
        img = self.normalize(img)
        img = torch.from_numpy(img).to(torch.float32)
        if img.shape[-1] != self.resolution:
            img = self.resizer(img)
        return {'target': img}

    def save_image(self, img, img_path):
        '''
        Save the image.
        Args:
            - img: image, (C, H, W), ndarray, np.uint8.
            - img_path: path to save the image, str.
        '''
        img = img.transpose(1, 2, 0)    # (C, H, W) -> (H, W, C)
        img = Image.fromarray(img)
        img.save(img_path)


    def load_raw_image(self, img_path):
        '''
        Load the image and convert it to CHW format.
        Args:
            - img_path: path to the image, str.
        Returns:
            - img: image, (C, H, W), ndarray, np.uint8.
        '''
        img = np.array(Image.open(img_path))
        img = img.transpose(2, 0, 1)    # (H, W, C) -> (C, H, W)
        return img


    def normalize(self, img):
        '''
        Normalize the image to [-1, 1].
        Args:
            - img: image, (C, H, W), numpy array.
        Returns:
            - img: image, (C, H, W), numpy array.
        '''
        img = img / 127.5 - 1.0
        return img


    def unnormalize(self, img):
        '''
        Normalize the image to [0, 1]
        Args:
            - img: image, (C, H, W), numpy array.
        Returns:
            - img: image, (C, H, W), numpy array.
        '''
        img = (img + 1.0) / 2.0
        return img


class LMDBData(Dataset):
    def __init__(self, root, 
                 resolution=128,
                 raw_resolution=128,
                 num_channels=1,
                 norm=True,
                 mean=0.0, std=5.0, id_list=None):
        super().__init__()
        self.root = root
        self.open_lmdb()
        self.resolution = resolution
        self.raw_resolution = raw_resolution
        self.num_channels = num_channels
        self.norm = norm
        if id_list is None:
            self.length = self.txn.stat()['entries']
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.idx_map(idx)
        key = f'{idx}'.encode('utf-8')
        img_bytes = self.txn.get(key)
        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(self.num_channels, self.raw_resolution, self.raw_resolution)
        if self.resolution != self.raw_resolution:
            img = TF.resize(torch.from_numpy(img.copy()), self.resolution, antialias=True)
        if self.norm:
            img = self.normalize(img)
        return {'target': img}

    def open_lmdb(self):
        self.env = lmdb.open(self.root, readonly=True, lock=False, create=False)
        self.txn = self.env.begin(write=False)

    def normalize(self, data):
        # By default, we normalize to zero mean and 0.5 std.
        return (data - self.mean) / (2 * self.std)

    def unnormalize(self, data):
        return data * 2 * self.std + self.mean


class BlackHole(Dataset):
    def __init__(self, root, resolution=64, original_resolution=400,
                 random_flip=True, zoom_in_out=True, zoom_range=[0.833, 1.145], id_list=None):
        super().__init__()
        self.root = root
        self.open_lmdb()
        self.resolution = resolution
        self.original_resolution = original_resolution
        self.length = self.txn.stat()['entries']
        self.random_flip = random_flip
        self.zoom_in_out = zoom_in_out
        self.zoom_range = zoom_range

        if id_list is None:
            self.length = self.txn.stat()['entries']
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list

    def __len__(self):
        return self.length

    def open_lmdb(self):
        self.env = lmdb.open(self.root, readonly=True, lock=False, create=False)
        self.txn = self.env.begin(write=False)

    def __getitem__(self, idx):
        key = f'{self.idx_map(idx)}'.encode('utf-8')
        img_bytes = self.txn.get(key)
        img = np.frombuffer(img_bytes, dtype=np.float64).reshape(1, self.original_resolution, self.original_resolution)
        img = torch.from_numpy(np.array(img, copy=True))
        if self.zoom_in_out:
            scale = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            zoom_shape = [
                int(self.resolution * scale),
                int(self.resolution * scale)
            ]
            img = TF.resize(img, zoom_shape, antialias=True)
            if zoom_shape[0] > self.resolution:
                img = TF.center_crop(img, self.resolution)
            elif zoom_shape[0] < self.resolution:
                diff = self.resolution - zoom_shape[0]
                img = TF.pad(
                    img,
                    (diff // 2 + diff % 2, diff // 2 + diff % 2, diff // 2, diff // 2)
                )
        else:
            img = TF.resize(img, (self.resolution, self.resolution), antialias=True)

        # normalize image
        img /= img.max()
        img = 2 * img - 1

        if self.random_flip and np.random.rand() < 0.5:
            img = torch.flip(img, [2])  # left-right flip
        if self.random_flip and np.random.rand() < 0.5:
            img = torch.flip(img, [1])  # top-down flip
        return {'target': img}


class ImageDataset(Dataset):
    """
        A concrete class for handling image datasets, inherits from DiffusionData.

        This class is responsible for loading images from a specified directory,
        applying transformations to center crop the squared images of given resolution.

        Supported extension : ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        Output data range   : [-1, 1]
    """

    def __init__(self, root, resolution=256, device='cuda', start_id=None, end_id=None):
        # Define the file extensions to search for
        extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.data = [file for ext in extensions for file in Path(root).rglob(ext)]
        self.data = sorted(self.data)

        # Subset the dataset
        self.data = self.data[start_id: end_id]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution)
        ])
        self.res = resolution
        self.device = device

    def __getitem__(self, i):
        img = (self.trans(Image.open(self.data[i])) * 2 - 1).to(self.device)
        return {'target': img}

    def __len__(self):
        return len(self.data)

    def unnormalize(self, data):
        return (data + 1.0) / 2


class MultiCoilMRIData(Dataset):
    def __init__(self, root, image_size, mvue_only=False, slice_range=[5, -5], id_list=None, simulated_kspace=False):
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.mvue_only = mvue_only
        self.simulated_kspace = simulated_kspace
        self.data = []
        for fname in tqdm(sorted(self.root.iterdir()), desc='Loading data'):
            if 'brain' in str(fname) and 'T2' not in str(fname):
                continue
            with h5py.File(fname, 'r') as f:
                for slice_idx in range(slice_range[0], len(f['s_maps'])+slice_range[1]):
                    self.data.append((fname, slice_idx))
            if 'brain' in str(fname) and len(self.data) > 500:
                break
        if id_list is None:
            self.length = len(self.data)
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list

    @staticmethod
    def get_rss(kspace):
        return np.sqrt(np.sum(np.square(np.abs(sp.ifft(kspace))), axis=0, keepdims=True))

    @staticmethod
    def get_mvue(kspace, s_maps):
        ''' Get mvue estimate from coil measurements '''
        return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))

    @staticmethod
    def unnormalize(gen_img, estimated_mvue):
        scaling = np.quantile(np.abs(estimated_mvue), 0.99)
        return gen_img * scaling

    @staticmethod
    def normalize(gen_img, estimated_mvue):
        scaling = np.quantile(np.abs(estimated_mvue), 0.99)
        return gen_img / scaling

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fname, slice_idx = self.data[self.idx_map(idx)]
        with h5py.File(str(fname).replace('_sens_maps_espirit', ''), 'r') as f:
            gt_ksp = f['kspace'][slice_idx]
        with h5py.File(fname, 'r') as f:
            maps = f['s_maps'][slice_idx]

        # Crop extra lines and reduce FoV in phase-encode
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0], gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space

        # Crop extra lines and reduce FoV in phase-encode
        maps = sp.fft(maps, axes=(-2, -1)) # These are now maps in k-space
        maps = sp.resize(maps, (maps.shape[0], maps.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0], maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,)) # Back to k-space
        maps = sp.ifft(maps, axes=(-2, -1)) # Finally convert back to image domain

        # Find mvue image
        mvue = self.get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape))
        mvue_scaled = self.normalize(mvue, mvue)

        if self.mvue_only:
            return np.concatenate([mvue_scaled.real, mvue_scaled.imag], axis=0)

        if self.simulated_kspace:
            gt_ksp_scaled = sp.fft(maps * mvue_scaled, axes=(-2, -1))
        else:
            gt_ksp_scaled = self.normalize(gt_ksp, mvue)

        # Find rss image
        rss_scaled = self.get_rss(gt_ksp_scaled).astype(np.float64)

        # Output
        return {
            'target': torch.view_as_real(torch.from_numpy(mvue_scaled).squeeze(0)).permute(2, 0, 1).contiguous(),
            'mvue': mvue_scaled,
            'rss': rss_scaled,
            'maps': maps,
            'kspace': gt_ksp_scaled,
            'fname': str(fname),
            'slice_idx': slice_idx
        }


class MultiCoilMRILMDBData(Dataset):
    def __init__(self, root, image_size, mvue_only=False, slice_range=[5, -5], id_list=None, simulated_kspace=False):
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.mvue_only = mvue_only
        self.simulated_kspace = simulated_kspace
        if not mvue_only:
            kspace_env = lmdb.open(str(self.root / "kspace"), readonly=True, lock=False, create=False)
            s_maps_env = lmdb.open(str(self.root / "s_maps"), readonly=True, lock=False, create=False)
            self.kspace_txn = kspace_env.begin(write=False)
            self.s_maps_txn = s_maps_env.begin(write=False)
        mvue_env = lmdb.open(str(self.root / "mvue"), readonly=True, lock=False, create=False)
        self.mvue_txn = mvue_env.begin(write=False)
        if id_list is None:
            self.length = self.mvue_txn.stat()['entries']
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = f'{self.idx_map(idx)}'.encode('utf-8')
        mvue_bytes = self.mvue_txn.get(key)
        mvue = np.frombuffer(mvue_bytes, dtype=np.complex64)
        mvue = mvue.reshape(1, self.image_size[0], self.image_size[1])
        mvue_scaled = MultiCoilMRIData.normalize(mvue, mvue)
        if self.mvue_only:
            return np.concatenate([mvue_scaled.real, mvue_scaled.imag], axis=0)
        else:
            s_maps_bytes = self.s_maps_txn.get(key)
            s_maps = np.frombuffer(s_maps_bytes, dtype=np.complex64)
            maps = s_maps.reshape(-1, self.image_size[0], self.image_size[1])
            if self.simulated_kspace:
                gt_ksp_scaled = sp.fft(maps * mvue_scaled, axes=(-2, -1))
            else:
                kspace_bytes = self.kspace_txn.get(key)
                kspace = np.frombuffer(kspace_bytes, dtype=np.complex64)
                gt_ksp = kspace.reshape(-1, self.image_size[0], self.image_size[1])
                gt_ksp_scaled = MultiCoilMRIData.normalize(gt_ksp, mvue)
            return {
                'target': torch.view_as_real(torch.from_numpy(mvue_scaled).squeeze(0)).permute(2, 0, 1).contiguous(),
                'mvue': mvue_scaled,
                'maps': maps,
                'kspace': gt_ksp_scaled,
            }