import logging
import h5py
import os
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import cv2
import glob

from .data_utils import read_paths

def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth

def _require(args, attr, dataset):
    val = getattr(args, attr, None)
    if val is None:
        print(f"[skip] --{attr} not set, skipping {dataset}.")
    return val


def change_to_sunrgbd(args):
    args.dataset = "sunrgbd"

    args.do_kb_crop = False
    args.max_depth_eval = 8
    args.min_depth_eval = 0.001
    args.garg_crop = False

    args.data_path = _require(args, "sunrgbd_root", "sunrgbd")
    if args.data_path is not None and not args.data_path.endswith('/'):
        args.data_path += '/'

    args.val_image_path_sunrgbd = "./data_splits/sunrgbd/sunrgbd_testing_images.txt"
    args.val_ground_truth_path_sunrgbd = "./data_splits/sunrgbd/sunrgbd_testing_depths.txt"
    args.txt_path_eval = "./text/text_all/sunrgbd/sunrgbd_all.txt"


def change_to_ibims(args):
    args.dataset = "ibims"

    args.max_depth_eval = 10
    args.min_depth_eval = 0.001

    args.ibims_root = _require(args, "ibims_root", "ibims")
    args.txt_path_eval = "./text/text_all/ibims1/ibims1_all.txt"

def change_to_diode_outdoor(args):
    args.dataset = "diode_outdoor"

    args.max_depth_eval = 80
    args.min_depth_eval = 0.001

    args.diode_root = _require(args, "diode_root", "diode_outdoor")
    args.txt_path_eval = "./text/text_all/diode_outdoor/diode_outdoor_all.txt"

def change_to_vkitti2(args):
    args.dataset = "vkitti2"
    args.split = 'test'

    args.max_depth_eval = 80
    args.min_depth_eval = 0.001

    args.do_kb_crop = True
    args.vkitti2_root = _require(args, "vkitti2_root", "vkitti2")
    args.txt_path_eval = "./text/text_all/vkitti2/vkitti2_all.txt"

def change_to_hypersim(args):
    args.dataset = "hypersim"

    args.max_depth_eval = 80
    args.min_depth_eval = 0.001

    args.hypersim_root = _require(args, "hypersim_root", "hypersim")
    args.txt_path_eval = "./text/text_all/hypersim/hypersim_all.txt"

class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.normalize = lambda x : x
        self.resize = transforms.Resize((480, 640))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        if sample['dataset'] == 'diode_outdoor' or sample['dataset'] == 'hypersim':
            image = self.resize(image)
        depth = self.to_tensor(depth)

        dataset = sample['dataset']
        return {**sample, 
                'image': image, 
                'depth': depth, 
                'dataset': dataset}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
        
class SUNRGBDDataset(Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) camera intrinsics matrix

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
    '''

    def __init__(self, args):

        self.args = args
        self.image_paths = read_paths(args.data_path, args.val_image_path_sunrgbd)
        self.depth_gt_paths = read_paths(args.data_path, args.val_ground_truth_path_sunrgbd)

        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = ToTensor()
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        depth_path = self.depth_gt_paths[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path), dtype=np.uint16) / 10000.0
        # print(depth.min(), depth.max())
        depth[depth > 8] = 0
        depth = depth[..., None]
        return self.transform(dict(image=image, depth=depth, image_path=image_path, dataset='sunrgbd'))

    def __len__(self):
        return len(self.image_paths)
    
class iBims(Dataset):
    def __init__(self, args):
        self.args = args
        root_folder = args.ibims_root
        with open(os.path.join(root_folder, "imagelist.txt"), 'r') as f:
            imglist = f.read().split()

        samples = []
        for basename in imglist:
            img_path = os.path.join(root_folder, 'rgb', basename + ".png")
            depth_path = os.path.join(root_folder, 'depth', basename + ".png")
            valid_mask_path = os.path.join(
                root_folder, 'mask_invalid', basename+".png")
            transp_mask_path = os.path.join(
                root_folder, 'mask_transp', basename+".png")

            samples.append(
                (img_path, depth_path, valid_mask_path, transp_mask_path))

        self.samples = samples
        # self.normalize = T.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x

    def __getitem__(self, idx):
        img_path, depth_path, valid_mask_path, transp_mask_path = self.samples[idx]

        img = np.asarray(Image.open(img_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path),
                           dtype=np.uint16).astype('float')*50.0/65535

        mask_valid = np.asarray(Image.open(valid_mask_path))
        mask_transp = np.asarray(Image.open(transp_mask_path))

        # depth = depth * mask_valid * mask_transp
        depth = np.where(mask_valid * mask_transp, depth, -1)

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.normalize(img)
        depth = torch.from_numpy(depth).unsqueeze(0)

        return dict(image=img, depth=depth, image_path=img_path, depth_path=depth_path, dataset=self.args.dataset)

    def __len__(self):
        return len(self.samples)


class DIODE(Dataset):
    def __init__(self, args):
        import glob
        self.args = args
        data_dir_root = args.diode_root
        # image paths are of the form <data_dir_root>/scene_#/scan_#/*.png
        self.image_files = glob.glob(
            os.path.join(data_dir_root, '*', '*', '*.png'))
        self.depth_files = [r.replace(".png", "_depth.npy")
                            for r in self.image_files]
        self.depth_mask_files = [
            r.replace(".png", "_depth_mask.npy") for r in self.image_files]
        self.transform = ToTensor()
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        depth_mask_path = self.depth_mask_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.load(depth_path)  # in meters
        valid = np.load(depth_mask_path)  # binary

        # depth[depth > 8] = -1
        # depth = depth[..., None]

        sample = dict(image=image, depth=depth, image_path=image_path, valid=valid, dataset=self.args.dataset)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)
    
class EndoSLAM(Dataset):
    def __init__(self, args):
        self.args = args
        data_dir_root = args.endoslam_root
        # image paths are of the form <data_dir_root>/rgb/<scene>/<variant>/frames/<rgb,depth>/Camera<0,1>/rgb_{}.jpg
        self.image_files = glob.glob(os.path.join(
            data_dir_root, "UnityCam", "Colon", "Frames", '*.jpg'), recursive=True)
        self.depth_files = [r.replace("/rgb/", "/depth/").replace(
            "rgb_", "depth_").replace(".jpg", ".png") for r in self.image_files]
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        # depth = Image.open(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        depth = Image.fromarray(depth)
        # print("dpeth min max", depth.min(), depth.max())

        # print(np.shape(image))
        # print(np.shape(depth))

        if self.do_kb_crop:
            if idx == 0:
                print("Using KB input crop")
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth = depth.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            # uv = uv[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]

        image = np.asarray(image, dtype=np.float32) / 255.0
        # depth = np.asarray(depth, dtype=np.uint16) /1.
        depth = np.asarray(depth, dtype=np.float32) / 1.
        depth[depth > 80] = -1

        depth = depth[..., None]
        sample = dict(image=image, depth=depth, dataset=self.args.dataset)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        return len(self.image_files)
    
class HyperSim(Dataset):
    def __init__(self, args):
        # image paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_final_preview/*.tonemap.jpg
        # depth paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_final_preview/*.depth_meters.hdf5
        self.args = args
        data_dir_root = args.hypersim_root
        self.image_files = glob.glob(os.path.join(
            data_dir_root, '*', 'images', 'scene_cam_*_final_preview', '*.tonemap.jpg'))
        self.image_files.sort()
        self.depth_files = [r.replace("_final_preview", "_geometry_hdf5").replace(
            ".tonemap.jpg", ".depth_meters.hdf5") for r in self.image_files]
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

        # depth from hdf5
        depth_fd = h5py.File(depth_path, "r")
        # in meters (Euclidean distance)
        distance_meters = np.array(depth_fd['dataset'])
        depth = hypersim_distance_to_depth(
            distance_meters)  # in meters (planar depth)

        # depth[depth > 8] = -1
        depth = depth[..., None]

        sample = dict(image=image, depth=depth, image_path=image_path, dataset=self.args.dataset)
        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_files)