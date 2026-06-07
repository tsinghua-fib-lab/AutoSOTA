import os
import copy
import torch
import numpy as np
import imageio.v3 as imageio
from glob import glob
from util.reader import *
from util.augmentor import Augmentor

class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, sparse=False, aug_params=None, reader=None, mask=None):
        self.augmentor = None
        self.reader = reader
        self.mask = mask
        self.image_list = []
        self.disp_list = []

        if aug_params:
            self.augmentor = Augmentor(sparse, aug_params)

    def __len__(self):
        return len(self.image_list)

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disp_list = v * copy_of_self.disp_list

        return copy_of_self

    def __getitem__(self, index):
        left = imageio.imread(self.image_list[index][0]).astype(np.float32)[..., :3]
        right = imageio.imread(self.image_list[index][1]).astype(np.float32)[..., :3]
        disp, valid = self.reader(self.disp_list[index], self.mask)

        if self.augmentor:
            left, right, disp, valid = self.augmentor(name=self.image_list[index][0], left=left, right=right, disp=disp, valid=valid)

        left = torch.from_numpy(left).permute(2, 0, 1).float()
        right = torch.from_numpy(right).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp)[None].float()
        valid = torch.from_numpy(valid)[None].float()

        return self.image_list[index][0], left, right, disp, valid

class SceneFlow(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/sceneflow', dstype='frames_finalpass', things_test=False, mask=None):
        super(SceneFlow, self).__init__(sparse=False, aug_params=aug_params, reader=sceneflow_disp_reader, mask=mask)
        assert os.path.exists(root)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things('TEST')
        else:
            self._add_things('TRAIN')
            self._add_monkaa('TRAIN')
            self._add_driving('TRAIN')

    def _add_things(self, split='TRAIN'):
        left_list = sorted(glob(os.path.join(self.root, self.dstype, split, '*/*/left/*.png')))
        right_list = [img.replace('left', 'right') for img in left_list]
        disp_list = [img.replace(self.dstype, 'disparity').replace('.png', '.pfm') for img in left_list]

        assert len(left_list) == len(right_list) == len(disp_list)

        for idx, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

    def _add_monkaa(self, split='TRAIN'):
        left_list = sorted(glob(os.path.join(self.root, self.dstype, split, '*/left/*.png')))
        right_list = [img.replace('left', 'right') for img in left_list]
        disp_list = [img.replace(self.dstype, 'disparity').replace('.png', '.pfm') for img in left_list]

        assert len(left_list) == len(right_list) == len(disp_list)

        for idx, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

    def _add_driving(self, split='TRAIN'):
        left_list = sorted(glob(os.path.join(self.root, self.dstype, split, '*/*/*/left/*.png')))
        right_list = [img.replace('left', 'right') for img in left_list]
        disp_list = [img.replace(self.dstype, 'disparity').replace('.png', '.pfm') for img in left_list]

        assert len(left_list) == len(right_list) == len(disp_list)

        for idx, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='data/datasets/kitti', year='2015', split='training', mask='all'):
        super(KITTI, self).__init__(sparse=True, aug_params=aug_params, reader=kitti_disp_reader, mask=mask)
        assert os.path.exists(root)

        if year == '2012':
            left_list = sorted(glob(os.path.join(root, year, split, 'colored_0/*_10.png')))
            right_list = sorted(glob(os.path.join(root, year, split, 'colored_1/*_10.png')))
            if split == 'training':
                disp_list = sorted(glob(os.path.join(root, year, 'training', 'disp_occ/*_10.png')))
            else:
                disp_list = [os.path.join(root, year, 'training', 'disp_occ/000000_10.png')] * len(left_list)

        if year == '2015':
            left_list = sorted(glob(os.path.join(root, year, split, 'image_2/*_10.png')))
            right_list = sorted(glob(os.path.join(root, year, split, 'image_3/*_10.png')))
            if split == 'training':
                disp_list = sorted(glob(os.path.join(root, year, 'training', 'disp_occ_0/*_10.png')))
            else:
                disp_list = [os.path.join(root, year, 'training', 'disp_occ_0/000000_10.png')] * len(left_list)

        if year == 'all':
            left_list = sorted(glob(os.path.join(root, '2012', split, 'colored_0/*_10.png')))
            right_list = sorted(glob(os.path.join(root, '2012', split, 'colored_1/*_10.png')))
            if split == 'training':
                disp_list = sorted(glob(os.path.join(root, '2012', 'training', 'disp_occ/*_10.png')))
            else:
                disp_list = [os.path.join(root, year, 'training', 'disp_occ/000000_10.png')] * len(left_list)

            left_list += sorted(glob(os.path.join(root, '2015', split, 'image_2/*_10.png')))
            right_list += sorted(glob(os.path.join(root, '2015', split, 'image_3/*_10.png')))
            if split == 'training':
                disp_list += sorted(glob(os.path.join(root, '2015', 'training', 'disp_occ_0/*_10.png')))
            else:
                disp_list += [os.path.join(root, year, 'training', 'disp_occ_0/000000_10.png')] * len(left_list)

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/middlebury', year='MiddEval3', split='training', resolution='H', mask='noc'):
        super(Middlebury, self).__init__(sparse=True, aug_params=aug_params, reader=middlebury_disp_reader, mask=mask)
        assert os.path.exists(root)

        if year == 'MiddEval3':
            left_list = sorted(glob(os.path.join(root, year, split + resolution, '*/im0.png')))
            right_list = sorted(glob(os.path.join(root, year, split + resolution, '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, year, split + resolution, '*/disp0GT.pfm')))
        elif year == '2021':
            left_list = sorted(glob(os.path.join(root, year, 'data', '*/im0.png')))
            right_list = sorted(glob(os.path.join(root, year, 'data', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, year, 'data', '*/disp0.pfm')))    

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/eth3d', split='training', mask='noc'):
        super(ETH3D, self).__init__(sparse=True, aug_params=aug_params, reader=eth3d_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, f'two_view_{split}', '*/im0.png')))
        right_list = sorted(glob(os.path.join(root, f'two_view_{split}', '*/im1.png')))
        disp_list = sorted(glob(os.path.join(root, 'two_view_training_gt', '*/disp0GT.pfm')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class DrivingStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/drivingstereo', split='cloudy', mask=None, resolution='H'):
        super().__init__(sparse=True, aug_params=aug_params, reader=drivingstereo_disp_reader, mask=mask)
        assert os.path.exists(root)

        if resolution == 'F':
            res = 'full'
        elif resolution == 'H':
            res = 'half'

        left_list = sorted(glob(os.path.join(root, split, f'left-image-{res}-size/*.jpg')))
        right_list = sorted(glob(os.path.join(root, split, f'right-image-{res}-size/*.jpg')))
        disp_list = sorted(glob(os.path.join(root, split, f'disparity-map-{res}-size/*.png')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class Booster(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/booster_q', split='train', light='balanced', mask=None):
        super().__init__(sparse=True, aug_params=aug_params, reader=booster_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, split, light, '*/camera_00/*.png')))
        right_list = sorted(glob(os.path.join(root, split, light, '*/camera_02/*.png')))

        assert len(left_list) == len(right_list)

        for _, (left, right) in enumerate(zip(left_list, right_list)):
            self.image_list += [[left, right]]
            self.disp_list += [os.path.join(os.path.dirname(left.replace('camera_00', '')), 'disp_00.npy')]

class FoundationStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/foundationstereo', mask=None):
        super().__init__(sparse=False, aug_params=aug_params, reader=foundationstereo_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, '*/dataset/data/left/rgb/*.jpg')))
        right_list = sorted(glob(os.path.join(root, '*/dataset/data/right/rgb/*.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/dataset/data/left/disparity/*.png')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/tartanair', mask=None):
        super().__init__(sparse=False, aug_params=aug_params, reader=tartanair_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, '*/*/*/*/image_left/*.png')))
        right_list = sorted(glob(os.path.join(root, '*/*/*/*/image_right/*.png')))
        disp_list = sorted(glob(os.path.join(root, '*/*/*/*/depth_left/*.npy')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class CREStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/crestereo', mask=None):
        super().__init__(sparse=False, aug_params=aug_params, reader=crestereo_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        right_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/*_left.disp.png')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='/data/datasets/fallingthings', mask=None):
        super().__init__(sparse=False, aug_params=aug_params, reader=fallingthings_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, '**/*.left.jpg'), recursive=True))
        right_list = sorted(glob(os.path.join(root, '**/*.right.jpg'), recursive=True))
        disp_list = sorted(glob(os.path.join(root, '**/*.left.depth.png'), recursive=True))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]