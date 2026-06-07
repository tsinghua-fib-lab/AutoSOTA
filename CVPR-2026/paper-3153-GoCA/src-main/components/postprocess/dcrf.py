# git clone https://github.com/lucasb-eyer/pydensecrf
# cd pydensecrf
# pip3 install --force-reinstall cython==0.29.36
# python3 setup.py install

import cv2
import torch
import pydensecrf
import numpy as np
import torch.nn.functional as F
from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax, unary_from_labels, create_pairwise_bilateral

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = unary_from_softmax(probmap)
        # U = pydensecrf.utils.unary_from_labels(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

def dcrf(pred, image_path):
    """
    CRF post-processing on pre-computed logits
    """

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )
    mean_bgr = (104.008, 116.669, 122.675)

    image = cv2.imread(image_path).astype(np.float32)
    # image = np.array(image)
    # Mean subtraction
    image -= mean_bgr

    pred = F.interpolate(pred.unsqueeze(0), image.shape[:2], mode='bicubic').clamp(min=0).squeeze(0)
    pred_min = torch.amin(pred, dim=[1, 2], keepdim=True)
    pred_max = torch.amax(pred, dim=[1, 2], keepdim=True)
    pred = (pred - pred_min) / (pred_max - pred_min)
    prob = pred.detach().cpu().numpy()

    image = image.astype(np.uint8)
    prob = postprocessor(image, prob)

    return torch.from_numpy(prob)
