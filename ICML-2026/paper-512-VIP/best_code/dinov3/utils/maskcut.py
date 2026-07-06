#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from tqdm import tqdm
import PIL
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pycocotools import mask
import pycocotools.mask as mask_util
from scipy import ndimage
from scipy.linalg import eigh


from dinov3.utils import metric
# from crf import densecrf


# -- key feat for cut off
# -- plug in vision tower get_class_and_patch_tokens() function:
# key_feat = self.backbone.blocks[-1].attn.key_feats[0].clone()
# feat_h, feat_w = 21, 21
# bs, nb_token = key_feat.shape[0], key_feat.shape[2]
# feat_dim = 1024
# k = key_feat.transpose(1, 2).reshape(bs, nb_token, -1)
# key_feat = k[:, 5:].transpose(1, 2).reshape(bs, feat_dim, feat_h * feat_w)  


def attn_vis(feat, second_eige, mask):
    import matplotlib.pyplot as plt
    """
    visualization of the attention map, between cls token and patch tokens
    and, between patch tokens
    """

    patch_patch = torch.einsum("d n, d m -> n m", feat, feat)
    

    # patch sample
    patch_window = 16
    patch_idxs = 198
    image_size = 336
    h = w = int(image_size/patch_window)

    save_path = ''

    plt.figure(figsize=(30,15))
    nrow = 1
    ncol = 4

    # inter-patch weights
    plt.subplot(nrow,ncol,1)
    plt.imshow(patch_patch.detach().cpu().numpy())
    plt.title('patch_patch')
    plt.xticks([])
    plt.yticks([])

    # selected patch weights
    plt.subplot(nrow,ncol,2)
    patch_sim = patch_patch[patch_idxs].reshape(h,w)
    patch_sim = F.interpolate(patch_sim.unsqueeze(0).unsqueeze(0), size=(image_size, image_size), mode='bilinear',align_corners=False)[0,0]
    plt.imshow(patch_sim.detach().cpu().numpy(),alpha=0.5,cmap='jet')
    col = (patch_idxs//w)*patch_window+patch_window//2
    row = (patch_idxs%w)*patch_window+patch_window//2
    plt.plot(row,col,'ro',markersize=15)
    plt.xticks([])
    plt.yticks([])
    plt.title('patch_idx'+str(patch_idxs)+' for key_feat')


    # second eige vec
    plt.subplot(nrow,ncol,3)
    second_eige = torch.tensor(second_eige)
    second_eige = second_eige.reshape(h,w)
    second_eige = F.interpolate(second_eige.unsqueeze(0).unsqueeze(0), size=(image_size, image_size), mode='bilinear',align_corners=False)[0,0]
    plt.imshow(second_eige.detach().cpu().numpy(),alpha=0.5,cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.title('second_vec'+str(patch_idxs))


    plt.subplot(nrow,ncol,4)
    mask = torch.tensor(mask)
    mask = mask.reshape(h,w)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(image_size, image_size), mode='nearest')[0,0]
    plt.imshow(mask.detach().cpu().numpy(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('cut_mask')
    

    plt.savefig(save_path)

def monotone_sharpen(v, kind="tanh", alpha=2.0, gamma=0.7):
    v = v.copy()
    if kind == "tanh":
        # z-score 后再 tanh，alpha 控制锐度
        v = (v - v.mean()) / (v.std() + 1e-12)
        v = np.tanh(alpha * v)
    elif kind == "gamma":
        # 对正负分开做 gamma，保持单调
        pos = v >= 0
        v[pos]  = np.sign(v[pos]) * (np.abs(v[pos]) ** gamma)
        v[~pos] = - (np.abs(v[~pos]) ** gamma)
    elif kind == "rank":
        # 秩变换 -> 线性映射到 [-1,1]（稳健、单调）
        r = np.argsort(np.argsort(v))
        v = (r - r.mean()) / (len(v)/2.0)
        v = np.clip(v, -1, 1)
    return v



def detect_box(bipartition, seed,  dims, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]

    if principle_object:
        mask = np.where(objects == cc)
        return mask
    else:
        raise NotImplementedError


def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    return A, D


def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    _, eigenvectors = eigh(D-A, D, subset_by_index=[0,1])
    eigenvec = np.copy(eigenvectors[:, 1])
    second_smallest_vec = eigenvectors[:, 1]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def get_masked_affinity_matrix(painting, feats, mask, ps):
    # mask out affinity matrix based on the painting matrix 
    dim, num_patch = feats.size()[0], feats.size()[1]
    painting = painting + mask.unsqueeze(0)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    feats = feats.clone().view(dim, ps, ps)
    feats = ((1 - painting) * feats).view(dim, num_patch)
    return feats, painting

def maskcut_forward(key_feat, dims=[21,21], tau=0.15, N=3, cpu=False):
    """
    Implementation of MaskCut.
    Inputs
      feats: the pixel/patche features of an image, [1,dim,h*w]
      dims: dimension of the map from which the features are used, [h,w]
      tau: thresold for graph construction
      N: number of pseudo-masks per image.
    """
    feats = key_feat.squeeze(0)

    bipartitions = []
    eigvecs = []

    for i in range(N):
        if i == 0:
            painting = torch.from_numpy(np.zeros(dims))
            if not cpu: painting = painting.cuda()
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, ps)

        # construct the affinity matrix
        A, D = get_affinity_matrix(feats, tau)
        # get the second smallest eigenvector
        eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)


        # visualization the feat_affinity and mask output by tokencut, ie second_smallest_vec and bipartition
        # attn_vis(feats, second_smallest_vec, bipartition) 


        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dims)
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            # reverse bipartition, eigenvector and get new seed
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
            seed = np.argmax(eigenvec)
        else:
            seed = np.argmax(second_smallest_vec)

        # get pxiels corresponding to the seed
        bipartition = bipartition.reshape(dims).astype(float)
        cc = detect_box(bipartition, seed, dims)
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0],cc[1]] = 1
        pseudo_mask = torch.from_numpy(pseudo_mask)
        if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = torch.sum(pseudo_mask) / pseudo_mask.size()[0] / pseudo_mask.size()[1]
            if metric.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = np.zeros(dims)
                pseudo_mask = torch.from_numpy(pseudo_mask)
                if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        current_mask = pseudo_mask

        # mask out foreground areas in previous stages
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition_masked = bipartition - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # unsample the eigenvec
        eigvec = second_smallest_vec.reshape(dims)
        eigvec = torch.from_numpy(eigvec)
        if not cpu: eigvec = eigvec.to('cuda')
        eigvecs.append(eigvec.cpu().numpy())

    return seed, bipartitions, eigvecs


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour




