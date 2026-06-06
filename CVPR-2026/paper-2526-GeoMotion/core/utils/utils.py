# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import torch
import einops
import cv2
import numpy as np
import torch.nn.functional as F
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
# from cvbase.optflow.visualize import flow2rgb

def get_feat(featmap_downscale_factor, track_2d, dino_list):
    """compute the nearest DINO feature for each pixel

    Args:
        track_2d (tensor): [2, N, T]  # (x, y) coordinates, N points, T time steps
        dino_list (list): List of DINO features for each time step, each element has shape [B, num_patches_h, num_patches_w, C]

    return:
        feature: [C, N, T]  # output features should have shape [C, N, T]
    """
    _, N, T = track_2d.shape

    features_over_time = []
    for t in range(T):
        #  (2, N) -> x = track_2d[0], y = track_2d[1]
        coords_x = track_2d[0, :, t]  # (N,)
        coords_y = track_2d[1, :, t]  # (N,)

        dino_x = (coords_x * featmap_downscale_factor[1]).long()
        dino_y = (coords_y * featmap_downscale_factor[0]).long()

        dino_x = torch.clamp(dino_x, 0, dino_list[t].shape[2] - 1)
        dino_y = torch.clamp(dino_y, 0, dino_list[t].shape[1] - 1)

        # (B, N, C)
        dino_feat = dino_list[t][:, dino_y, dino_x, :]

        # (B, N, C)
        features_over_time.append(dino_feat)

    # (B, N, T, C)
    features_over_time = torch.stack(features_over_time, dim=2)

    # (B, C, N, T)
    features_over_time = features_over_time.permute(0,3,1,2)
    
    return features_over_time

# def load_config_file(name):
#     """
#     Load configuration from YAML file (compatible with original function)
#     """
#     cfg = yaml.safe_load(open(name, 'r'))
    
#     class pObject(object):
#         def __init__(self):
#             pass
            
#         def get(self, key, default=None):
#             """Add get method for compatibility"""
#             return getattr(self, key, default)
    
#     cfg_new = pObject()
#     for attr in list(cfg.keys()):
#         setattr(cfg_new, attr, cfg[attr])
    
#     return cfg_new

# Move the class definition outside the function to make it pickleable
class ConfigObject(object):
    """Configuration object that can be pickled for multiprocessing"""
    def __init__(self):
        pass
    
    def get(self, key, default=None):
        """Add get method for compatibility"""
        return getattr(self, key, default)
    
    def __getattr__(self, name):
        """Only return None for user-defined attrs, not for system methods"""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"{name} not found")
        return None
from types import SimpleNamespace

def load_config_file(name):
    with open(name, 'r') as f:
        if name.endswith('.yaml') or name.endswith('.yml'):
            cfg = yaml.safe_load(f)
        elif name.endswith('.json'):
            cfg = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {name}")
    
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        else:
            return d
    
    return dict_to_namespace(cfg)

# def load_config_file(name):
#     """
#     Load configuration from YAML file (compatible with original function)
#     Fixed to be pickleable for multiprocessing
#     """
#     with open(name, 'r') as f:
#         if name.endswith('.yaml') or name.endswith('.yml'):
#             cfg = yaml.safe_load(f)
#         elif name.endswith('.json'):
#             cfg = json.load(f)
#         else:
#             raise ValueError(f"Unsupported config file format: {name}")
    
#     cfg_new = ConfigObject()
#     for attr in list(cfg.keys()):
#         value = cfg[attr]
#         # Handle nested dictionaries recursively
#         if isinstance(value, dict):
#             nested_obj = ConfigObject()
#             for nested_key, nested_value in value.items():
#                 setattr(nested_obj, nested_key, nested_value)
#             setattr(cfg_new, attr, nested_obj)
#         else:
#             setattr(cfg_new, attr, value)
    
#     return cfg_new

def save_model(model, log_root, epoch, test_iou1, test_iou2):
    log_model = os.path.join(log_root, 'checkpoints')
    if not os.path.exists(log_model):
        os.makedirs(log_model)
    filename = os.path.join(log_model, 'checkpoint_{}_iou1_{:.5f}_iou2_{:.5f}.pth'.format(epoch, np.round(test_iou1, 3), np.round(test_iou2, 3)))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        }, filename)
    
def save_motion_seg_model(model, log_dir, epoch, test_results):
    """
    Save model checkpoint with test results
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Get best IoU across all test sets
    best_iou = 0.0
    if test_results:
        best_iou = max([results.get('iou', 0.0) for results in test_results.values()])
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'test_results': test_results,
        'best_iou': best_iou
    }
    
    # Save current checkpoint
    save_path = os.path.join(log_dir, f'motion_seg_model_epoch_{epoch}.pth')
    torch.save(checkpoint, save_path)
    
    # Save best model
    best_model_path = os.path.join(log_dir, 'best_motion_seg_model.pth')
    should_save_best = True
    
    if os.path.exists(best_model_path):
        try:
            best_checkpoint = torch.load(best_model_path, map_location='cpu')
            if best_iou <= best_checkpoint.get('best_iou', 0.0):
                should_save_best = False
        except:
            # If loading fails, save anyway
            pass
    
    if should_save_best:
        torch.save(checkpoint, best_model_path)
        print(f"New best model saved with IoU: {best_iou:.4f}")

def set_learning_rate(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def convert_for_vis(inp, use_flow=False):
    dim = len(inp.size())
    if not use_flow:
        return torch.clamp((0.5*inp+0.5)*255,0,255).type(torch.ByteTensor)
    else:
        if dim == 4:
            inp = einops.rearrange(inp, 'b c h w -> b h w c').detach().cpu().numpy()
            rgb = [flow2rgb(inp[x]) for x in range(np.shape(inp)[0])]
            rgb = np.stack(rgb, axis=0)
            rgb = einops.rearrange(rgb, 'b h w c -> b c h w')
        if dim == 5:
            b, s, w, h, c = inp.size()
            inp = einops.rearrange(inp, 'b s c h w -> (b s) h w c').detach().cpu().numpy()
            rgb = [flow2rgb(inp[x]) for x in range(np.shape(inp)[0])]
            rgb = np.stack(rgb, axis=0)
            rgb = einops.rearrange(rgb, '(b s) h w c -> b s c h w', b=b, s=s)
        return torch.Tensor(rgb*255).type(torch.ByteTensor)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def heuristic_fg_bg(mask):
    mask = mask.copy()
    h, w = mask.shape
    mask[1:-1, 1:-1] = 0
    borders = 2*h+2*w-4
    return np.sum(mask>0.5)/borders

def rectangle_iou(masks, gt):
    t, s, c, H_, W_ = masks.size()
    H, W = gt.size()
    masks = F.interpolate(masks, size=(1, H, W))
    ms = []
    for t_ in range(t):
        m = masks[t_,0,0] #h w
        m = m.detach().cpu().numpy()
        if heuristic_fg_bg(m) > 0.5: m = 1-m
        ms.append(m)
    masks = np.stack(ms, 0)
    gt = gt.detach().cpu().numpy()
    for idx, m in enumerate([masks[0], masks.mean(0)]):
        m[m>0.1]=1
        m[m<=0.1]=0
        contours = cv2.findContours((m*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        area = 0
        for cnt in contours:
            (x_,y_,w_,h_) = cv2.boundingRect(cnt)
            if w_*h_ > area:
                x=x_; y=y_; w=w_; h=h_;
                area = w_ * h_
        if area>0:
            bbox = np.array([x, y, x+w, y+h],dtype=float)
            #if the size reference for the annotation (the original jpg image) is different than the size of the mask
            i, j = np.where(gt==1.)
            bbox_gt = np.array([min(j), min(i), max(j)+1, max(i)+1],dtype=float)
            iou = bb_intersection_over_union(bbox_gt,bbox)
        else:
            iou = 0.
        if idx == 0: iou_single = iou
        if idx == 1: iou_mean = iou
    masks = np.expand_dims(masks, 1)
    return masks, masks.mean(0), iou_mean, iou_single

def iou(masks, gt, thres=0.5):
    masks = (masks>thres).float()
    intersect = torch.tensordot(masks, gt, dims=([-2, -1], [0, 1]))
    union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    return intersect/(union + 1e-12)

def ensemble_hungarian_iou(masks, gt, moca=False):
    thres = 0.5
    b, c, h, w = gt.size()
    gt = gt[0,0,:,:] #h ,w

    if moca:
        #return masks, masks.mean(0), 0, rectangle_iou(masks[0], gt) 
        masks, mean_mask, iou_mean, iou_single_gap = rectangle_iou(masks, gt)
    else:
        masks = F.interpolate(masks, size=(1, h, w))  # t s 1 h w
        mask_iou = iou(masks[:,:,0], gt, thres)  # t s # t s
        iou_max, slot_max = mask_iou.max(dim=1)
        masks = masks[torch.arange(masks.size(0)), slot_max]  # pick the slot for each mask
        mean_mask = masks.mean(0)
        gap_1_mask = masks[0]  # note last frame will use gap of -1, not major.
        iou_mean = iou(mean_mask, gt, thres).detach().cpu().numpy()
        iou_single_gap = iou(gap_1_mask, gt, thres).detach().cpu().numpy()
        mean_mask = mean_mask.detach().cpu().numpy()  # c h w
        masks = masks.detach().cpu().numpy()

    return masks, mean_mask, iou_mean, iou_single_gap


def hungarian_iou(masks, gt):
    thres = 0.5
    masks = (masks>thres).float()
    gt = gt[:,0:1,:,:]
    b, c, h, w = gt.size()
    
    mask = F.interpolate(masks, size=(h, w))
    #IOU
    intersect = (mask*gt).sum(dim=[-2, -1])
    union = mask.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    iou = intersect/(union + 1e-12)
    return iou.mean()

def cls_iou(pred, label, thres = 0.7):
    mask = (pred>thres).float()
    b, c, n = label.shape 
    # IOU
    intersect = (mask * label).sum(-1)
    union = mask.sum(-1) + label.sum(-1) - intersect
    iou = intersect / (union + 1e-12)
    return iou.mean()

def cal_roc(pred, label, output_path, epoch):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    thresholds = np.linspace(0, 1, num=100)
    tprs = []  
    fprs = []  
    recalls = []  
    precisions = []  
    
    for thres in thresholds:
        binarized_pred = (pred > thres).astype(float)
        
        tp = np.sum((binarized_pred == 1) & (label == 1))
        fp = np.sum((binarized_pred == 1) & (label == 0))
        tn = np.sum((binarized_pred == 0) & (label == 0))
        fn = np.sum((binarized_pred == 0) & (label == 1))
        
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        
        tprs.append(tpr)
        fprs.append(fpr)
        
        recall = tp / (tp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        
        recalls.append(recall)
        precisions.append(precision)
    
    plt.figure(figsize=(10, 5))
    auc_res = auc(fprs, tprs)
    plt.plot(fprs, tprs, label='ROC Curve (area = %0.5f)' % auc_res)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, f'ROC_Curve_{epoch}.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(recalls, precisions, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_path, f'Precision_Recall_Curve_{epoch}.png'))
    plt.close()

    return fprs, tprs, recalls, precisions, auc_res

def gt_iou(mask, label, gt):
    # mask: [N, L], label: [N], gt: [H, W, L]
    N, L = mask.shape
    sum_iou = 0
    for i in range(L):
        valid = 1.0 - mask[:,i]
        intersect = (valid * label).sum()
        union = gt[:,:,i].sum()
        if union == 0:
            continue
        iou = intersect / (union+1e-12)
        sum_iou += iou
    mean_iou = sum_iou / L
    return mean_iou

def gt_ratio(mask, label, gt):
    # mask: [N, L], label: [N], gt: [H, W, L]
    N, L = mask.shape
    h, w = gt.shape[0], gt.shape[1]
    sum_ratio = 0
    for i in range(L):
        valid = 1.0 - mask[:,i]
        ratio = valid.sum() / (h*w)
        sum_ratio += ratio
    mean_ratio = sum_ratio / L
    return mean_ratio

TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

def save_img_pred_gt_horizon(path, idx, img, pred, gt):
    # img: [N, H, W, 3], pred & GT: [N, H, W] 
    n, h, w, _ = img.shape
    img = np.reshape(np.transpose(img, (1,0,2,3)), (h, n*w, 3))
    pred = np.reshape(np.transpose(pred > 0.5, (1,0,2)), (h, n*w))
    gt = np.reshape(np.transpose(gt, (1,0,2)), (h, n*w))
    cv2.imwrite(os.path.join(path, "{:0>6d}_img.png".format(idx)), 255.0*img)
    cv2.imwrite(os.path.join(path, "{:0>6d}_pred.png".format(idx)), 255.0*pred)
    cv2.imwrite(os.path.join(path, "{:0>6d}_gt.png".format(idx)), 255.0*gt)


color_bank = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]


def draw_traj_cls(img, traj, mask, label, gt_label):
    # img: [L, H, W, 3], traj: [N, L, 2], mask: [N, L], label: [N]
    L, h, w, _ = img.shape
    N = traj.shape[0]
    vis_imgs = [img[i][:,:,::-1].copy() for i in range(L)]
    gt_imgs = [img[i][:,:,::-1].copy() for i in range(L)]
    traj_imgs = [img[i][:,:,::-1].copy() for i in range(L)]
    for i in range(N):
        single_traj = traj[i]
        single_mask = mask[i]
        for j in range(L):
            if single_mask[j] != 1:
                x, y = single_traj[j]
                x, y = int(x), int(y)
                l = int(label[i] > 0.5)
                gt_l = int(gt_label[i] > 0.5)
                color = color_bank[l]
                gt_color = color_bank[gt_l]
                cv2.circle(vis_imgs[j], center=(x,y), radius=1, color=color, thickness=2)
                cv2.circle(gt_imgs[j], center=(x,y), radius=1, color=gt_color, thickness=1)
    for i in range(100):
        idx = np.random.randint(N)
        single_traj = traj[idx]
        single_mask = mask[idx]
        for j in range(L):
            if single_mask[j] != 1:
                x, y = single_traj[j]
                x, y = int(x), int(y)
                cv2.circle(traj_imgs[j], center=(x,y), radius=3, color=color_bank[idx%6], thickness=4)
    vis_imgs = np.concatenate(vis_imgs, axis=1)
    gt_imgs = np.concatenate(gt_imgs, axis=1)
    traj_imgs = np.concatenate(traj_imgs, axis=1)
    concat_imgs = np.concatenate([img[i][:,:,::-1].copy() for i in range(L)], axis=1)
    combine_imgs = np.concatenate([concat_imgs, vis_imgs, gt_imgs, traj_imgs], 0)
    return combine_imgs
    
def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor([2 / size for size in reversed(sizes)], device=coords.device)

    coords -= 1

    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)
