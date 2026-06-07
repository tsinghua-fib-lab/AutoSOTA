import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import torch.cuda.amp as amp

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]

class SigLoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(
        self, valid_mask=True, loss_weight=1.0, warm_up=False, warm_iter=100, loss_name="sigloss"
    ):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.loss_name = loss_name

        self.eps = 1e-7  # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def forward(self, input, target, max_depth):
        if self.valid_mask:
            valid_mask = target > 0
            if max_depth is not None:
                valid_mask = torch.logical_and(target > 0.1, target <= max_depth)
                # valid_mask = torch.logical_and(valid_mask, input > 0)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)

        loss = self.loss_weight * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        return loss

class GradientLoss(nn.Module):
    """GradientLoss.

    Adapted from https://www.cs.cornell.edu/projects/megadepth/

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
    """

    def __init__(self, valid_mask=True, loss_weight=1.0, loss_name="loss_grad"):
        super(GradientLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.loss_name = loss_name

        self.eps = 1e-7  # avoid grad explode

    def forward(self, input, target, max_depth):
        if self.valid_mask:
            valid_mask = target > 0
            if max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= max_depth)

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(valid_mask)

        gradient_loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        gradient_loss = gradient_loss + \
            nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        
        return self.loss_weight * gradient_loss

    
class L1Loss(nn.Module):
    def __init__(self, max_depth=10, min_depth=1e-3, loss_type = "L1"):
        super(L1Loss, self).__init__()

        self.max_depth = max_depth
        self.min_depth = min_depth
        if loss_type == "L1":
            self.loss = torch.nn.L1Loss(reduction = 'none')
        elif loss_type == "Huber":
            self.loss = torch.nn.HuberLoss(reduction = 'none', delta=1.0)
        elif loss_type == "L2":
            self.loss = torch.nn.MSELoss(reduction = 'none')
        else:
            raise()



    def forward(self, depth_prediction, gts):

        weight_depth = torch.where(
            (gts > self.min_depth) * (gts < self.max_depth),
            torch.ones_like(gts),
            torch.zeros_like(gts)
        )

        # gts = weight_depth * gts  # remove invalid gt



        diff = self.loss(depth_prediction, gts)

        num_pixels = (gts > self.min_depth) * (gts < self.max_depth)


        diff = torch.where(
        (gts > self.min_depth) * (gts < self.max_depth),
        diff,
        torch.zeros_like(diff)
        )

        diff = diff.reshape(diff.shape[0], -1)
        num_pixels = num_pixels.reshape(num_pixels.shape[0], -1).sum(dim=-1) + 1e-6

        loss1 = (diff).sum(dim=-1) / num_pixels

        total_pixels = gts.shape[1] * gts.shape[2] * gts.shape[3]

        weight = num_pixels.to(diff.dtype) / total_pixels

        loss = (loss1 * weight).sum()

        return loss

class SelfThresholdScaleLoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(
        self, valid_mask=True, loss_weight=1.0, warm_up=False, warm_iter=100, loss_name="stsloss"
    ):
        super(SelfThresholdScaleLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.loss_name = loss_name

        self.eps = 1e-7  # avoid grad explode
        self.max_factor = 3
        
        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0
        self.thershold = 0.99

    def forward(self, scale_pred, shift_pred, gt_depth, relative_depth, max_depth, dataset):

        pred_h, pred_w = scale_pred.shape[1], scale_pred.shape[2]
        if self.valid_mask:
            valid_mask = gt_depth > 0
            if max_depth is not None:
                valid_mask = torch.logical_and(gt_depth > 0, gt_depth <= max_depth)
            if dataset == "kitti":
                gt_height, gt_width = gt_depth.shape[1], gt_depth.shape[2]
                eval_mask = torch.zeros(valid_mask.shape).cuda()
                eval_mask[:, int(0.40810811 * gt_height):int(0.99189189 * gt_height), 
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                valid_mask = torch.logical_and(valid_mask, eval_mask)
            elif dataset == 'nyu' or dataset == 'void':
                gt_height, gt_width = gt_depth.shape[1], gt_depth.shape[2]
                eval_mask = torch.zeros(valid_mask.shape).cuda()
                eval_mask[45:471, 41:601] = 1
                valid_mask = torch.logical_and(valid_mask, eval_mask)     

        gt_depth_rev = gt_depth.clone()
        gt_depth_rev[valid_mask] = 1 / gt_depth_rev[valid_mask]
        gt_depth_rev[~valid_mask] = 0

        scale, shift = compute_scale_and_shift(relative_depth, gt_depth_rev, valid_mask)
        rel_abs_depth = 1 / (scale.view(-1, 1, 1) * relative_depth + shift.view(-1, 1, 1))

        ratio = (gt_depth + 1e-8) / rel_abs_depth  # (B, H, W)
        ratio[torch.isnan(ratio)] = self.eps
        ratio[torch.isinf(ratio)] = self.eps

        condition = (ratio < 1.25) & (1 / ratio < 1.25) & valid_mask
        count_per_sample = condition.view(gt_depth.size(0), -1).sum(dim=1).float()
        total_valid_per_sample = valid_mask.view(gt_depth.size(0), -1).sum(dim=1).float()

        percentage = (count_per_sample / (total_valid_per_sample + self.eps))

        if any(x > self.thershold for x in percentage):
            percentage = (percentage > self.thershold).view(-1, 1, 1).expand(-1, pred_h, pred_w)

            scale_map = scale.view(-1, 1, 1).expand(-1, pred_h, pred_w)
            shift_map = shift.view(-1, 1, 1).expand(-1, pred_h, pred_w)

            scale_loss = nn.functional.l1_loss(scale_pred[percentage], scale_map[percentage])
            shift_loss = nn.functional.l1_loss(shift_pred[percentage], shift_map[percentage])

            loss = self.loss_weight * (scale_loss + shift_loss)
            # loss = self.loss_weight * scale_loss
            # loss = self.loss_weight * shift_loss
                                                   
            if torch.isnan(loss):
                print("Nan SILog loss")
                print("scale:", scale_pred.shape)
                print("shift:", shift_pred.shape)
                print("target:", rel_abs_depth.shape)
                print("scale min max", torch.min(scale_pred), torch.max(scale_pred))
                print("shift min max", torch.min(shift_pred), torch.max(shift_pred))
                print("Target min max", torch.min(rel_abs_depth), torch.max(rel_abs_depth))
                print("scale_loss", torch.isnan(scale_loss))
                print("shift_loss", torch.isnan(shift_loss))

            return loss
        else:
            return torch.tensor(0).cuda()

class SelfThresholdScaleDLoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(
        self, valid_mask=True, loss_weight=1.0, warm_up=False, warm_iter=100, loss_name="stsloss"
    ):
        super(SelfThresholdScaleDLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.loss_name = loss_name

        self.eps = 1e-7  # avoid grad explode
        self.max_factor = 3

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0
        self.thershold = 0.99

    def forward(self, pred_depth, pred_scale, pred_shift, gt_depth, relative_depth, max_depth, dataset):

        pred_h, pred_w = pred_depth.shape[1], pred_depth.shape[2]
        if self.valid_mask:
            valid_mask = gt_depth > 0
            if max_depth is not None:
                valid_mask = torch.logical_and(gt_depth > 0.1, gt_depth <= max_depth)
            if dataset == "kitti":
                gt_height, gt_width = gt_depth.shape[1], gt_depth.shape[2]
                eval_mask = torch.zeros(valid_mask.shape).cuda()
                eval_mask[:, int(0.40810811 * gt_height):int(0.99189189 * gt_height), 
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
                valid_mask = torch.logical_and(valid_mask, eval_mask)
            elif dataset == 'nyu' or dataset == 'void':
                gt_height, gt_width = gt_depth.shape[1], gt_depth.shape[2]
                eval_mask = torch.zeros(valid_mask.shape).cuda()
                eval_mask[45:471, 41:601] = 1
                valid_mask = torch.logical_and(valid_mask, eval_mask)        

        gt_depth_rev = gt_depth.clone()
        gt_depth_rev[valid_mask] = 1 / gt_depth_rev[valid_mask]
        gt_depth_rev[~valid_mask] = 0

        scale, shift = compute_scale_and_shift(relative_depth, gt_depth_rev, valid_mask)
        rel_abs_depth = 1 / (scale.view(-1, 1, 1) * relative_depth + shift.view(-1, 1, 1))

        ratio = (gt_depth + 1e-8) / rel_abs_depth  # (B, H, W)
        ratio[torch.isnan(ratio)] = self.eps
        ratio[torch.isinf(ratio)] = self.eps

        condition = (ratio < 1.25) & (1 / ratio < 1.25) & valid_mask
        count_per_sample = condition.view(gt_depth.size(0), -1).sum(dim=1).float()
        total_valid_per_sample = valid_mask.view(gt_depth.size(0), -1).sum(dim=1).float()

        percentage = (count_per_sample / (total_valid_per_sample + self.eps))

        if any(x > self.thershold for x in percentage):
            percent_single = percentage > self.thershold
            single_scale_loss = torch.abs(torch.log(scale[percent_single] + self.eps) - torch.log(pred_scale[percent_single] + self.eps))
            single_shift_loss = torch.abs(torch.log(shift[percent_single] + self.eps) - torch.log(pred_shift[percent_single] + self.eps))
            
            percentage = (percentage > self.thershold).view(-1, 1, 1).expand(-1, pred_h, pred_w)

            rel_abs_mask = torch.logical_and(rel_abs_depth > 0.1, rel_abs_depth <= max_depth)

            g = torch.log(pred_depth[percentage &  rel_abs_mask] + self.eps) - \
            torch.log(rel_abs_depth[percentage &  rel_abs_mask] + self.eps)
            Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
            loss = self.loss_weight * torch.sqrt(Dg)
            loss += self.loss_weight(single_scale_loss + single_shift_loss)
            
            if torch.isnan(loss):
                print("Nan SelfThreshold loss")
                print("pred_depth:", pred_depth.shape)
                print("target:", rel_abs_depth.shape)
                print("pred_depth min max", torch.min(pred_depth), torch.max(pred_depth))
                print("Target min max", torch.min(rel_abs_depth), torch.max(rel_abs_depth))
                print("loss", torch.isnan(loss))

            return loss
        else:
            return torch.tensor(0).cuda()
        
class EdgeAwareSmoothLoss(nn.Module):
    """EdgeAwareSmoothLoss for scale and shift prediction.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
    """

    def __init__(self, loss_weight=1.0, loss_name="loss_smooth"):
        super(EdgeAwareSmoothLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, scale_pred, shift_pred, image):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_scale_x = torch.abs(scale_pred[:, :, :, :-1] - scale_pred[:, :, :, 1:])
        grad_scale_y = torch.abs(scale_pred[:, :, :-1, :] - scale_pred[:, :, 1:, :])

        grad_shift_x = torch.abs(shift_pred[:, :, :, :-1] - shift_pred[:, :, :, 1:])
        grad_shift_y = torch.abs(shift_pred[:, :, :-1, :] - shift_pred[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)

        grad_scale_x *= torch.exp(-grad_img_x)
        grad_scale_y *= torch.exp(-grad_img_y)

        grad_shift_x *= torch.exp(-grad_img_x)
        grad_shift_y *= torch.exp(-grad_img_y)

        loss = grad_scale_x.mean() + grad_scale_y.mean() + grad_shift_x.mean() + grad_shift_y.mean()

        return self.loss_weight * loss
    