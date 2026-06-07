import os
import PIL
import torch
import numpy as np
from PIL import Image
from collections import Counter
import torch.nn.functional as F

from .downstream.palette import palette
from .postprocess.hungarian import hungarian_matching

def downstream(task, data, final_call=False):
    if task == 'visualize_class_map':
        return visualize_class_map(data['pred'], data['save_path'], final_call)
    elif task == 'quantitative_evaluation':
        return quantitative_evaluation(
            data['pred'], data['label'], data['save_path'],
            data['present_objects'], data['class_count'],
            final_call, data['method'], data['background_threshold'],
        )
    elif task == 'visualize_segmentation':
        return visualize_segmentation(
            data['pred'], data['save_path'],
            data['class_count'], data['present_objects'],
            final_call, data['method'], data['background_threshold'],
        )
    elif task == 'hungarian_evaluate':
        return hungarian_evaluate(
            data['pred'], data['label'], data['class_count'],
            data['save_path'],
            final_call,
        )
    elif task == 'hungarian_visualization':
        return hungarian_visualization(
            data['pred'], data['save_path'], final_call,
            )
    else:
        raise NotImplementedError


def visualize_class_map(pred, save_path, final_call):
    '''
    example:
    downstream_setting = {
        'task': 'visualize_class_map',
        'save_path_root': '../baseline_outputs/test',
    }
    '''
    # class x h x w
    if final_call:
        return

    pred = pred.detach().cpu()
    for i in range(pred.shape[0]):
        image = pred[i].detach()
        # image = 255 * ((image - image.min()) / (image.max() - image.min()))
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image).resize((512, 512), PIL.Image.NEAREST)
        os.makedirs(save_path, exist_ok=True)
        image.save(os.path.join(save_path, f'{i}.png'))


# background_threshold = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
# background_threshold = [0]
# background_threshold = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
# background_threshold = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
unions = None
intersections = None
def quantitative_evaluation(
    pred, label, save_path, present_objects, class_count, final_call, bg_method, background_threshold
):
    global unions
    global intersections
    if unions is None:
        unions = {threshold: Counter() for threshold in background_threshold}
        intersections = {threshold: Counter() for threshold in background_threshold}
    class_num = class_count

    if not final_call:
        pred_full = torch.zeros(class_num, label.shape[0], label.shape[1])

        # print(pred.shape)  # n x h x w
        # pred_max = pred.amax(dim=[1,2], keepdim=True)
        # pred_min = pred.amin(dim=[1,2], keepdim=True)
        # pred = (pred - pred_min) / (pred_max - pred_min)

        if pred is not None:
            pred = F.interpolate(pred.unsqueeze(0), label.shape, mode='bilinear').squeeze(0).detach().cpu()
            supporting_target = pred[-1]
            for idx in range(pred.shape[0] - 1):
                class_id = present_objects[idx]
                pred_full[class_id,:,:] = pred[idx,:,:]

        for threshold in background_threshold:
            # fill in background and non-present objects
            # method vanilla
            if bg_method == 'vanilla':
                pred_full[0,:,:] = threshold  # background threshold
            # method#1
            elif bg_method == 'max':
                supporting_target = supporting_target.clamp(min=threshold)
                pred_full[0,:,:] = supporting_target
            # method#2
            elif bg_method == 'exact':
                pred_full[0,:,:] = supporting_target
            # method#3
            elif bg_method == 'avg':
                pred_full[0,:,:] = (threshold + supporting_target) / 2
            # method#4
            elif bg_method == 'offset':
                pred_full[0,:,:] = threshold + supporting_target

            pred_class = pred_full.argmax(dim=0).numpy()

            for class_index in range(class_num):
                preds_tmp = (pred_class == class_index).astype(int)
                gts_tmp = (label == class_index).astype(int)
                unions[threshold][class_index] += (preds_tmp | gts_tmp).sum()
                intersections[threshold][class_index] += (preds_tmp & gts_tmp).sum()

    else:
        all_miou = ''
        for threshold in background_threshold:
            ious = []
            for class_index in range(class_num):
                iou = intersections[threshold][class_index] / (1e-8 + unions[threshold][class_index])
                ious.append(iou)
            miou = np.array(ious).mean()
            print(f'mIoU: {miou} at background_threshold {threshold}')
            all_miou += f'mIoU: {miou} at background_threshold {threshold}\n'
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'evaluation.txt'), 'w') as f:
            f.write(all_miou)


def visualize_segmentation(
    pred, save_path, class_count, present_objects, final_call, bg_method, background_threshold
):
    '''
    example:
    downstream_setting = {
        'task': 'visualize_segmentation',
        'save_path_root': '../baseline_outputs/test',
    }
    '''
    # class x h x w
    global unions
    global intersections
    if unions is None:
        unions = {threshold: Counter() for threshold in background_threshold}
        intersections = {threshold: Counter() for threshold in background_threshold}

    if final_call:
        return

    to_out = pred.detach().cpu()
    for i in range(to_out.shape[0]):
        image = to_out[i].detach()
        # image = 255 * ((image - image.min()) / (image.max() - image.min()))
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.float().cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image).resize((512, 512), PIL.Image.NEAREST)
        os.makedirs(save_path, exist_ok=True)
        image.save(os.path.join(save_path, f'{i}.png'))

    class_num = class_count
    pred = F.interpolate(pred.unsqueeze(0), 512, mode='bilinear').squeeze(0).detach().cpu()
    supporting_target = pred[-1]
    pred_full = torch.zeros(class_num, pred.shape[1], pred.shape[1])
    for idx in range(pred.shape[0] - 1):
        class_id = present_objects[idx]
        pred_full[class_id,:,:] = pred[idx,:,:]

    for threshold in background_threshold:
        # fill in background and non-present objects
        # method vanilla
        if bg_method == 'vanilla':
            pred_full[0,:,:] = threshold  # background threshold
        # method#1
        elif bg_method == 'max':
            supporting_target = supporting_target.clamp(min=threshold)
            pred_full[0,:,:] = supporting_target
        # method#2
        elif bg_method == 'exact':
            pred_full[0,:,:] = supporting_target
        # method#3
        elif bg_method == 'avg':
            pred_full[0,:,:] = (threshold + supporting_target) / 2
        # method#4
        elif bg_method == 'offset':
            pred_full[0,:,:] = threshold + supporting_target

        pred_class = pred_full.argmax(dim=0).numpy()

        segments = pred_class
        img = palette[segments]
        img = Image.fromarray(img)
        os.makedirs(save_path, exist_ok=True)
        img.save(os.path.join(save_path, f'segment-{threshold}.png'))


hungarian_first_run = True
TP = None
FP = None
FN = None
ALL = None
def hungarian_evaluate(pred, label, n_class, save_path, final_call):
    global hungarian_first_run
    global TP
    global FP
    global FN
    global ALL

    label = label - 1

    if final_call:
        acc = TP.sum() / ALL
        iou = TP / (TP + FP + FN)
        print(iou)
        miou = np.nanmean(iou)

        print(f'acc {acc}, miou {miou}')
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'evaluation.txt'), 'w') as f:
            f.write(f'acc {acc}, miou {miou}')
        return

    if hungarian_first_run:
        hungarian_first_run = False
        TP = np.zeros(n_class)
        FP = np.zeros(n_class)
        FN = np.zeros(n_class)
        ALL = 0

    tp, fp, fn, all, _, _ = hungarian_matching(pred.detach().cpu(), label, n_class)
    TP += tp
    FP += fp
    FN += fn
    ALL += all


def hungarian_visualization(pred, save_path, final_call):
    if final_call:
        return

    pred_class = pred.detach().cpu().numpy()

    segments = pred_class
    img = palette[segments]
    img = Image.fromarray(img)
    os.makedirs(save_path, exist_ok=True)
    img.save(os.path.join(save_path, f'segment.png'))
