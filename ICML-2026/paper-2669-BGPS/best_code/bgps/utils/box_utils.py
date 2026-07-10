from typing import Union
import numpy as np
import torch

BBOX_DTYPE = Union[np.ndarray, torch.Tensor]

__all__ = ["boxes_area", "xywh_to_xyxy", "xyxy_to_xywh",
           "boxes_to_deltas", "deltas_to_boxes"]


def boxes_area(boxes: BBOX_DTYPE) -> Union[float, torch.Tensor]:
    # boxes: (x1, y1, x2, y2)
    width = (boxes[:, 2] - boxes[:, 0] + 1)
    height = (boxes[:, 3] - boxes[:, 1] + 1)
    areas = width * height
    return areas


def xywh_to_xyxy(boxes: BBOX_DTYPE) -> BBOX_DTYPE:
    # (x1, y1, w, h) -> (x1, y1, x2, y2)
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2] + x_min - 1  # width = x_max - x_min + 1
    y_max = boxes[:, 3] + y_min - 1  # height = y_max - y_min + 1
    # assume that width and height are positive values.

    if isinstance(boxes, np.ndarray):
        new_boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)  # (N, 4)
    elif isinstance(boxes, torch.Tensor):
        new_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # (N, 4)
    else:
        raise ValueError(f"xywh2xyxy requires np.ndarray or torch.Tensor, got {type(boxes)}.")
    return new_boxes


def xyxy_to_xywh(boxes: BBOX_DTYPE) -> BBOX_DTYPE:
    # (x1, y1, x2, y2) -> (x1, y1, w, h)
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    width = boxes[:, 2] - x_min + 1  # width = x_max - x_min + 1
    height = boxes[:, 3] - y_min + 1  # height = y_max - y_min + 1
    # assume that x1 < x2 and y1 < y2.

    if isinstance(boxes, np.ndarray):
        new_boxes = np.stack([x_min, y_min, width, height], axis=-1)  # (N, 4)
    elif isinstance(boxes, torch.Tensor):
        new_boxes = torch.stack([x_min, y_min, width, height], dim=-1)  # (N, 4)
    else:
        raise ValueError(f"xyxy2xywh requires np.ndarray or torch.Tensor, got {type(boxes)}.")
    return new_boxes


def boxes_to_deltas(anchors: BBOX_DTYPE, gt_boxes: BBOX_DTYPE, weights=(10.0, 10.0, 5.0, 5.0)):
    """Calculate target deltas for object detection.
    :param anchors:     (num_boxes, 4)  x1, y1, x2, y2
    :param gt_boxes:    (num_boxes, 4)  x1, y1, x2, y2
    :param weights:     scaling for deltas
    :return:            (num_boxes, 4)  dx, dy, dw, dh
    :note:
        make sure that input arguments are xyxy format.
    """
    anchors = anchors.float()
    gt_boxes = gt_boxes.float()

    anchor_widths = anchors[:, 2] - anchors[:, 0] + 1.0
    anchor_heights = anchors[:, 3] - anchors[:, 1] + 1.0
    anchor_cx = anchors[:, 0] + 0.5 * anchor_widths
    anchor_cy = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_cx - anchor_cx) / anchor_widths
    targets_dy = wy * (gt_cy - anchor_cy) / anchor_heights
    targets_dw = ww * np.log(gt_widths / anchor_widths)
    targets_dh = wh * np.log(gt_heights / anchor_heights)

    if isinstance(anchors, np.ndarray):
        deltas = np.stack([targets_dx, targets_dy, targets_dw, targets_dh], axis=-1)  # (N, 4)
    elif isinstance(anchors, torch.Tensor):
        deltas = torch.stack([targets_dx, targets_dy, targets_dw, targets_dh], dim=-1)  # (N, 4)
    else:
        raise ValueError(f"boxes_to_deltas requires np.ndarray or torch.Tensor, got {type(anchors)}.")
    return deltas


def deltas_to_boxes(anchors: BBOX_DTYPE, deltas: BBOX_DTYPE, weights=(10.0, 10.0, 5.0, 5.0)):
    """Calculate predicted boxes from anchors and deltas.
    :param anchors:     (num_boxes, 4)  x1, y1, x2, y2
    :param deltas:      (num_boxes, 4)  dx, dy, dw, dh
    :param weights:     scaling for deltas
    :return:            (num_boxes, 4)  x1, y1, x2, y2
    :note:
        make sure that input arguments are xyxy format.
        make sure to use the same weights as in 'boxes_to_deltas'.
    """
    anchors = anchors.float()
    deltas = deltas.float()

    anchor_widths = anchors[:, 2] - anchors[:, 0] + 1.0
    anchor_heights = anchors[:, 3] - anchors[:, 1] + 1.0
    anchor_cx = anchors[:, 0] + 0.5 * anchor_widths
    anchor_cy = anchors[:, 1] + 0.5 * anchor_heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh

    # prevent sending too large values into exp
    if isinstance(anchors, np.ndarray):
        dw = np.minimum(dw, 4.0)
        dh = np.minimum(dh, 4.0)
    elif isinstance(anchors, torch.Tensor):
        dw = torch.clamp_max(dw, 4.0)
        dh = torch.clamp_max(dh, 4.0)
    else:
        raise ValueError(f"deltas_to_boxes requires np.ndarray or torch.Tensor, got {type(anchors)}.")

    pred_cx = dx * anchor_widths + anchor_cx
    pred_cy = dy * anchor_heights + anchor_cy
    pred_widths = dw.exp() * anchor_widths
    pred_heights = dh.exp() * anchor_heights

    if isinstance(anchors, np.ndarray):
        pred_boxes = np.zeros_like(deltas)
    else:  # torch.Tensor
        pred_boxes = torch.zeros_like(deltas)

    pred_boxes[:, 0] = pred_cx - 0.5 * pred_widths
    pred_boxes[:, 1] = pred_cy - 0.5 * pred_heights
    pred_boxes[:, 2] = pred_cx + 0.5 * pred_widths - 1
    pred_boxes[:, 3] = pred_cy + 0.5 * pred_heights - 1
    return pred_boxes
