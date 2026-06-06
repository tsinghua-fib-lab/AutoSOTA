# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamHierarchicalMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        high_score_thresh: float = 8.5,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          high_score_thresh (float): A filtering threshold in [-inf,inf], to find out
            the unmasked area for the next generation.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.high_score_thresh = high_score_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        
    def set_point_grids(self, point_grids):
        self.point_grids = point_grids
        
    def set_points_per_side(self, points_per_side):
        self.point_grids = build_all_layer_point_grids(
                points_per_side,
                0,
                1,
            )

    @torch.no_grad()
    def set_image(self, image: np.ndarray) -> MaskData:
        # Crop the image and calculate embeddings
        self.predictor.set_image(image)

    @torch.no_grad()
    def hierarchical_generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        self.set_image(image)
        self.set_points_per_side(self.points_per_side // 4)
        ori_masks, or_results = self.generate(image, True)

        ih, iw, _ = image.shape
        hstride = ih // self.points_per_side
        wstride = iw // self.points_per_side
        new_points = []

        pass_counter = 0
        full_point_grids = np.array(self.point_grids)

        for mask in range(full_point_grids.shape[1]):
            point_coords = [full_point_grids[0, mask, 0] * iw, full_point_grids[0, mask, 1] * ih]
            for sy in [-1, 0, 1]:
                for sx in [-1, 0, 1]:
                    if (sy == 0 and sx == 0) or or_results[int(point_coords[0] + wstride * sy), int(point_coords[1] + hstride * sx)]:
                        continue
                    new_points.append([(point_coords[0] + wstride * sy) / iw, (point_coords[1] + hstride * sx) / ih])
            if point_coords[0] + wstride * 2 < iw:
                for sx in [-1, 0, 1]:
                    if or_results[int(point_coords[0] + wstride * 2), int(point_coords[1] + hstride * sx)]:
                        continue
                    new_points.append([(point_coords[0] + wstride * 2) / iw, (point_coords[1] + hstride * sx) / ih])
            if point_coords[1] + hstride * 2 < ih:
                for sy in [-1, 0, 1]:
                    if or_results[int(point_coords[0] + wstride * sy), int(point_coords[1] + hstride * 2)]:
                        continue
                    new_points.append([(point_coords[0] + wstride * sy) / iw, (point_coords[1] + hstride * 2) / ih])
            if point_coords[0] + wstride * 2 < iw and point_coords[1] + hstride * 2 < ih:
                if or_results[int(point_coords[0] + wstride * 2), int(point_coords[1] + hstride * 2)]:
                    continue
                new_points.append([(point_coords[0] + wstride * 2) / iw, (point_coords[1] + hstride * 2) / ih])

        self.set_point_grids([np.array(new_points)])
        new_masks = self.generate(image, False)

        new_masks.cat(ori_masks)
        new_masks = self.post_process(image, new_masks)
        return new_masks

    @torch.no_grad()
    def generate(self, image: np.ndarray, need_high: bool) -> MaskData:
        orig_size = image.shape[:2]
        # Get points for this crop
        points_scale = np.array(orig_size)[None, ::-1]
        points_for_image = self.point_grids[0] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            orig_h, orig_w = orig_size
            # Run model on this batch
            transformed_points = self.predictor.transform.apply_coords(points, orig_size)
            in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            masks, iou_preds, _ = self.predictor.predict_torch(
                in_points[:, None, :],
                in_labels[:, None],
                return_logits=True,
            )
            
            # Serialize predictions and store in MaskData
            batch_data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
            )
            del masks
            
            if self.pred_iou_thresh > 0.0:
                keep_mask = batch_data["iou_preds"] > self.pred_iou_thresh
                batch_data.filter(keep_mask)

            # Calculate stability score
            batch_data["stability_score"] = calculate_stability_score(
                batch_data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = batch_data["stability_score"] >= self.stability_score_thresh
                batch_data.filter(keep_mask)
            
            if need_high:
                batch_data["high_masks"] = batch_data["masks"] > self.high_score_thresh
            batch_data["masks"] = batch_data["masks"] > self.predictor.model.mask_threshold
            batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])
            keep_mask = ~is_box_near_crop_edge(batch_data["boxes"], [0, 0, orig_w, orig_h], [0, 0, orig_w, orig_h])
            if not torch.all(keep_mask):
                batch_data.filter(keep_mask)

            # Compress to RLE
            batch_data["rles"] = mask_to_rle_pytorch(batch_data["masks"])
            data.cat(batch_data)
            del batch_data
            
        if need_high:
            high_masks = data["high_masks"]
            or_results = torch.zeros([high_masks.shape[1], high_masks.shape[2]]).to(high_masks.device)
            for mask in high_masks:
                or_results = torch.logical_or(or_results, mask)
            del data["high_masks"]
                
            or_results = or_results.permute(1, 0)

            del data['masks']
            return data, or_results
        else:
            del data['masks']
            return data
    
    @torch.no_grad()
    def reset_image(self):
        self.predictor.reset_image()
    
    @torch.no_grad()
    def post_process(self, image: np.ndarray, data: MaskData) -> List[Dict[str, Any]]:
        orig_size = image.shape[:2]
        orig_h, orig_w = orig_size

        
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            data = self.postprocess_small_regions(
                data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        
        # Encode masks
        if self.output_mode == "coco_rle":
            data["segmentations"] = [coco_encode_rle(rle) for rle in data["rles"]]
        elif self.output_mode == "binary_mask":
            data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]]
        else:
            data["segmentations"] = data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(data["segmentations"])):
            ann = {
                "segmentation": data["segmentations"][idx],
                "area": area_from_rle(data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(data["boxes"][idx]).tolist(),
                "predicted_iou": data["iou_preds"][idx].item(),
                "point_coords": [data["points"][idx].tolist()],
                "stability_score": data["stability_score"][idx].item(),
            }
            curr_anns.append(ann)
            
        # print("post use time: {}".format(time.time() - st))
        return curr_anns

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
