import json
import numpy as np
from PIL import Image, ImageDraw
import os
from typing import List, Dict, Any, Union


def convert_coco_to_binary_masks(coco_json_path: str) -> List[Dict[str, Any]]:
    """
    Convert COCO annotation JSON file to a list of dictionaries with file names and binary segmentation masks.

    Args:
        coco_json_path: Path to the COCO annotation JSON file

    Returns:
        List of dictionaries, each with keys:
            - "file_name": Name of the image file
            - "binary_segmentation_mask": Binary mask as numpy array (1 for object, 0 for background)
    """
    # Load COCO annotations
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Create a mapping from image_id to image details
    image_map = {img["id"]: img for img in coco_data.get("images", [])}

    # Initialize result list
    result = []

    # Process annotations
    for annotation in coco_data.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id not in image_map:
            continue

        image_info = image_map[image_id]
        width = image_info.get("width")
        height = image_info.get("height")
        file_name = image_info.get("file_name")

        # Create an empty binary mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Get segmentation data
        segmentation = annotation.get("segmentation", [])

        # Process segmentation polygons
        for polygon in segmentation:
            # Convert polygon to a list of (x, y) points
            if len(polygon) >= 6:  # At least 3 points (x1,y1,x2,y2,x3,y3)
                points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]

                # Create a PIL Image to draw the polygon
                pil_mask = Image.new("L", (width, height), 0)
                ImageDraw.Draw(pil_mask).polygon(points, outline=1, fill=1)

                # Convert PIL mask to numpy array and combine with existing mask
                polygon_mask = np.array(pil_mask)
                mask = np.logical_or(mask, polygon_mask).astype(np.uint8)

        # Add to result
        result.append({"file_name": file_name, "binary_segmentation_mask": mask})

    return result


def save_masks_to_files(masks_data: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Save binary masks to image files.

    Args:
        masks_data: List of dictionaries with file_name and binary_segmentation_mask
        output_dir: Directory to save mask images
    """
    os.makedirs(output_dir, exist_ok=True)

    for item in masks_data:
        file_name = item["file_name"]
        mask = item["binary_segmentation_mask"]

        # Convert binary mask to image (255 for foreground)
        mask_img = Image.fromarray(mask * 255).convert("L")

        # Create output file name (replace extension with .png)
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}_mask.png")

        # Save mask image
        mask_img.save(output_path)


# Example usage
if __name__ == "__main__":
    # Example path to COCO annotation file
    coco_json_path = "path/to/coco_annotations.json"

    # Convert COCO annotations to binary masks
    masks_data = convert_coco_to_binary_masks(coco_json_path)

    # Print information about the first few masks
    for i, item in enumerate(masks_data[:3]):
        print(f"Image: {item['file_name']}")
        mask = item["binary_segmentation_mask"]
        print(f"Mask shape: {mask.shape}, Unique values: {np.unique(mask)}")
        print(f"Foreground pixels: {np.sum(mask == 1)}")
        print("-" * 40)

    # Optionally save masks to files
    # save_masks_to_files(masks_data, "output_masks")
