# steering_utils/data_loader.py
import json
import os
from PIL import Image
from typing import List, Dict, Any, Tuple

def load_steering_data(json_path: str) -> List[Dict[str, Any]]:
    """
    load data for steering.
    """
    if not os.path.exists(json_path):
        print(f"Error: Steering data file not found: {json_path}")
        raise FileNotFoundError(f"Steering data file not found: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f) 
    
    #  { "annotations": [...] }
    if isinstance(data, dict) and 'annotations' in data:
        dataset = data['annotations']
    #  [...]
    elif isinstance(data, list):
        dataset = data
    else:
        raise ValueError("JSON file is not in the expected format (must be a list, or a dict with an 'annotations' key)")

    print(f"Loaded {len(dataset)} samples from {json_path}")
    return dataset

def load_image_and_mask(image_path: str) -> Tuple[Image.Image, Image.Image]:
    """create black mask"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    pil_image = Image.open(image_path).convert('RGB')
    masked_pil_image = Image.new('RGB', pil_image.size, (0, 0, 0))
    return pil_image, masked_pil_image