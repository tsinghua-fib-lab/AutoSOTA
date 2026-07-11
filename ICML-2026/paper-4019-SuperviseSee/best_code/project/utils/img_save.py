import os
import cv2
import json
import numpy as np
from pycocotools import mask

# =========== save mask as json =========== 
def save_json(masks, path):
    """
    save binary masks as COCO-style RLE JSON file
    """
    rles = []
    for m in masks:
        rle = mask.encode(np.asfortranarray(m.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        rles.append(rle)

    with open(path, 'w') as f:
        json.dump(rles, f)


# =========== save json as instance mask png =========== 
def load_rles(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    
def decode_rles(rles):
    masks = []
    for rle in rles:
        rle_obj = {'size': rle['size'], 'counts': rle['counts'].encode('ascii')}
        masks.append(mask.decode(rle_obj).astype(bool))
    return masks

def colorize_label_mask(label_mask):
    n_labels = int(label_mask.max())
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(n_labels)]
    h, w = label_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, n_labels + 1):
        color_img[label_mask == i] = colors[i - 1]
    return color_img

def assemble_instance_mask(masks):
    if not masks:
        return None
    h, w = masks[0].shape
    inst_mask = np.zeros((h, w), dtype=np.int32)
    for idx, m in enumerate(masks, start=1):
        inst_mask[m & (inst_mask == 0)] = idx
    return inst_mask


def save_instance(json_path, output_dir, label):
    """Save both instance masks as colorized PNGs."""
    rles = load_rles(json_path)
    masks = decode_rles(rles)
    inst_mask = assemble_instance_mask(masks)
    if inst_mask is None:
        print(f"No instances in {json_path}")
        return

    # save original mask
    img = colorize_label_mask(inst_mask)
    output_path = os.path.join(output_dir, f"{label}_mask.png")
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

