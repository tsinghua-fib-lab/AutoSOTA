import torch
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score


def load_coco_captions(json_file):
    """
    Load COCO captions from the Karpathy format JSON file.
    Limits to first 5 captions per image for consistency with dataset loader.
    
    Returns:
        image_ids: List of image identifiers in order
        captions: List of all captions (max 5 per image)
        pairs_idx: List of (image_idx, caption_idx) pairs
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_ids = []
    captions = []
    pairs_idx = []
    
    for img_data in data['images']:
        img_id = img_data['filename']  # Use filename as unique image identifier
        image_idx = len(image_ids)
        image_ids.append(img_id)
        
        # Add captions for this image (limit to first 5)
        caption_count = 0
        for sentence in img_data['sentences']:
            if caption_count >= 5:  # Only keep first 5 captions
                break
            caption = sentence['raw'].strip()
            caption_idx = len(captions)
            captions.append(caption)
            pairs_idx.append((image_idx, caption_idx))
            caption_count += 1
    
    return image_ids, captions, pairs_idx


def evaluate_coco_retrieval(image_embeds, text_embeds, json_file):
    """
    Evaluate retrieval performance on COCO dataset.
    
    Args:
        image_embeds: Image embeddings tensor (num_images, embedding_dim)
        text_embeds: Text embeddings tensor (num_captions, embedding_dim)
        json_file: Path to COCO metadata JSON file
        
    Returns:
        dict: Evaluation metrics including i2t_mAP, i2t_p1, i2t_p5, t2i_p1
    """
    # Load captions from JSON
    image_ids, captions, pairs_idx = load_coco_captions(json_file)
    
    # Ensure we have the right number of embeddings
    if len(image_embeds) != len(image_ids):
        print(f"Warning: Mismatch between image embeddings ({len(image_embeds)}) and image IDs ({len(image_ids)})")
        min_len = min(len(image_embeds), len(image_ids))
        image_embeds = image_embeds[:min_len]
        image_ids = image_ids[:min_len]
        print(f"Truncated to {min_len} images")
    
    if len(text_embeds) != len(captions):
        print(f"Warning: Mismatch between text embeddings ({len(text_embeds)}) and captions ({len(captions)})")
        min_len = min(len(text_embeds), len(captions))
        text_embeds = text_embeds[:min_len]
        captions = captions[:min_len]
        # Rebuild pairs_idx with valid indices
        pairs_idx = [(img_idx, cap_idx) for img_idx, cap_idx in pairs_idx if cap_idx < min_len]
        print(f"Truncated to {min_len} captions")
    
    # Convert to float for computation
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()
    
    num_images = len(image_ids)
    
    # Image-to-text retrieval
    APs = []
    P1_i2t = 0
    P5_total = 0
    
    # Build ground truth captions per image
    gt_caps = {i: [] for i in range(num_images)}
    for cap_idx, (img_idx, _) in enumerate(pairs_idx):
        gt_caps[img_idx].append(cap_idx)
    
    # Evaluate I2T
    for i in tqdm(range(num_images), desc="Eval I2T"):
        sims = (text_embeds @ image_embeds[i]).cpu().numpy()
        gt = np.zeros(len(sims))
        gt[gt_caps[i]] = 1
        APs.append(average_precision_score(gt, sims))
        order = np.argsort(-sims)
        
        # P@1: 1 if any ground truth caption is in top 1, 0 otherwise
        if any(j in order[:1] for j in gt_caps[i]):
            P1_i2t += 1
        
        # P@5: (number of ground truth captions in top 5) / (total ground truth captions for this image)
        num_gt_in_top5 = sum(1 for j in gt_caps[i] if j in order[:5])
        num_gt_total = len(gt_caps[i])
        p5_for_image = num_gt_in_top5 / num_gt_total if num_gt_total > 0 else 0
        P5_total += p5_for_image
    
    mAP = np.mean(APs)
    p1_i2t = P1_i2t / num_images
    p5_i2t = P5_total / num_images
    
    # Text-to-image retrieval
    P1_t2i = 0
    for j in tqdm(range(len(pairs_idx)), desc="Eval T2I"):
        sims = (image_embeds @ text_embeds[j]).cpu().numpy()
        order = np.argsort(-sims)
        img_idx, _ = pairs_idx[j]
        if img_idx == order[0]:
            P1_t2i += 1
    p1_t2i = P1_t2i / len(pairs_idx)
    
    return {
        "i2t_mAP": mAP,
        "i2t_p1": p1_i2t,
        "i2t_p5": p5_i2t,
        "t2i_p1": p1_t2i
    }
