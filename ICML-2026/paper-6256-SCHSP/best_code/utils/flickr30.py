import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

def load_captions(captions_file):
    # Returns list of (image_id, caption)
    pairs = []
    with open(captions_file, 'r', encoding='utf-8') as f:
        # Skip header if present
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            assert len(line.split(".jpg,")) == 2, f"Invalid line format in captions file: {line}"
            key, caption = line.split(".jpg,")
            # Remove leading/trailing whitespace
            caption = caption.strip()
            # Remove leading/trailing quotes if present
            caption = caption.strip('"').strip("'").strip()
            pairs.append((key, caption))
    return pairs

def build_index(pairs):
    # Map unique images and captions to indices
    image_ids = sorted(set(img for img, _ in pairs))
    image2idx = {img: i for i, img in enumerate(image_ids)}
    captions = [cap for _, cap in pairs]
    pairs_idx = [(image2idx[img], cap) for img, cap in pairs]
    return image_ids, captions, pairs_idx

def evaluate(image_embeds, text_embeds, pairs_idx, num_images, top_k=(1,5)):
    # image_embeds and text_embeds to float
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()
    # Image-to-text
    APs = []
    P1 = 0
    P5_total = 0  # Sum of P@5 scores for each image
    n = num_images
    # Build ground truth captions per image
    gt_caps = {i: [] for i in range(n)}
    for idx, (_, cap_idx) in enumerate(pairs_idx):
        image_idx, _ = pairs_idx[idx]
        gt_caps[image_idx].append(idx)
    for i in tqdm(range(n), desc="Eval I2T"):
        sims = (text_embeds @ image_embeds[i]).cpu().numpy()
        gt = np.zeros(len(sims))
        gt[gt_caps[i]] = 1
        APs.append(average_precision_score(gt, sims))
        order = np.argsort(-sims)
        
        # P@1: 1 if any ground truth caption is in top 1, 0 otherwise
        if any(j in order[:1] for j in gt_caps[i]):
            P1 += 1
        
        # P@5: (number of ground truth captions in top 5) / (total ground truth captions for this image)
        num_gt_in_top5 = sum(1 for j in gt_caps[i] if j in order[:5])
        num_gt_total = len(gt_caps[i])
        p5_for_image = num_gt_in_top5 / num_gt_total if num_gt_total > 0 else 0
        P5_total += p5_for_image
        
    mAP = np.mean(APs)
    p1 = P1 / n
    p5 = P5_total / n  # Average P@5 across all images

    # Text-to-image
    P1_t2i = 0
    # Only iterate over the number of captions in the captions file
    for j in tqdm(range(len(pairs_idx)), desc="Eval T2I"):
        sims = (image_embeds @ text_embeds[j]).cpu().numpy()
        order = np.argsort(-sims)
        img_idx, _ = pairs_idx[j]
        if img_idx == order[0]:
            P1_t2i += 1
    p1_t2i = P1_t2i / len(pairs_idx)

    return {"i2t_mAP": mAP, "i2t_p1": p1, "i2t_p5": p5, "t2i_p1": p1_t2i}

def filter_and_reorder_embeds(image_embeds, text_embeds, all_image_ids, all_captions, used_image_ids, used_captions):
    # Map from image_id to index in all_image_ids
    image_id_to_idx = {img: i for i, img in enumerate(all_image_ids)}
    # Map from caption to index in all_captions
    caption_to_idx = {cap: i for i, cap in enumerate(all_captions)}
    # Reorder image_embeds to match used_image_ids
    image_indices = [image_id_to_idx[img] for img in used_image_ids]
    filtered_image_embeds = image_embeds[image_indices]
    # Reorder text_embeds to match used_captions
    text_indices = [caption_to_idx[cap] for cap in used_captions]
    filtered_text_embeds = text_embeds[text_indices]
    return filtered_image_embeds, filtered_text_embeds

def evaluate_flickr_retrieval(image_embeds, text_embeds, captions_file, actual_image_ids=None, actual_captions=None):
    """
    Args:
        image_embeds: Image embeddings tensor
        text_embeds: Text embeddings tensor  
        captions_file: Path to captions file
        actual_image_ids: List of image IDs in the same order as image_embeds (from dataloader)
        actual_captions: List of captions in the same order as text_embeds (from text processing)
    """
    # Load data from captions file
    pairs = load_captions(captions_file)
    
    if actual_image_ids is not None and actual_captions is not None:
        print("Using provided actual image and caption ordering")
        
        # Create mapping for the provided actual order
        image_ids = actual_image_ids
        captions = actual_captions
        
        # Build pairs_idx based on actual ordering
        image2idx = {img: i for i, img in enumerate(image_ids)}
        pairs_idx = []
        
        # Map each caption to its corresponding image index
        caption_to_image = {cap: img for img, cap in pairs}
        for cap_idx, cap in enumerate(captions):
            if cap in caption_to_image:
                img_id = caption_to_image[cap]
                if img_id in image2idx:
                    img_idx = image2idx[img_id]
                    pairs_idx.append((img_idx, cap_idx))
                else:
                    print(f"Warning: Image ID {img_id} not found in actual_image_ids")
            else:
                print(f"Warning: Caption '{cap[:50]}...' not found in captions file")
        
        print(f"Mapped {len(pairs_idx)} caption-image pairs")
        
    else:
        print("Warning: No actual ordering provided - using original method (may have alignment issues)")
        image_ids, captions, pairs_idx = build_index(pairs)
        
        print(f"Loaded {len(image_ids)} images and {len(captions)} captions from file order.")
        
        # Check if we have filtering parameters (the old way)
        # Note: This is kept for backward compatibility but may have alignment issues
    
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
        print(f"Truncated to {min_len} captions")
        
        # Update pairs_idx to only include valid indices
        pairs_idx = [(img_idx, cap_idx) for img_idx, cap_idx in pairs_idx 
                     if img_idx < len(image_ids) and cap_idx < len(captions)]
    
    print(f"Final evaluation: {len(image_ids)} images, {len(captions)} captions, {len(pairs_idx)} pairs")
    
    results = evaluate(image_embeds, text_embeds, pairs_idx, len(image_ids))
    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results

def create_aligned_flickr_embeddings(backbone, text_encoder, dataloader, captions_file, device='cuda', max_images=None):
    """
    Create properly aligned image and text embeddings from a Flickr30K dataloader.
    
    Args:
        backbone: Image encoder model
        text_encoder: Text encoder function
        dataloader: DataLoader for images
        captions_file: Path to captions file
        device: Device to run on
        max_images: Maximum number of images to process (None for all)
    
    Returns:
        tuple: (image_embeds, text_embeds, actual_image_ids, actual_captions)
    """
    print("Creating aligned Flickr30K embeddings...")
    
    # Load captions from file
    pairs = load_captions(captions_file)
    
    # Get actual image order from dataloader
    from dataloaders.datasets_and_dataloaders import get_flickr30k_dataloaders
    # We need to get the dataset to access image paths
    # This assumes the dataloader comes from get_flickr30k_dataloaders
    testset = dataloader.dataset
    
    actual_image_ids = []
    if hasattr(testset, 'image_paths'):
        actual_image_paths = testset.image_paths
        if max_images:
            actual_image_paths = actual_image_paths[:max_images]
        actual_image_ids = [path.split('/')[-1].replace('.jpg', '') for path in actual_image_paths]
        print(f"Found {len(actual_image_ids)} images in dataloader order")
    else:
        print("ERROR: Dataset doesn't have image_paths attribute!")
        return None, None, None, None
    
    # Compute image embeddings in dataloader order
    print("Computing image embeddings...")
    image_embeds = []
    count = 0
    
    backbone.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            embeddings = backbone(images)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embeddings = embeddings.float()  # Ensure float32
            image_embeds.append(embeddings.cpu())
            
            count += len(images)
            if max_images and count >= max_images:
                break
    
    image_embeds = torch.cat(image_embeds, dim=0)
    if max_images:
        image_embeds = image_embeds[:max_images]
        actual_image_ids = actual_image_ids[:max_images]
    
    # Reorder captions to match actual image order
    print("Aligning captions with image order...")
    aligned_captions = []
    
    for img_id in actual_image_ids:
        # Find all captions for this image
        captions_for_image = [cap for img, cap in pairs if img == img_id]
        
        if not captions_for_image:
            print(f"WARNING: No captions found for image {img_id}")
            # Add dummy captions to maintain structure
            captions_for_image = [f"No caption found for {img_id}"] * 5
        
        aligned_captions.extend(captions_for_image)
    
    # Compute text embeddings for aligned captions
    print("Computing text embeddings for aligned captions...")
    text_embeds = []
    batch_size = 50
    
    for i in tqdm(range(0, len(aligned_captions), batch_size), desc="Processing captions"):
        batch_captions = aligned_captions[i:i+batch_size]
        embeddings = text_encoder(batch_captions)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        embeddings = embeddings.float()  # Ensure float32
        text_embeds.append(embeddings.cpu())
    
    text_embeds = torch.cat(text_embeds, dim=0)
    
    print(f"Created aligned embeddings:")
    print(f"  Images: {len(image_embeds)} embeddings")
    print(f"  Captions: {len(text_embeds)} embeddings")
    print(f"  Average captions per image: {len(text_embeds)/len(image_embeds):.1f}")
    
    return image_embeds, text_embeds, actual_image_ids, aligned_captions