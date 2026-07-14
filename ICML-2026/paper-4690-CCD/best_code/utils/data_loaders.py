import os
import numpy as np
from PIL import Image

def load_tv_data(metric_dir, jsonl_data, metadata, metric_name="score_diff"):
    raw_maps = []
    masks = []
    
    BASE_DIR = os.getcwd()
    METADATA_PATH = os.path.join(BASE_DIR, "templates/metadata.parquet")

    for i, item in enumerate(jsonl_data):
        prompt_idx = i
        gen_seeds_encoded = item.get("gen_seeds", [])
        parsed_pairs = []
        if gen_seeds_encoded:
            for x in gen_seeds_encoded:
                if isinstance(x, list) or isinstance(x, tuple):
                    parsed_pairs.append(x)
                else:
                    parsed_pairs.append([x, -1])
        
        if not parsed_pairs:
            continue
            
        valid_pairs = []
        for seed, t_idx in parsed_pairs:
            metric_filename = f"prompt_{prompt_idx:04d}_seed_{seed:02d}_{metric_name}.npy"
            metric_path = os.path.join(metric_dir, metric_filename)

            if os.path.exists(metric_path):
                valid_pairs.append((seed, t_idx, metric_path))
        
        if not valid_pairs:
            continue

        for seed, t_idx, metric_path in valid_pairs:
            try:
                m_map = np.load(metric_path)
            except Exception:
                continue
            
            if np.isinf(m_map).any() or np.isnan(m_map).any():
                continue
            
            if m_map.shape != (256, 256):
                img_map = Image.fromarray(m_map.astype(np.float32), mode="F")
                img_map = img_map.resize((256, 256), Image.BILINEAR)
                m_map = np.array(img_map)
            
            if t_idx < 0 or t_idx >= len(metadata):
                gt_mask = np.ones((256, 256), dtype=float)
            else:
                mask_rel = metadata.iloc[t_idx]["mask_file"]
                mask_abs = os.path.join(BASE_DIR, mask_rel)
                if not os.path.exists(mask_abs):
                    mask_abs_alt = os.path.join(os.path.dirname(METADATA_PATH), os.path.basename(mask_rel))
                    if os.path.exists(mask_abs_alt):
                        mask_abs = mask_abs_alt
                    else:
                        continue
                
                try:
                    img = Image.open(mask_abs).convert("L")
                    if img.size != (256, 256):
                        img = img.resize((256, 256), Image.NEAREST)
                    mask_arr = np.array(img).astype(np.float32) / 255.0
                    gt_mask = (mask_arr > 0.5).astype(float)
                except Exception:
                    continue
            
            raw_maps.append(m_map)
            masks.append(gt_mask)
            
    return raw_maps, masks

def load_nmem_data(metric_dir, prompts, metric_name="score_diff", indices=None):
    raw_maps = []
    masks = []
    
    if indices is None:
        indices = range(len(prompts))
    
    for i in indices:
        metric_filename = f"nmem_prompt_{i:04d}_seed_00_{metric_name}.npy"
        metric_path = os.path.join(metric_dir, metric_filename)
        
        if not os.path.exists(metric_path):
            continue
        
        try:
            m_map = np.load(metric_path)
        except Exception:
            continue
        
        if np.isinf(m_map).any() or np.isnan(m_map).any():
            continue
        
        if m_map.shape != (256, 256):
            img_map = Image.fromarray(m_map.astype(np.float32), mode="F")
            img_map = img_map.resize((256, 256), Image.BILINEAR)
            m_map = np.array(img_map)
            
        raw_maps.append(m_map)
        masks.append(np.zeros((256, 256), dtype=float))
        
    return raw_maps, masks
