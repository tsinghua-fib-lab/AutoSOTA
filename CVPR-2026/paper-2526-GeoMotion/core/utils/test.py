import random
import os
from glob import glob
from tqdm import tqdm

q_ts = list(range(8, 17, 4))

data_dir = "current-data-dir/waymo/drivestudio/data/waymo/processed/train_new"

sample_list = [item for item in sorted(os.listdir(data_dir))
            if os.path.isdir(os.path.join(data_dir, item))]

img_dirs_list, mask_dirs_list, track_dirs_list = [], [], []    

for seq_name in (sample_list):
    # Load image list.
    img_dir = os.path.join(data_dir,seq_name, "images")
    img_dirs_list.append(img_dir)

for idx in tqdm(range(len(img_dirs_list)), desc="checking..."):
    
    img_dir = img_dirs_list[idx]
    depth_dir = img_dir.replace("images", "depth_anything_v2")
    depth_paths = sorted(glob(os.path.join(depth_dir, "*.png"))) + sorted(
        glob(os.path.join(depth_dir, "*.jpg")) + sorted(glob(os.path.join(depth_dir, "*.jpeg")))
    )
    frame_names = [os.path.splitext(os.path.basename(p))[0] for p in depth_paths]

    num_frames = len(frame_names)
    # q_ts = [x for x in q_ts if x <= num_frames]

    for q_t in q_ts:
        name = frame_names[q_t]