import cv2
import os
from glob import glob
from tqdm import tqdm
import shutil

oridir = "/data0/hexiankang/code/HOI4D_release"

mp4_files = sorted(glob(f'{oridir}/**/*.mp4', recursive=True))

images_folder = '/data0/hexiankang/code/SegAnyMo/data/HOI4D/images'
mask_folder = "/data0/hexiankang/code/SegAnyMo/data/HOI4D/dynamic_masks"

for i, video_path in tqdm(enumerate(mp4_files), "converting:"):
    # save images from video
    output_folder = os.path.join(images_folder, f'{i:05d}')
    os.makedirs(output_folder, exist_ok=True)

    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0
    success, frame = video_capture.read()

    interval = 4

    while success:
        if frame_count % interval == 0:
            image_path = os.path.join(output_folder, f'{frame_count:05d}.png')
            
            cv2.imwrite(image_path, frame)
        
        success, frame = video_capture.read()
        frame_count += 1

    video_capture.release()
    
    # move dynamic mask
    out_mask_folder = os.path.join(mask_folder, f'{i:05d}')
    os.makedirs(out_mask_folder, exist_ok=True)
    dir_name = os.path.dirname(os.path.dirname(video_path))
    ori_mask_dir = dir_name.replace('HOI4D_release',"HOI4D_annotations")
    
    shift_mask_dir = os.path.join(ori_mask_dir, "2Dseg", "shift_mask")
    mask_dir = os.path.join(ori_mask_dir, "2Dseg", "mask")

    if os.path.isdir(shift_mask_dir) and len(glob(os.path.join(shift_mask_dir, "*.png"))) > 0:
        ori_mask_dir = shift_mask_dir
    else:
        ori_mask_dir = mask_dir

    mask_paths = sorted(glob(os.path.join(ori_mask_dir, "*.png")))
    mask_paths = mask_paths[::interval]
    # import pdb;pdb.set_trace()
    
    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path)
        out_mask_path = os.path.join(out_mask_folder, mask_name)
        shutil.move(mask_path, out_mask_path)
        assert os.path.exists(out_mask_path), f"File move failed: {out_mask_path} does not exist."