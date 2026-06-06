import os
import cv2
import glob
import torch
import numpy as np
import argparse
import pickle
import json
from PIL import Image
from tqdm import tqdm
from glob import glob

def load_depth(depth_dir):
    # [1,1,H,W,1], unnormilized but scaled
    depth_paths = sorted(glob(os.path.join(depth_dir, "*.png"))) + sorted(
        glob(os.path.join(depth_dir, "*.jpg")) + sorted(glob(os.path.join(depth_dir, "*.jpeg")))
    )
    depth_list = []
    for path in depth_paths:
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        depth_tensor = torch.from_numpy(depth / 65535.0 * 1000.0)
        depth_list.append(depth_tensor)
    depths = torch.stack(depth_list, dim=0).permute(1,2,0).unsqueeze(0).unsqueeze(0)
    return depths

def read_imgs_from_path(img_dir):
    # Helper function placeholder if needed, 
    # assuming logic similar to typical image reading
    img_paths = sorted(glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.jpg")))
    imgs = []
    for p in img_paths:
        imgs.append(cv2.imread(p)) # logic depends on specific usage
    return imgs

def main(args):
    # load 3d track, camera, depth, 2d track -> 3d track
    if args.dataset == "pointodyssey":
        anno_path = os.path.join(args.data_dir, "anno.npz")
        anno = np.load(anno_path)
        # [t, n, 3]
        trajs_3d = anno['trajs_3d'].astype(np.float32)
        trajs_2d = anno['trajs_2d'].astype(np.float32)
        visible = anno['visibs']
        valid = anno['valids']
        n_mask = np.all(valid, axis=0)
        trajs_2d = trajs_2d[:,n_mask]
        
        # Note: PointOdyssey logic kept as is, but assuming img loading is handled
        # img_dir = os.path.join(args.data_dir, "rgbs")
        # imgs = read_imgs_from_path(img_dir)
        
    elif args.dataset == "dynamic_stereo":
        # load instance id mask
        instance_dir = os.path.join(args.data_dir, "instance_id_maps")
        map_png = sorted(glob(os.path.join(instance_dir, "*.png")))
        map_pkl = sorted(glob(os.path.join(instance_dir, "*.pkl")))

        # --- FIX 1: 加载时保持 Int 类型，不要用 ToTensor() 归一化 ---
        obj_list = []
        for png_path in map_png:
            with Image.open(png_path) as img:
                # 转换为 numpy 数组，保持 uint8 或 int32
                arr = np.array(img)
                tensor_image = torch.from_numpy(arr).long() # [H, W]
                obj_list.append(tensor_image)
                
        pkl_list = []
        for pkl_path in map_pkl:
            with open(pkl_path, 'rb') as file:
                data = pickle.load(file)
                pkl_list.append(data)
        
        # Process logic for identifying dynamic objects
        directory_record_path = os.path.join(os.path.dirname(args.data_dir), "processed_dir.json")
        
        # Handle "right" view by borrowing logic from "left" view
        if "right" in args.data_dir:
            if os.path.exists(directory_record_path):
                name = os.path.basename(args.data_dir.replace("right", "left"))
                with open(directory_record_path, 'r') as file:
                    data = json.load(file)
                if name in data:
                    all_dynamic = False
                    # Make sure dynamic_id is LongTensor for comparison
                    dynamic_id = torch.tensor(data[name], dtype=torch.long)
                else:
                    all_dynamic = True
            else:
                all_dynamic = True
        else:
            # Process "left" view (Main Logic)
            # import pdb; pdb.set_trace()
            obj_masks = torch.stack(obj_list, dim=0) # [T, H, W] LongTensor
            
            # load track and find dynamic
            track_dir = os.path.join(args.data_dir, "trajectories")
            track_paths = sorted(glob(os.path.join(track_dir, "*.pth")))

            trajs_2d, trajs_3d, imgs, visibles, instances = [],[],[],[],[]
            for path in tqdm(track_paths, desc="loading info...."):
                # [N,3],[H,W,3],[N]
                info = torch.load(path)
                traj_3d = info['traj_3d_world']
                traj_2d = info['traj_2d']
                # only for the left view
                img = info['img']
                visible = info['verts_inds_vis']
                instance = info['instances']
                import pdb;pdb.set_trace()
                trajs_2d.append(traj_2d)
                trajs_3d.append(traj_3d)
                imgs.append(img)
                visibles.append(visible)
                instances.append(instance)
            
            # Stack data
            trajs_2d = torch.stack(trajs_2d, dim=0)
            trajs_3d = torch.stack(trajs_3d, dim=0)
            # imgs = torch.stack(imgs, dim=0) # Optional, huge memory usage
            visibles = torch.stack(visibles, dim=0)
            instances = torch.stack(instances, dim=0)

            # calculate distance to determine dynamic
            distances = torch.sqrt(torch.sum(torch.diff(trajs_3d, dim=0) ** 2, dim=2))
            threshold = 0.0001
            
            # Determine whether the maximum moving distance is greater than threshold
            is_dynamic = (torch.max(distances, dim=0).values) >= threshold
            
            # Filter instances based on movement
            d_instance = instances[:, is_dynamic]
            
            # Determine whether an obj is dynamic using voting
            # Now inputs are proper Integers, so bincount works correctly
            total_counts = torch.bincount(instances.view(-1).long())
            # Ensure minlength covers all IDs
            max_id = max(total_counts.shape[0], d_instance.max() + 1 if d_instance.numel() > 0 else 0)
            counts = torch.bincount(d_instance.view(-1).long(), minlength=max_id)
            
            # Use strict comparison (half) to decide if an ID is dynamic
            # Note: total_counts might be shorter than counts if d_instance has weird IDs, fix shape
            if total_counts.shape[0] < counts.shape[0]:
                padding = torch.zeros(counts.shape[0] - total_counts.shape[0], device=total_counts.device)
                total_counts = torch.cat([total_counts, padding])
            elif counts.shape[0] < total_counts.shape[0]:
                 padding = torch.zeros(total_counts.shape[0] - counts.shape[0], device=counts.device)
                 counts = torch.cat([counts, padding])
                
            half_total_counts = total_counts / 2
            id_mask = (counts > half_total_counts) & (total_counts != 0)
        
            # Extract the actual dynamic IDs
            obj_id = torch.unique(obj_masks) # List of all IDs in the masks
            
            # Filter dynamic IDs
            # Ensure we don't index out of bounds if obj_id is larger than id_mask
            valid_ids = []
            for oid in obj_id:
                if oid < id_mask.shape[0] and id_mask[oid]:
                    valid_ids.append(oid)
            dynamic_id = torch.tensor(valid_ids, dtype=torch.long)

            # Check if all objects are dynamic (excluding background 0 usually, but logic depends on dataset)
            # Assuming 0 is background and it might be static.
            # Here we check if the number of dynamic IDs roughly equals total IDs.
            if dynamic_id.shape[0] >= (obj_id.shape[0] - 1): # -1 for potential background
                all_dynamic = True
            else:
                all_dynamic = False
        
        # --- FIX 2: 保存只包含动态物体的 Instance ID Maps ---
        # 改名保存目录，避免混淆二值 mask
        save_dir = os.path.join(args.data_dir, "instance_id_maps_dynamic")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Processing Masks... All Dynamic: {all_dynamic}")
        
        # 即使是 all_dynamic，为了保险起见，我们也重新生成一遍，确保只有我们在 dynamic_id 里的物体
        # 如果你想直接复制，也可以写 shutil.copy，但过滤一遍更稳妥
        
        for i, raw_mask_tensor in enumerate(obj_list):
            # raw_mask_tensor is [H, W] LongTensor
            
            # 创建一个全0 (背景) 的画布
            final_mask = torch.zeros_like(raw_mask_tensor)

            # 将属于动态 ID 的像素填回去
            # 使用 Tensor 操作一次性完成，比 Python 循环快
            # 创建一个 mask，标记哪些像素属于 dynamic_id
            # method: isin (torch >= 1.10)
            if hasattr(torch, 'isin'):
                is_dyn = torch.isin(raw_mask_tensor, dynamic_id)
            else:
                # manual implementation for older torch
                is_dyn = torch.zeros_like(raw_mask_tensor, dtype=torch.bool)
                for d_id in dynamic_id:
                    is_dyn |= (raw_mask_tensor == d_id)
            
            # 在这些位置，把原始 ID 填进去
            final_mask[is_dyn] = raw_mask_tensor[is_dyn]
            
            base_name = os.path.basename(map_png[i])
            save_path = os.path.join(save_dir, base_name)
            
            # 保存
            # 转换为 numpy uint8 (如果 ID < 255)
            mask_np = final_mask.cpu().numpy().astype(np.uint8)
            img = Image.fromarray(mask_np, 'L')
            img.save(save_path)
            
        # Record processed info
        if not "right" in args.data_dir:
            if os.path.exists(directory_record_path):
                with open(directory_record_path, 'r') as json_file:
                    existing_data = json.load(json_file)
            else:
                existing_data = {}
            
            dirname = os.path.basename(args.data_dir)
            existing_data[dirname] = dynamic_id.tolist()
            with open(directory_record_path, 'w') as json_file:
                json.dump(existing_data, json_file)            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate dynamic instance mask")
    parser.add_argument("--data_dir", type=str, default="/data0/hexiankang/code/SegAnyMo/data/dynamic_replica_data/02daee-3_obj_source_left", help="images")
    parser.add_argument("--dataset", type=str, default="dynamic_stereo", help="[pointodyssey, dynamic_stereo]")
    args = parser.parse_args()
    main(args)