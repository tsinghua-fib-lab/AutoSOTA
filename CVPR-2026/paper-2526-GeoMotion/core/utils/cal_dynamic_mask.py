import argparse
import os
from glob import glob
import numpy as np
from PIL import Image
import torch
import cv2
from visualize import read_video_from_path, Visualizer, read_imgs_from_path
from tqdm import tqdm
import pickle
from torchvision import transforms
import json

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
        
        img_dir = os.path.join(args.data_dir, "rgbs")
        imgs = read_imgs_from_path(img_dir)
        
    elif args.dataset == "dynamic_stereo":
        # load instance id mask
        instance_dir = os.path.join(args.data_dir, "instance_id_maps")
        map_png = sorted(glob(os.path.join(instance_dir, "*.png")))
        map_pkl = sorted(glob(os.path.join(instance_dir, "*.pkl")))
        transform = transforms.ToTensor()

        obj_list = []
        for png_path in map_png:
            with Image.open(png_path) as img:
                tensor_image = transform(img)
                obj_list.append(tensor_image)
                
        pkl_list = []
        for pkl_path in map_pkl:
            with open(pkl_path, 'rb') as file:
                data = pickle.load(file)
                pkl_list.append(data)
        # only left camera have trajs, so 
        directory_record_path = os.path.join(os.path.dirname(args.data_dir), "processed_dir.json")
        if "right" in args.data_dir:
            if os.path.exists(directory_record_path):
                name = os.path.basename(args.data_dir.replace("right", "left"))
                with open(directory_record_path, 'r') as file:
                    data = json.load(file)
                if name in data:
                    all_dynamic = False
                    dynamic_id = torch.tensor(data[name], dtype=torch.int16)
                else:
                    all_dynamic = True
            else:
                all_dynamic = True
        else:
            obj_masks = torch.stack(obj_list, dim=0)
            
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
                trajs_2d.append(traj_2d)
                trajs_3d.append(traj_3d)
                imgs.append(img)
                visibles.append(visible)
                instances.append(instance)
            import pdb;pdb.set_trace()
            trajs_2d = torch.stack(trajs_2d, dim=0)
            trajs_3d = torch.stack(trajs_3d, dim=0)
            imgs = torch.stack(imgs, dim=0)
            visibles = torch.stack(visibles, dim=0)
            instances = torch.stack(instances, dim=0)

            # calculate distance
            distances = torch.sqrt(torch.sum(torch.diff(trajs_3d, dim=0) ** 2, dim=2))
            threshold = 0.0001
            # Determine whether the maximum moving distance of each point between all time frames is greater than the threshold
            is_dynamic = (torch.max(distances, dim=0).values) >= threshold
            d_instance = instances[:, is_dynamic]
            # Determine whether an obj is dynamic
            total_counts = torch.bincount(instances.view(-1).to(torch.int64))
            counts = torch.bincount(d_instance.view(-1).to(torch.int64), minlength=total_counts.shape[0])
                
            half_total_counts = total_counts / 2
            id_mask = (counts > half_total_counts)[total_counts != 0]
        
            # obj id image -> dynamic mask
            obj_id = torch.unique(obj_masks)
            if id_mask.shape[0] == obj_id.shape[0]:
                dynamic_id = obj_id[id_mask]
                all_dynamic = (dynamic_id.shape[0] == (obj_id.shape[0]-1))
            else:
                all_dynamic = False
                if all(id_mask[1:]):
                    all_dynamic = True
        
        # [1, H, W]
        save_dir = os.path.join(args.data_dir, "dynamic_masks")
        if all_dynamic:
            src_dir = os.path.join(args.data_dir, "masks")
            os.rename(src_dir, save_dir)
        else:
            dirname = os.path.basename(args.data_dir)
            os.makedirs(save_dir, exist_ok=True)
            for i, obj_mask in enumerate(obj_list):
                obj_mask = obj_mask.squeeze(0)
                dynamic_mask = torch.zeros_like(obj_mask, dtype=torch.bool)

                for d_id in dynamic_id:
                    dynamic_mask |= (obj_mask == d_id)      
                    
                base_name = os.path.basename(map_png[i])
                save_path = os.path.join(save_dir, base_name)
                dynamic_mask_img = dynamic_mask.to(torch.uint8) * 255
                img = Image.fromarray(dynamic_mask_img.cpu().numpy(), 'L')
                img.save(save_path)
                
            if os.path.exists(directory_record_path):
                with open(directory_record_path, 'r') as json_file:
                    existing_data = json.load(json_file)
            else:
                existing_data = {}
            
            existing_data[dirname] = dynamic_id.tolist()
            with open(directory_record_path, 'w') as json_file:
                json.dump(existing_data, json_file)            
    # vis = Visualizer(args.data_dir, pad_value=120, linewidth=3)

    # points = (trajs_2d).unsqueeze(0)
    # visible_mask = (visibles).unsqueeze(0).unsqueeze(-1)
    # imgs = imgs.unsqueeze(0).permute(0,1,4,2,3)
    # vis.visualize(imgs, points, visible_mask, filename="original")
    
    # dynamic_points = trajs_2d[:, is_static, :]
    # visible = visible[:, is_static]

    # dynamic_points = (dynamic_points).unsqueeze(0)
    # visible = (visible).unsqueeze(0).unsqueeze(-1)
    # vis.visualize(imgs, dynamic_points, visible, filename="dynamic")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate dynamic mask")
    parser.add_argument("--data_dir", type=str, default="/data0/hexiankang/code/SegAnyMo/data/dynamic_replica_data/02daee-3_obj_source_left", help="images")
    parser.add_argument("--dataset", type=str, default="dynamic_stereo", help="[pointodyssey, dynamic_stereo]")
    args = parser.parse_args()
    main(args)

'''
some sequence have problem, obj list lacks an id
    img_names = [
        "2960d2-7_obj_source_left",
        "746af7-7_obj_source_left",
        "174239-7_obj_source_left",
        "4e331d-7_obj_source_left",
        "2960d2-7_obj_source_right",
        "746af7-7_obj_source_right",
        "174239-7_obj_source_right",
        "4e331d-7_obj_source_right",
    ]
'''
