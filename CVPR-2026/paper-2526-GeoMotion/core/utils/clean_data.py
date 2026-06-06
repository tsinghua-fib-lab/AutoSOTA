import argparse
import os
from glob import glob
import numpy as np
import h5py
import shutil
from tqdm import tqdm

def delete_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

def main(args):
    if args.stereo:
        seq_dir = os.path.dirname(args.data_dir)
        # delete flow related data
        flow_forward_dir = os.path.join(seq_dir, "flow_forward")
        flow_backward_dir = os.path.join(seq_dir, "flow_backward")
        flow_forward_mask_dir = os.path.join(seq_dir, "flow_forward_mask")
        flow_backward_mask_dir = os.path.join(seq_dir, "flow_backward_mask")
        
        delete_dir(flow_forward_dir)
        delete_dir(flow_backward_dir)
        delete_dir(flow_forward_mask_dir)
        delete_dir(flow_backward_mask_dir)
        
        # delete cotraker
        bootstapir_dir = os.path.join(seq_dir, "cotracker")
        delete_dir(bootstapir_dir)
        
        # transfer dino feat from .npy to h5py
        dino_dir = os.path.join(seq_dir, "dinos")
        dino_paths = sorted(glob(os.path.join(dino_dir, "*.npy")))
        if len(dino_paths) > 0:
            for path in tqdm(dino_paths, "clean dino ..."):
                # [103, 183, 768], [num_patches_h, num_patches_w, C]
                features = np.load(path).squeeze()
                features = features.astype(np.float16)
                
                save_path = os.path.splitext(path)[0] + ".h5"
                with h5py.File(save_path, 'w') as hf:
                    hf.create_dataset('dinos', data=features, compression='gzip')
                os.remove(path)
    elif args.waymo:
        print("processing waymo sequence...")
        seq_dir = os.path.dirname(args.data_dir)
        # transfer dino feat from .npy to h5py
        dino_dir = os.path.join(seq_dir, "dinos")
        delete_dir(dino_dir)
        # dino_paths = sorted(glob(os.path.join(dino_dir, "*.npy")))
        # if len(dino_paths) > 0:
        #     for path in tqdm(dino_paths, "clean dino ..."):
        #         # [103, 183, 768], [num_patches_h, num_patches_w, C]
        #         features = np.load(path).squeeze()
        #         features = features.astype(np.float16)
                
        #         save_path = os.path.splitext(path)[0] + ".h5"
        #         with h5py.File(save_path, 'w') as hf:
        #             hf.create_dataset('dinos', data=features, compression='gzip')
        #         os.remove(path)

    else:
        print("processing kubric sequence...")
        dino_dir = args.data_dir.replace("images", "dinos")
        dino_paths = sorted(glob(os.path.join(dino_dir, "*.npy")))
        if len(dino_paths) > 0:
            for path in dino_paths:
                # [103, 183, 768], [num_patches_h, num_patches_w, C]
                features = np.load(path).squeeze()
                features = features.astype(np.float16)
                
                save_path = os.path.splitext(path)[0] + ".h5"
                with h5py.File(save_path, 'w') as hf:
                    hf.create_dataset('dinos', data=features, compression='gzip')
                os.remove(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate dynamic mask")
    # parser.add_argument("--data_dir", type=str, default="current-data-dir/dynamic_stereo/dynamic_replica_data/valid/0cde48-3_obj_source_left/images", help="images")
    # parser.add_argument("--data_dir", type=str, default="current-data-dir/kubric/movie_f/validation/images/000000", help="images")
    parser.add_argument("--data_dir", type=str, default="current-data-dir/waymo/drivestudio/data/waymo/processed/trainseg/001_0/images", help="images")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    args = parser.parse_args()
    # args.waymo = True
    
    main(args)