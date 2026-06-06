import torch
import argparse
import os
import numpy as np
from visualize import read_video_from_path, Visualizer, read_imgs_from_path

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train trajectory-based motion segmentation network')
    parser.add_argument('--video_path', type=str,help='path to config file')
    parser.add_argument('--save_dir', type=str,default="current-data-dir/kubric/movie_e/validation/cotracker/000000",help='path to config file')
    parser.add_argument('--imgs_dir', type=str,default="current-data-dir/kubric/movie_e/validation/images/000000")
    args = parser.parse_args()

    device = 'cuda'

    if args.video_path is not None:
        video = read_video_from_path(args.video_path)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(device) # B T C H W  

    if args.imgs_dir is not None:
        # (B,T,C,H,W)
        video = read_imgs_from_path(args.imgs_dir).float().to(device)
    
    grid_size = 16
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(device)

    # Run Online CoTracker, the same model with a different API:
    # Initialize online processing
    cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  

    # Process the video
    for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
        pred_tracks, pred_visibility = cotracker(
            video_chunk=video[:, ind : ind + cotracker.step * 2]
        )  # B T N 2,  B T N 1
        
    # save video
    track_dir = os.path.join(args.save_dir)
    os.makedirs(track_dir,exist_ok=True)
    # vis = Visualizer(save_dir=track_dir, pad_value=100)
    # vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='cotracker')

    pred_tracks = pred_tracks.squeeze(0).cpu().detach().numpy()
    pred_visibility = pred_visibility.squeeze(0).cpu().detach().numpy()
    
    np.save(os.path.join(track_dir,'pred_tracks.npy'), pred_tracks) 
    np.save(os.path.join(track_dir, 'pred_visibility.npy'), pred_visibility) 
    
# CUDA_VISIBLE_DEVICES=3 python core/utils/cotracker.py