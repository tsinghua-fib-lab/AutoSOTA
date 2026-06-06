import time
import numpy as np
import torch
import argparse
import os
import cv2
import matplotlib.pylab as plt
from datasets import create_dataset
from models import build_MixerCSeg as build_model
from main import get_args_parser



parser = argparse.ArgumentParser('newcseg', parents=[get_args_parser()])
args = parser.parse_args()
args.phase = 'test'
args.dataset_path = '/home/linux/code/sod/dataset/Crack500'

if __name__ == '__main__':
    args.batch_size = 1
    t_all = []
    device = torch.device(args.device)
    test_dl = create_dataset(args)
    load_model_file = "/home/linux/code/sod/MixerCSeg/checkpoints/weights/nbins_36_Crack500/checkpoint_best.pth"
    data_size = len(test_dl)
    model, criterion = build_model(args)
    state_dict = torch.load(load_model_file)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("Load Model Successful!")
    suffix = load_model_file.split('/')[-2]
    save_root = "./results/results_test/" + suffix
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    with torch.no_grad():
        model.eval()
        for batch_idx, (data) in enumerate(test_dl):

            x = data["image"]
            target = data["label"]
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
   
            out = model(x)
        
            target = target[0, 0, ...].cpu().numpy()
            out = out[0, 0, ...].cpu().numpy()
            root_name = data["A_paths"][0].split("/")[-1][0:-4]
            target = 255 * (target / np.max(target))
            out = 255 * (out / np.max(out))

            # out[out >= 0.5] = 255
            # out[out < 0.5] = 0

            print('----------------------------------------------------------------------------------------------')
            print(os.path.join(save_root, "{}_lab.png".format(root_name)))
            print(os.path.join(save_root, "{}_pre.png".format(root_name)))
            print('----------------------------------------------------------------------------------------------')
            cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
            cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)

    print("Finished!")
