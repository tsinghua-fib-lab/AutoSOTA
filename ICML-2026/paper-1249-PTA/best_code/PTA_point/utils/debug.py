import argparse

import open_clip
import torch

from utils import *

from models import uni3d
from datasets.modelnet_c_sdxl import ModelNet_C_SDXL


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./data/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], help='CLIP model backbone to use: RN50 or ViT-B/16.')

    parser.add_argument("--pc-model", type=str, default="eva_giant_patch14_560", help="Name of pointcloud backbone to use.",)
    parser.add_argument("--pretrained-pc", default='', type=str, help="Use a pretrained CLIP model vision weights with the specified tag or file path.",)
    
    # Model
    parser.add_argument('--model', default='create_uni3d', type=str)
    parser.add_argument("--clip-model", type=str, default="EVA02-E-14-plus", help="Name of the vision and text backbone to use.",)
    parser.add_argument("--pretrained", default='weights/open_clip_pytorch_model_laion2b_s9b_b144k.bin', type=str, help="Use a pretrained CLIP model weights with the specified tag or file path.",)
    parser.add_argument("--device", default=0, type=int, help="The GPU device id to use.",)
    parser.add_argument('--ckpt_path', default='', help='the ckpt to test 3d zero shot')

    # point encoder
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    parser.add_argument("--pc-feat-dim", type=int, default=768, help="Pointcloud feature dimension.")
    parser.add_argument("--group-size", type=int, default=32, help="Pointcloud Transformer group size.")
    parser.add_argument("--num-group", type=int, default=512, help="Pointcloud Transformer number of groups.")
    parser.add_argument("--pc-encoder-dim", type=int, default=512, help="Pointcloud Transformer encoder dimension.")
    parser.add_argument("--embed-dim", type=int, default=512, help="teacher embedding dimension.")
    parser.add_argument("--patch-dropout", type=float, default=0., help="flip patch dropout.")
    parser.add_argument('--drop-path-rate', default=0.0, type=float)
    
    parser.add_argument('--modelnet_c_sdxl_root', type=str, default='data/diffusion/modelnet_c_sdxl', help='')

    parser.add_argument('--distributed', action='store_true', default=False)

    args = parser.parse_args()

    return args

def run(args):
    # 0. create CLIP model and load its weights
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained) 
    clip_model.to(args.device)
    print(clip_model.visual.state_dict().keys())

    # 1. create 3D model
    lm3d_model = uni3d.create_uni3d(args)
    lm3d_model.to(args.device)
    lm3d_model.eval()

    # 2. load 3D model pre-trained weights
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    sd = checkpoint['module']

    print(sd.keys())

    # NOTE `args.distributed` 这个参数到底是哪来的？
    #   1. 目前没看到它从哪定义的，只是看到了对它赋值
    #   2. 终究得自己定义，否则报错
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    lm3d_model.load_state_dict(sd)
    
    # 3. check ModelNet_C_SDXL definition and the order of classes
    test_loader, _, _ = build_test_data_loader(args, 'modelnet_c_sdxl', None, None)
    print('len(test_loader):', len(test_loader))
    
    for i, (images, target) in enumerate(test_loader):
        print(f'{i}'.zfill(2), f'images.shape: {images.shape}', f'target: {target}')


if '__main__' == __name__:
    args = get_arguments()

    run(args)