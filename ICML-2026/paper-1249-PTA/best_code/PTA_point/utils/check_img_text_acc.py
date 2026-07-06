import argparse

import open_clip
import torch

from utils import *


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, help="Dataset to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./data/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], help='CLIP model backbone to use: RN50 or ViT-B/16.')
    
    # point encoder & clip & weights
    parser.add_argument('--model', default='create_uni3d', type=str)
    parser.add_argument("--pc-model", type=str, default="eva_giant_patch14_560", help="Name of pointcloud backbone to use.",)
    parser.add_argument("--pretrained-pc", default='', type=str, help="Use a pretrained CLIP model vision weights with the specified tag or file path.",)
    parser.add_argument("--clip-model", type=str, default="EVA02-E-14-plus", help="Name of the vision and text backbone to use.",)
    parser.add_argument("--pretrained", default='weights/open_clip_pytorch_model_laion2b_s9b_b144k.bin', type=str, help="Use a pretrained CLIP model weights with the specified tag or file path.",)
    parser.add_argument("--device", default=0, type=int, help="The GPU device id to use.",)
    parser.add_argument('--ckpt_path', default='weights/uni3d_g_ensembled_model.pt', help='the ckpt to test 3d zero shot')

    # point encoder config
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    parser.add_argument("--pc-feat-dim", type=int, default=768, help="Pointcloud feature dimension.")
    parser.add_argument("--group-size", type=int, default=32, help="Pointcloud Transformer group size.")
    parser.add_argument("--num-group", type=int, default=512, help="Pointcloud Transformer number of groups.")
    parser.add_argument("--pc-encoder-dim", type=int, default=512, help="Pointcloud Transformer encoder dimension.")
    parser.add_argument("--embed-dim", type=int, default=512, help="teacher embedding dimension.")
    parser.add_argument("--patch-dropout", type=float, default=0., help="flip patch dropout.")
    parser.add_argument('--drop-path-rate', default=0.0, type=float)

    parser.add_argument('--distributed', action='store_true', default=False, help='whether use distributed inference')

    # data
    parser.add_argument('--openshape_setting', action='store_true', default=False, 
                        help='whether to use osaug, by default enabled with openshape.')
    parser.add_argument('--objaverse_lvis_root', type=str, default='data/objaverse_lvis', help='')
    parser.add_argument('--scanobjnn_root', type=str, default='data/scanobjectnn', help='')
    parser.add_argument('--sonn_c_root', type=str, default='data/sonn_c', help='')
    parser.add_argument('--sonn_variant', type=str, default='hardest', help='')
    parser.add_argument('--modelnet40_root', type=str, default='data/modelnet40', help='')
    parser.add_argument('--modelnet_c_root', type=str, default='data/modelnet_c', help='')
    parser.add_argument('--modelnet40_c_root', type=str, default='data/modelnet40_c', help='')
    parser.add_argument('--modelnet40_sdxl_root', type=str, default='data/diffusion/modelnet40_sdxl', help='')
    parser.add_argument('--cor_type', type=str, default='add_global_2', help='data corruption type')

    parser.add_argument("--p_thres", type=float, default=0.1, help="take how many confident images from all images")
    parser.add_argument('--imsize', type=int, default=224, help='image resolution')

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()

    # 0. create CLIP model and load its weights
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained, device='cpu') 
    clip_model.half().to(args.device)
    clip_model.eval()
    
    dataset_name = args.dataset
    preprocess = None
    
    test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, preprocess)
    # `clip_txt_weights`: (emb_dim, n_cls)
    clip_txt_weights = clip_classifier(classnames, template, clip_model)
    emb_dim, n_cls = clip_txt_weights.size()

    if 'modelnet' in dataset_name:
        diff_prefix = 'modelnet40'
    img_cache = build_img_cache(args, f'{diff_prefix}_sdxl', clip_model)

    # (n_cls*3, emb_dim)     (n_cls*3, n_cls)
    keys, values = img_cache['keys'], img_cache['values']
    print('keys.shape:', keys.shape)
    print('values.shape:', values.shape)
    
    # keys = keys.reshape(n_cls, -1, emb_dim).mean(dim=1)
    # values = values.reshape(n_cls, -1, n_cls).mean(dim=1)
    
    cnt = 0
    total = len(values)
    for key, val in zip(keys, values):
        pred = torch.argmax(key @ clip_txt_weights)
        target = val.argmax(dim=-1)        
        if pred == target:
            cnt += 1
        print(f"pred: {pred}\t target: {target}")
    print('img-text match acc (%):', cnt/total * 100)

if __name__ == "__main__":
    main()