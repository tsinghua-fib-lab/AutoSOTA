import os
import re
import yaml
import math
import numpy as np
import clip
import random
import argparse
import open_clip
from collections import OrderedDict
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.modelnet_c import ModelNet_C
from datasets.modelnet_c_views import ModelNet_C_Views
from datasets.modelnet40_sdxl import ModelNet40_SDXL
from datasets.modelnet40_c import ModelNet40_C
from datasets.modelnet40 import ModelNet40
from datasets.modelnet40_views import ModelNet40_Views
from datasets.scanobjnn import ScanObjNN
from datasets.scanobjectnn import ScanObjectNN
from datasets.sonn_c import SONN_C
from datasets.snv2_c import SNV2_C
from datasets.objaverse_lvis import Objaverse_LVIS
from datasets.omniobject3d import OmniObject3D
from datasets.sim2real_sonn import Sim2Real_SONN
from datasets.pointda_modelnet import PointDA_ModelNet
from datasets.pointda_scannet import PointDA_ScanNet
from datasets.pointda_shapenet import PointDA_ShapeNet

from datasets.utils import AugMixAugmenter
import torchvision.transforms as transforms

from models import uni3d
from models import openshape
from models import ulip

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    
    # system settings
    parser.add_argument('--config', dest='config', help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--lm3d', default='uni3d', type=str, help='which large multi-modal 3d model to use')
    parser.add_argument('--seed', type=int, default=1, help='experiment seed')
    parser.add_argument("--device", default=0, type=int, help="The GPU device id to use.",)
    parser.add_argument('--distributed', action='store_true', default=False, help='whether use distributed inference')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--print-freq', type=int, default=500, help='result print frequency')
    parser.add_argument("--k_shot", type=int, default=3, help="number of shots cached in per class")
    parser.add_argument("--n_cluster", type=int, default=3, help="number of local clustered parts for a 3D object")
    parser.add_argument('--alpha', default=4.0, type=float, help="a balance factor to adjust the weights of cached logits")
    parser.add_argument('--beta', default=3.0, type=float, help="a sharpness factor to adjust query-key attention computation")
    
    # uni3d
    parser.add_argument("--pc-model", type=str, default="eva_giant_patch14_560", help="Name of pointcloud backbone to use.",)
    parser.add_argument("--pretrained-pc", default='', type=str, help="Use a pretrained CLIP model vision weights with the specified tag or file path.",)
    parser.add_argument("--clip-model", type=str, default="EVA02-E-14-plus", help="Name of the vision and text backbone to use.",)
    parser.add_argument("--pretrained", default='weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin', type=str, help="open clip version",)
    parser.add_argument('--ckpt_path', default='weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt', help='the ckpt to test 3d zero shot')
    parser.add_argument('--drop-path-rate', default=0.0, type=float, help="passed by uni3d and ulip")
    
    # openshape
    parser.add_argument("--oshape-version", type=str, choices=["vitg14", "vitl14"], default="vitg14")
    parser.add_argument('--npoints', default=1024, type=int, help='number of points used for pre-train and test.')
    parser.add_argument("--pc-feat-dim", type=int, default=768, help="Pointcloud feature dimension.")
    parser.add_argument("--group-size", type=int, default=32, help="Pointcloud Transformer group size.")
    parser.add_argument("--num-group", type=int, default=512, help="Pointcloud Transformer number of groups.")
    parser.add_argument("--pc-encoder-dim", type=int, default=512, help="Pointcloud Transformer encoder dimension.")
    parser.add_argument("--embed-dim", type=int, default=512, help="teacher embedding dimension.")
    parser.add_argument("--patch-dropout", type=float, default=0., help="flip patch dropout.")
    
    # ulip: Share ***point encoder*** config with openshape since both of them use `PointBERT`
    parser.add_argument("--ulip-version", type=str, choices=["ulip1", "ulip2"], default="ulip2")
    parser.add_argument("--pc-depth", type=int, default=12, help="number of layers of PointTransformer")
    parser.add_argument("--num-head", type=int, default=6, help="number of heads in PointTransformer attention")
    parser.add_argument("--encoder-dim", type=int, default=256, help="dimensions of the encoder before feeding  PointTransformer")
    parser.add_argument("--slip-ckpt-path", type=str, default="weights/ulip/slip_base_100ep.pt")

    # data
    parser.add_argument('--dataset', default='modelnet40', type=str, help="Datasets to process")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./data/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--objaverse_lvis_root', type=str, default='data/objaverse_lvis', help='')
    parser.add_argument('--omniobject3d_root', type=str, default='data/omniobject3d', help='')
    parser.add_argument('--scanobjnn_root', type=str, default='data/scanobjnn', help='')
    parser.add_argument('--scanobjectnn_root', type=str, default='data/scanobjectnn', help='')
    parser.add_argument('--sonn_c_root', type=str, default='data/sonn_c', help='')
    parser.add_argument('--sonn_variant', type=str, default='hardest', help='')
    parser.add_argument('--modelnet40_root', type=str, default='data/modelnet40', help='')
    parser.add_argument('--modelnet_c_root', type=str, default='data/modelnet_c', help='')
    parser.add_argument('--modelnet40_c_root', type=str, default='data/modelnet40_c', help='')
    parser.add_argument('--snv2_c_root', type=str, default='data/snv2_c', help='')
    parser.add_argument('--cor_type', type=str, default='add_global_2', help='data corruption type')
    parser.add_argument('--sim2real_type', type=str, default='so_obj_only_9', choices=['so_obj_only_9', 'so_obj_only_11', 
                        'so_obj_bg_9', 'so_obj_bg_11', 'so_hardest_9', 'so_hardest_11'])
    parser.add_argument('--pointda_type', type=str, default='so_obj_only_9', choices=['modelnet', 'scannet', 'shapenet'])
    parser.add_argument('--cname', type=str, default='airplane', help='specify class name for visualization')

    parser.add_argument("--p_thres", type=float, default=0.1, help="take how many confident images from all images")
    parser.add_argument("--obj-id", type=int, default=0, help="object id when visualizing all patches and the clustering"
                        "centers of a 3D object")

    args = parser.parse_args()

    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_entropy(loss, clip_weights):
    # clip_weights: (emb_dim, n_cls)
    # so `max_entropy` is a scalar
    max_entropy = math.log2(clip_weights.size(1))
    # NOTE float(x) requires x is a scalar, so `loss` is a scalar
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    '''
        Do not understand why the following operations can compute `average entropy`?
    '''
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    # output.topk(...)[0] -> topk values
    # output.topk(...)[1] -> topk indices
    pred = output.topk(topk, dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


@torch.no_grad()
def build_img_cache(args, dataset, clip_model):
    cache_keys = []
    cache_values = []
    test_feats = []
    
    test_loader, _, _ = build_test_data_loader(args, dataset, None, None)
    for images, target in test_loader:
        # (3, 3, 224, 224)
        images = images.reshape(-1, 3, args.imsize, args.imsize).half().cuda()
        # (3, 512)
        img_features = clip_model.encode_image(images)
        test_feats.append(img_features)
        cache_values.append(target.cuda().repeat(images.shape[0]))
    # (n_cls*3, 512)
    cache_keys = torch.cat(test_feats, dim=0)
    # (n_cls*3, )
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        
    return {'keys': cache_keys, 'values': cache_values}
    
    
@torch.no_grad()
def clip_classifier_img_weights(args, dataset, clip_model):
    clip_weights = []

    test_loader, _, _ = build_test_data_loader(args, dataset, None, None)
    
    for images, _ in test_loader:
        images = images.reshape(-1, 3, args.imsize, args.imsize).cuda()
        img_cls_embeds = clip_model.encode_image(images)
        img_cls_embed = img_cls_embeds.mean(dim=0)
        img_cls_embed /= img_cls_embed.norm()
        clip_weights.append(img_cls_embed)

    # NOTE torch.stack along the 
    #       i. `dim=1` will make `clip_weights` have shape (emb_dim, n_cls)
    #      ii. `dim=0` will make `clip_weights` have shape (n_cls, emb_dim)
    # clip_weights: (emb_dim, n_cls)
    clip_weights_img_weights = torch.stack(clip_weights, dim=1).cuda()
    
    return clip_weights_img_weights


@torch.no_grad()
def clip_classifier(args, classnames, template, clip_model):
    clip_weights = []

    for classname in classnames:
        # option 1: use the manual template from `templates.py`
        classname = classname.replace('_', ' ')
        texts = [t.format(classname) for t in template]
        # option 2: use the responses from the LLM
        # texts = template[classname]
        
        if args.lm3d == 'uni3d' or args.lm3d == 'ulip':
            texts = clip.tokenize(texts).cuda()
        elif args.lm3d == 'openshape':
            texts = open_clip.tokenizer.tokenize(texts).cuda()
        # prompt ensemble for ImageNet
        # class_embeddings: (n_temp, emb_dim)
        class_embeddings = clip_model.encode_text(texts)
        # class_embeddings: (n_temp, emb_dim)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        # class_embeddings: (emb_dim,)
        class_embedding = class_embeddings.mean(dim=0)
        # class_embeddings: (emb_dim,)
        class_embedding /= class_embedding.norm()
        clip_weights.append(class_embedding)

    # NOTE torch.stack along the 
    #       i. `dim=1` will make `clip_weights` have shape (emb_dim, n_cls)
    #      ii. `dim=0` will make `clip_weights` have shape (n_cls, emb_dim)
    # clip_weights: (emb_dim, n_cls)
    clip_weights = torch.stack(clip_weights, dim=1).cuda()
    
    return clip_weights


mn40_vit_b16_view_weight = torch.tensor([0.75, 0.75, 0.75, 0.25, 0.75, 1.0, 0.25, 1.0, 0.75, 0.25]).cuda()

@torch.no_grad()
def get_clip_logits(images, clip_model, clip_weights, p_thres):
    ''' NOTE PointCLIP V1 & V2 use this function'''
    
    if isinstance(images, list):
        images = torch.cat(images, dim=0).cuda()
    else:
        images = images.cuda()

    ''' NOTE --- option 1 '''
    image_features = clip_model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    ''' NOTE --- option 2 '''
    # image_features = clip_model.encode_image(images)
    # image_features *= mn40_vit_b16_view_weight.reshape(-1, 1)
    # image_features /= image_features.norm(dim=-1, keepdim=True)

    # clip_logits: (batch*n_views, n_cls) for a multi-view point cloud
    clip_logits = 100. * image_features @ clip_weights

    '''NOTE for multi views of a point cloud, choose this branch '''
    if image_features.size(0) > 1:
        ''' NOTE --- option 1: using predictions with low entropy'''
        # batch_entropy: (batch*n_views,)
        batch_entropy = softmax_entropy(clip_logits)            
        # selected_idx: (n_selected_idx,)
        selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * p_thres)]
        # output: (n_selected_idx, n_cls)
        output = clip_logits[selected_idx]
        
        # image_features: (1, emb_dim)
        image_features = image_features[selected_idx].mean(0).unsqueeze(0)
        # clip_logits: (1, n_cls)
        clip_logits = output.mean(0).unsqueeze(0)

        # `loss` is a scalar
        loss = avg_entropy(output)
        # prob_map: (1, n_cls)
        prob_map = output.softmax(1).mean(0).unsqueeze(0)
        # `torch.topk` returns k largest elements of the tensor along a given dimension
        # pred: is a scalar, indicating the index of largest class probability
        pred = int(output.mean(0).unsqueeze(0).topk(1, dim=1, largest=True, sorted=True)[1].t())

        ''' NOTE --- option 2: using all predictions and give them different weights'''
        # image_features = image_features.mean(0).unsqueeze(0)
        # clip_logits = clip_logits.mean(0).unsqueeze(0)

        # loss = avg_entropy(clip_logits)
        # prob_map = clip_logits.softmax(1).mean(0).unsqueeze(0)
        # pred = int(clip_logits.mean(0).unsqueeze(0).topk(1, dim=1, largest=True, sorted=True)[1].t())

    else:
        loss = softmax_entropy(clip_logits)
        prob_map = clip_logits.softmax(1)
        pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

    return image_features, clip_logits, loss, prob_map, pred
    

@torch.no_grad()
def get_uni3d_logits(cache_type, feat, lm3d_model, clip_weights, p_thres=0.1):
    if isinstance(feat, list):
        feat = torch.cat(feat, dim=0).cuda()
    else:
        feat = feat.cuda()

    # NOTE feat: (batch, N, 3+3) -> 3 coordinates + 3 color channels
    if cache_type == 'global':
        pc_feats = lm3d_model.encode_pc(feat)
        pc_feats /= pc_feats.norm(dim=-1, keepdim=True)
        # 100 times by jerry
        clip_logits = 100. * pc_feats @ clip_weights
    elif cache_type == 'local':
        patch_centers = lm3d_model.encode_pc(feat)
        patch_centers /= patch_centers.norm(dim=-1, keepdim=True)
        clip_logits = 100. * patch_centers.mean(0, keepdim=True) @ clip_weights
    else: # NOTE hierarchical caches
        pc_feats, patch_centers = lm3d_model.encode_pc(feat)
        pc_feats /= pc_feats.norm(dim=-1, keepdim=True)
        patch_centers /= patch_centers.norm(dim=-1, keepdim=True)
        clip_logits = 100. * pc_feats @ clip_weights

    loss = softmax_entropy(clip_logits)
    prob_map = clip_logits.softmax(1)   # normalize logits to [0, 1]
    # pred: is a scalar, indicating the index of largest class probability
    pred = int(clip_logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0])

    if cache_type == 'global':
        return pc_feats, clip_logits, loss, prob_map, pred
    elif cache_type == 'local':
        return patch_centers, clip_logits, loss, prob_map, pred
    else: # NOTE hierarchical caches
        return pc_feats, patch_centers, clip_logits, loss, prob_map, pred


@torch.no_grad()
def get_openshape_logits(cache_type, feat, lm3d_model, clip_weights, p_thres=0.1):
    if isinstance(feat, list):
        feat = torch.cat(feat, dim=0).cuda()
    else:
        feat = feat.cuda()
    
    # feat: (batch, npoints, 6)
    xyz = feat[:, :, :3]
    if cache_type == 'global':
        pc_feats = lm3d_model(xyz, feat)
        pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True)
        # 100 times by jerry
        clip_logits = 100. * pc_feats @ clip_weights
    elif cache_type == 'local':
        patch_centers = lm3d_model(xyz, feat)
        patch_centers = patch_centers / patch_centers.norm(dim=-1, keepdim=True)
        clip_logits = 100. * patch_centers.mean(0, keepdim=True) @ clip_weights
    else: # NOTE hierarchical caches
        pc_feats, patch_centers = lm3d_model(xyz, feat)
        pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True)
        patch_centers = patch_centers / patch_centers.norm(dim=-1, keepdim=True)
        clip_logits = 100. * pc_feats @ clip_weights
    
    """ NOTE F.normalize(pc_feats, dim=1) is equivalent to `pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True)`
        according to https://pytorch.org/docs/stable/generated/torch.norm.html
    """
    
    loss = softmax_entropy(clip_logits)
    prob_map = clip_logits.softmax(1)   # normalize logits to [0, 1]
    # pred: is a scalar, indicating the index of largest class probability
    pred = int(clip_logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0])

    if cache_type == 'global':
        return pc_feats, clip_logits, loss, prob_map, pred
    elif cache_type == 'local':
        return patch_centers, clip_logits, loss, prob_map, pred
    else: # NOTE hierarchical caches
        return pc_feats, patch_centers, clip_logits, loss, prob_map, pred


@torch.no_grad()
def get_ulip_logits(feat, lm3d_model, clip_weights, p_thres=0.1):
    if isinstance(feat, list):
        feat = torch.cat(feat, dim=0).cuda()
    else:
        feat = feat.cuda()
    
    # feat: (batch, npoints, 6)
    xyz = feat[:, :, :3]
    pc_feats = lm3d_model(xyz)
    pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True)
    # 100 times by jerry
    clip_logits = 100. * pc_feats @ clip_weights
    
    loss = softmax_entropy(clip_logits)
    prob_map = clip_logits.softmax(1)   # normalize logits to [0, 1]
    # pred: is a scalar, indicating the index of largest class probability
    pred = int(clip_logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0])

    return pc_feats, clip_logits, loss, prob_map, pred

def get_logits(args, feat, lm3d_model, clip_weights):
    if args.lm3d == 'ulip':
        return get_ulip_logits(feat, lm3d_model, clip_weights, args.p_thres)
    else:
        raise NotImplementedError(f'[get_logits] This lm3d {args.lm3d} is not supported!')


def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(args, config_path, dataset_name):
    config_name = f"{dataset_name}.yaml"
    config_file = os.path.join(config_path, config_name)
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)
        # NOTE cfg is a `dict`, override default values in cfg by command-line arguments
        cfg['positive']['shot_capacity'] = args.k_shot
        print("\ncfg['positive']['shot_capacity']:", cfg['positive']['shot_capacity'], "\n")
        print("n_cluster in KMeans:", args.n_cluster, "\n")
        cfg['positive']['alpha'] = args.alpha
        print("`alpha` for cache weights:", args.alpha, "\n")
        cfg['positive']['beta'] = args.beta
        print("`beta` for cache attention sharpness:", args.beta, "\n")

    return cfg


def build_test_data_loader(args, dataset_name, root_path, preprocess_val):
    '''
    NOTE Here all `test_loader` have `batch_size=1`
        so how to handle when `batch_size=1` -> Do NOT need to consider this for training-free method
    '''

    # NOTE how to implement the data augmentations
    if dataset_name == 'modelnet40':
        dataset = ModelNet40(args)      # NOTE this dataset from OpenShape github repository
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)

    elif dataset_name == 'modelnet40_views':
        dataset = ModelNet40_Views(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
    
    elif dataset_name == 'scanobjnn':
        dataset = ScanObjNN(args)       # NOTE this dataset from OpenShape github repository
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        
    elif dataset_name == 'scanobjectnn':
        dataset = ScanObjectNN(args)    # NOTE this dataset from the original paper - `sonn_h5_files`
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        
    elif dataset_name == 'sonn_c':
        dataset = SONN_C(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        
    elif dataset_name == 'snv2_c':
        dataset = SNV2_C(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)

    elif dataset_name == 'objaverse_lvis':
        dataset = Objaverse_LVIS(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        
    elif dataset_name == 'omniobject3d':
        dataset = OmniObject3D(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)

    elif dataset_name == 'modelnet_c':
        dataset = ModelNet_C(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)

    elif dataset_name == 'modelnet_c_views':
        dataset = ModelNet_C_Views(args)
        # dataset = ModelNet_C_Views(args, preprocess_val)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        
    elif dataset_name == 'modelnet40_sdxl': # generate images for modelnet_c classes using sdxl
        dataset = ModelNet40_SDXL(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
    
    elif dataset_name == 'modelnet40_c':
        dataset = ModelNet40_C(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        
    elif dataset_name == 'sim2real_sonn':
        dataset = Sim2Real_SONN(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)

    elif dataset_name == 'pointda_modelnet':
        dataset = PointDA_ModelNet(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        
    elif dataset_name == 'pointda_scannet':
        dataset = PointDA_ScanNet(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        
    elif dataset_name == 'pointda_shapenet':
        dataset = PointDA_ShapeNet(args)
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template


def count_params(clip_model, lm3d_model):
    clip_params = sum([param.numel() for param in clip_model.parameters() if param.requires_grad])
    print('\n', '-'*20)
    print('\tclip_params:', clip_params)
    
    lm3d_params = sum([param.numel() for param in lm3d_model.parameters() if param.requires_grad])
    print('\n', '-'*20)
    print('\tlm3d_params:', lm3d_params)
    
    print('\n', '-'*20)
    print('\ttotal_params:', clip_params+lm3d_params, '\n')
    
    
def load_uni3d(args):
    # 0. create CLIP model and load its weights
    open_clip_model, _, _ = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained, device='cpu') 

    # remove the image encoder weights 
    clip_state_dict = open_clip_model.state_dict()
    keys_to_delete = [key for key in clip_state_dict if key.startswith('visual')]
    for key in keys_to_delete:
        del clip_state_dict[key]
    # only move text encoder to a gpu device
    open_clip_model.text.half().to(args.device)
    open_clip_model.eval()

    # 1. create 3D model
    lm3d_model = uni3d.create_uni3d(args)

    # 2. load model pre-trained weights
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    sd = checkpoint['module']
    # NOTE `args.distributed`: need to define in advance
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    lm3d_model.load_state_dict(sd)
    
    lm3d_model.half().to(args.device)
    lm3d_model.eval()
    
    count_params(open_clip_model.text, lm3d_model)
    
    return open_clip_model, lm3d_model
    
    
def load_config(*yaml_files, cli_args=[], extra_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    yaml_confs += [OmegaConf.from_cli(extra_args)]
    conf = OmegaConf.merge(*yaml_confs, cli_args)
    OmegaConf.resolve(conf)
    return conf


def load_openshape(args):
    if args.oshape_version == 'vitg14':
        clip_name = 'ViT-bigG-14'
        open_clip_model, _, _ = open_clip.create_model_and_transforms(clip_name, 
                                pretrained='weights/openshape/open_clip_pytorch_model/vit-bigG-14/laion2b_s39b_b160k.bin')
    elif args.oshape_version == 'vitl14':
        clip_name = 'ViT-L-14'
        open_clip_model, _, _ = open_clip.create_model_and_transforms(clip_name, 
                                pretrained='weights/openshape/open_clip_pytorch_model/vit-l-14/laion2b_s32b_b82k.bin')
    
    open_clip_model.half().to(args.device)
    open_clip_model.eval()
    
    # read config file
    config = load_config("models/openshape/config.yaml", cli_args = vars(args))
    lm3d_model = openshape.create_openshape(config)
    lm3d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lm3d_model)

    checkpoint = torch.load(f"weights/openshape/openshape-pointbert-{args.oshape_version}-rgb/model.pt")
    model_dict = OrderedDict()
    
    if args.oshape_version == "vitg14":
        pattern = re.compile('module.')
        for k,v in checkpoint['state_dict'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
        lm3d_model.load_state_dict(model_dict)  # NOTE previously lose this operation, which is a severe bug
    elif args.oshape_version == "vitl14":
        pattern = re.compile('pc_encoder.')
        for k,v in checkpoint.items():
            if re.search("pc_encoder", k):
                model_dict[re.sub(pattern, '', k)] = v
        lm3d_model.load_state_dict(model_dict)
    else:
        raise NotImplementedError(f'OpenShape with {args.oshape_version} is not supported!')
    
    lm3d_model.half().to(args.device)
    lm3d_model.eval()
    
    count_params(open_clip_model, lm3d_model)
    
    return open_clip_model, lm3d_model


def load_ulip(args):
    '''
        NOTE load model weights here
    '''
    # --- 1. load clip text encoder
    clip_model = ulip.create_clip_text_encoder(args)
    print('len(clip_model.state_dict()):', len(dict(clip_model.state_dict())))
    # *** NOTE for debug purpose
    # print('='*10, 'Before loading the weights', '='*10)
    # clip_model_sd = clip_model.state_dict()
    # print(clip_model_sd['ln_final.bias'].shape, '\n')
    # print(clip_model_sd['ln_final.bias'], '\n')
    
    pretrain_slip = torch.load(args.slip_ckpt_path, map_location=torch.device('cpu'))
    pretrain_slip_sd = pretrain_slip['state_dict']
    pretrain_slip_sd = {k.replace('module.', ''): v for k, v in pretrain_slip_sd.items()}
    pretrain_slip_sd = {k:v for k,v in pretrain_slip_sd.items()
                        if k.startswith('positional_embedding') or 
                           k.startswith('text_projection') or 
                           k.startswith('logit_scale') or 
                           k.startswith('transformer') or 
                           k.startswith('token_embedding') or 
                           k.startswith('ln_final')}
    print('len(pretrain_slip_sd):', len(pretrain_slip_sd), '\n')

    clip_model_dict = OrderedDict(pretrain_slip_sd)
    clip_model.load_state_dict(clip_model_dict)
    # *** NOTE for debug purpose
    # print('='*10, 'After loading the weights', '='*10)
    # print(clip_model_sd['ln_final.bias'].shape, '\n')
    # print(clip_model_sd['ln_final.bias'], '\n')
        
    clip_model.half().to(args.device)
    clip_model.eval()
    
    # --- 2. load point encoder
    lm3d_model = ulip.create_ulip(args)
    print('len(lm3d_model.state_dict()):', len(lm3d_model.state_dict()))
    # *** NOTE for debug purpose
    # lm3d_sd = lm3d_model.state_dict()
    # print('='*10, 'Before loading the weights', '='*10)
    # print(lm3d_sd['point_encoder.encoder.first_conv.1.running_mean'].shape, '\n')
    # print(lm3d_sd['point_encoder.encoder.first_conv.1.running_mean'], '\n')
    
    ulip_ckpt_path = f'weights/ulip/pointbert_{args.ulip_version}.pt'
    pretrain_point = torch.load(ulip_ckpt_path, map_location=torch.device('cpu'))
    pretrain_point_sd = pretrain_point['state_dict']
    pretrain_point_sd = {k.replace('module.', ''): v for k, v in pretrain_point_sd.items()}
    pretrain_point_sd = {k:v for k, v in pretrain_point_sd.items() 
                         if k.startswith('pc_projection') or k.startswith('point_encoder')}
    print('len(pretrain_point_sd):', len(pretrain_point_sd), '\n')
    
    lm3d_model_dict = OrderedDict(pretrain_point_sd)
    lm3d_model.load_state_dict(lm3d_model_dict)
    # *** NOTE for debug purpose
    # print('='*10, 'After loading the weights', '='*10)
    # print(lm3d_sd['point_encoder.encoder.first_conv.1.running_mean'].shape, '\n')
    # print(lm3d_sd['point_encoder.encoder.first_conv.1.running_mean'], '\n')

    lm3d_model.half().to(args.device)
    lm3d_model.eval()
    
    count_params(clip_model, lm3d_model)
    
    return clip_model, lm3d_model


def load_models(args):
    if args.lm3d == 'uni3d':
        return load_uni3d(args)
    elif args.lm3d == 'openshape':
        return load_openshape(args)
    elif args.lm3d == 'ulip':
        return load_ulip(args)
    else:
        raise NotImplementedError(f'[LM3D in `load_models`]: {args.lm3d} is not supported!')
    