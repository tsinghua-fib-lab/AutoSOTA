# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
import sys

sys.path.append(os.curdir)

from mmengine.config import Config
from mmseg.utils import get_classes, get_palette
from mmengine.runner.checkpoint import _load_checkpoint
from rein.utils import init_model
import rein
import time
from mmseg.structures import SegDataSample
import torch
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument("config", help="Path to the training configuration file.")
    parser.add_argument("--checkpoint", help="Path to the checkpoint file for both the REIN and head models.")
    parser.add_argument("--images", help="Directory or file path of images to be processed.")
    parser.add_argument("--suffix", default=".png", help="File suffix to filter images in the directory. Default is '.png'.")
    parser.add_argument("--not-recursive", action='store_false', help="Whether to search images recursively in subfolders. Default is recursive.")
    parser.add_argument("--search-key", default="", help="Keyword to filter images within the directory. Default is no filtering.")
    parser.add_argument(
        "--backbone",
        default="checkpoints/dinov2_vitl14_converted_1024x1024.pth",
        help="Path to the backbone model checkpoint. Default is 'checkpoints/dinov2_vitl14_converted_1024x1024.pth'."
    )
    parser.add_argument("--save_dir", default="work_dirs/show", help="Directory to save the output images. Default is 'work_dirs/show'.")
    parser.add_argument("--tta", action="store_true", help="Enable test time augmentation. Default is disabled.")
    parser.add_argument("--device", default="cuda:0", help="Device to use for computation. Default is 'cuda:0'.")
    args = parser.parse_args()
    return args

def load_backbone(checkpoint: dict, backbone_path: str) -> None:
    converted_backbone_weight = _load_checkpoint(backbone_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint["state_dict"].update(
            {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
        )
    else:
        checkpoint.update(
            {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
        )


classes = get_classes("cityscapes")
palette = get_palette("cityscapes")

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, device=args.device)
    print(model)
    input_shape = (3, 512, 512)
    result = {}
    result['ori_shape'] = input_shape[-2:]
    result['img_shape'] = input_shape[-2:]
    data_batch = {
        'inputs': [torch.rand(input_shape)],
        'data_samples': [SegDataSample(metainfo=result)]
    }
    data = model.data_preprocessor(data_batch)
    # inputs = torch.rand((1, 3, 1024, 1024)).to(args.device)
    start = time.time()
    for i in range(10):
        with torch.no_grad():
            result = model.predict(data['inputs'], data['data_samples'])
    end = time.time()
    print(f"fps {10 / (end - start)}")


if __name__ == "__main__":
    main()
