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
from mmseg.apis import inference_model
import rein
import tqdm
import mmengine
import torch
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument("config", help="Path to the training configuration file.")
    parser.add_argument("checkpoint", help="Path to the checkpoint file for both the REIN and head models.")
    parser.add_argument("images", help="Directory or file path of images to be processed.")
    parser.add_argument("--suffix", default=".jpg", help="File suffix to filter images in the directory. Default is '.png'.")
    parser.add_argument("--not-recursive", action='store_false', help="Whether to search images recursively in subfolders. Default is recursive.")
    parser.add_argument("--search-key", default="", help="Keyword to filter images within the directory. Default is no filtering.")

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


def draw_sem_seg(sem_seg: torch.Tensor):
    num_classes = len(classes)
    sem_seg = sem_seg.data.squeeze(0)
    H, W = sem_seg.shape
    ids = torch.unique(sem_seg).cpu().numpy()
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)
    colors = [palette[label] for label in labels]
    colors = [torch.tensor(color, dtype=torch.uint8).view(1, 1, 3) for color in colors]
    result = torch.zeros([H, W, 3], dtype=torch.uint8)
    for label, color in zip(labels, colors):
        result[sem_seg == label, :] = color
    return result.cpu().numpy()


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if "test_pipeline" not in cfg:
        cfg.test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1920,
                    1080,
                ),
                type="Resize",
            ),
            dict(type="PackSegInputs"),
        ]
    model = init_model(cfg, args.checkpoint, device=args.device)
    model=model.cuda(args.device)
    # state_dict = model.state_dict()
    # load_backbone(state_dict, args.backbone)
    # model.load_state_dict(state_dict)
    mmengine.mkdir_or_exist(args.save_dir)
    images = []
    if osp.isfile(args.images):
        images.append(args.images)
    elif osp.isdir(args.images):
        for im in mmengine.scandir(args.images, suffix=args.suffix, recursive=args.not_recursive):
            if args.search_key in im:
                images.append(osp.join(args.images, im))
    else:
        raise NotImplementedError()
    print(f"Collect {len(images)} images")
    for im_path in tqdm.tqdm(images):
        print(im_path)
        result = inference_model(model, im_path)
        pred = draw_sem_seg(result.pred_sem_seg)
        img = Image.open(im_path).convert("RGB")
        pred = Image.fromarray(pred).resize(
            [img.width, img.height], resample=Image.NEAREST
        )
        # vis = Image.new("RGB", [img.width * 2, img.height])
        # vis.paste(img, (0, 0))
        # vis.paste(pred, (img.width, 0))
        pred.save(osp.join(args.save_dir, osp.basename(im_path)))
    print(f"Results are saved in {args.save_dir}")


if __name__ == "__main__":
    main()
