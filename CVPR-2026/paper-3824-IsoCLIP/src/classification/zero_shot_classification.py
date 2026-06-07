import argparse
import sys
import os, csv
from pathlib import Path

import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger
from dotmap import DotMap 
PROJECT_ROOT = Path(__file__).absolute().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
import os
import csv
import random
import string
import datetime

from data_utils import CLASSIFICATION_SPLTS
import classification.datasets.caltech101
import classification.datasets.dtd
import classification.datasets.eurosat
import classification.datasets.fgvc_aircraft
import classification.datasets.food101
import classification.datasets.imagenet
import classification.datasets.oxford_flowers
import classification.datasets.oxford_pets
import classification.datasets.stanford_cars
import classification.datasets.sun397
import classification.datasets.ucf101

import classification.trainers

lower_to_name = {
    "caltech101": 'Caltech101',
    'dtd': 'DescribableTextures',
    'eurosat': 'EuroSAT',
    'fgvc_aircraft': 'FGVCAircraft',
    'food101': 'Food101',
    'imagenet': 'ImageNet',
    'oxford_flowers': 'OxfordFlowers',
    'oxford_pets': 'OxfordPets',
    'stanford_cars': 'StanfordCars',
    'sun397': 'SUN397',
    'ucf101': 'UCF101',
}


def update_cfg(cfg, args):
    if args.dataroot:
        cfg.DATASET.ROOT = args.dataroot

    if args.out_path:
        cfg.OUTPUT_PATH = args.out_path

    if args.seed:
        cfg.SEED = args.seed

    cfg.DATASET.NAME = lower_to_name[args.dataset_name]
    cfg.MODEL.BACKBONE.NAME = args.clip_model_name
    cfg.MODEL.BACKBONE.OPEN_CLIP_PRETRAINED = args.open_clip_pretrained
    cfg.MODEL.BACKBONE.USE_OPEN_CLIP = args.use_open_clip

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    
    if args.eval_type == "zeroshot" or args.eval_type == "ncm":
        cfg.iso_ktop, cfg.iso_kbottom = -1, -1 
    elif args.eval_type == "iso_ncm":
        cfg.iso_ktop = args.iso_ktop 
        cfg.iso_kbottom = args.iso_kbottom 


def setup_cfg(args):
    cfg = get_cfg_default()

    update_cfg(cfg, args)

    return cfg


def main(args):
    
    cfg = setup_cfg(args)


    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    dataset_name = args.dataset_name
    
    split = CLASSIFICATION_SPLTS[dataset_name]['split']

    if args.eval_type == "zeroshot":
        cfg.TRAINER.NAME = "ClipZeroshot"
    elif args.eval_type == "ncm":
        cfg.TRAINER.NAME = "ClipNCM"
    elif args.eval_type == "iso_ncm": 
        cfg.TRAINER.NAME = "IsoNCM" 
    

    trainer = build_trainer(cfg)
    accuracy = trainer.test(split=split)
    
        
    # --------------------------------------------------
    # Base output directory
    # --------------------------------------------------
    if args.out_path is not None:
        base_dir = args.out_path
    else:
        base_dir = "local_run_classification" 

    os.makedirs(base_dir, exist_ok=True)
    
    # --------------------------------------------------
    # Unique run folder name
    # --------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_tag = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    run_dir = os.path.join(base_dir, f"exp_{timestamp}_{rand_tag}")
    os.makedirs(run_dir, exist_ok=True)
    
    
    # --------------------------------------------------
    # Row data
    # --------------------------------------------------
    row_data = {
        "Model": args.clip_model_name,
        "Dataset": args.dataset_name,
        "Eval_type": args.eval_type,
        "K_top": cfg.iso_ktop,
        "K_bottom": cfg.iso_kbottom,
        "Accuracy": accuracy,
        "timestamp": timestamp,
        "folder_path": os.path.abspath(run_dir),
    }
            
    fieldnames = list(row_data.keys())
    
    csv_path = os.path.join(run_dir, "classification_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row_data)
   
    print(f"\n✅ Summary saved at: {csv_path}\n")

   




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True, help="Root directory containing all datasets.")
    
    parser.add_argument("--out_path", type=str, default=None, help="Path to save results")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use.", required=True)
 
    parser.add_argument("--eval-type", type=str, choices=['zeroshot', 'ncm', 'iso_ncm'], default='iso_ncm',
                        help="Evaluation mode:\n"
                             "'zeroshot' – use CLIP's original image/text features for zero-shot classification,\n"
                              "'ncm' - use CLIP's original image features for ncm classification,\n"
                              "'iso_ncm' - use IsoCLIP's image features for ncm classification."
    )
    parser.add_argument("--seed", type=int, default=1,
                    help="Set a fixed random seed for reproducibility (only applied if > 0).")
                          
    parser.add_argument("--iso_ktop", type=int, default=150, help="ISO Projection M Top (default: 0)")
    parser.add_argument("--iso_kbottom", type=int, default=50, help="ISO Projection M Bottom (default: 0)") 
    parser.add_argument("--clip_model_name", default="ViT-B/32", type=str, help="CLIP model variant to use, e.g. 'ViT-B/32'.")
    parser.add_argument("--use_open_clip", action='store_true', help="Enable to use OpenCLIP instead of OpenAI CLIP.",
                        default=False)
    parser.add_argument("--open_clip_pretrained", type=str,
                        help="Name of the pretrained weights for OpenCLIP (e.g., 'laion2b_s34b_b79k').", default=None)
    
    args = parser.parse_args() 

    print(args)

    main(args)
