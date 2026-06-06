import torch
import os.path as osp
from collections import OrderedDict
from torch import Tensor
import torch.nn.functional as F
import sys
import numpy as np
import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("pretrained", type=str)
    args.add_argument("converted", type=str)
    args.add_argument("--kernel", default=16, type=int)
    args.add_argument("--height", default=512, type=int)
    args.add_argument("--width", default=512, type=int)
    return args.parse_args()

def convert_weight(ckpt):
    new_state_dict = OrderedDict()
    for k,v in ckpt.items():
        if 'student' in k:
            print(k)
            new_state_dict[k.replace("student.backbone.", "")] = v

    return new_state_dict

def load_weight(pretrained_path):
    if not osp.isfile(pretrained_path):
        raise FileNotFoundError(
            f"{pretrained_path} dont exist(absolute path: {osp.abspath(pretrained_path)})"
        )
    state_dict = torch.load(pretrained_path, map_location="cpu")['model']
    weight = convert_weight(state_dict)
    if len(weight.keys()) <= 10:
        print(f"The read weights may be abnormal, as shown below:")
        print(weight.keys())
        raise KeyError()
    return weight


def main():
    args = parse_args()
    pretrained_path = args.pretrained
    converted_path = args.converted
    weight = load_weight(pretrained_path)
    print("Load from", pretrained_path)
    torch.save(weight, converted_path)
    print("Save to", converted_path)
    return args


# Check if the script is run directly (and not imported)
if __name__ == "__main__":
    main()
