import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from glob import glob

import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data
import torchvision.transforms.functional as F

# from basicsr.utils import DiffJPEG, USMSharp
# from basicsr.utils.img_process_util import filter2D
# from basicsr.data.transforms import paired_random_crop, triplet_random_crop
# from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian
#
# from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
# from basicsr.data.transforms import augment
# from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
# from basicsr.utils.registry import DATASET_REGISTRY
from .degGen import DegradationGeneration

noise_params = {
            'sg_high': 0.10,
            'sg_low': 0.03,

            'sb_high': 5,
            'sb_low': 0.1,

            'o_high': 75,
            'o_low': 15,

            'var_high': 20 ** 2,
            'var_low': 5 ** 2,

            'scale_high': 1.5,
            'scale_low': 0.01,

            'noise_prob': 0.5
        }


new_blur_params = {
            'ksize_high': 23,
            'ksize_low': 7,

            'sigma_high': 3,
            'sigma_low': 1,

            'scale_choice': [1, 2, 4],
            'interpolation_choice': [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA],
        }

lc_params = {
            'alpha_high': 0.2,
            'alpha_low': 0.8,

            'beta_high': 0.4,
            'beta_low': 0.2,

            'brightness_by_max': True
        }

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)
    # parser.add_argument("--base_config", default="./configs/sr.yaml", type=str)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_randomcrop_256x256_hflip", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_val", default=True, action="store_false")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--sd_path")
    parser.add_argument("--pretrained_path", type=str, default=None,)
    parser.add_argument("--pretrain_dem_path", type=str, default=None,)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)
    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=50000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "piecewise_constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=0.1, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep == "randomcrop_512x512_hflip":
        T = transforms.Compose([
            transforms.RandomCrop((512, 512)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep == "resized_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T



class InfraredPairedDataset(data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        super().__init__()
        if split == "train":
            with open(dataset_folder + '/train.txt', 'r') as f:
                self.img_names = f.read().splitlines()
        elif split == "val":
            with open(dataset_folder + '/test.txt', 'r') as f:
                self.img_names = f.read().splitlines()
        print(len(self.img_names))
        self.data_folder = os.path.join(dataset_folder, "imgs")
        self.label_folder = os.path.join(dataset_folder, "Label")
        self.T = build_transform(image_prep)
        # self.level_T = build_transform("randomcrop_256x256_hflip")
        self.tokenizer = tokenizer
        self.degGenerate = DegradationGeneration(noise_params, new_blur_params, lc_params)
        self.deg = None
        self.deg_params = None
        
        self.split = split
        self.img_ref = np.array(self.T(Image.open(os.path.join(self.data_folder, self.img_names[0]))).convert("L"), dtype=np.uint8)
        self.level_num = 2
        # label = Image.open(os.path.join(self.label_folder, img_name))

        # input images scaled to -1,1
        # img_t = self.T(input_img)
        # deg_t = self.degGenerate.comp_all(np.array(img_t.convert("L"), dtype=np.uint8))
        # img_t = F.to_tensor(img_t)
        # # output images scaled to -1,1
        # # output_t = self.T(output_img)
        # deg_t = F.to_tensor(Image.fromarray(deg_t).convert("RGB"))
        # img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        # deg_t = F.normalize(deg_t, mean=[0.5], std=[0.5])
        
        


    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.data_folder, img_name)).convert("RGB")
        caption = ""
        # label = Image.open(os.path.join(self.label_folder, img_name))

        # input images scaled to -1,1
        # img_t = self.T(input_img)
        # deg_t = self.degGenerate.comp_all(np.array(img_t.convert("L"), dtype=np.uint8))
        # img_t = F.to_tensor(img_t)
        # # output images scaled to -1,1
        # # output_t = self.T(output_img)
        # deg_t = F.to_tensor(Image.fromarray(deg_t).convert("RGB"))
        # img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        # deg_t = F.normalize(deg_t, mean=[0.5], std=[0.5])
        
        
        img_t = self.T(input_img)
        if self.deg == "level_deg":
            # level_img_t = self.level_T(input_img)
            level_img_t = img_t
            level_deg_t = self.level_deg_img(level_img_t)
            level_caption = [caption] * level_deg_t.shape[0]
            # level_img_t = F.to_tensor(level_img_t)
            # level_img_t = F.normalize(level_img_t, mean=[0.5], std=[0.5])
            
        deg_t = self.degGenerate.comp1_all(np.array(img_t.convert("L"), dtype=np.uint8))
        deg_t = F.to_tensor(Image.fromarray(deg_t).convert("RGB"))
        deg_t = F.normalize(deg_t, mean=[0.5], std=[0.5])
        
        img_t = F.to_tensor(img_t)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        # if self.split == "train":
            # print(self.deg)
            
        

        # input_ids = self.tokenizer(
        #     caption, max_length=self.tokenizer.model_max_length,
        #     padding="max_length", truncation=True, return_tensors="pt"
        # ).input_ids
        if self.deg == "level_deg":
            return level_deg_t, deg_t, img_t, caption, level_caption
            # return level_deg_t, deg_t, img_t, level_img_t, caption, level_caption
        else:
            return deg_t, img_t, caption #, label
        #     {
        #     "output_pixel_values": img_t,
        #     "conditioning_pixel_values": deg_t,
        #     "input_ids": input_ids,
        #     "label": label
        # }
        
    
    def set_level_deg(self):
        self.deg = "level_deg"
        self.deg_params = []
        for _ in range(self.level_num):
            self.deg_params.append(self.degGenerate.get_comp1_all_params(self.img_ref))
        
        
    def set_level_deg_params(self):
        self.deg_params = None

    def set_deg(self):
        self.deg = "deg"

    def level_deg_img(self, img):
        # deg_weak, deg_heavy = self.degGenerate.comp1_level_all(np.array(img.convert("L"), dtype=np.uint8))
        # deg_weak = F.normalize(F.to_tensor(Image.fromarray(deg_weak).convert("RGB")), mean=[0.5], std=[0.5])
        # deg_heavy = F.normalize(F.to_tensor(Image.fromarray(deg_heavy).convert("RGB")), mean=[0.5], std=[0.5])
        # deg_t = torch.stack([deg_weak, deg_heavy], dim=0)
        
        deg_level = []
        for idx in range(self.level_num):
            deg_level_s = self.degGenerate.comp1_all_params(np.array(img.convert("L"), dtype=np.uint8), **(self.deg_params[idx]))
            deg_level_s = F.normalize(F.to_tensor(Image.fromarray(deg_level_s).convert("RGB")), mean=[0.5], std=[0.5])
            deg_level.append(deg_level_s)
        deg_t = torch.stack(deg_level, dim=0)
        
        return deg_t