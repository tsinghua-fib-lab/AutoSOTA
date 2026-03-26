import torch

import os

from . import models_mae
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
from . import util

from huggingface_hub import snapshot_download
import os
from pathlib import Path

MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"

class VisionTS(nn.Module):

    def __init__(self, arch='mae_base', finetune_type='ln', ckpt_dir='./ckpt/', load_ckpt=True):
        super(VisionTS, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")

        self.vision_model = MAE_ARCH[arch][0]()

        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[arch][1]
                util.download_file(remote_url, ckpt_path)
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                self.vision_model.load_state_dict(checkpoint['model'], strict=True)
            except:
                print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")
        
        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n

    
    def update_config(self, context_len, pred_len, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear'):
        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.input_resize = util.safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const
        
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()
    

    def forward(self, x, export_image=False, fp64=False):
        # Forecasting using visual model.
        # x: look-back window, size: [bs x context_len x nvars]
        # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
        # return: forecasting window, size: [bs x pred_len x nvars]

        # 1. Normalization
        means = x.mean(1, keepdim=True).detach()  # [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [bs x 1 x nvars]
        stdev /= self.norm_const
        x_enc /= stdev
        # Channel Independent
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]

        # 2. Segmentation
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate') # [b n s]
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)

        # 3. Render & Alignment
        x_resize = self.input_resize(x_2d)
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)

        # 4. Reconstruction
        _, y, mask = self.vision_model(
            image_input, 
            mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
        )
        image_reconstructed = self.vision_model.unpatchify(y) # [(bs x nvars) x 3 x h x w]
        
        # 5. Forecasting
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # color image to grey
        y_segmentations = self.output_resize(y_grey) # resize back
        y_flatten = einops.rearrange(
            y_segmentations, 
            '(b n) 1 f p -> b (p f) n', 
            b=x_enc.shape[0], f=self.periodicity
        ) # flatten
        y = y_flatten[:, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :] # extract the forecasting window

        # 6. Denormalization
        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        if export_image:
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.vision_model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = self.vision_model.unpatchify(mask)  # 1 is removing, 0 is keeping
            # mask = torch.einsum('nchw->nhwc', mask)
            image_reconstructed = image_input * (1 - mask) + image_reconstructed * mask
            green_bg = -torch.ones_like(image_reconstructed) * 2
            image_input = image_input * (1 - mask) + green_bg * mask
            image_input = einops.rearrange(image_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            
            image_reconstructed = einops.rearrange(image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            return y, image_input, image_reconstructed
        return y


# huggingface repository name:
VISIONTSPP_REPO_ID = "Lefei/VisionTSpp"

class VisionTSpp(nn.Module):
    def __init__(self, arch='mae_base', finetune_type='ln', ckpt_dir='./ckpt/', ckpt_path=None, load_ckpt=True,
                 quantile=True, clip_input=0, complete_no_clip=False, color=True, quantile_head_num=9):
        super(VisionTSpp, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")

        self.vision_model = MAE_ARCH[arch][0](quantile=quantile, quantile_head_num=quantile_head_num)
        self.quantile = quantile
        self.clip_input = clip_input
        self.complete_no_clip = complete_no_clip
        self.color = color

        if load_ckpt:
            if ckpt_path is None:
                ckpt_path = os.path.join(ckpt_dir, "visiontspp_model.ckpt")
            
            if not os.path.isfile(ckpt_path):
                # local directory to save the model
                local_dir = Path(ckpt_path).parent

                # Download model from HuggingFace
                snapshot_download(
                    repo_id=VISIONTSPP_REPO_ID,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
            
            try:
                print(f"Load {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                # quantile model:
                if not quantile:
                    for k in list(checkpoint['model'].keys()):
                        if "decoder_pred_1" in k:
                            new_k = k.replace("decoder_pred_1", "decoder_pred")
                            checkpoint['model'][new_k] = checkpoint['model'][k]
                            del checkpoint['model'][k]
                        if "decoder_pred_2" in k:
                            del checkpoint['model'][k]
                        if "decoder_pred_3" in k:
                            del checkpoint['model'][k]
                else:
                    for k in list(checkpoint['model'].keys()):
                        # quantile model also change 'decoder_pred_1' into 'decoder_pred'
                        if "decoder_pred_1" in k:
                            new_k = k.replace("decoder_pred_1", "decoder_pred")
                            checkpoint['model'][new_k] = checkpoint['model'][k]
                            del checkpoint['model'][k]
                # Load the model
                self.vision_model.load_state_dict(checkpoint['model'], strict=True)
            except:
                print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")
                raise
        
        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n

    
    def update_config(self, context_len, pred_len, num_patch_input=None, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear', padding_mode='replicate'):

        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity
        self.padding_mode = padding_mode
        
        if num_patch_input is not None:
            # extra padding
            extra_padding = pred_len / (self.num_patch - num_patch_input) * num_patch_input - self.context_len
            if extra_padding > 0:
                self.context_len += int(np.ceil(extra_padding))

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        if num_patch_input is None:
            self.num_patch_input = int(input_ratio * self.num_patch * align_const)
            if self.num_patch_input == 0:
                self.num_patch_input = 1
        else:
            self.num_patch_input = num_patch_input

        self.num_patch_output = self.num_patch - self.num_patch_input
        self.adjust_input_ratio = self.num_patch_input / self.num_patch

        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        
        
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * self.adjust_input_ratio))
        self.norm_const = norm_const
        
        
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()
    

    def forward(self, x, export_image=False, fp64=False, multivariate=False, color_list=None, LOOKBACK_LEN_VISUAL=300):
        # Forecasting using visual model.
        # x: look-back window, shape: [bs x context_len x nvars]
        # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
        # return: forecasting window, size: [bs x pred_len x nvars]

        # get nvars from input here
        self.nvars = x.shape[-1]
        
        # in case of input cannot be evenly divided by nvars, pad with zeros
        self.image_size_per_var = int(self.image_size / self.nvars)  # 224 // nvars
        self.input_resize = util.safe_resize((self.image_size_per_var, int(self.image_size * self.adjust_input_ratio)), interpolation=self.interpolation)
        
        self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=self.interpolation)


        # 1. Normalization
        means = x.mean(1, keepdim=True).detach()  # x: [bs x seq_len x nvars], means: [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [bs x 1 x nvars]
        # norm_const: ususally 0.4
        stdev /= self.norm_const
        x_enc /= stdev
        # Channel Independent
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')  # [bs x nvars x seq_len]

        if x_enc.shape[-1] < self.context_len:
            extra_padding = self.context_len - x_enc.shape[-1]
        else:
            extra_padding = 0

        # 2. Segmentation
        x_pad = F.pad(x_enc, (self.pad_left + extra_padding, 0), mode=self.padding_mode) # [b n s]
        
        # f is periodicity, p is the number of periods
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> b n f p', f=self.periodicity)  # x_2d.shape: [bs, nvars, f, p]

        # 3. Render & Alignment
        x_resize = self.input_resize(x_2d)  # x_resize.shape: [b, nvars, 224//nvars, p->112]
        x_resize = einops.rearrange(x_resize, 'b n h w -> b 1 (n h) w')  # x_resize.shape: [bs, 1, nvars*224//nvars, 112]
        # in case of input cannot be evenly divided by nvars, there may exist pad_down
        pad_down = self.image_size - x_resize.shape[2]  # 224 - nvars*224//nvars
        
        if pad_down > 0:
            x_resize = torch.concat([
                    x_resize, 
                    torch.zeros((x_resize.shape[0], x_resize.shape[1], pad_down, x_resize.shape[3]), 
                        device=x_resize.device, dtype=x_resize.dtype)
                ], 
                dim=2)  # [bs, 1, 224, 112]
        # assert x_resize.shape[2] == self.image_size, f"image size mismatch: {x_resize.shape[2]} vs {self.image_size}"
        
        
        # masked: [bs, 1, 224, 112]，right-half mask
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)  # x_concat_with_masked.shape = [bs, 1, 224, 224]
        
        
        # We recommend to use fixed color order, such as RGB color order cycling during inference.
        if not self.color:
            image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)
        else:
            image_input = torch.zeros((x_concat_with_masked.shape[0], 3, x_concat_with_masked.shape[2], x_concat_with_masked.shape[3]), 
                                    device=x_concat_with_masked.device, 
                                    dtype=x_concat_with_masked.dtype)  # [bs, 3, 224, 224]
            
            if color_list is None:
                color_list = [i % 3 for i in range(self.nvars)]
            
            for i in range(self.nvars):
                color = color_list[i]
                image_input[:, color, i*self.image_size_per_var:(i+1)*self.image_size_per_var, :] = \
                    x_concat_with_masked[:, 0, i*self.image_size_per_var:(i+1)*self.image_size_per_var, :]
        
        
        if self.clip_input == 0:  # if not self.clip_input:
            if not self.complete_no_clip:
                image_input = torch.clip(image_input, -5, 5)
        else:
            # mean and std of ImageNet dataset
            image_mean = [0.485,0.456,0.406]
            image_std = [0.229,0.224,0.225]
            # Calculate threshold for each channel, [-mean/std, (1-mean)/std]. The result is as follows:
            thres_down_list = [-2.1179039301310043, -2.0357142857142856, -1.8044444444444445]
            thres_up_list = [2.2489082969432315, 2.428571428571429, 2.6399999999999997]
            thres_down = max(thres_down_list)
            thres_up = min(thres_up_list)
            
            image_input = torch.clip(image_input, thres_down, thres_up)
            

        # 4. Reconstruction
        _, y, mask = self.vision_model(
            image_input, 
            mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
        )
        
        if self.quantile:
            # including two parts
            y, y_quantile_list = y
            
            image_reconstructed_quantile_list = []
            for y_quantile in y_quantile_list:
                image_reconstructed_quantile = self.vision_model.unpatchify(y_quantile)  # [(bs x nvars) x 1 x h x w]
                image_reconstructed_quantile_list.append(image_reconstructed_quantile)
        
        # main data
        image_reconstructed = self.vision_model.unpatchify(y) # [(bs x nvars) x 3 x h x w]


        # extract each color channel from image_reconstructed
        def process_images(image_reconstructed, nvars, color_list):
            """
            Args:
                image_reconstructed: tensor, [batch, 3, 224, 224] input image
                nvars: int, number of variables
                color_list: list, color index list, length is nvars
            Returns:
                output: tensor, [batch, 1, 224, 224] processed image
            """
            batch_size = image_reconstructed.shape[0]
            height, width = image_reconstructed.shape[2], image_reconstructed.shape[3]
            output = torch.zeros((batch_size, 1, height, width), 
                                device=image_reconstructed.device)
            
            nvar = nvars
            for i in range(batch_size):
                # calculate height of each variable block
                h_per_var = height // nvar
                remainder = height % nvar
                
                for k in range(nvar):
                    # calculate start and end height of current block
                    start_h = k * h_per_var
                    end_h = (k + 1) * h_per_var
                    
                    # If it is the last block and there is a remainder, adjust the end height
                    if k == nvar - 1 and remainder != 0:
                        end_h = height
                    
                    # get current color channel
                    color_channel = color_list[k]
                    
                    # extract and fill to output
                    output[i, 0, start_h:end_h, :] = \
                        image_reconstructed[i, color_channel, start_h:end_h, :]
            
            return output  # [batch, 1, 224, 224]
        
        
        # 5. Forecasting
        if not self.color:
            # deal with RGB color channels
            y_grey = torch.mean(image_reconstructed, 1, keepdim=True)  # color image to grey, [B, 1, H, W]
            y_before_resize = y_grey
        else:
            y_grey = process_images(image_reconstructed, self.nvars, color_list)  # [B, 1, H, W]

        
        if self.quantile:
            y_grey_quantile_list = []
            for image_reconstructed_quantile in image_reconstructed_quantile_list:
                if not self.color:
                    y_grey_quantile = torch.mean(image_reconstructed_quantile, 1, keepdim=True) # color image to grey, [B, 1, H, W]
                else:
                    y_grey_quantile = process_images(image_reconstructed_quantile, self.nvars, color_list) # [B, 1, H, W]
                y_grey_quantile_list.append(y_grey_quantile)
        
        
        def extract_TS_from_image(y_grey):
            if pad_down > 0: 
                y_grey = y_grey[:, :, :-pad_down, :]  # [b, 1, 224-pad_down, 224]
            y_grey = einops.rearrange(y_grey, 'b 1 (n h) w -> b n h w', n=self.nvars)  # [b, nvars, 224//nvars, 224]
            y_segmentations = self.output_resize(y_grey)  # shape: [b, n, periodicity, num_periods]
            
            y_flatten = einops.rearrange(
                y_segmentations, 
                'b n f p -> b (p f) n', 
                # f=self.periodicity,
                # n=self.nvars,
            ) # flatten -> shape changes to [bs, total_len, nvars]
            
            start_idx = self.pad_left + self.context_len
            end_idx = self.pad_left + self.context_len + self.pred_len
            y_pred = y_flatten[:, start_idx: end_idx, :]
            
            return y_pred  # final shape is [bs, pred_len, nvars]

        # get the transformed time series
        # the following predictions are all [bs, pred_len, nvars]
        y_pred = extract_TS_from_image(y_grey)
        if self.quantile:
            y_pred_quantile_list = []
            for y_grey_quantile in y_grey_quantile_list:
                y_pred_quantile = extract_TS_from_image(y_grey_quantile)
                y_pred_quantile_list.append(y_pred_quantile)
        

        if self.quantile:
            y = y_pred
            y_quantile_list = y_pred_quantile_list
        else:
            y = y[:, 0]

        # 6. Denormalization
        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        if self.quantile:
            y_quantile_list = [
                y_quantile * (stdev.repeat(1, self.pred_len, 1)) + means.repeat(1, self.pred_len, 1)
                for y_quantile in y_quantile_list
            ]
            y = [y, y_quantile_list]
            

        if export_image:
            period_num = LOOKBACK_LEN_VISUAL // self.periodicity
            x_2d_visual = x_2d[:, :, :, -period_num:]  # x_2d.shape: [bs, nvars, f, p]
            
            x_resize = self.input_resize(x_2d_visual)  # x_resize.shape: [b, nvars, 224//nvars, p->112]
            x_resize = einops.rearrange(x_resize, 'b n h w -> b 1 (n h) w')  # x_resize.shape: [bs, 1, nvars*224//nvars, 112]
            # pad_down
            pad_down = self.image_size - x_resize.shape[2]  # 224 - nvars*224//nvars
            
            if pad_down > 0:
                x_resize = torch.concat([
                        x_resize, 
                        torch.zeros((x_resize.shape[0], x_resize.shape[1], pad_down, x_resize.shape[3]), 
                            device=x_resize.device, dtype=x_resize.dtype)
                    ], 
                    dim=2)  # [bs, 1, 224, 112]
            # assert x_resize.shape[2] == self.image_size, f"image size mismatch: {x_resize.shape[2]} vs {self.image_size}"
            
            # masked: [bs, 1, 224, 112]，right-half mask
            masked = torch.zeros((x_2d_visual.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d_visual.device, dtype=x_2d_visual.dtype)
            x_concat_with_masked = torch.cat([
                x_resize, 
                masked
            ], dim=-1)  # x_concat_with_masked.shape = [bs, 1, 224, 224]
            

            if not self.color:
                image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)
            else:
                image_input = torch.zeros((x_concat_with_masked.shape[0], 3, x_concat_with_masked.shape[2], x_concat_with_masked.shape[3]), 
                                        device=x_concat_with_masked.device, 
                                        dtype=x_concat_with_masked.dtype)  # [bs, 3, 224, 224]
                
                if color_list is None:
                    color_list = [i % 3 for i in range(self.nvars)]
                
                for i in range(self.nvars):
                    color = color_list[i]
                    image_input[:, color, i*self.image_size_per_var:(i+1)*self.image_size_per_var, :] = \
                        x_concat_with_masked[:, 0, i*self.image_size_per_var:(i+1)*self.image_size_per_var, :]
            
            if self.clip_input == 0:  # if not self.clip_input:
                if not self.complete_no_clip:
                    image_input = torch.clip(image_input, -5, 5)
            else:
                # mean and std of ImageNet dataset
                image_mean = [0.485,0.456,0.406]
                image_std = [0.229,0.224,0.225]
                # Calculate threshold for each channel, [-mean/std, (1-mean)/std]. The result is as follows:
                thres_down_list = [-2.1179039301310043, -2.0357142857142856, -1.8044444444444445]
                thres_up_list = [2.2489082969432315, 2.428571428571429, 2.6399999999999997]
                thres_down = max(thres_down_list)
                thres_up = min(thres_up_list)
                
                image_input = torch.clip(image_input, thres_down, thres_up)
            
            
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.vision_model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = self.vision_model.unpatchify(mask)  # 1 means removing, 0 means keeping
            # mask = torch.einsum('nchw->nhwc', mask)
            image_reconstructed = image_input * (1 - mask) + image_reconstructed * mask
            green_bg = -torch.ones_like(image_reconstructed) * 2
            image_input = image_input * (1 - mask) + green_bg * mask
            image_input = einops.rearrange(image_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            
            image_reconstructed = einops.rearrange(image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            
            # return y, image_input, image_reconstructed
            return y, image_input, image_reconstructed, self.nvars, color_list

        return y
