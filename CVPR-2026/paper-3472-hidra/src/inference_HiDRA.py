import os
import gc
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms

import diffusers

from diffusers.utils.import_utils import is_xformers_available

from my_utils.testing_utils import parse_args_paired_testing, InfraredLQDataset
from DEM import DEM
from StableDiffusionDAloRA import StableDiffusionDAloRA


def main(args):
    pretrained_path = args.pretrained_path
    sd_path = args.sd_path

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # initialize net_enh
    net_enh = StableDiffusionDAloRA(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=sd_path,
                      pretrained_path=pretrained_path)
    net_enh.set_eval()

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').cuda().eval()
    net_hyper = DEM(backbone)
    net_hyper.load_model(args.dem_path)
    net_hyper = net_hyper.cuda()
    net_hyper.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_enh.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_enh.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_val = InfraredLQDataset(args.lq_path)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_enh, net_hyper = accelerator.prepare(net_enh, net_hyper)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_enh.to(accelerator.device, dtype=weight_dtype)
    net_hyper.to(accelerator.device, dtype=weight_dtype)

    for step, batch_val in enumerate(dl_val):
        lq_path = batch_val['lq_path'][0]
        (path, name) = os.path.split(lq_path)

        im_lq = batch_val['lq'].cuda()
        im_lq = im_lq.to(memory_format=torch.contiguous_format).float()
        ori_h, ori_w = im_lq.shape[2:]
        im_lq_resize = F.interpolate(
            im_lq,
            size=(ori_h * args.scale,
                  ori_w * args.scale),
            mode='bilinear',
            align_corners=False
            ) if args.scale != 1 else im_lq

        im_lq_resize = im_lq_resize.contiguous()
        im_lq_resize_norm = im_lq_resize * 2 - 1.0
        im_lq_resize_norm = torch.clamp(im_lq_resize_norm, -1.0, 1.0)
        resize_h, resize_w = im_lq_resize_norm.shape[2:]

        pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
        pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
        im_lq_resize_norm = F.pad(im_lq_resize_norm, pad=(0, pad_w, 0, pad_h), mode='reflect')

        B = im_lq_resize.size(0)
        with torch.no_grad():
            # forward pass
            prompt = net_hyper(im_lq_resize_norm)
            text_prompt = ["" for _ in range(B)]
            x_tgt_pred = accelerator.unwrap_model(net_enh)(im_lq_resize_norm, prompt, text_prompt)
            x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            out_img = (x_tgt_pred * 0.5 + 0.5).cpu().detach()

        output_pil = transforms.ToPILImage()(out_img[0]).convert('L')
        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.output_dir, fname + '.png')
        output_pil.save(outf)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args_paired_testing()
    main(args)
