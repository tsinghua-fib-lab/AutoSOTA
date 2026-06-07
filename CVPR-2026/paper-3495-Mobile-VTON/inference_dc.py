import os
import sys
import os.path as osp
import argparse
import json
from typing import Literal, Tuple

import torch
import torch.utils.data as data
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKL
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

WORK_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
if WORK_DIR not in sys.path:
    sys.path.append(WORK_DIR)

from Mobile_VTON.utils.misc import get_real_path
from Mobile_VTON.models.autoencoders.vae import Decoder

from Mobile_VTON.models.unets.unet_2d_condition_tryon import (
    UNet2DConditionModel as Unet_Tryon
)
from Mobile_VTON.models.unets.unet_2d_condition_garment import (
    UNet2DConditionModel as Unet_Garment
)

from Mobile_VTON.pipelines.tryon_pipeline_full_cat import T2IMobilePipelineV1_3_NotLoadingT5_Decoder as TryonPipeline


class DresscodeTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        category: str = "upper_body",
        size: Tuple[int, int] = (512, 384),
        file: str = "train_unpairs",
        image_encoder_path: str = "ckpt/DINO_V2",
        person_image_name: str = "",
    ):
        super(DresscodeTestDataset, self).__init__()
        self.dataroot = os.path.join(dataroot_path,category)
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()
        self.order = order
        self.radius = 5
        self.category = category
        
        im_names = []
        c_names = []

        filename = os.path.join(dataroot_path,category, f"{file}.txt")
        with open(filename, "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()

                if person_image_name:
                    im_name = person_image_name
                im_names.append(im_name)
                c_names.append(c_name)

        self.descriptions = dict()
        with open(os.path.join(dataroot_path, category, "dc_descriptions.txt"), "r") as f:
            for line in f:
                parts = line.strip().split(': ', 1)
                if len(parts) == 2:
                    filename, description = parts
                    self.descriptions[filename] = description

        self.im_names = im_names
        self.c_names = c_names
        self.image_processor = AutoImageProcessor.from_pretrained(image_encoder_path)
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        
        cloth = Image.open(os.path.join(self.dataroot, "images", c_name))

        im_pil_big = Image.open(
            os.path.join(self.dataroot, "images", im_name)
        ).resize((self.width,self.height))
        image = self.transform(im_pil_big)
        if self.category == "upper_body":
            instruction = "upper body"
        if self.category == "lower_body":
            instruction = "lower body"
        if self.category == "dresses":
            instruction = "whole body"
 
        result = {}
        result["c_name"] = c_name
        result["im_name"] = im_name
        result["image"] = image
        result["cloth_pure"] = self.transform(cloth)
        result["cloth"] = self.image_processor(images=cloth, return_tensors="pt").pixel_values
        result["caption_cloth"] = self.descriptions[c_name]
        result["caption"] = f"Replace the {instruction} with " + " ".join(self.descriptions[c_name].split()[1:])

        return result
    
    def __len__(self):
        return len(self.im_names)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def parse_args():
    p = argparse.ArgumentParser("HSW Try-on: infer all test pairs (full pipeline)")
    p.add_argument("--data_dir", type=str, default="../IDM-VTON/Dataset/zalando")
    p.add_argument("--output_dir", type=str, default="output/infer_test_full")
    p.add_argument("--order", type=str, choices=["paired", "unpaired"], default="paired")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--test_batch_size", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    p.add_argument("--file", type=str, default=None)
    p.add_argument("--person_image_name", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, required=True,)
    p.add_argument("--scheduler_shift", type=float, default=3.0,)


    return p.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    if accelerator.is_main_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    test_dataset = DresscodeTestDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order=args.order,
        size=(args.height, args.width),
        file=args.file,
        image_encoder_path=args.checkpoint_path + "/image_encoder",
        person_image_name=args.person_image_name,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        shuffle=False, 
        batch_size=args.test_batch_size, 
        num_workers=4, 
        pin_memory=True
    )

    # Scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=args.scheduler_shift,
    )

    # Tokenizer & Text Encoder
    tokenizer_one = CLIPTokenizer.from_pretrained(args.checkpoint_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(args.checkpoint_path, subfolder="tokenizer_2")

    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(args.checkpoint_path, subfolder="text_encoder")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.checkpoint_path, subfolder="text_encoder_2")

    # VAE encoder（SD3.5-large's vae）
    vae = AutoencoderKL.from_pretrained(args.checkpoint_path, subfolder="vae")

    # VAE decoder
    vd_cfg = get_real_path(args.checkpoint_path + "/vae_decoder/decoder.json")
    vd_ckpt = get_real_path(args.checkpoint_path + "/vae_decoder/decoder.pt")
    with open(vd_cfg, "r") as f:
        vd_cfg_json = json.load(f)
    vae_decoder = Decoder(**vd_cfg_json)
    vae_decoder.load_state_dict(torch.load(vd_ckpt, map_location="cpu"), strict=True)

    # DINOv2
    image_encoder = AutoModel.from_pretrained(args.checkpoint_path, subfolder="image_encoder")

    # garmentnet and tryonnet
    denoiser = Unet_Tryon.from_pretrained(args.checkpoint_path, subfolder="denoiser")
    denoiser_garment = Unet_Garment.from_pretrained(args.checkpoint_path, subfolder="denoiser_garment")

    vae.to(accelerator.device, dtype=weight_dtype)
    vae_decoder.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    denoiser.to(accelerator.device, dtype=weight_dtype)
    denoiser_garment.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    for m in [vae, vae_decoder, text_encoder_one, text_encoder_two, denoiser, denoiser_garment, image_encoder]:
        m.eval().requires_grad_(False)

    pipe = TryonPipeline(
        vae=vae,
        vae_decoder=vae_decoder,
        scheduler=noise_scheduler,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        image_encoder=image_encoder,
        denoiser=denoiser,
        denoiser_garment=denoiser_garment,
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    neg_prompt = (
        "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, "
        "missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, "
        "mutated, ugly, disgusting, blurry, amputation, NSFW"
    )

    test_loader = accelerator.prepare(test_loader)
    
    pipe_device = accelerator.device
    progress = tqdm(total=len(test_loader), disable=not accelerator.is_local_main_process, desc="Inferring")

    with torch.no_grad():
        autocast_dtype = None
        if accelerator.mixed_precision == "fp16":
            autocast_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            autocast_dtype = torch.bfloat16

        for i, batch in enumerate(test_loader):
            if i >= 4:
                break
            B = batch["cloth"].shape[0]

            ip_imgs = torch.cat([batch["cloth"][i] for i in range(B)], dim=0)  # [B,3,h,w]

            prompt = batch["caption"]
            if not isinstance(prompt, list):
                prompt = [prompt] * B
            negative_prompt = [neg_prompt] * B

            prompt_c = batch["caption_cloth"]
            if not isinstance(prompt_c, list):
                prompt_c = [prompt_c] * B
            negative_prompt_c = [neg_prompt] * B 

            with torch.autocast(device_type=pipe_device.type, dtype=autocast_dtype) if autocast_dtype else torch.no_grad():

                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                    device=pipe_device,
                )


                (
                    prompt_embeds_c,
                    negative_prompt_embeds_c,
                    pooled_prompt_embeds_c,
                    negative_pooled_prompt_embeds_c,
                ) = pipe.encode_prompt(
                    prompt=prompt_c,
                    prompt_2=prompt_c,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt_c,
                    device=pipe_device,
                )


                images = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    text_embeds_cloth=prompt_embeds_c,
                    negative_text_embeds_cloth=negative_prompt_embeds_c,
                    cloth=batch["cloth_pure"].to(pipe_device), 
                    image=(batch["image"].to(pipe_device) + 1.0) / 2.0, 
                    ip_adapter_image=ip_imgs.to(pipe_device), 
                    device=pipe_device,
                )[0]

            for i, pil_img in enumerate(images):
                im_name = batch["im_name"][i]
                c_name = batch["c_name"][i]
                out_name = f"{im_name[:-4]}_{c_name}"
                x = pil_to_tensor(pil_img)
                torchvision.utils.save_image(x, os.path.join(args.output_dir, out_name))

            progress.update(1)

    progress.close()
    if accelerator.is_main_process:
        print(f"Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
