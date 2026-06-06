import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset, TextMultiTaskVideoDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, init_distributed_mode, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD, BiDMD_GAN
import torch
import wandb
import time
from einops import rearrange
import subprocess
import os
import io

import sys
import math
import random
import types
import traceback
from contextlib import contextmanager
from functools import partial

from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import imageio
import torchvision

from utils.preprocess import VaceVideoProcessor
from petrel_client.client import Client
import time
from petrel_client.common.exception import NetworkConnectionError

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True

        #launch_distributed_job()

        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # Configuration for discriminator warmup
        self.discriminator_warmup_steps = getattr(config, "discriminator_warmup_steps", 0)
        self.in_discriminator_warmup = self.step < self.discriminator_warmup_steps
        if self.in_discriminator_warmup and self.is_main_process:
            print(f"Starting with discriminator warmup for {self.discriminator_warmup_steps} steps")
        self.loss_scale = getattr(config, "loss_scale", 1.0)

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        elif config.distribution_loss == "bi_dmd_gan":
            self.model = BiDMD_GAN(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            # min_num_params=1e5
            cpu_offload=True
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=True
        )

        for name, param in self.model.generator.named_parameters():
            print(name, param.shape, param.requires_grad)

        print(
            f"  Total generator parameters per FSDP shard = {sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)} M"
        )
        print(
            f"  Total fake parameters per FSDP shard = {sum(p.numel() for p in self.model.fake_score.parameters() if p.requires_grad) / 1e9} B"
        )
        print(
            f"  Total real parameters per FSDP shard = {sum(p.numel() for p in self.model.real_score.parameters() if p.requires_grad) / 1e9} B"
        )

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Create separate parameter groups for the fake_score network
        # One group for parameters with "_cls_pred_branch" or "_gan_ca_blocks" in the name
        # and another group for all other parameters
        fake_score_params = []
        discriminator_params = []

        for name, param in self.model.fake_score.named_parameters():
            if param.requires_grad:
                if "_cls_pred_branch" in name or "_gan_ca_blocks" in name:
                    discriminator_params.append(param)
                else:
                    fake_score_params.append(param)

        # Use the special learning rate for the special parameter group
        # and the default critic learning rate for other parameters
        self.critic_param_groups = [
            {'params': fake_score_params, 'lr': config.lr_critic},
            {'params': discriminator_params, 'lr': config.lr_critic * config.discriminator_lr_multiplier}
        ]
        if self.in_discriminator_warmup:
            self.critic_optimizer = torch.optim.AdamW(
                self.critic_param_groups,
                betas=(0.9, config.beta2_critic)
            )
        else:
            self.critic_optimizer = torch.optim.AdamW(
                self.critic_param_groups,
                betas=(config.beta1_critic, config.beta2_critic)
            )

        # Step 3: Initialize the dataloader
        dataset = TextMultiTaskVideoDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        # rename_param = (
        #     lambda name: name.replace("_fsdp_wrapped_module.", "")
        #     .replace("_checkpoint_wrapped_module.", "")
        #     .replace("_orig_mod.", "")
        # )
        # self.name_to_trainable_params = {}
        # for n, p in self.model.generator.named_parameters():
        #     if not p.requires_grad:
        #         continue

        #     renamed_n = rename_param(n)
        #     self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        # if getattr(config, "generator_ckpt", False):
        #     print(f"Loading pretrained generator from {config.generator_ckpt}")
        #     state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        #     if "generator" in state_dict:
        #         state_dict = state_dict["generator"]
        #     elif "model" in state_dict:
        #         state_dict = state_dict["model"]
        #     self.model.generator.load_state_dict(
        #         state_dict, strict=True
        #     )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 5.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 5.0)
        self.previous_time = None


        vae_stride = (4, 8, 8)
        patch_size = (1, 2, 2)
        sample_fps = 16 
        self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(vae_stride, patch_size)]),
            min_area=480 * 832,
            max_area=480 * 832,
            min_fps=sample_fps,
            max_fps=sample_fps,
            zero_start=True,
            seq_len=32760,
            keep_last=True)
        
        self.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

        #self.client = Client('petreloss.conf', enable_mc=True)
        self.client = None


    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        # if self.config.use_lora:
        #     generator_state_dict = {k: v for k, v in generator_state_dict.items() if "lora_" in k}

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                # "critic": critic_state_dict,
                # "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                # "critic": critic_state_dict,
            }

        if self.is_main_process:
            proxy_url = "xxxx"
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)
            
            weight_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt")
            # with io.BytesIO() as buffer:
            torch.save(state_dict, "ckpts/model.pt")

            result = subprocess.run(
                ["aws", "s3", 
                 "cp",
                 "--endpoint-url=xxx",
                 "ckpts/model.pt",
                 weight_path], stderr=None,
                stdout=None)

            os.environ['HTTP_PROXY'] = proxy_url
            os.environ['HTTPS_PROXY'] = proxy_url
            os.environ['http_proxy'] = proxy_url
            os.environ['https_proxy'] = proxy_url
            print("Model saved to", weight_path)

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames)
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive)
            reactive = vae.encode(reactive)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs)
                else:
                    ref_latent = vae.encode(refs)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=(4, 8, 8)):
        #vae_stride = self.vae_stride if vae_stride is None else vae_stride
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // vae_stride[0])
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, vae_stride[1], width, vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                vae_stride[1] * vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m, dtype=torch.bfloat16):
        return [torch.cat([zz, mm], dim=0).to(dtype=dtype) for zz, mm in zip(z, m)]
    
    def robust_get(self, client, input_path, max_retries=0, retry_interval=3):
        """
        Robustly read remote data, automatically retry on network errors.
        
        Parameters:
            client: petrel_client instance
            input_path: remote path (e.g. s3://xxx/yyy.mp4)
            max_retries: maximum number of retries, 0 means infinite retries until success
            retry_interval: wait time between retries in seconds
        
        Returns:
            data: binary data read from remote
        """
        # if input_path is a local path, return the data directly
        # if not input_path.startswith("s3://"):
        return open(input_path, "rb").read()

        # # if input_path is a remote s3 path, use petrel_client to read the data
        # attempt = 0
        # while True:
        #     try:
        #         data = client.get(input_path)
        #         return data  
        #     except NetworkConnectionError as e:
        #         attempt += 1
        #         print(f"[Warning] Network error ({attempt} attempts): {e}")
        #         if max_retries > 0 and attempt >= max_retries:
        #             raise RuntimeError(f"Failed to read {input_path} after {max_retries} retries: {e}")
        #         # exponential backoff: extend wait time on failure
        #         time.sleep(retry_interval * min(4, attempt))
        #     except Exception as e:
        #         # capture other potential exceptions (e.g. file not found)
        #         raise RuntimeError(f"Failed to read {input_path}: {e}")


    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device, task_names):
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720*1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480*832:
            self.vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(f'image_size {image_size} is not supported')

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video[0], sub_src_mask[0], task_names)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video[0], task_names)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])


        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        img_bytes = self.robust_get(self.client, "gcc:" + ref_img)
                        ref_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                            scale = min(canvas_height / ref_height, canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images


    def read_in_gt_videos(self, video_file_path_list):
        artgrid_base = "xxx/artgrid/resize/"
        ret_video_files = []
        for video_file_path in video_file_path_list:
            path_str = str(video_file_path)
            # if the path is a local path, read the video from the local path
            with open(path_str, "rb") as f:
                videofile = io.BytesIO(f.read())
            ret_video_files.append(videofile)

            # if the path is a remote path, read the video from the remote path
            # video_file_name = os.path.basename(path_str)
            # video_file_path = os.path.join(artgrid_base, video_file_name)
            # videofile = io.BytesIO(self.client.get(video_file_path))
            # ret_video_files.append(videofile)

        return ret_video_files


    def fwdbwd_one_step(self, batch, train_generator, exit_flag, step_now):

        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        pose_video_path = batch["video_path"]
        task_names = batch["task_name"]
        mask_video_path = batch["mask_video_path"]
        ref_image_path = batch["ref_image_path"]

        print(f"task name: {task_names}, video_path: {pose_video_path}")

        gt_videos_path = [] 
        artgrid_gt_base = "xxx/artgrid/resize/"
        if task_names[0] != 'reference': 
            for idx in range(len(pose_video_path)):
                input_path = pose_video_path[idx] 
                # print(input_path)
                # print(type(input_path))
                input_path = str(input_path)
                video_file_name = os.path.basename(input_path)
                video_file_path = os.path.join(artgrid_gt_base, video_file_name)
                gt_videos_path.append(video_file_path)
        else: 
            for idx in range(len(ref_image_path)):
                input_path = ref_image_path[idx].split(',')[0] 
                video_file_name = os.path.basename(input_path)
                video_file_path = os.path.join(artgrid_gt_base, video_file_name.split('.')[0].split('_')[0] + '.mp4')
                gt_videos_path.append(video_file_path)

        image_latent = None 
        clean_latent_list = [] 
        with torch.no_grad():
            for gt_video in gt_videos_path: 
                src_video, src_mask, src_ref_images = self.prepare_source(
                    [[gt_video]],
                    [None],
                    [None],
                    81, (832, 480), 
                    self.device,
                    task_names
                )
                clean_latent_ = self.model.vae.encode_to_latent(
                        src_video[0].unsqueeze(0)).to(device=self.device, dtype=self.dtype).squeeze(0)
                clean_latent_list.append(clean_latent_)

            clean_latent = torch.stack(clean_latent_list).float()
            print('clean latent shape:', clean_latent.shape)

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            time0 = time.time()
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)
            time1 = time.time()
            print(f"text encode cost {time1-time0}s.")

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.sample_neg_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict
            
            time0 = time.time()
            src_video = pose_video_path if isinstance(pose_video_path[0], str) else None
            src_mask = mask_video_path if isinstance(mask_video_path[0], str) else None
            src_ref_images = ref_image_path[0].split(',') if isinstance(ref_image_path[0], str) else None

            src_video, src_mask, src_ref_images = self.prepare_source([src_video],
                                                                      [src_mask],
                                                                      [src_ref_images],
                                                                      81, (832, 480), self.device, task_names)

            z0 = self.vace_encode_frames(src_video, src_ref_images, masks=src_mask, vae=self.model.vae)
            m0 = self.vace_encode_masks(src_mask, src_ref_images)
            z = self.vace_latent(z0, m0, self.dtype)
            time1 = time.time()
            print(f"pose encode cost {time1-time0}s.")

            target_shape = list(z0[0].shape)
            target_shape[0] = int(target_shape[0] / 2)
            patch_size = (1, 2, 2)
            seq_len = (target_shape[2] * target_shape[3]) / (patch_size[1] * patch_size[2]) * target_shape[1]

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            G_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=None,
                exit_flag=exit_flag,
                vace_context=z,
                step_now=step_now,
            )

            total_loss = G_loss * self.loss_scale / self.config.gradient_accumulation_steps
            total_loss.backward()

            generator_log_dict.update({"G_loss": G_loss})

            return generator_log_dict
        else:
            generator_log_dict = {}
        
        torch.cuda.empty_cache()

        # Step 4: Store gradients for the critic (if training the critic)
        (denoising_loss, gan_D_loss, r1_loss, r2_loss), critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            real_image_or_video=clean_latent,
            initial_latent=None,
            exit_flag=exit_flag,
            vace_context=z,
            step_now=step_now,
        )

        total_loss = (denoising_loss + gan_D_loss + 0.5 * (r1_loss + r2_loss)) * self.loss_scale / self.config.gradient_accumulation_steps
        total_loss.backward()

        critic_log_dict.update({"D_loss": gan_D_loss + denoising_loss,
                                "r1_loss": r1_loss,
                                "r2_loss": r2_loss})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def generate_and_sync_list(self, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device
            )
        else:
            indices = torch.empty(1, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def train(self):
        start_step = self.step

        while True:
            if self.is_main_process:
                print('-'*80)
                print('Now step: ', self.step)
                print('-'*80)


            if self.step == self.discriminator_warmup_steps and self.discriminator_warmup_steps != 0:
                print("Resetting critic optimizer")
                del self.critic_optimizer
                torch.cuda.empty_cache()
                # Create new optimizers
                self.critic_optimizer = torch.optim.AdamW(
                    self.critic_param_groups,
                    betas=(self.config.beta1_critic, self.config.beta2_critic)
                )
                # Update checkpointer references
                # self.checkpointer_critic.optimizer = self.critic_optimizer
            # Check if we're in the discriminator warmup phase
            self.in_discriminator_warmup = self.step < self.discriminator_warmup_steps

            # self.model.update_step(self.step)
            exit_flag = self.generate_and_sync_list(4, device=self.device)
            TRAIN_GENERATOR = not self.in_discriminator_warmup and self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.model.fake_score.requires_grad_(False)
                self.model.generator.requires_grad_(True)
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for _ in range(self.config.gradient_accumulation_steps):
                    batch = next(self.dataloader)
                    extra = self.fwdbwd_one_step(batch, True, exit_flag, self.step)
                    extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                        self.max_grad_norm_generator)
                generator_log_dict["generator_grad_norm"] = generator_grad_norm
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                

            # Train the critic/discriminator
            if self.in_discriminator_warmup:
                # During warmup, only allow gradient for discriminator params
                self.model.generator.requires_grad_(False)
                self.model.fake_score.requires_grad_(False)

                # Enable gradient only for discriminator params
                for name, param in self.model.fake_score.named_parameters():
                    if "_cls_pred_branch" in name or "_gan_ca_blocks" in name:
                        param.requires_grad_(True)
            else:
                # Normal training mode
                self.model.generator.requires_grad_(False)
                self.model.fake_score.requires_grad_(True)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            for _ in range(self.config.gradient_accumulation_steps):
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, False, exit_flag, self.step)
                extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)
            critic_log_dict["critic_grad_norm"] = critic_grad_norm
            self.critic_optimizer.step()

            self.step += 1
            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.step % self.config.gc_interval == 0:
                # if dist.get_rank() == 0:
                #     logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
