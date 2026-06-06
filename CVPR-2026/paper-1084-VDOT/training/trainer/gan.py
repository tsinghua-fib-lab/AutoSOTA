import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle, ODEfromCeph, collate_fn
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import GAN
import torch
import wandb
import time
import os, io
from einops import rearrange
import subprocess

import numpy as np
import imageio
import torchvision

class Trainer:
    def __init__(self, config):
        self.config = config
        print('config:', self.config)
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # launch_distributed_job()
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
        self.model = GAN(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=True,
        )

        # self.model.vae = fsdp_wrap(
        #     self.model.vae,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
        #     cpu_offload=True,
        # )
        # if not config.no_visualize or config.load_raw_video:
        #     self.model.vae = self.model.vae.to(
        #         device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
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
        self.data_path = config.data_path
        # dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        dataset = ODEfromCeph(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset), flush=True)

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
        #
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
        # if hasattr(config, "load"):
        #     resume_ckpt_path_critic = os.path.join(config.load, "critic")
        #     resume_ckpt_path_generator = os.path.join(config.load, "generator")
        # else:
        #     resume_ckpt_path_critic = "none"
        #     resume_ckpt_path_generator = "none"
        #
        # _, _ = self.checkpointer_critic.try_best_load(
        #     resume_ckpt_path=resume_ckpt_path_critic,
        # )
        # self.step, _ = self.checkpointer_generator.try_best_load(
        #     resume_ckpt_path=resume_ckpt_path_generator,
        #     force_start_w_ema=config.force_start_w_ema,
        #     force_reset_zero_step=config.force_reset_zero_step,
        #     force_reinit_ema=config.force_reinit_ema,
        #     skip_optimizer_scheduler=config.skip_optimizer_scheduler,
        # )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 5.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 5.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model)

        # generator_state_dict = {k: v for k, v in generator_state_dict.items() if "camera" in k}

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            proxy_url = "xxx"
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)
            weight_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt")
            # with io.BytesIO() as buffer:
            torch.save(state_dict, "ckpts/model.pt")
            result = subprocess.run(
                ["aws", "s3", "cp",
                 "ckpts/model.pt",
                 weight_path], stderr=None,
                stdout=None)
            # client.put(weight_path, buffer.getvalue())

            os.environ['HTTP_PROXY'] = proxy_url
            os.environ['HTTPS_PROXY'] = proxy_url
            os.environ['http_proxy'] = proxy_url
            os.environ['https_proxy'] = proxy_url


    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompt_embeds"]  # next(self.dataloader)
        clean_latent = batch["ode_latent"].to(self.device).to(self.dtype)
        clean_latent = rearrange(clean_latent, "b c t h w -> b t c h w")

        batch_size = len(text_prompts)
        image_or_video_shape = list(clean_latent.shape)
        image_or_video_shape[0] = batch_size

        text_prompt_on_device = [item.to(self.device).to(self.dtype) for item in text_prompts]
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # conditional_dict = self.model.text_encoder(
            #     text_prompts=text_prompts)
            conditional_dict = {
                'prompt_embeds': text_prompt_on_device,
            }

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict
        # mini_bs, full_bs = (
        #     batch["mini_bs"],
        #     batch["full_bs"],
        # )

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            gan_G_loss = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=None
            )

            # loss_ratio = mini_bs * self.world_size / full_bs
            # loss_ratio = 1.0
            total_loss = gan_G_loss * self.loss_scale / self.config.gradient_accumulation_steps

            total_loss.backward()
            # generator_grad_norm = self.model.generator.clip_grad_norm_(
            #     self.max_grad_norm_generator)

            # generator_log_dict = {"generator_grad_norm": generator_grad_norm,
            #                       "gan_G_loss": gan_G_loss}
            generator_log_dict = {"gan_G_loss": gan_G_loss}

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        (gan_D_loss, r1_loss, r2_loss), critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            real_image_or_video=clean_latent,
            initial_latent=None
        )

        # loss_ratio = mini_bs * dist.get_world_size() / full_bs
        total_loss = (gan_D_loss + 0.5 * (r1_loss + r2_loss)) * self.loss_scale / self.config.gradient_accumulation_steps

        total_loss.backward()
        # critic_grad_norm = self.model.fake_score.clip_grad_norm_(
        #     self.max_grad_norm_critic)

        # critic_log_dict.update({"critic_grad_norm": critic_grad_norm,
        #                         "gan_D_loss": gan_D_loss,
        #                         "r1_loss": r1_loss,

        critic_log_dict.update({"gan_D_loss": gan_D_loss,
                                "r1_loss": r1_loss,
                                "r2_loss": r2_loss})
        return critic_log_dict

    def generate_video(self, pipeline, prompts):
        with torch.no_grad():
            batch_size = len(prompts)
            sampled_noise = torch.randn(
                [batch_size, 21, 16, 60, 104], device=self.device, dtype=self.dtype
            )
            conditional_dict = self.model.text_encoder(
                text_prompts=prompts)
            print(conditional_dict['prompt_embeds'].shape, flush=True)
            video_latent, _, _ = pipeline.inference_with_trajectory(
                noise=sampled_noise,
                **conditional_dict,
            )
            return video_latent

    def train(self):
        start_step = self.step
        self.model._initialize_inference_pipeline()

        while True:
            # if self.is_main_process:
            if self.step % 5 == 0:
                video_latent = self.generate_video(self.model.inference_pipeline, ["A woman reading a flaming burning book while many people are walking around her in a the busy city."])
                if self.is_main_process:
                    with torch.no_grad():
                        self.model.vae = self.model.vae.to(
                            device=self.device)
                        output_video = self.model.vae.decode_to_pixel(video_latent)
                        self.model.vae = self.model.vae.to(
                            device=torch.device('cpu'))
                    print('vae decode:', output_video.shape, flush=True)
                    output_video_2 = rearrange(output_video, "b t c h w -> b t c h w")
                    output_video_2 = (output_video_2.cpu() + 1) / 2
                    output_video_2 = (output_video_2[0] * 255).numpy().astype(np.uint8)
                    wandb.log({"video": wandb.Video(output_video_2, fps=16, format="mp4")}, step=self.step, commit=True)

                    # output_video_3 = rearrange(output_video, "b t c h w -> t b c h w")
                    # output_video_3 = (output_video_3.cpu() + 1) / 2
                    # # print(videos.max(), videos.min())
                    # outputs = []
                    # for x in output_video_3:
                    #     x = torchvision.utils.make_grid(x, nrow=6)
                    #     x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    #     outputs.append((x * 255).numpy().astype(np.uint8))
                    # imageio.mimsave(
                    #     "test.mp4", outputs, fps=16
                    # )

            # dist.barrier()
            # quit()

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

            # Only update generator and critic outside the warmup phase
            TRAIN_GENERATOR = not self.in_discriminator_warmup and self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator (only outside warmup phase)
            if TRAIN_GENERATOR:
                self.model.fake_score.requires_grad_(False)
                self.model.generator.requires_grad_(True)
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for _ in range(self.config.gradient_accumulation_steps):
                    batch = next(self.dataloader)
                    extra = self.fwdbwd_one_step(batch, True)
                    extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                        self.max_grad_norm_generator)
                generator_log_dict["generator_grad_norm"] = generator_grad_norm
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
            else:
                generator_log_dict = {}

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

            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            for _ in range(self.config.gradient_accumulation_steps):
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)
            critic_log_dict["critic_grad_norm"] = critic_grad_norm
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # If we just finished warmup, print a message
            if self.is_main_process and self.step == self.discriminator_warmup_steps:
                print(f"Finished discriminator warmup after {self.discriminator_warmup_steps} steps")

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            wandb_loss_dict = {
                # "generator_grad_norm": generator_log_dict["generator_grad_norm"],
                "critic_grad_norm": critic_log_dict["critic_grad_norm"].to(self.device),
                "real_logit": critic_log_dict["noisy_real_logit"].to(self.device),
                "fake_logit": critic_log_dict["noisy_fake_logit"].to(self.device),
                "r1_loss": critic_log_dict["r1_loss"].to(self.device),
                "r2_loss": critic_log_dict["r2_loss"].to(self.device),
                "gan_D_loss": critic_log_dict["gan_D_loss"].to(self.device),
            }
            if TRAIN_GENERATOR:
                wandb_loss_dict.update({
                    "generator_grad_norm": generator_log_dict["generator_grad_norm"].to(self.device),
                    "gan_G_loss": generator_log_dict["gan_G_loss"].to(self.device),
                })
            self.all_gather_dict(wandb_loss_dict)
            wandb_loss_dict["diff_logit"] = wandb_loss_dict["real_logit"] - wandb_loss_dict["fake_logit"]
            wandb_loss_dict["reg_loss"] = 0.5 * (wandb_loss_dict["r1_loss"] + wandb_loss_dict["r2_loss"])

            if self.is_main_process:
                if self.in_discriminator_warmup:
                    warmup_status = f"[WARMUP {self.step}/{self.discriminator_warmup_steps}] Training only discriminator params"
                    print(warmup_status)
                    if not self.disable_wandb:
                        wandb_loss_dict.update({"warmup_status": 1.0})

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
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

    def all_gather_dict(self, target_dict):
        for key, value in target_dict.items():
            gathered_value = torch.zeros(
                [self.world_size, *value.shape],
                dtype=value.dtype, device=self.device)
            dist.all_gather_into_tensor(gathered_value, value)
            avg_value = gathered_value.mean().item()
            target_dict[key] = avg_value
