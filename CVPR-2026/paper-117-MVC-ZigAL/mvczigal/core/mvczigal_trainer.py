import os
import contextlib
from collections import defaultdict
from functools import partial
import tqdm
import torch
from safetensors.torch import save_file
from accelerate.utils import broadcast
from accelerate.logging import get_logger
from peft.utils import get_peft_model_state_dict
from mvczigal.utils.wandb_utils import log_images_to_wandb
from mvczigal.utils.training_utils import run_pipeline

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

logger = get_logger(__name__)


def mvczigal_trainer(accelerator, cfg, pipeline, unet, reward_fn, mv_reward_fn, prompt_set):
    # initialize optimizer
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.training.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )

    # initialize Lagrange multiplier
    lagrangian_multiplier = torch.nn.Parameter(
        torch.tensor(cfg.training.lambda_init, dtype=torch.float32, device=accelerator.device),
        requires_grad=True,
    )
    lambda_max = cfg.training.lambda_max if cfg.training.lambda_max else None
    lambda_optimizer = torch.optim.SGD([lagrangian_multiplier], lr=cfg.training.lambda_lr)
    constrain_threshold = cfg.training.constrain_threshold_init

    # for some reason, autocast is necessary for non-lora training but not for lora training, and it uses
    # more memory
    autocast = contextlib.nullcontext if cfg.training.use_lora else accelerator.autocast

    # prepare unet and optimizer
    unet, optimizer = accelerator.prepare(unet, optimizer)

    num_timesteps = cfg.training.num_inference_steps

    logger.info("***** Starting Training *****")

    logging_step = 0

    for epoch in range(cfg.training.num_epochs):

        for iter in range(cfg.training.batches_per_epoch):
            # =================== SAMPLING ===================
            unet.eval()
            pipeline.unet.eval()
            samples1 = []
            samples2 = []
            prompts_to_log, images_to_log, rewards_to_log = None, None, None
            mv_rewards_0, mv_rewards_1, mv_rewards_2 = None, None, None
            for _ in tqdm(
                range(cfg.training.num_sample_iters),
                desc=f"Epoch {epoch}.{iter}: Sampling",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                # sample from prompt_set
                prompts = prompt_set.sample()
                if prompts_to_log is None:
                    prompts_to_log = prompts.copy()

                # sample images
                with autocast():
                    sample_dict1 = run_pipeline(
                        pipeline,
                        prompts=prompts,
                        height=cfg.training.image_height,
                        width=cfg.training.image_width,
                        num_inference_steps=num_timesteps,
                        guidance_scale=cfg.training.guidance_scale,
                        negative_prompt=cfg.training.negative_prompt,
                        num_views=cfg.training.num_views,
                        lora_scale=cfg.training.lora_scale,
                        eta=cfg.training.eta,
                        device=accelerator.device,
                        zmv_sampling=False,
                    )

                if images_to_log is None:
                    images_to_log = sample_dict1["images"].clone()

                # compute rewards synchronously
                rewards = reward_fn(
                    sample_dict1["images"],
                    [p for p in prompts for _ in range(cfg.training.num_views)],
                )
                if cfg.training.mv_reward_fn == "hyper_score":
                    mv_rewards_0, mv_rewards_1, mv_rewards_2, mv_rewards = mv_reward_fn(
                        sample_dict1["images"],
                        prompts,
                        cfg.training.num_views,
                    )
                else:
                    mv_rewards = mv_reward_fn(
                        sample_dict1["images"],
                        prompts,
                    )

                del sample_dict1["images"]

                if rewards_to_log is None:
                    rewards_to_log = rewards.clone()

                # add rewards to sample dict
                sample_dict1["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
                sample_dict1["mv_rewards"] = torch.as_tensor(mv_rewards, device=accelerator.device)
                if cfg.training.mv_reward_fn == "hyper_score":
                    sample_dict1["mv_rewards_0"] = torch.as_tensor(mv_rewards_0, device=accelerator.device)
                    sample_dict1["mv_rewards_1"] = torch.as_tensor(mv_rewards_1, device=accelerator.device)
                    sample_dict1["mv_rewards_2"] = torch.as_tensor(mv_rewards_2, device=accelerator.device)

                initial_latents = sample_dict1["states"][:, 0]

                samples1.append(sample_dict1)
                del sample_dict1
                torch.cuda.empty_cache()

                # sample images
                with autocast():
                    sample_dict2 = run_pipeline(
                        pipeline,
                        prompts=prompts,
                        height=cfg.training.image_height,
                        width=cfg.training.image_width,
                        num_inference_steps=num_timesteps,
                        guidance_scale=cfg.training.guidance_scale,
                        negative_prompt=cfg.training.negative_prompt,
                        num_views=cfg.training.num_views,
                        lora_scale=cfg.training.lora_scale,
                        eta=cfg.training.eta,
                        device=accelerator.device,
                        latents=initial_latents,
                        zmv_sampling=True,
                    )
                del initial_latents

                # compute rewards synchronously
                rewards = reward_fn(
                    sample_dict2["images"],
                    [p for p in prompts for _ in range(cfg.training.num_views)],
                )
                if cfg.training.mv_reward_fn == "hyper_score":
                    mv_rewards_0, mv_rewards_1, mv_rewards_2, mv_rewards = mv_reward_fn(
                        sample_dict2["images"],
                        prompts,
                        cfg.training.num_views,
                    )
                else:
                    mv_rewards = mv_reward_fn(
                        sample_dict2["images"],
                        prompts,
                    )

                del sample_dict2["images"]

                # add rewards to sample dict
                sample_dict2["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
                sample_dict2["mv_rewards"] = torch.as_tensor(mv_rewards, device=accelerator.device)
                if cfg.training.mv_reward_fn == "hyper_score":
                    sample_dict2["mv_rewards_0"] = torch.as_tensor(mv_rewards_0, device=accelerator.device)
                    sample_dict2["mv_rewards_1"] = torch.as_tensor(mv_rewards_1, device=accelerator.device)
                    sample_dict2["mv_rewards_2"] = torch.as_tensor(mv_rewards_2, device=accelerator.device)

                samples2.append(sample_dict2)
                del sample_dict2
                torch.cuda.empty_cache()

            # create super batch
            samples1 = {k: torch.cat([s[k] for s in samples1]) for k in samples1[0].keys()}
            samples2 = {k: torch.cat([s[k] for s in samples2]) for k in samples2[0].keys()}

            # gather rewards across processes
            rewards = accelerator.gather(torch.cat([samples1["rewards"], samples2["rewards"]])).cpu().numpy()
            mv_rewards = accelerator.gather(torch.cat([samples1["mv_rewards"], samples2["mv_rewards"]])).cpu().numpy()
            if cfg.training.mv_reward_fn == "hyper_score":
                mv_rewards_0 = accelerator.gather(
                    torch.cat([samples1["mv_rewards_0"], samples2["mv_rewards_0"]])
                ).cpu().numpy()
                mv_rewards_1 = accelerator.gather(
                    torch.cat([samples1["mv_rewards_1"], samples2["mv_rewards_1"]])
                ).cpu().numpy()
                mv_rewards_2 = accelerator.gather(
                    torch.cat([samples1["mv_rewards_2"], samples2["mv_rewards_2"]])
                ).cpu().numpy()

            del samples1["rewards"], samples1["mv_rewards"], samples2["rewards"], samples2["mv_rewards"]
            if cfg.training.mv_reward_fn == "hyper_score":
                del samples1["mv_rewards_0"], samples1["mv_rewards_1"], samples1["mv_rewards_2"]
                del samples2["mv_rewards_0"], samples2["mv_rewards_1"], samples2["mv_rewards_2"]

            # update Lagrange multiplier
            mv_rewards_mean = mv_rewards.mean()
            ema_beta = cfg.training.ema_beta
            if constrain_threshold > 0:
                constrain_threshold = ema_beta * constrain_threshold + (1 - ema_beta) * mv_rewards_mean
            else:
                constrain_threshold = mv_rewards_mean
            if accelerator.is_main_process and epoch >= cfg.training.lambda_update_delay_steps:
                lambda_loss = (mv_rewards_mean - constrain_threshold) * lagrangian_multiplier
                lambda_loss.backward()
                if mv_rewards_mean < constrain_threshold:
                    lambda_optimizer.param_groups[0]['lr'] = cfg.training.lambda_lr
                else:
                    lambda_optimizer.param_groups[0]['lr'] = cfg.training.lambda_lr * 0.1
                lambda_optimizer.step()
                lambda_optimizer.zero_grad()
                with torch.no_grad():
                    lagrangian_multiplier.clamp_(min=0, max=lambda_max)
            with torch.no_grad():
                broadcast(lagrangian_multiplier, from_process=0)
            lam = lagrangian_multiplier.detach()

            # log images and rewards
            if cfg.wandb.log:
                log_images_to_wandb(
                    accelerator=accelerator,
                    prompts=[p for p in prompts_to_log for _ in range(cfg.training.num_views)],
                    images=images_to_log,
                    rewards=rewards_to_log,
                    step=logging_step,
                )
                log_dict = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr'],
                    "rewards": rewards,
                    "rewards_mean": rewards.mean(),
                    "rewards_std": rewards.std(),
                    "mv_rewards": mv_rewards,
                    "mv_rewards_mean": mv_rewards_mean,
                    "mv_rewards_std": mv_rewards.std(),
                    "constrain_threshold": constrain_threshold,
                    "lambda": lam,
                    "lambda_lr": lambda_optimizer.param_groups[0]['lr'],
                }
                if cfg.training.mv_reward_fn == "hyper_score":
                    log_dict.update({
                        "mv_rewards_0": mv_rewards_0,
                        "mv_rewards_0_mean": mv_rewards_0.mean(),
                        "mv_rewards_0_std": mv_rewards_0.std(),
                        "mv_rewards_1": mv_rewards_1,
                        "mv_rewards_1_mean": mv_rewards_1.mean(),
                        "mv_rewards_1_std": mv_rewards_1.std(),
                        "mv_rewards_2": mv_rewards_2,
                        "mv_rewards_2_mean": mv_rewards_2.mean(),
                        "mv_rewards_2_std": mv_rewards_2.std(),
                    })
                accelerator.log(log_dict, step=logging_step)
                del log_dict

            del prompts_to_log, images_to_log, rewards_to_log

            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            mv_advantages = (mv_rewards - mv_rewards.mean()) / (mv_rewards.std() + 1e-8)

            del rewards, mv_rewards
            del mv_rewards_0, mv_rewards_1, mv_rewards_2

            # ungather advantages
            advantages = (
                torch.as_tensor(advantages)
                .reshape(accelerator.num_processes, -1)[accelerator.process_index]
                .to(accelerator.device)
            )
            samples1["advantages"], samples2["advantages"] = torch.chunk(advantages, chunks=2)
            mv_advantages = (
                torch.as_tensor(mv_advantages)
                .reshape(accelerator.num_processes, -1)[accelerator.process_index]
                .to(accelerator.device)
            )
            samples1["mv_advantages"], samples2["mv_advantages"] = torch.chunk(
                mv_advantages.repeat_interleave(cfg.training.num_views, dim=0), chunks=2
            )

            del advantages, mv_advantages
            torch.cuda.empty_cache()

            total_batch_size = samples1["states"].shape[0]

            # =================== TRAINING ===================
            for _ in range(cfg.training.num_inner_epochs):
                # shuffle along timestep dimension independently for each sample
                perms = torch.stack([
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size // cfg.training.num_views)
                ]).repeat_interleave(cfg.training.num_views, dim=0)
                for key in ["step_index", "timesteps", "states", "next_states", "log_probs"]:
                    samples1[key] = samples1[key][
                        torch.arange(total_batch_size, device=accelerator.device)[:, None], perms
                    ]
                    samples2[key] = samples2[key][
                        torch.arange(total_batch_size, device=accelerator.device)[:, None], perms
                    ]

                # rebatch for training
                ex_keys = ["add_time_ids", "adapter_state_0", "adapter_state_1", "adapter_state_2", "adapter_state_3"]
                samples_batched1 = {
                    k: (
                        v.reshape(-1, cfg.training.train_batch_size_per_gpu * 2 * cfg.training.num_views, *v.shape[1:])
                        if pipeline.do_classifier_free_guidance and k in ex_keys else
                        v.reshape(-1, cfg.training.train_batch_size_per_gpu * cfg.training.num_views, *v.shape[1:])
                    )
                    for k, v in samples1.items()
                }
                samples_batched2 = {
                    k: (
                        v.reshape(-1, cfg.training.train_batch_size_per_gpu * 2 * cfg.training.num_views, *v.shape[1:])
                        if pipeline.do_classifier_free_guidance and k in ex_keys else
                        v.reshape(-1, cfg.training.train_batch_size_per_gpu * cfg.training.num_views, *v.shape[1:])
                    )
                    for k, v in samples2.items()
                }

                # dict of lists -> list of dicts for easier iteration
                samples_batched1 = [dict(zip(samples_batched1, x)) for x in zip(*samples_batched1.values())]
                samples_batched2 = [dict(zip(samples_batched2, x)) for x in zip(*samples_batched2.values())]

                # get Guidance Scale Embedding
                if pipeline.unet.config.time_cond_proj_dim is not None:
                    guidance_scale_tensor = torch.tensor(cfg.training.guidance_scale - 1).repeat(
                        cfg.training.train_batch_size_per_gpu * cfg.training.num_views)
                    timestep_cond = pipeline.get_guidance_scale_embedding(
                        guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
                    ).to(device=accelerator.device, dtype=samples1["states"].dtype)
                else:
                    timestep_cond = None

                # begin training
                unet.train()
                pipeline.unet.train()
                info = defaultdict(list)
                for i, (sample1, sample2) in tqdm(
                    enumerate(zip(samples_batched1, samples_batched2)),
                    desc=f"Epoch {epoch}.{iter}: Training",
                    disable=not accelerator.is_local_main_process,
                    position=0,
                ):
                    if pipeline.do_classifier_free_guidance:
                        prompt_embeds1 = torch.cat([sample1["negative_prompt_embeds"], sample1["prompt_embeds"]])
                        prompt_embeds2 = torch.cat([sample2["negative_prompt_embeds"], sample2["prompt_embeds"]])
                        added_cond_kwargs1 = {
                            "text_embeds": torch.cat([
                                sample1["negative_pooled_prompt_embeds"], sample1["add_text_embeds"]
                            ]),
                            "time_ids": sample1["add_time_ids"],
                        }
                        added_cond_kwargs2 = {
                            "text_embeds": torch.cat([
                                sample2["negative_pooled_prompt_embeds"], sample2["add_text_embeds"]
                            ]),
                            "time_ids": sample2["add_time_ids"],
                        }
                    else:
                        prompt_embeds1 = sample1["prompt_embeds"]
                        prompt_embeds2 = sample2["prompt_embeds"]
                        added_cond_kwargs1 = {
                            "text_embeds": sample1["add_text_embeds"],
                            "time_ids": sample1["add_time_ids"],
                        }
                        added_cond_kwargs2 = {
                            "text_embeds": sample2["add_text_embeds"],
                            "time_ids": sample2["add_time_ids"],
                        }

                    for j in tqdm(
                        range(num_timesteps),
                        desc="Timestep",
                        disable=not accelerator.is_local_main_process,
                        position=1,
                        leave=False,
                        dynamic_ncols=True,
                    ):
                        with accelerator.accumulate(unet):
                            with autocast():
                                if pipeline.do_classifier_free_guidance:
                                    latents = torch.cat([sample1["states"][:, j]] * 2)
                                    timesteps = torch.cat([sample1["timesteps"][:, j]] * 2)
                                else:
                                    latents = sample1["states"][:, j]
                                    timesteps = sample1["timesteps"][:, j]

                                noise_pred1 = unet(
                                    latents,
                                    timesteps,
                                    prompt_embeds1,
                                    timestep_cond=timestep_cond,
                                    cross_attention_kwargs={
                                        "mv_scale": 1.0,
                                        **(pipeline.cross_attention_kwargs or {}),
                                    },
                                    down_intrablock_additional_residuals=[
                                        sample1[f"adapter_state_{i}"].clone()
                                        for i in range(4)
                                    ],
                                    added_cond_kwargs=added_cond_kwargs1,
                                ).sample

                                if pipeline.do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = noise_pred1.chunk(2)
                                    noise_pred1 = noise_pred_uncond + pipeline.guidance_scale * (
                                        noise_pred_text - noise_pred_uncond
                                    )

                                log_probs = []
                                for k in range(cfg.training.train_batch_size_per_gpu * cfg.training.num_views):
                                    log_prob = pipeline.scheduler.step(
                                        model_output=noise_pred1[k][None],
                                        step_index=sample1["step_index"][k, j],
                                        timestep=sample1["timesteps"][k, j],
                                        sample=sample1["states"][k, j][None],
                                        eta=cfg.training.eta,
                                        prev_sample=sample1["next_states"][k, j][None],
                                        return_dict=False,
                                    )[2]
                                    log_probs.append(log_prob)
                                log_prob1 = torch.cat(log_probs)

                                if pipeline.do_classifier_free_guidance:
                                    latents = torch.cat([sample2["states"][:, j]] * 2)
                                    timesteps = torch.cat([sample2["timesteps"][:, j]] * 2)
                                else:
                                    latents = sample2["states"][:, j]
                                    timesteps = sample2["timesteps"][:, j]

                                noise_pred2 = unet(
                                    latents,
                                    timesteps,
                                    prompt_embeds2,
                                    timestep_cond=timestep_cond,
                                    cross_attention_kwargs={
                                        "mv_scale": 1.0,
                                        **(pipeline.cross_attention_kwargs or {}),
                                    },
                                    down_intrablock_additional_residuals=[
                                        sample2[f"adapter_state_{i}"].clone()
                                        for i in range(4)
                                    ],
                                    added_cond_kwargs=added_cond_kwargs2,
                                ).sample

                                if pipeline.do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = noise_pred2.chunk(2)
                                    noise_pred2 = noise_pred_uncond + pipeline.guidance_scale * (
                                        noise_pred_text - noise_pred_uncond
                                    )

                                log_probs = []
                                for k in range(cfg.training.train_batch_size_per_gpu * cfg.training.num_views):
                                    log_prob = pipeline.scheduler.step(
                                        model_output=noise_pred2[k][None],
                                        step_index=sample2["step_index"][k, j],
                                        timestep=sample2["timesteps"][k, j],
                                        sample=sample2["states"][k, j][None],
                                        eta=cfg.training.eta,
                                        prev_sample=sample2["next_states"][k, j][None],
                                        return_dict=False,
                                    )[2]
                                    log_probs.append(log_prob)
                                log_prob2 = torch.cat(log_probs)

                            # compute the advantage difference
                            advantage_diff = sample1["advantages"] - sample2["advantages"]
                            mv_advantage_diff = sample1["mv_advantages"] - sample2["mv_advantages"]
                            advantage_diff = (advantage_diff + lam * mv_advantage_diff) / (1 + lam)

                            # compute the log prob difference
                            log_ratio1 = (log_prob1 - sample1["log_probs"][:, j])
                            log_ratio2 = (log_prob2 - sample2["log_probs"][:, j])
                            log_diff = log_ratio1 - log_ratio2

                            loss_unclipped = torch.square(log_diff - advantage_diff)

                            log_prob1_clipped = torch.clamp(
                                log_prob1,
                                sample1["log_probs"][:, j] - cfg.training.clip_range,
                                sample1["log_probs"][:, j] + cfg.training.clip_range
                            )
                            log_prob2_clipped = torch.clamp(
                                log_prob2,
                                sample2["log_probs"][:, j] - cfg.training.clip_range,
                                sample2["log_probs"][:, j] + cfg.training.clip_range
                            )
                            log_ratio1 = (log_prob1_clipped - sample1["log_probs"][:, j])
                            log_ratio2 = (log_prob2_clipped - sample2["log_probs"][:, j])
                            log_diff_clipped = log_ratio1 - log_ratio2

                            loss_clipped = torch.square(log_diff_clipped - advantage_diff)
                            loss = torch.mean(torch.maximum(loss_unclipped, loss_clipped))

                            # debugging values
                            info["approx_kl"].append(0.5 * 0.5 * torch.mean(
                                (log_prob1 - sample1["log_probs"][:, j]) ** 2
                                + (log_prob2 - sample2["log_probs"][:, j]) ** 2
                            ))
                            info["clipfrac"].append(0.5 * torch.mean(
                                (torch.abs(log_prob1 - sample1["log_probs"][:, j]) > cfg.training.clip_range).float()
                                + (torch.abs(log_prob2 - sample2["log_probs"][:, j]) > cfg.training.clip_range).float()
                            ))
                            info["loss"].append(loss)

                            # backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                # log training-related stuff
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                if cfg.wandb.log:
                    accelerator.log(info, step=logging_step)
                logging_step += 1
                info = defaultdict(list)

                # make sure we did an optimization step at the end of the inner epoch
                assert accelerator.sync_gradients

                del samples_batched1, samples_batched2
                torch.cuda.empty_cache()

            del samples1, samples2
            del latents, timesteps, prompt_embeds1, prompt_embeds2, added_cond_kwargs1, added_cond_kwargs2
            del noise_pred1, noise_pred2, log_probs, log_prob1, log_prob2, log_prob1_clipped, log_prob2_clipped
            del log_ratio1, log_ratio2, log_diff, log_diff_clipped, loss_unclipped, loss_clipped, loss
            if pipeline.do_classifier_free_guidance:
                del noise_pred_uncond, noise_pred_text
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
        if (epoch + 1) % cfg.training.save_interval == 0 and accelerator.is_main_process:
            save_dir = os.path.join(accelerator.project_dir, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            if cfg.training.use_lora:
                unet_lora_layers = accelerator.unwrap_model(unet)
                unet_lora_layers = get_peft_model_state_dict(unet_lora_layers)
                weight_name = "mvczigal_lcm_sdxl_lora.safetensors"
                save_file(unet_lora_layers, os.path.join(save_dir, weight_name))
                logger.info(f"LoRA weights saved in: {os.path.join(save_dir, weight_name)}")
                del unet_lora_layers
                torch.cuda.empty_cache()
            else:
                # TODO: Enable model saving for non-LoRA training
                logger.info("Model saving for non-LoRA training is not implemented yet.")
