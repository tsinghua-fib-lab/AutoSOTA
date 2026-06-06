import os
import datetime
import torch
from safetensors.torch import load_file
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
import hydra
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMInverseScheduler
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvczigal.diffusers_patch.lcm_scheduler import LCMScheduler
from mvczigal.diffusers_patch.lcm_inv_scheduler import LCMInverseScheduler
from mvczigal.diffusers_patch.ddim_scheduler import DDIMScheduler
from mvczigal.diffusers_patch.pipeline import MVAdapterT2MVSDXLPipeline
from mvczigal.rewards import rewards
from mvczigal.data import prompts

logger = get_logger(__name__)

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@hydra.main(
    version_base=None, config_path="../mvczigal/configs", config_name="lcm_sdxl_mate3d"
)
def main(cfg: DictConfig) -> None:
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not cfg.wandb.run_name:
        cfg.wandb.run_name = unique_id
    else:
        cfg.wandb.run_name += "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(cfg.training.logdir, cfg.wandb.run_name),
        automatic_checkpoint_naming=True,
        total_limit=99999,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision="fp16",
        project_config=accelerator_config,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
    )

    if cfg.wandb.log and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.wandb.project_name,
            config=OmegaConf.to_container(cfg),
            init_kwargs={"wandb": {"name": cfg.wandb.run_name}},
        )

    # set seed (device_specific is very important to get different prompts on different devices)
    # set_seed(cfg.training.seed, device_specific=True, deterministic=True)
    set_seed(cfg.training.seed, device_specific=True)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and non-lora unet) to
    # half-precision as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load pipeline
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained.vae_model if cfg.pretrained.vae_model else cfg.pretrained.base_model,
        subfolder=None if cfg.pretrained.vae_model else "vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained.unet_model if cfg.pretrained.unet_model and not os.path.exists(cfg.pretrained.unet_model)
        else cfg.pretrained.base_model,
        subfolder=None if cfg.pretrained.unet_model and not os.path.exists(cfg.pretrained.unet_model) else "unet",
        torch_dtype=inference_dtype,
    )
    if cfg.pretrained.unet_model and os.path.exists(cfg.pretrained.unet_model):
        unet.load_state_dict(load_file(cfg.pretrained.unet_model))
    pipeline = MVAdapterT2MVSDXLPipeline.from_pretrained(
        cfg.pretrained.base_model,
        torch_dtype=inference_dtype,
        vae=vae,
        unet=unet,
    )
    pipeline.vae = vae
    pipeline.unet = unet

    if cfg.training.scheduler == "lcm":
        scheduler_class = LCMScheduler
        inv_scheduler_class = LCMInverseScheduler
    elif cfg.training.scheduler == "ddim":
        scheduler_class = DDIMScheduler
        inv_scheduler_class = DDIMInverseScheduler
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.training.scheduler}")

    # load scheduler
    scheduler = ShiftSNRScheduler.from_scheduler(
        pipeline.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    # load inverse scheduler
    pipeline.inv_scheduler = ShiftSNRScheduler.from_scheduler(
        pipeline.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=inv_scheduler_class,
    )
    pipeline.scheduler = scheduler

    # enable memory-efficient operators for VAE or attention
    pipeline.enable_vae_slicing()
    # pipeline.enable_vae_tiling()
    # pipeline.enable_attention_slicing()
    # pipeline.enable_xformers_memory_efficient_attention()

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    if cfg.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        desc="Timestep",
        disable=not accelerator.is_local_main_process,
        position=1,
        leave=False,
        dynamic_ncols=True,
    )

    # move unet, vae, text_encoder and cond_encoder to device and cast to inference_dtype
    if cfg.pretrained.vae_model:
        pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    else:
        pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    unet.to(accelerator.device, dtype=inference_dtype)

    # load multi-view adapter
    pipeline.init_custom_adapter(num_views=cfg.training.num_views)
    pipeline.load_custom_adapter(cfg.pretrained.custom_adapter, "mvadapter_t2mv_sdxl.safetensors")
    if cfg.training.use_lora:
        unet.requires_grad_(False)
        unet.to(accelerator.device, dtype=inference_dtype)
    pipeline.cond_encoder.requires_grad_(False)
    pipeline.cond_encoder.to(accelerator.device, dtype=inference_dtype)

    if cfg.training.use_lora:
        unet_lora_config = LoraConfig(
            r=cfg.training.lora_rank,
            lora_alpha=cfg.training.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "to_k_mv", "to_q_mv", "to_v_mv", "to_out_mv.0"],
        )
        unet.add_adapter(unet_lora_config)
        if accelerator.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA layers) into fp32
            cast_training_params(unet, dtype=torch.float32)

    # enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

    # load reward function and prompt set
    reward_fn = getattr(rewards, cfg.training.reward_fn)(torch.float32, accelerator.device)
    mv_reward_fn = getattr(rewards, cfg.training.mv_reward_fn)(torch.float32, accelerator.device)
    prompt_set = getattr(prompts, cfg.training.prompt_set)(cfg.training.sample_batch_size_per_gpu)

    # start training
    from mvczigal.core.mvczigal_trainer import mvczigal_trainer
    mvczigal_trainer(accelerator, cfg, pipeline, unet, reward_fn, mv_reward_fn, prompt_set)

    accelerator.end_training()


if __name__ == "__main__":
    main()
