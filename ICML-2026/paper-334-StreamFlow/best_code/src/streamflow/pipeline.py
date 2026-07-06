import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from .image_utils import postprocess_image


class StreamFlow:
    """
    PeRFlow优化的流式扩散管道
    
    解决了StreamDiffusion的几个关键问题：
    1. 使用PeRFlow的原生scheduler.step()而不是自定义的scheduler_step_batch()
    2. 正确的VAE解码后处理，避免图像偏暗
    3. 针对PeRFlow的时间步优化
    """
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "full",
        use_original_scheduler: bool = True,  # 新增：强制使用原始调度器
        vae_decode_method: str = "normalize",  # 新增：VAE解码后处理方法
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)
        self.cfg_type = cfg_type
        self.use_original_scheduler = use_original_scheduler
        self.vae_decode_method = vae_decode_method

        # 为PeRFlow优化的批大小计算
        self.batch_size = frame_buffer_size
        self.t_list = t_index_list

        self.do_add_noise = do_add_noise

        # 相似图像过滤（从StreamDiffusion保留）
        self.similar_image_filter = False
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        # 保持原始调度器！不替换为LCMScheduler
        self.scheduler = pipe.scheduler
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.inference_time_ema = 0

        # 缓存变量
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.guidance_scale = 7.5

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 7.5,
        delta: float = 1.0,
    ) -> None:
        """
        为PeRFlow优化的准备函数
        """
        self.guidance_scale = guidance_scale
        
        # 编码提示词
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        
        self.prompt_embeds = prompt_embeds[0]
        self.negative_prompt_embeds = prompt_embeds[1] if do_classifier_free_guidance else None

        # 关键：使用PeRFlow的原生时间步设置
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)
        
        # 根据t_index_list选择子时间步
        self.sub_timesteps = []
        for t_idx in self.t_list:
            if t_idx < len(self.timesteps):
                self.sub_timesteps.append(self.timesteps[t_idx])
            else:
                print(f"警告: t_index {t_idx} 超出时间步范围 {len(self.timesteps)}")
                # 使用最后一个时间步
                self.sub_timesteps.append(self.timesteps[-1])
        
        self.sub_timesteps_tensor = torch.stack(self.sub_timesteps)
        
        print(f"PeRFlow时间步: {self.timesteps.tolist()}")
        print(f"选择的子时间步: {[t.item() for t in self.sub_timesteps]}")

    def predict_x0_perflow(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        """
        使用PeRFlow原生调度器的去噪预测
        """
        latents = x_t_latent
        
        # 修复的CFG条件：同时检查guidance_scale和cfg_type
        use_cfg = self.guidance_scale > 1.0 and self.cfg_type != "none"
        
        # 使用选择的时间步进行去噪
        for i, t in enumerate(self.sub_timesteps):
            # 修复的CFG处理
            if use_cfg:
                latent_model_input = torch.cat([latents] * 2)
                if self.negative_prompt_embeds is not None:
                    prompt_embeds = torch.cat([self.negative_prompt_embeds, self.prompt_embeds])
                else:
                    prompt_embeds = torch.cat([self.prompt_embeds, self.prompt_embeds])
            else:
                latent_model_input = latents
                prompt_embeds = self.prompt_embeds

            # UNet预测
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            # 修复的CFG后处理
            if use_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 关键：使用PeRFlow的原生step方法！
            step_result = self.scheduler.step(noise_pred, t, latents, return_dict=False)
            latents = step_result[0]

        return latents

    def decode_image_perflow(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        """
        为PeRFlow优化的VAE解码
        """
        with torch.no_grad():
            # VAE解码
            output_latent = self.vae.decode(
                x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
            )[0]
            
            # 使用优化的后处理方法
            output_latent = postprocess_image(
                output_latent, 
                output_type="pt", 
                denormalize_method=self.vae_decode_method
            )
        
        return output_latent

    @torch.no_grad()
    def __call__(
        self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None
    ) -> torch.Tensor:
        """
        图像到图像生成
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        if x is not None:
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            x_t_latent = self.encode_image(x)
        else:
            x_t_latent = torch.randn(
                (1, 4, self.latent_height, self.latent_width),
                device=self.device,
                dtype=self.dtype
            )
        
        # 使用PeRFlow优化的去噪
        x_0_pred_out = self.predict_x0_perflow(x_t_latent)
        x_output = self.decode_image_perflow(x_0_pred_out).detach().clone()

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        """
        文本到图像生成
        """
        # 生成初始噪声
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype
        )
        
        # 使用PeRFlow优化的去噪
        x_0_pred_out = self.predict_x0_perflow(x_t_latent)
        x_output = self.decode_image_perflow(x_0_pred_out).detach().clone()
        return x_output

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        图像编码（从StreamDiffusion保留）
        """
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        
        # 添加噪声（如果需要）
        if self.do_add_noise and len(self.sub_timesteps) > 0:
            noise = torch.randn_like(img_latent)
            img_latent = self.scheduler.add_noise(img_latent, noise, self.sub_timesteps[0])
        
        return img_latent

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        加载LoRA权重
        """
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False) -> None:
        """
        融合LoRA权重
        """
        self.pipe.fuse_lora(lora_scale=lora_scale, safe_fusing=safe_fusing)

    def enable_similar_image_filter(
        self, threshold: float = 0.98, max_skip_frame: int = 10
    ) -> None:
        """
        启用相似图像过滤（从StreamDiffusion保留）
        """
        self.similar_image_filter = True
        # 这里可以添加过滤逻辑

    def get_inference_time(self) -> float:
        """
        获取平均推理时间
        """
        return self.inference_time_ema