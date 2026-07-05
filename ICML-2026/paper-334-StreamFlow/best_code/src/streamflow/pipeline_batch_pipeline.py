"""
基于StreamDiffusion思想的PeRFlow流水线批量去噪

关键洞察：
- 使用latent buffer维护多个图像的中间状态
- 流水线处理：不同图像同时处于不同的去噪阶段
- 一次UNet调用处理所有阶段，但每个阶段对应不同的图像
"""

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


class PipelineBatchStreamFlow:
    """
    基于流水线的PeRFlow批量去噪
    
    原理：
    - 维护4个latent buffer，分别对应4个去噪阶段
    - 每次调用处理4张不同图像的不同阶段
    - 实现真正的流水线并行，避免破坏序列依赖
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
        use_pipeline_batch: bool = True,
        vae_decode_method: str = "normalize",
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
        self.vae_decode_method = vae_decode_method
        self.use_pipeline_batch = use_pipeline_batch

        self.t_list = t_index_list
        self.do_add_noise = do_add_noise

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        # 保持原始调度器
        self.scheduler = pipe.scheduler
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.inference_time_ema = 0

        # 缓存变量
        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.guidance_scale = 7.5

        # 流水线状态
        self.pipeline_initialized = False
        self.latent_buffer = None  # 存储不同阶段的latent
        self.step_counter = 0

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 7.5,
        delta: float = 1.0,
    ) -> None:
        """准备函数"""
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

        # 使用PeRFlow的原生时间步设置
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)
        
        # 根据t_index_list选择子时间步
        self.sub_timesteps = []
        for t_idx in self.t_list:
            if t_idx < len(self.timesteps):
                self.sub_timesteps.append(self.timesteps[t_idx])
            else:
                self.sub_timesteps.append(self.timesteps[-1])
        
        self.sub_timesteps_tensor = torch.stack(self.sub_timesteps)
        
        # 初始化流水线buffer
        if self.use_pipeline_batch and not self.pipeline_initialized:
            self._initialize_pipeline()
        
        print(f"PeRFlow时间步: {self.timesteps.tolist()}")
        print(f"选择的子时间步: {[t.item() for t in self.sub_timesteps]}")
        print(f"流水线批处理: {'✅ 启用' if self.use_pipeline_batch else '❌ 禁用'}")

    def _initialize_pipeline(self):
        """初始化流水线buffer"""
        # 创建随机的初始latent buffer
        # 这些buffer代表不同图像的不同阶段
        self.latent_buffer = torch.randn(
            (self.denoising_steps_num - 1, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype
        )
        self.pipeline_initialized = True
        print(f"✅ 流水线buffer初始化: {self.latent_buffer.shape}")

    def predict_x0_pipeline_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        """
        流水线批量去噪
        
        关键思想：
        - x_t_latent: 新图像的第1阶段
        - latent_buffer[0]: 某图像的第2阶段
        - latent_buffer[1]: 某图像的第3阶段  
        - latent_buffer[2]: 某图像的第4阶段
        
        一次UNet调用处理所有这些不同阶段
        """
        if not self.use_pipeline_batch or not self.pipeline_initialized:
            return self.predict_x0_perflow_original(x_t_latent)
        
        # 构建流水线批量输入
        if self.denoising_steps_num > 1:
            # 将新输入与buffer中的中间状态组合
            pipeline_latents = torch.cat([x_t_latent, self.latent_buffer], dim=0)
        else:
            pipeline_latents = x_t_latent
        
        # 时间步对应不同的阶段
        timestep_batch = self.sub_timesteps_tensor
        
        # CFG处理
        use_cfg = self.guidance_scale > 1.0 and self.cfg_type != "none"
        
        if use_cfg:
            latent_model_input = torch.cat([pipeline_latents, pipeline_latents], dim=0)
            timestep_input = torch.cat([timestep_batch, timestep_batch], dim=0)
            
            if self.negative_prompt_embeds is not None:
                batch_prompt_embeds = torch.cat([
                    self.negative_prompt_embeds.repeat(self.denoising_steps_num, 1, 1),
                    self.prompt_embeds.repeat(self.denoising_steps_num, 1, 1)
                ], dim=0)
            else:
                batch_prompt_embeds = self.prompt_embeds.repeat(self.denoising_steps_num * 2, 1, 1)
        else:
            latent_model_input = pipeline_latents
            timestep_input = timestep_batch
            batch_prompt_embeds = self.prompt_embeds.repeat(self.denoising_steps_num, 1, 1)
        
        # 🚀 流水线UNet调用：一次处理所有阶段
        with torch.inference_mode():
            noise_pred_batch = self.unet(
                latent_model_input,
                timestep_input,
                encoder_hidden_states=batch_prompt_embeds,
                return_dict=False,
            )[0]
        
        # CFG后处理
        if use_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred_batch.chunk(2)
            noise_pred_batch = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # 应用scheduler步骤到每个阶段
        processed_latents = []
        
        for i in range(self.denoising_steps_num):
            latent = pipeline_latents[i].unsqueeze(0)
            noise_pred = noise_pred_batch[i].unsqueeze(0)
            t = self.sub_timesteps[i]
            
            # 应用PeRFlow scheduler
            step_result = self.scheduler.step(noise_pred, t, latent, return_dict=False)
            processed_latents.append(step_result[0])
        
        # 更新buffer：向前移动流水线
        if self.denoising_steps_num > 1:
            # 输出是最后一个阶段的结果
            output = processed_latents[-1]
            
            # 更新buffer：前移流水线
            # 新的buffer[0] = 当前输入的第1步结果（将进入第2阶段）
            # 新的buffer[1] = 之前buffer[0]的结果（将进入第3阶段）
            # 新的buffer[2] = 之前buffer[1]的结果（将进入第4阶段）
            new_buffer = []
            
            # 当前输入处理后进入下一阶段
            new_buffer.append(processed_latents[0])
            
            # 之前buffer中的状态继续前进
            for i in range(self.denoising_steps_num - 2):
                new_buffer.append(processed_latents[i + 1])
            
            self.latent_buffer = torch.cat(new_buffer, dim=0)
            
            return output
        else:
            return processed_latents[0]

    def predict_x0_perflow_original(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        """原始逐步去噪方法"""
        latents = x_t_latent
        use_cfg = self.guidance_scale > 1.0 and self.cfg_type != "none"

        for i, t in enumerate(self.sub_timesteps):
            if use_cfg:
                latent_model_input = torch.cat([latents] * 2)
                if self.negative_prompt_embeds is not None:
                    prompt_embeds = torch.cat([self.negative_prompt_embeds, self.prompt_embeds])
                else:
                    prompt_embeds = torch.cat([self.prompt_embeds, self.prompt_embeds])
            else:
                latent_model_input = latents
                prompt_embeds = self.prompt_embeds

            # 🔧 TensorRT兼容：确保timestep是正确shape的tensor [batch_size]
            batch_size = latent_model_input.shape[0]
            if isinstance(t, torch.Tensor) and t.dim() == 0:
                # 标量tensor -> [batch_size] tensor
                timestep = t.unsqueeze(0).repeat(batch_size)
            elif isinstance(t, torch.Tensor):
                timestep = t
            else:
                # 如果是Python数值，转换为tensor
                timestep = torch.tensor([t] * batch_size, device=self.device, dtype=torch.long)

            with torch.inference_mode():
                noise_pred = self.unet(
                    latent_model_input,
                    timestep,  # 🔧 使用正确shape的timestep
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            if use_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            step_result = self.scheduler.step(noise_pred, t, latents, return_dict=False)
            latents = step_result[0]

        return latents

    def decode_image_perflow(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        """VAE解码"""
        # 确保VAE与输入在同一device/dtype，避免CPU/GPU混用
        if hasattr(self.vae, "to"):
            try:
                # 一些TensorRT VAE的to()返回None，避免将self.vae覆盖为None
                moved = self.vae.to(device=x_0_pred_out.device, dtype=x_0_pred_out.dtype)
                if moved is not None:
                    self.vae = moved
            except Exception:
                pass
        with torch.inference_mode():
            # Check if this is WSG VAE and handle dtype properly
            if hasattr(self.vae, 'decode_rgb_only_by_default'):
                # WSG VAE might use different dtype (e.g., float32)
                # Convert input to VAE's dtype if needed
                if hasattr(self.vae, 'dtype'):
                    vae_dtype = self.vae.dtype
                else:
                    try:
                        vae_dtype = next(self.vae.parameters()).dtype
                    except StopIteration:
                        vae_dtype = self.dtype
                
                if x_0_pred_out.dtype != vae_dtype:
                    x_0_pred_out = x_0_pred_out.to(dtype=vae_dtype)
                
                output_latent = self.vae.decode(
                    x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
                )[0]
                
                # WSG VAE outputs are already properly normalized
                # Keep the output in VAE's dtype
                return output_latent
            else:
                # Standard VAE decode
                output_latent = self.vae.decode(
                    x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
                )[0]
                
                # Standard VAE needs postprocessing
                output_latent = postprocess_image(
                    output_latent, 
                    output_type="pt", 
                    denormalize_method=self.vae_decode_method
                )
                
                return output_latent

    @torch.inference_mode()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        """文本到图像生成"""
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype
        )

        # 使用流水线批量去噪
        x_0_pred_out = self.predict_x0_pipeline_batch(x_t_latent)
        x_output = self.decode_image_perflow(x_0_pred_out).detach().clone()

        self.step_counter += 1

        return x_output

    @torch.inference_mode()
    def generate_latent(self, batch_size: int = 1) -> torch.Tensor:
        """只生成latent，不解码（用于批量VAE解码优化）"""
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype
        )

        # 使用流水线批量去噪
        x_0_pred_out = self.predict_x0_pipeline_batch(x_t_latent)

        self.step_counter += 1

        return x_0_pred_out

    @torch.inference_mode()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """批量解码latents（支持batch）"""
        return self.decode_image_perflow(latents).detach().clone()

    def get_inference_time(self) -> float:
        """获取平均推理时间"""
        return self.inference_time_ema
    
    def reset_pipeline(self):
        """重置流水线状态"""
        self.pipeline_initialized = False
        self.latent_buffer = None
        self.step_counter = 0
