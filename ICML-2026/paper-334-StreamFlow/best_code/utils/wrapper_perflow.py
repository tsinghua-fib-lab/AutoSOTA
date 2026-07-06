import gc
import os
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image

from src.streamflow import StreamFlow
from src.streamflow.image_utils import postprocess_image
from src.scheduler_perflow import PeRFlowScheduler

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PeRFlowWrapper:
    def __init__(
        self,
        model_id_or_path: str = "hansyan/perflow-sd15-dreamshaper",
        t_index_list: List[int] = [0, 1, 2, 3],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "txt2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        vae_decode_method: Literal["normalize", "dynamic", "clamp"] = "normalize",
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 5,
        acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
        do_add_noise: bool = True,
        use_tiny_vae: bool = False,
        cfg_type: Literal["none", "full", "self", "initialize"] = "full",
        seed: int = 2,
        num_inference_steps: int = 4,
        guidance_scale: float = 7.5,
    ):
        """
        PeRFlow专用包装器，优化了质量和性能
        
        Parameters
        ----------
        model_id_or_path : str
            PeRFlow模型路径，默认 "hansyan/perflow-sd15-dreamshaper"
        t_index_list : List[int]
            时间步索引列表，PeRFlow推荐 [0, 1, 2, 3]
        vae_decode_method : Literal["normalize", "dynamic", "clamp"]
            VAE解码后处理方法：
            - "normalize": 标准归一化 (image / 2 + 0.5) - 推荐
            - "dynamic": 动态范围归一化 - 最大动态范围但可能有色偏
            - "clamp": 直接截断 - 会偏暗，不推荐
        use_tiny_vae : bool
            是否使用TinyVAE加速，False使用原始VAE保证质量
        num_inference_steps : int
            推理步数，PeRFlow推荐4步
        guidance_scale : float
            引导缩放，PeRFlow推荐7.5
        """
        
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.output_type = output_type
        self.vae_decode_method = vae_decode_method
        self.warmup = warmup
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        print(f"🚀 初始化PeRFlowWrapper...")
        print(f"   模型: {model_id_or_path}")
        print(f"   模式: {mode}")
        print(f"   VAE解码方法: {vae_decode_method}")
        print(f"   时间步: {t_index_list}")
        print(f"   推理步数: {num_inference_steps}")
        
        # 加载管道
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id_or_path, 
            torch_dtype=dtype,
            use_safetensors=True,
        )
        
        # 设置PeRFlow调度器
        self.pipe.scheduler = PeRFlowScheduler.from_config(
            self.pipe.scheduler.config, 
            prediction_type="diff_eps", 
            num_time_windows=4
        )
        
        self.pipe.to(device, dtype)
        
        # 应用加速
        if acceleration == "xformers":
            self.pipe.enable_xformers_memory_efficient_attention()
        elif acceleration == "tensorrt":
            # 这里可以添加TensorRT加速逻辑
            print("⚠️  TensorRT加速暂未实现，使用xformers")
            self.pipe.enable_xformers_memory_efficient_attention()
        
        # 可选使用TinyVAE
        if use_tiny_vae:
            print("🔄 加载TinyVAE...")
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd"
            ).to(device, dtype)
            print("✅ TinyVAE加载完成")
        
        # 加载LoRA
        if lora_dict:
            for lora_name, lora_scale in lora_dict.items():
                self.pipe.load_lora_weights(lora_name, adapter_name=lora_name)
                print(f"✅ 加载LoRA: {lora_name} (scale: {lora_scale})")
        
        # 创建StreamFlow
        self.stream = StreamFlow(
            self.pipe,
            t_index_list=t_index_list,
            torch_dtype=dtype,
            width=width,
            height=height,
            do_add_noise=do_add_noise,
            frame_buffer_size=frame_buffer_size,
            cfg_type=cfg_type,
            use_original_scheduler=True,
            vae_decode_method=vae_decode_method,
        )
        
        print("✅ PeRFlowWrapper初始化完成")
    
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ):
        """
        准备推理
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
            
        self.stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # 预热
        print(f"🔥 预热中... ({self.warmup}次)")
        if self.mode == "txt2img":
            for _ in range(self.warmup):
                self.stream.txt2img()
        else:
            # img2img模式的预热需要输入图像
            dummy_image = torch.randn(1, 3, 512, 512).to(self.device, self.dtype)
            for _ in range(self.warmup):
                self.stream(dummy_image)
        
        print("✅ 预热完成")
    
    def __call__(
        self, 
        image: Optional[Union[str, Image.Image, torch.Tensor, np.ndarray]] = None
    ) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        执行推理
        """
        if self.mode == "txt2img":
            result = self.stream.txt2img()
        else:
            if image is None:
                raise ValueError("img2img模式需要输入图像")
            
            # 处理不同类型的输入图像
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            result = self.stream(image)
        
        # 后处理输出格式
        if self.output_type == "pt":
            return result
        elif self.output_type == "np":
            return result.cpu().numpy()
        elif self.output_type == "pil":
            # 转换为PIL
            result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result_np = (result_np * 255).astype(np.uint8)
            return Image.fromarray(result_np)
        elif self.output_type == "latent":
            # 返回潜在表示需要修改StreamFlow
            raise NotImplementedError("latent输出类型暂未实现")
    
    def txt2img(self) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        文本到图像生成
        """
        return self.__call__()
    
    def img2img(self, image) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        """
        图像到图像生成
        """
        return self.__call__(image)
    
    def batch_generate(
        self,
        num_images: int = 1,
        show_progress: bool = True,
    ) -> List[Union[Image.Image, torch.Tensor, np.ndarray]]:
        """
        批量生成
        """
        results = []
        for i in range(num_images):
            result = self.__call__()
            results.append(result)
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"📸 已生成 {i + 1}/{num_images} 张图像")
        
        return results
    
    def change_vae_decode_method(self, method: Literal["normalize", "dynamic", "clamp"]):
        """
        动态修改VAE解码方法
        """
        self.vae_decode_method = method
        self.stream.vae_decode_method = method
        print(f"🔄 VAE解码方法已修改为: {method}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取性能统计
        """
        return {
            "inference_time_ema": self.stream.get_inference_time(),
        }