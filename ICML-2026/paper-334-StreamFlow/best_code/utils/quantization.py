"""
GPU兼容的VAE INT8量化工具
"""

import torch
import torch.nn as nn
import os
from diffusers import AutoencoderTiny


class QuantizedLayer(nn.Module):
    """
    量化层：存储INT8权重，推理时动态反量化
    """
    def __init__(self, original_layer, weight_int8, scale):
        super().__init__()
        self.layer_type = type(original_layer)

        # 存储INT8权重和scale
        self.register_buffer('weight_int8', weight_int8)
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

        # 保存bias
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.data)
        else:
            self.bias = None

        # 保存层参数
        if isinstance(original_layer, (nn.Conv2d, nn.ConvTranspose2d)):
            self.stride = original_layer.stride
            self.padding = original_layer.padding
            self.dilation = original_layer.dilation
            self.groups = original_layer.groups
            if isinstance(original_layer, nn.ConvTranspose2d):
                self.output_padding = original_layer.output_padding

        # 缓存反量化后的权重（避免重复转换）
        self._cached_weight = None
        self._cached_dtype = None

    def forward(self, x):
        """前向：使用缓存的FP16权重"""
        # 检查是否需要重新反量化
        if self._cached_weight is None or self._cached_dtype != x.dtype:
            # 第一次或dtype改变时反量化
            self._cached_weight = self.weight_int8.to(x.dtype) * self.scale
            self._cached_dtype = x.dtype

        weight = self._cached_weight

        # 根据层类型计算
        if self.layer_type == nn.Conv2d:
            return nn.functional.conv2d(
                x, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups
            )
        elif self.layer_type == nn.ConvTranspose2d:
            return nn.functional.conv_transpose2d(
                x, weight, self.bias,
                self.stride, self.padding, self.output_padding,
                self.groups, self.dilation
            )
        elif self.layer_type == nn.Linear:
            return nn.functional.linear(x, weight, self.bias)
        else:
            raise NotImplementedError(f"Unsupported layer: {self.layer_type}")


class Int8QuantizedVAE(nn.Module):
    """
    INT8量化VAE（GPU兼容）
    decoder权重量化为INT8，推理时动态反量化
    """
    def __init__(self, vae):
        super().__init__()
        self.config = vae.config

        # 复制各个组件
        self.decoder = vae.decoder
        if hasattr(vae, 'encoder'):
            self.encoder = vae.encoder
        if hasattr(vae, 'quant_conv'):
            self.quant_conv = vae.quant_conv
        if hasattr(vae, 'post_quant_conv'):
            self.post_quant_conv = vae.post_quant_conv

    def decode(self, latent, **kwargs):
        """解码latent为图像"""
        if hasattr(self, 'post_quant_conv'):
            latent = self.post_quant_conv(latent)
        image = self.decoder(latent)
        return (image,)

    def encode(self, image, **kwargs):
        """编码图像为latent"""
        if hasattr(self, 'encoder'):
            latent = self.encoder(image)
            if hasattr(self, 'quant_conv'):
                latent = self.quant_conv(latent)
            return latent
        raise NotImplementedError("Encoder not available")

    def forward(self, *args, **kwargs):
        return self.decode(*args, **kwargs)


def quantize_vae_decoder(vae):
    """
    量化VAE的decoder为INT8

    参数:
        vae: AutoencoderTiny或AutoencoderKL

    返回:
        Int8QuantizedVAE
    """
    quantized_vae = Int8QuantizedVAE(vae)

    # 量化decoder
    quantized_vae.decoder = _quantize_module(vae.decoder)

    return quantized_vae


def _quantize_module(module):
    """递归量化模块中的Conv和Linear层"""
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            # 量化这一层
            setattr(module, name, _quantize_layer(child))
        else:
            # 递归处理子模块
            _quantize_module(child)
    return module


def _quantize_layer(layer):
    """量化单个层的权重为INT8"""
    weight = layer.weight.data

    # 计算量化scale（per-tensor）
    weight_max = weight.abs().max()
    scale = weight_max / 127.0  # INT8范围 [-127, 127]

    # 量化权重
    weight_int8 = torch.clamp(
        torch.round(weight / scale), -127, 127
    ).to(torch.int8)

    return QuantizedLayer(layer, weight_int8, scale)


def load_quantized_tinyvae(device, dtype):
    """
    加载预量化的TinyVAE

    参数:
        device: 目标设备 (cuda/cpu)
        dtype: 数据类型 (torch.float16等)

    返回:
        量化后的VAE模型
    """
    model_path = "models/tinyvae_int8.pth"

    if not os.path.exists(model_path):
        print(f"❌ 量化模型不存在: {model_path}")
        print(f"   请先运行: python -m utils.quantization")
        print(f"   回退到在线量化...")

        # 在线量化
        print("🔬 在线量化TinyVAE...")
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
        quantized_vae = quantize_vae_decoder(vae)
        quantized_vae = quantized_vae.to(device=device, dtype=dtype)
        print("   ✅ 在线量化完成")
        return quantized_vae

    print(f"📦 加载预量化TinyVAE: {model_path}")

    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # 创建VAE并量化结构（这样结构就和保存的匹配）
    temp_vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
    quantized_vae = quantize_vae_decoder(temp_vae)  # 量化结构

    # 加载量化权重
    quantized_vae.load_state_dict(checkpoint['model_state_dict'])

    # 移到目标设备
    quantized_vae = quantized_vae.to(device=device, dtype=dtype)

    print(f"   ✅ INT8量化VAE加载成功")
    print(f"   精度: INT8权重 + {dtype}推理")

    return quantized_vae


def quantize_and_save_tinyvae():
    """
    离线量化TinyVAE并保存
    运行: python -m utils.quantization
    """
    print("=" * 60)
    print("TinyVAE INT8 量化工具")
    print("=" * 60)

    # 加载原始TinyVAE
    print("\n📦 加载TinyVAE...")
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
    vae.eval()

    print(f"   模型: {type(vae)}")
    print(f"   参数量: {sum(p.numel() for p in vae.parameters()):,}")

    # 量化
    print("\n🔬 开始量化...")
    quantized_vae = quantize_vae_decoder(vae)

    # 测试量化模型
    print("\n🧪 测试量化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantized_vae = quantized_vae.to(device=device, dtype=torch.float16)

    test_latent = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)

    with torch.no_grad():
        output = quantized_vae.decode(test_latent / quantized_vae.config.scaling_factor)

    image = output[0] if isinstance(output, tuple) else output
    print(f"   ✅ 测试通过")
    print(f"   输入: {test_latent.shape}")
    print(f"   输出: {image.shape}")
    print(f"   范围: [{image.min():.3f}, {image.max():.3f}]")

    # 对比质量
    print("\n📊 质量对比...")
    vae_original = vae.to(device=device, dtype=torch.float16)
    with torch.no_grad():
        output_original = vae_original.decode(test_latent / vae.config.scaling_factor)

    # 提取图像tensor
    if hasattr(output_original, 'sample'):
        image_original = output_original.sample
    elif isinstance(output_original, tuple):
        image_original = output_original[0]
    else:
        image_original = output_original

    mse = ((image - image_original) ** 2).mean().item()
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(mse))
    print(f"   MSE: {mse:.6f}")
    print(f"   PSNR: {psnr:.2f} dB")
    print(f"   {'✅ 质量损失可接受' if mse < 0.001 else '⚠️  质量损失较大'}")

    # 显存对比
    print("\n💾 显存对比...")
    original_size = sum(p.numel() * p.element_size() for p in vae.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_vae.parameters())

    print(f"   原始: {original_size / 1024 / 1024:.2f} MB")
    print(f"   量化: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"   节省: {(1 - quantized_size / original_size) * 100:.1f}%")

    # 保存
    os.makedirs("models", exist_ok=True)
    output_path = "models/tinyvae_int8.pth"
    print(f"\n💾 保存量化模型到: {output_path}")

    # 移回CPU保存
    quantized_vae_cpu = quantized_vae.cpu()

    torch.save({
        'model_state_dict': quantized_vae_cpu.state_dict(),
        'config': vae.config,
        'quantization_info': {
            'method': 'per-tensor INT8',
            'target': 'decoder weights only',
            'dtype': 'int8',
            'mse': mse,
            'psnr': psnr.item(),
        }
    }, output_path)

    print(f"   ✅ 保存成功")
    print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 60)
    print("✅ 量化完成！")
    print("=" * 60)
    print(f"\n使用方法:")
    print(f"1. 在test_demo_gen.py中设置: USE_INT8_VAE = True")
    print(f"2. 运行测试，自动加载 {output_path}")
    print()


# 允许作为脚本运行
if __name__ == "__main__":
    quantize_and_save_tinyvae()
