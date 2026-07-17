"""
BitNet 量化+剪枝 融合脚本
直接从 BF16 稠密权重出发，计算 Scale，量化为 Ternary，然后应用 Z:L 稀疏掩码，最后输出指定格式。

优势：
1. Scale 计算基于原始稠密权重，避免了先剪枝导致的 Scale 漂移问题。
2. 利用量化过程天然产生的零，减少强制剪枝带来的精度损失。
3. 支持多种输出格式 (8I, 16F, 16BF, E4M3, E5M2, E2M1)。

输入：
- 标准的PyTorch bf16检查点文件

输出（按选择的dtype追加后缀，不覆盖原文件）：
- _16F.pt   : FP16存储的三元权重 (-1,0,1)
- _16BF.pt  : BF16存储的三元权重 (-1,0,1)
- _8I.pt    : INT8存储的三元权重 (-1,0,1)
- _E4M3.pt  : FP8 E4M3存储的三元权重 (-1,0,1)
- _E5M2.pt  : FP8 E5M2存储的三元权重 (-1,0,1)
- _E2M1.pt  : FP4 E2M1三元权重（4bit打包）

- _pruned_Z_L_mode_dtype.pt: 保存量化+剪枝后的模型，其中 Z, L, mode, dtype 为参数值

magnitude模式：基于权重的绝对值，选择重要性最低的元素进行剪枝。
random模式：基于随机选择。Z:L约束仅用于控制最终的稀疏度。

对于 K 无法被 L 整除的情况，会自动填充0 最后再去掉padding，确保分组后每组元素个数为 L。


使用示例（2:8剪枝，magnitude模式，INT8输出）：
python convert_checkpoint_quant_prune_to_Z_L.py --input ./checkpoints/model_state.pt --Z 2 --L 8 --mode magnitude --dtype 8I

python convert_checkpoint_quant_prune_to_Z_L.py --input ./checkpoints/model_state.pt --Z 2 --L 6 --mode magnitude --dtype E5M2

"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import model_allint8 as fast

def _validate_params(z: int, l: int) -> None:
    if z <= 0 or l <= 0:
        raise ValueError("Z和L必须为正整数")
    #if l % z != 0:
    #    raise ValueError("L必须能被Z整除，例如2:8或1:4")
    if z > l:
        raise ValueError("Z不能大于L")

def to_fp4_e2m1_pack(ternary_int8: torch.Tensor):
    """
    将三元int8 (-1,0,1) 编码为FP4 E2M1 (4bit)，并打包。
    按行独立打包，处理奇数列的情况。
    """
    device = ternary_int8.device
    lut = torch.tensor([0xE, 0x0, 0x6], device=device, dtype=torch.uint8)
    idx = (ternary_int8.to(torch.long) + 1) # 先转 long 再加，或者加完转 long 都可以
    encoded = lut[idx]
    
    rows, cols = encoded.shape
    
    # 如果列数是奇数，先补一列 0
    if cols % 2 != 0:
        encoded = torch.cat([encoded, torch.zeros((rows, 1), device=device, dtype=torch.uint8)], dim=1)
    
    # 现在的 shape 是 (rows, even_cols)，直接在该维度上 view 和打包
    # view 成 (rows, cols//2, 2) -> 最后一维是成对的两个元素
    pairs = encoded.view(rows, -1, 2)
    
    # 打包：第一个元素在高4位，第二个元素在低4位
    packed = (pairs[:, :, 0] << 4) | (pairs[:, :, 1])
    
    return packed

def _build_prune_mask(ternary_grouped: torch.Tensor, original_grouped: torch.Tensor, z: int, l: int, mode: str) -> torch.Tensor:
    """
    基于Ternary权重的零分布，利用Original权重的幅度信息，生成剪枝掩码。
    """
    # 1. 统计Ternary中的非零情况
    nonzero_mask = (ternary_grouped != 0)
    nonzero_count = nonzero_mask.sum(dim=1)

    # 2. 计算需要额外剪掉的数量
    # 目标：每组最多保留 L-Z 个非零值
    prune_count = (nonzero_count - (l - z)).clamp(min=0)

    max_prune = int(z)
    if max_prune == 0 or prune_count.max() == 0:
        return torch.zeros_like(ternary_grouped, dtype=torch.bool)

    # 3. 确定剪枝候选位置 (基于原始权重的重要性)
    if mode == "magnitude":
        importance = original_grouped.abs()
        # 关键：已经为0的位置（在Ternary中）不需要再剪，设为无限大防止被选中
        importance[~nonzero_mask] = float("inf") 
        
        # 选出重要性最小的 max_prune 个位置
        _, candidate_idx = importance.topk(k=max_prune, dim=1, largest=False)
    else:  # random
        rand_scores = torch.rand(ternary_grouped.shape, device=ternary_grouped.device, dtype=torch.float32)
        rand_scores[~nonzero_mask] = float("inf")
        _, candidate_idx = rand_scores.topk(k=max_prune, dim=1, largest=False)

    # 4. 生成掩码
    prune_mask = torch.zeros_like(ternary_grouped, dtype=torch.bool)
    for k in range(max_prune):
        # 只对需要剪掉 > k 个元素的组进行操作
        needs = prune_count > k
        if needs.any():
            rows = torch.where(needs)[0]
            cols = candidate_idx[needs, k]
            prune_mask[rows, cols] = True

    return prune_mask

def quant_and_prune_tensor(weight: torch.Tensor, z: int, l: int, mode: str, dtype: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    核心逻辑：
    1. 计算Scale (基于Dense Weight)
    2. Quantize -> Ternary
    3. Prune Ternary (基于Z:L约束)
    4. Convert Format
    """
    # 1. Scale & Ternarize
    s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    ternary = (weight * s).round().clamp(-1, 1)
    
    # 2. Prune
    rows, cols = ternary.shape
    
    # 计算需要补多少列才能被 L 整除
    pad_cols = (l - (cols % l)) % l
    
    if pad_cols > 0:
        ternary_padded = torch.cat([ternary, torch.zeros((rows, pad_cols), device=ternary.device, dtype=ternary.dtype)], dim=1)
        original_padded = torch.cat([weight, torch.zeros((rows, pad_cols), device=weight.device, dtype=weight.dtype)], dim=1)
    else:
        ternary_padded = ternary
        original_padded = weight
        
    # Reshape to (-1, l)
    # 先按行 padding，再 reshape，保证每一组都在同一行内，不跨行
    ternary_grouped = ternary_padded.reshape(-1, l)
    original_grouped = original_padded.reshape(-1, l)
    
    mask = _build_prune_mask(ternary_grouped, original_grouped, z, l, mode)
    
    if mask.any():
        ternary_grouped.masked_fill_(mask, 0)
        
    # Reshape back to (rows, padded_cols)
    ternary_final = ternary_grouped.reshape(rows, -1)
    
    # Remove padding if added
    if pad_cols > 0:
        ternary_final = ternary_final[:, :cols]

    # 3. Convert Format
    scale = (1.0 / s).to(torch.bfloat16).reshape(1)
    
    if dtype == "8I":
        return ternary_final.to(torch.int8), scale
    elif dtype == "16F":
        return ternary_final.to(torch.float16), scale
    elif dtype == "16BF":
        return ternary_final.to(torch.bfloat16), scale
    elif dtype == "E4M3":
        if not hasattr(torch, "float8_e4m3fn"): raise RuntimeError("No float8_e4m3fn")
        return ternary_final.to(torch.float8_e4m3fn), scale
    elif dtype == "E5M2":
        if not hasattr(torch, "float8_e5m2"): raise RuntimeError("No float8_e5m2")
        return ternary_final.to(torch.float8_e5m2), scale
    elif dtype == "E2M1":
        return to_fp4_e2m1_pack(ternary_final.to(torch.int8)), scale
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

@torch.inference_mode()
def convert_checkpoint(
    *,
    input_path: str,
    z: int = 2,
    l: int = 8,
    mode: str = "magnitude",
    dtype: str = "8I",
    seed: int = None,
) -> None:
    
    _validate_params(z, l)
    if seed is not None:
        torch.manual_seed(seed)

    # 加载模型配置参数
    config = fast.ModelArgs()
    print(f"Model config {config.__dict__}")
    print(f"Config: Z={z}, L={l}, mode={mode}, dtype={dtype}")

    merged_result = torch.load(input_path, map_location="cpu", mmap=True)
    print(f"Loaded checkpoint from {input_path}")

    result = {}
    zero = torch.zeros(1).to(torch.bfloat16)

    for key, value in merged_result.items():
        if not torch.is_tensor(value) or value.dim() != 2:
            # 这里避免使用value.clone从而保证权重共享，同一块storage只写一次
            result[key] = value
            continue

        # 仅处理目标层
        if 'wqkv' in key:
            wq = value[:config.dim]
            wk = value[config.dim:config.dim // config.n_heads * config.n_kv_heads + config.dim]
            wv = value[config.dim // config.n_heads * config.n_kv_heads + config.dim:]
            
            wq_w, wq_s = quant_and_prune_tensor(wq, z, l, mode, dtype)
            wk_w, wk_s = quant_and_prune_tensor(wk, z, l, mode, dtype)
            wv_w, wv_s = quant_and_prune_tensor(wv, z, l, mode, dtype)
            
            result[key] = torch.cat([wq_w, wk_w, wv_w], dim=0)
            result[key.replace('weight', 'weight_scale')] = torch.cat([wq_s, wq_s, wq_s, wq_s, wk_s, wv_s], dim=0)

        elif 'w13' in key:
            w1 = value[:config.ffn_dim]
            w3 = value[config.ffn_dim:]
            
            w1_w, w1_s = quant_and_prune_tensor(w1, z, l, mode, dtype)
            w3_w, w3_s = quant_and_prune_tensor(w3, z, l, mode, dtype)
            
            result[key] = torch.cat([w1_w, w3_w], dim=0)
            result[key.replace('weight', 'weight_scale')] = torch.cat([w1_s, w3_s, zero, zero, zero, zero], dim=0)

        elif 'w2' in key or 'wo' in key:
            w, s = quant_and_prune_tensor(value, z, l, mode, dtype)
            result[key] = w
            result[key.replace('weight', 'weight_scale')] = torch.cat([s, zero, zero, zero, zero, zero], dim=0)
            
        else:
            # 这里避免使用value.clone从而保证权重共享，同一块storage只写一次
            result[key] = value

    output_dir = os.path.dirname(input_path)
    stem = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{stem}_{dtype}_pruned_{z}_{l}_{mode}.pt")
    
    print(f"Saving to {output_path}")
    torch.save(result, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BitNet 量化+剪枝融合脚本")
    default_input = Path(__file__).resolve().parent / "checkpoints" / "model_state.pt"
    parser.add_argument("--input", type=str, default=str(default_input))
    parser.add_argument("--Z", type=int, default=2)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--mode", type=str, default="magnitude", choices=["magnitude", "random"])
    parser.add_argument("--dtype", type=str, default="8I", choices=['16F','16BF','8I','E4M3','E5M2','E2M1'])
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    convert_checkpoint(
        input_path=args.input,
        z=args.Z,
        l=args.L,
        mode=args.mode,
        dtype=args.dtype,
        seed=args.seed,
    )
