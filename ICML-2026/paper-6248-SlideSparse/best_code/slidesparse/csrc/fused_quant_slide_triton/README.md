# SlideSparse Fused Quant + Slide Triton Kernel

## 1. 功能概述

本 kernel 实现了 **Fused Quantization + SlideSparse Slide** 操作，将 per-token 量化和滑动窗口重排融合到单个 Triton kernel 中执行。

### 输入/输出

| 项目 | 描述 |
|------|------|
| **输入** | `[M, K]` BF16/FP16/FP32 tensor (行主序, contiguous) |
| **输出** | `[M_pad, K_out_pad]` FP8E4M3 或 INT8 tensor |
| **Scale** | `[M_pad]` FP32 per-token scale |

### 稀疏格式

SlideSparse 是一种结构化稀疏格式，通过滑动窗口方式压缩权重矩阵：

| 参数 | L=6 (2:6) | L=8 (2:8) | L=10 (2:10) |
|------|-----------|-----------|-------------|
| NUM_WINDOWS | 2 | 3 | 4 |
| expand_ratio | 1.333x | 1.500x | 1.600x |

**Slide 操作示例 (L=8)**:
```
输入:  [0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15, ...]
        \____group0____/  \_____group1_______/

输出:  [0,1,2,3], [2,3,4,5], [4,5,6,7], [8,9,10,11], [10,11,12,13], [12,13,14,15], ...
       \_win0_/   \_win1_/   \_win2_/   \__win0__/    \___win1___/    \___win2___/
       \_______group0_______/           \_____________group1______________/

每组 8 元素 → 3 个 window × 4 元素 = 12 输出元素 (expand 1.5x)
```

---

## 2. 核心设计

### 2.1 两遍扫描架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Kernel Architecture                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Grid: (M,)  ─ 每行一个 program                                      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Pass 1: Absmax 计算                                          │   │
│  │   for k in range(0, K_in_padded, BLOCK_K):                   │   │
│  │       load K 方向一块 → 计算局部 max(abs(x))                  │   │
│  │   scale = max(absmax / QMAX, MIN_SCALE)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Pass 2: Output-Oriented Quant + Slide (核心优化)              │   │
│  │   for out_start in range(0, total_out, BLOCK_OUT):           │   │
│  │       g = offs_out // NUM_WINDOWS   # group index            │   │
│  │       w = offs_out % NUM_WINDOWS    # window index           │   │
│  │       base_in = g * L + 2 * w       # 输入起始位置            │   │
│  │       load 4 elements → quantize → pack int32 → store        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Output-Oriented 优化（关键改进）

**原始设计（嵌套循环）:**
```python
for g_start in range(0, num_groups, BLOCK_GROUPS):
    for w in tl.static_range(NUM_WINDOWS):  # 编译期展开
        # Load → Quant → Store
```

**优化设计（单层循环）:**
```python
for out_start in range(0, total_out, BLOCK_OUT):
    g = offs_out // NUM_WINDOWS    # 运行时 div
    w = offs_out % NUM_WINDOWS     # 运行时 mod
    base_in = g * L + 2 * w        # 直接计算输入位置
    # Load → Quant → Store
```

**优势:**
- 消除嵌套循环的静态展开开销
- 更细粒度的并行控制
- 更好的输出访存合并
- 中等 M (128-4096) 平均 **12% 加速**，最佳可达 **21%**

---

## 3. 关键实现细节

### 3.1 边界处理（无 F.pad）

```python
# Python wrapper: 计算逻辑维度，但不实际 pad
K_in_padded = ((K_in_orig + L - 1) // L) * L  # 向上取整到 L 的倍数

# Kernel: 用 mask 处理越界
x0 = tl.load(x_row + base_in + 0,
             mask=mask_out & ((base_in + 0) < K_in_orig),  # 边界检查
             other=0.0)  # 越界读 0
```

**为什么不用 F.pad:** `F.pad` 会触发额外的内存拷贝，开销可能比整个量化 kernel 还大。

### 3.2 打包输出

每个输出位置写入一个 `int32`，包含 4 个量化值：

```python
# Pack 4 x FP8/INT8 → 1 x int32 (little-endian)
packed = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
tl.store(out_row32 + offs_out, packed, mask=mask_out)
```

### 3.3 输出 Padding

```python
# cuSPARSELt 要求 K 方向 32 对齐
K_out_padded = ((K_out + 31) // 32) * 32

# 算法要求 M 方向 16 对齐
M_padded = ((M + 15) // 16) * 16

# Padding 区域: out = 0, scale = 1.0 (避免 dequant 除零)
out = torch.zeros(M_padded, K_out_padded, ...)
scale = torch.ones(M_padded, ...)
```

---

## 4. 配置参数

### 4.1 自动配置策略

```python
def _get_config(M: int, K: int) -> Tuple[int, int, int]:
    """Returns: (BLOCK_OUT, num_warps, num_stages)"""
    if M <= 64:
        return 128, 4, 2    # 小 batch: 保守配置
    elif M <= 1024:
        return 256, 8, 2    # 中等 batch: 平衡配置
    else:
        return 512, 8, 2    # 大 batch: 高并行度
```

### 4.2 参数说明

| 参数 | 含义 | 影响 |
|------|------|------|
| `BLOCK_OUT` | Pass 2 每次处理的输出 int32 数量 | 并行度 vs 寄存器压力 |
| `BLOCK_K` | Pass 1 每次处理的 K 元素数量 | 循环次数 vs 向量化效率 |
| `num_warps` | Warp 数量 | SM 利用率 |
| `num_stages` | Pipeline stages | 访存隐藏 |

---

## 5. API 使用

### 5.1 FP8 量化

```python
from basic_quant_slide_triton import quant_slide_fp8_triton

# 输入: [M, K] bf16
x = torch.randn(1024, 2560, dtype=torch.bfloat16, device='cuda')

# 输出: out [M_pad, K_out_pad] fp8, scale [M_pad] fp32
out, scale = quant_slide_fp8_triton(x, L=8)

# 查看维度
print(f"Input:  {x.shape}")           # [1024, 2560]
print(f"Output: {out.shape}")          # [1024, 3840] (expand 1.5x, 32-aligned)
print(f"Scale:  {scale.shape}")        # [1024]
```

### 5.2 INT8 量化

```python
from basic_quant_slide_triton import quant_slide_int8_triton

out, scale = quant_slide_int8_triton(x, L=10)  # 2:10 sparsity
```

### 5.3 辅助函数

```python
from basic_quant_slide_triton import _compute_output_k, _get_num_windows

# 计算输出维度
K_in_padded, K_out, num_groups = _compute_output_k(K_in=2560, L=8)
# K_in_padded=2560, K_out=3840, num_groups=320

# 计算窗口数
num_windows = _get_num_windows(L=8)  # 3
```

---

## 6. 性能特性

### 6.1 测试结果 (RTX 5080, FP8, L=10)

| M 范围 | 相对 Basic | 说明 |
|--------|------------|------|
| Small (M ≤ 128) | 1.03x | 小幅提升 |
| **Medium (128 < M ≤ 4096)** | **1.12x** | **主要收益区间** |
| Large (M > 4096) | 1.01x | 接近带宽上限 |

### 6.2 最佳案例

- **M=1024, K=6912, L=10**: 达到 **1.21x** 加速
- 这正是推理 decode 阶段的典型 batch size

### 6.3 相对 memcpy baseline

由于 slide 操作本身会扩展输出（L=10 时 1.6x），相比纯 memcpy baseline 约为 **0.85x**，这接近理论上限。

---

## 7. 文件结构

```
fused_quant_slide_triton/
├── basic_quant_slide_triton.py     # 本文件：优化后的 kernel 实现
├── autotune_autogen_quant_slide.py # Autotune 脚本，生成固定配置 kernel
├── run_benchmark.py                # 性能测试脚本
├── benchmark_optimization.py       # 优化版本对比脚本
├── README.md                       # 本文档
├── build/                          # Autotune 生成的 kernel
│   └── {hw_dir}/
│       └── quant_slide_tuned_*.py
└── _archived/                      # 归档的旧实现和实验代码
    ├── basic_quant_slide_triton.py # 原始实现（Group-Window 嵌套循环）
    └── README.md
```

---

## 8. 与其他组件的集成

### 8.1 与 autotune 的关系

`autotune_autogen_quant_slide.py` 基于本文件的 kernel 进行参数搜索，生成针对特定模型的固定配置版本：

```bash
python3 autotune_autogen_quant_slide.py --model Llama3.2-1B-FP8
```

生成的文件保存在 `build/{hw_dir}/quant_slide_tuned_{model}.py`。

### 8.2 与 vLLM 的集成

通过 `slidesparse/core/kernels.py` 加载：

```python
# kernels.py 会自动搜索:
# 1. 优先使用 tuned kernel (build 目录)
# 2. 回退到 basic kernel (本文件)
from slidesparse.core.kernels import quant_slide_fp8_kernel
```

---

## 9. 注意事项

### 9.1 Triton 版本要求

- 需要 Triton >= 3.0 以支持 `tl.float8e4nv`
- 不同 Triton 版本可能生成不同的 PTX，建议固定版本

### 9.2 GPU 兼容性

| GPU | FP8 支持 | 说明 |
|-----|----------|------|
| A100 (sm_80) | ❌ | 需要 FP16 fallback |
| H100 (sm_90) | ✅ | 原生支持 |
| RTX 4090 (sm_89) | ✅ | 原生支持 |
| RTX 5080 (sm_120) | ✅ | 原生支持 |

### 9.3 数值精度

- FP8E4M3: 动态范围 [-448, 448]，适合激活值
- INT8: 动态范围 [-128, 127]，需要 rounding
- Scale 使用 FP32 以保证精度

---

## 10. 参考

- [Triton Documentation](https://triton-lang.org/)
- [vLLM Quantization Kernels](https://github.com/vllm-project/vllm)
- SlideSparse 论文/技术报告
