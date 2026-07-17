# SlideSparse Fused Quant + Slide Kernel 实现文档

## 1. 概述

SlideSparse 是一种结构化稀疏格式，通过滑动窗口方式压缩权重矩阵。本项目实现了 **Fused Quantization + Slide** Triton kernel，将量化和滑动窗口重排融合到一个 kernel 中执行。

### 稀疏格式定义

| 参数 | L=6 (2:6) | L=8 (2:8) |
|------|-----------|-----------|
| N (半窗口) | 3 | 4 |
| NUM_WINDOWS | 2 | 3 |
| expand_ratio | 1.333x | 1.500x |

**Slide 操作示例 (L=6)**:
```
输入:  [0,1,2,3,4,5, 6,7,8,9,10,11, ...]
        \_group0_/   \__group1__/

输出:  [0,1,2,3], [2,3,4,5], [6,7,8,9], [8,9,10,11], ...
       \window0/ \window1/  \window0/  \window1/
```

---

## 2. 文件结构

```
slidesparse/kernels/slide_kernals/
├── codegen_unified.py           # 统一的 kernel 代码生成器
├── autotune_slide_kernels.py    # Autotune 脚本，生成固定配置 kernel
├── slide_L6_int8.py             # 生成的 L=6 INT8 kernel (带 autotune)
├── slide_L6_fp8.py              # 生成的 L=6 FP8 kernel (带 autotune)
├── slide_L8_int8.py             # 生成的 L=8 INT8 kernel (带 autotune)
├── slide_L8_fp8.py              # 生成的 L=8 FP8 kernel (带 autotune)
├── slide_L6_int8_autotuned.py   # 固定配置版本 (autotune 生成)
├── slide_L6_fp8_autotuned.py    # 固定配置版本 (autotune 生成)
├── slide_L8_int8_autotuned.py   # 固定配置版本 (autotune 生成)
├── slide_L8_fp8_autotuned.py    # 固定配置版本 (autotune 生成)
├── triton_quant_int8.py         # Triton 纯量化 baseline (INT8)
├── triton_quant_fp8.py          # Triton 纯量化 baseline (FP8)
├── run_benchmark.py             # 性能测试脚本
└── README.md                    # 本文档
```

---

## 3. Kernel 生成方法

### 3.1 使用 codegen_unified.py 生成 kernel

```bash
cd /root/vllmbench/slidesparse/kernels/slide_kernals

# 生成 L=6 kernels
python3 codegen_unified.py 6 --int8 -o slide_L6_int8.py
python3 codegen_unified.py 6 --fp8 -o slide_L6_fp8.py

# 生成 L=8 kernels
python3 codegen_unified.py 8 --int8 -o slide_L8_int8.py
python3 codegen_unified.py 8 --fp8 -o slide_L8_fp8.py
```

### 3.2 Codegen 统一设计

所有 L 值共享相同的 autotune 配置和代码结构：

```python
# 统一的 autotune 配置 (BLOCK_GROUPS 为 2 的幂次)
BLOCK_GROUPS: [64, 128, 256, 512, 1024, 2048]
num_warps: [4, 8]
num_stages: [2, 3, 4, 5]

# Pass 1 统一使用 L_PAD (向上取整到 2 的幂次)
L_PAD = 1 << (L - 1).bit_length()  # L=6 -> 8, L=8 -> 8
BLOCK_K = BLOCK_GROUPS * L_PAD     # 总是 2 的幂次，保证向量化
```

### 3.3 使用 autotune_slide_kernels.py 生成固定配置 kernel

原始的 slide kernel 每次运行都会执行 autotune，这会带来额外开销。使用 `autotune_slide_kernels.py` 可以预先执行 autotune，生成带有固定最优配置的 kernel 文件。

```bash
cd /root/vllmbench/slidesparse/kernels/slide_kernals

# 调优 slide_L6_fp8.py，自动生成 slide_L6_fp8_autotuned.py
python3 autotune_slide_kernels.py slide_L6_fp8.py

# 调优并指定输出文件名
python3 autotune_slide_kernels.py slide_L8_int8.py -o my_tuned_kernel.py

# 调优所有 4 个 kernel
python3 autotune_slide_kernels.py slide_L6_fp8.py
python3 autotune_slide_kernels.py slide_L6_int8.py
python3 autotune_slide_kernels.py slide_L8_fp8.py
python3 autotune_slide_kernels.py slide_L8_int8.py
```

**生成的文件**:
- `slide_L6_fp8_autotuned.py`
- `slide_L6_int8_autotuned.py`  
- `slide_L8_fp8_autotuned.py`
- `slide_L8_int8_autotuned.py`

**使用固定配置版本**:
```python
# 替换原始导入
# from slide_L8_int8 import fused_quant_slide  # 原始 (带 autotune)
from slide_L8_int8_autotuned import fused_quant_slide  # 固定配置 (无 autotune)
```

**Autotune 脚本工作流程**:
1. 对各种 K_in 和 M 组合运行原始 kernel 触发 autotune
2. 收集每个 num_groups 对应的最优配置
3. 分析结果，构建 if-else 分支策略
4. 生成不依赖运行时 autotune 的固定配置 kernel

---

## 4. Kernel 实现细节

### 4.1 整体架构

```
                    ┌─────────────────────────────────────┐
                    │     fused_quant_slide(x)            │
                    │     Python wrapper                   │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    _fused_quant_slide_kernel                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Pass 1: Absmax 计算                                          │    │
│  │   - 向量化 load (BLOCK_K = BLOCK_GROUPS * L_PAD)             │    │
│  │   - mask = offs < K_in_orig (处理边界)                       │    │
│  │   - 计算 scale = absmax / 127.0 (INT8) 或 448.0 (FP8)       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Pass 2: Quant + Slide                                        │    │
│  │   - 按 group 遍历 (步长 = BLOCK_GROUPS)                      │    │
│  │   - Load L 个元素 (mask 处理越界，other=0.0)                 │    │
│  │   - 量化: rint(x * inv_scale) & 0xFF                        │    │
│  │   - Pack 4 bytes 成 int32，写入 NUM_WINDOWS 个 window        │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 关键实现

#### 边界处理 (无 F.pad)

```python
# Python wrapper: 计算逻辑 K，但不实际 pad
K_in = ((K_in_orig + L - 1) // L) * L  # 向上取整到 L 的倍数
# 不调用 F.pad！直接传 x

# Kernel: 用 mask 处理越界
x0 = tl.load(x_row + base + 0, 
             mask=mask_group & ((base + 0) < K_in_orig),  # 边界检查
             other=0.0)  # 越界读 0
```

**为什么不用 F.pad**：`F.pad` 触发额外内存拷贝，开销比整个 vLLM quant 还大 (17µs vs 10µs)。

#### Slide 窗口打包

```python
# L=6: 每组输出 2 个 window (int32)
# Window 0: [b0, b1, b2, b3]
# Window 1: [b2, b3, b4, b5]  (重叠 b2, b3)
out_base = gid * 2
tl.store(y_row32 + out_base + 0, (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)))
tl.store(y_row32 + out_base + 1, (b2 | (b3 << 8) | (b4 << 16) | (b5 << 24)))
```

---

## 5. Benchmark 试验

### 5.1 试验条件

| 项目 | 配置 |
|------|------|
| **GPU** | NVIDIA H100 PCIe |
| **测试层** | Wqkv (K=2560), W2 (K=6912) |
| **Batch Size (M)** | 1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096 |
| **数据类型** | bfloat16 输入 → int8/fp8 输出 |
| **Baseline** | vLLM `scaled_int8_quant` / `scaled_fp8_quant` |
| **Warmup** | 25 次 |
| **Repetitions** | 100 次 |

### 5.2 运行 Benchmark

```bash
cd /root/vllmbench/slidesparse/kernels/slide_kernals

# L=6 INT8
python3 run_benchmark.py --L 6 --dtype int8 --tuned

# L=8 INT8
python3 run_benchmark.py --L 8 --dtype int8 --tuned

# FP8
python3 run_benchmark.py --L 6 --dtype fp8 --tuned
python3 run_benchmark.py --L 8 --dtype fp8 --tuned

# 使用 autotuned 版本
python3 run_benchmark.py --tuned
```

### 5.3 Benchmark 结果

#### INT8 结果

| 配置 | Avg Slide/vLLM | 状态 |
|------|----------------|------|
| **L=8 int8** | **1.06x** | ✅ 优秀 (只比纯量化慢 6%) |
| **L=6 int8** | **1.19x** | ✅ 良好 (比纯量化慢 19%) |

#### 详细数据 (L=8 INT8)

| Layer | M | K | vLLM | Slide | Slide/vLLM |
|-------|---|---|------|-------|------------|
| Wqkv | 1 | 2560 | 4.99µs | 4.58µs | **0.92x** |
| Wqkv | 1024 | 2560 | 9.54µs | 10.27µs | 1.08x |
| W2 | 1 | 6912 | 5.47µs | 5.60µs | 1.02x |
| W2 | 4096 | 6912 | 51.46µs | 59.71µs | 1.16x |

#### 详细数据 (L=6 INT8)

| Layer | M | K | vLLM | Slide | Slide/vLLM |
|-------|---|---|------|-------|------------|
| Wqkv | 1 | 2560 | 4.77µs | 6.46µs | 1.36x |
| Wqkv | 1024 | 2560 | 9.34µs | 11.87µs | 1.27x |
| W2 | 1 | 6912 | 5.63µs | 5.31µs | **0.94x** |
| W2 | 4096 | 6912 | 51.39µs | 57.31µs | 1.12x |

---

## 6. 性能差异原因分析

### 6.1 L=8 vs L=6 性能差异

| 因素 | L=8 | L=6 | 影响 |
|------|-----|-----|------|
| **内存对齐** | `base = gid * 8` (100% 8字节对齐) | `base = gid * 6` (12.5% 8字节对齐) | **主要原因** |
| **Load 数量** | 8 个 | 6 个 | 影响小 |
| **Store 数量** | 3 个 window | 2 个 window | 影响小 |

```
L=6 内存访问模式:
  Thread 0: base=0   (8对齐 ✓)
  Thread 1: base=6   (不对齐 ✗)
  Thread 2: base=12  (不对齐 ✗)
  Thread 3: base=18  (不对齐 ✗)
  Thread 4: base=24  (8对齐 ✓)
  ...
  对齐率: 1/4 = 25% (考虑 bf16 = 2 bytes)

L=8 内存访问模式:
  Thread 0: base=0   (8对齐 ✓)
  Thread 1: base=8   (8对齐 ✓)
  Thread 2: base=16  (8对齐 ✓)
  ...
  对齐率: 100%
```

### 6.2 K=2560 vs K=6912 性能差异 (L=6)

| K | Slide/vLLM | 原因 |
|---|------------|------|
| 2560 | 1.27-1.40x | groups 少 (427), kernel launch 开销比例高 |
| 6912 | 0.94-1.12x | groups 多 (1152), 更好的并行度和 cache 利用 |

### 6.3 优化历程总结

| 版本 | L=6 性能 | 问题 | 解决方案 |
|------|----------|------|----------|
| v1 | 2.2x | F.pad 触发内存拷贝 (17µs) | 移除 F.pad，用 mask 处理边界 |
| v2 | 1.96x | Pass 1 用固定 ABSMAX_BLOCK=1024 | 改用 BLOCK_K=BLOCK_GROUPS*L_PAD |
| v3 (当前) | **1.19x** | Pass 2 strided access | 无法完全解决 (L 本身限制) |

---

## 7. 使用方法

### 7.1 Python API

```python
from slide_L8_int8 import fused_quant_slide, get_config

# 输入: [M, K] bf16/fp16/fp32
x = torch.randn(1024, 2560, dtype=torch.bfloat16, device='cuda')

# 输出: y [M, K_out_padded] int8, scale [M] fp32
y, scale = fused_quant_slide(x)

# 查看配置
print(get_config())
# {'L': 8, 'L_PAD': 8, 'N': 4, 'NUM_WINDOWS': 3, 'EXPAND_RATIO': 1.5, 'DTYPE': 'int8'}
```

### 7.2 输出格式

```python
# 输入 K=2560, L=8
# num_groups = ceil(2560/8) = 320
# K_out = 320 * 3 * 4 = 3840 bytes
# K_out_padded = ceil(3840/16)*16 = 3840 (已对齐)

# 输出 shape: [M, 3840] dtype=int8
# 每 12 bytes 表示一个 group 的 3 个 window
```

---

## 8. 后续优化方向

1. **2D Block 策略**: 把 M 维度也拆分到 block 里，增加并行度
2. **权重预处理**: 把权重预处理成对齐格式，避免 strided access
3. **Warp Shuffle**: 用 warp shuffle 优化非对齐访问
4. **专注 L=8**: 如果 L=6 优化困难，可考虑只支持 L=8 (性能更好)

---

## 9. 参考

- [Triton Documentation](https://triton-lang.org/)
- [vLLM Quantization Kernels](https://github.com/vllm-project/vllm)
- SlideSparse 论文/文档
