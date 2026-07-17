# SlideSparse 集成 vLLM 框架详解

本文档详细阐述 SlideSparse 稀疏加速方法的理论原理、工具开发流程、以及在 vLLM 框架中的工程实现方案。本文档旨在为后续的开发工作提供完整的技术指导和实施手册。

---

## 目录

1. [SlideSparse 理论概述与创新点](#1-slidesparse-理论概述与创新点)
2. [工程实现流程总览（七阶段）](#2-工程实现流程总览七阶段)
3. [Phase 1：环境准备](#3-phase-1环境准备)
4. [Phase 2：vLLM 基线测试](#4-phase-2vllm-基线测试)
5. [Phase 3：Dense 基线搭建和 Kernel 替换（cuBLASLt）](#5-phase-3dense-基线搭建和-kernel-替换cublaslt)
6. [Phase 4：离线工具链](#6-phase-4离线工具链)
7. [Phase 5：模型加载（在线 Model Loader 修改）](#7-phase-5模型加载在线-model-loader-修改)
8. [Phase 6：Sparse 搭建和 Kernel 替换（cuSPARSELt）](#8-phase-6sparse-搭建和-kernel-替换cusparselt)
9. [Phase 7：新模型引入（BitNet-1.58b）](#9-phase-7新模型引入bitnet-158b)
10. [代码组织结构](#10-代码组织结构)
11. [测试与验证](#11-测试与验证)
12. [附录：关键文件路径速查表](#12-附录关键文件路径速查表)

---

## 1. SlideSparse 理论概述与创新点

### 1.1 背景与动机

在大语言模型（LLM）推理中，**GEMM（通用矩阵乘法）占据了约 70-80% 的计算时间**。稀疏计算是加速 GEMM 的重要途径之一。NVIDIA 的 Ampere 及后续架构 GPU 提供了 2:4 结构化稀疏的硬件加速支持，可以实现理论上 2 倍的计算吞吐提升。

然而，现有的 2:4 稀疏方案存在以下限制：
- **稀疏度固定**：必须严格满足每 4 个元素中有 2 个为零（50% 稀疏度）
- **精度损失**：强制剪枝可能导致较大的精度下降
- **灵活性不足**：无法适应不同稀疏度的模型

**SlideSparse** 提出了一种创新的解决方案，通过权重和激活的重排策略，使**更低稀疏度、更粗粒度**的稀疏模型也能利用 2:4 结构化稀疏硬件加速。

### 1.2 SlideSparse 核心原理

#### 1.2.1 松弛结构化稀疏 (Relaxed Structured Sparse)

SlideSparse 定义了一种更宽松的稀疏格式：**Z:L' 稀疏**，其中：
- L' = L + k×(L-Z)，k = 0, 1, 2, 3...
- L 是硬件支持的稀疏窗口大小（如 2:4 中的 4）
- Z 是窗口内的零元素个数（如 2:4 中的 2）
- k 是滑框次数

例如，2:8 稀疏（25% 稀疏度）可以通过 SlideSparse 映射到 2:4 硬件进行加速。

#### 1.2.2 重叠滑动窗口机制 (Overlapping Sliding Windows)

SlideSparse 的核心机制是通过**重叠滑动窗口**将原始权重矩阵分解为多个子权重：

```
原始权重序列：    1  2  3  4  5  6  7  8
                 ↓  重叠滑动窗口分解
拓展后的序列：    1  2  3  4 | 3  4  5  6 | 5  6  7  8
                 └─窗口1─┘   └─窗口2─┘   └─窗口3─┘

窗口参数：
- 窗口大小 (Window Size) = L = 4
- 滑框步长 (Stride) = L - Z = 2
- 重叠宽度 (Overlap) = Z = 2
```

#### 1.2.3 贪心残差分配 (Greedy Residual Allocation)

SlideSparse 采用贪心残差分配策略，确保：
1. 所有零元素被完全覆盖
2. 非零元素被分配到符合 Z:L 稀疏格式的位置
3. 最小化拓展后的总长度

**示例分析**（2:8 稀疏 → 2:4 硬件）：
```
原始稀疏：2:8（25% 稀疏度，每 8 个元素有 2 个零，6 个非零）
所需窗口数：⌈6/2⌉ = 3 组 2:4 稀疏
拓展总长度：3 × 4 = 12
预期延时：(6/2 × 4) / 2 / 8 = 75%（相比 dense 计算）
```

#### 1.2.4 通用性公式

对于硬件支持 Z:L 稀疏加速（L 个元素中至少 Z 个零，最多 N 个非零，Z + N = L）：

| 参数 | 公式 | 说明 |
|------|------|------|
| 加速比 | L / N | 如 2:4 → 2x，3:4 → 4x |
| 支持的稀疏格式 | Z:L'，L' = L + k×N | k = 0, 1, 2, 3... |
| 窗口大小 | L（或 L 的因数） | |
| 滑框步长 | N = L - Z | |
| 重叠宽度 | Z | |
| 延时比例 | (L' - Z) / L' | |

### 1.3 SlideSparse 创新点总结

1. **任意稀疏度适配**：打破 2:4 稀疏的 50% 稀疏度限制，支持 2:4, 2:6, 2:8, 2:10, 2:12 等多种稀疏格式，以及 1:2, 1:3, 1:4, 1:5 等更细粒度的稀疏

2. **理论最优利用率**：确保 X% 的稀疏度一定能减少 X% 的计算延时，完全享受稀疏度带来的"零值跳跃"吞吐收益

3. **硬件兼容性**：无需修改硬件，复用现有的 NVIDIA 2:4 稀疏硬件加速单元

4. **端到端加速**：通过算子融合（fused_quant_slide + sparse_GEMM + fused_dequant_transpose），最小化额外开销

### 1.4 SlideSparse 与 vLLM 集成的目标

本项目的目标是将 SlideSparse 集成到 vLLM 推理框架中，实现：

1. **吞吐量提升**：在 Prefill 和 Decode 阶段验证 token/s 的提升百分比
2. **模型支持**：支持 Qwen3、Llama3.2 等 dense 小模型的端到端加速
3. **量化兼容**：支持 FP8 和 INT8 两种 GEMM 精度
4. **规模覆盖**：测试 ~1B、~3B、~7B、~14B 四种典型模型尺寸

**测试模型计划**：
- FP8：Llama3.2-1B、Llama3.2-3B、Qwen3-8B、Qwen3-14B
- INT8：需寻找 W8A8 版本或自行量化

---

## 2. 工程实现流程总览（七阶段）

本项目的工程实现分为七个阶段，每个阶段有明确的目标、输入输出和验证标准。

### 2.1 阶段总览图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SlideSparse vLLM 集成工程流程                            │
└─────────────────────────────────────────────────────────────────────────────────┘

Phase 1: 环境准备
├── Fork vLLM 0.13.0 稳定分支
├── Docker 环境配置（CUDA 12.9、cuSPARSELt）
└── 开发模式安装 vLLM

Phase 2: vLLM 基线测试（原生代码）
├── 测试 INT8 W8A8 模型（RedHat 发布的 quantized.w8a8 系列）
├── 测试 FP8 W8A8 模型（FP8-dynamic 系列）
├── 运行 vllm bench、vllm chat、lm_eval 精度测试
└── 通过埋点/nsys 分析当前调用的 GEMM Kernel 后端

Phase 3: Dense 基线搭建和 Kernel 替换（cuBLASLt）
├── 新建 cuBLASLtLinearMethod 继承 LinearMethodBase
├── 实现 cuBLASLt 的 INT8 和 FP8 两个版本的 GEMM+Dequant 融合
├── 保持 quant 不变，仅替换 apply 中的 GEMM 部分
└── 验证性能精度与 vLLM 原生接近（作为后续 cuSPARSELt 的试水）

Phase 4: 离线工具链
├── 权重 Prune 脚本（magnitude/random 剪枝）
├── 权重 Slide 脚本（滑动拓展，K → K'，注意 padding 到 4L）
├── 权重 Compress 脚本（cuSPARSELt 2:4 稀疏压缩）
├── 算法搜索 Search 脚本（离线搜索最优 algo_id）
├── AutoTune 脚本
└── 生成 .safetensors 格式的处理后权重

Phase 5: 模型加载（在线 Model Loader 修改）
├── 利用 DefaultModelLoader 的宽容性
├── 在 SlideSparseLinearMethod.create_weights() 中定义正确的 shape
├── 离线权重 key 名称不变，shape 从 [N, K] 变为 [N, K/2]（Compressed）
└── 最小侵入修改，无需重写 Loader 核心逻辑

Phase 6: Sparse 搭建和 Kernel 替换（cuSPARSELt）
├── 新建 SlideSparseLinearMethod 继承 LinearMethodBase
├── Quant+Slide 融合：开发 FP8 和 INT8 的 Triton kernel
├── GEMM+Dequant 融合：开发 cuSPARSELt 的 INT8 和 FP8 版本
├── 处理 A100 的 Row-Major 输出性能问题（放弃 Transpose，接受诚实的提升）
└── 修改 apply 方法，替换 quant 和 GEMM 的 Kernel

Phase 7: 新模型引入（BitNet-1.58b）
├── 在 vllm/model_executor/models/ 下新建 bitnet.py
├── 实现 ternary 量化（+1/0/-1）的 quant 方法
├── 复用 Llama 大部分代码，替换 MLP 和 Norm（BitLinear、ReLU²）
└── 确保兼容 cuBLASLtLinearMethod、SlideSparseLinearMethod

最后: 速度精度测试和实验
├── A100 / H100 / B200 显卡测试
├── 各种 model（Qwen2.5、Llama3.2、BitNet）
├── 各种稀疏度（2:4, 2:6, 2:8, 2:10, 2:12）
└── Prefill/Decode 吞吐、PPL 精度评估
```

### 2.2 各阶段依赖关系

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 6
                │                      │
                │                      ▼
                └──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7
                        │
                        └──► 离线脚本可独立开发测试
```

**关键说明**：
- Phase 3（cuBLASLt Dense）是 Phase 6（cuSPARSELt Sparse）的试水，提前探索换 Kernel 的坑
- Phase 4（离线工具链）可以独立开发，不依赖在线代码
- Phase 5（模型加载）依赖 Phase 4 生成的处理后权重
- Phase 6 是核心工作，依赖 Phase 3 的经验和 Phase 4/5 的支持
- Phase 7（BitNet）依赖 Phase 6 完成的 LinearMethod 框架

---

## 3. Phase 1：环境准备

### 3.1 代码框架和 Docker 环境

**环境配置步骤**：
- Fork vLLM 0.13.0 稳定分支作为本地 main 分支
- 基于 `vllm/vllm-openai:v0.13.0` Docker 镜像（Ubuntu 22.04、CUDA 12.9、PyTorch 2.9、Flash-Attention）
- 通过 Dockerfile 添加 cuSPARSELt 依赖
- 卸载预装的 vllm（`pip uninstall vllm`）
- 使用开发模式安装（`pip install -e .`）建立源码软链接

**Dockerfile 示例片段**：
```dockerfile
# 安装 cuSPARSELt
RUN apt-get update && apt-get install -y libcusparselt0 libcusparselt-dev
```

**验证标准**：
- 能够直接修改 `vllm/` 目录下的源代码并立即生效
- `import vllm` 成功，版本号正确

---

## 4. Phase 2：vLLM 基线测试

### 4.1 测试目标

使用 vLLM 官方原生代码，不做任何修改，测试 HuggingFace 上已量化好的 W8A8 模型，获取原始的性能和精度基线数据。

### 4.2 测试模型列表

#### 4.2.1 INT8 W8A8 模型（RedHat 发布）

| 模型 | HuggingFace 路径 |
|------|-----------------|
| Qwen2.5-0.5B | `RedHat/Qwen2.5-0.5B-Instruct-quantized.w8a8` |
| Qwen2.5-1.5B | `RedHat/Qwen2.5-1.5B-Instruct-quantized.w8a8` |
| Qwen2.5-3B | `RedHat/Qwen2.5-3B-Instruct-quantized.w8a8` |
| Qwen2.5-7B | `RedHat/Qwen2.5-7B-Instruct-quantized.w8a8` |
| Qwen2.5-14B | `RedHat/Qwen2.5-14B-Instruct-quantized.w8a8` |
| Llama3.2-1B | `RedHat/Llama3.2-1B-Instruct-quantized.w8a8` |
| Llama3.2-3B | `RedHat/Llama3.2-3B-Instruct-quantized.w8a8` |

#### 4.2.2 FP8 W8A8 模型（FP8-dynamic）

| 模型 | HuggingFace 路径 |
|------|-----------------|
| Qwen2.5-0.5B | `Qwen2.5-0.5B-Instruct-FP8-dynamic` |
| Qwen2.5-1.5B | `Qwen2.5-1.5B-Instruct-FP8-dynamic` |
| Qwen2.5-3B | `Qwen2.5-3B-Instruct-FP8-dynamic` |
| Qwen2.5-7B | `Qwen2.5-7B-Instruct-FP8-dynamic` |
| Qwen2.5-14B | `Qwen2.5-14B-Instruct-FP8-dynamic` |
| Llama3.2-1B | `Llama3.2-1B-Instruct-FP8-dynamic` |
| Llama3.2-3B | `Llama3.2-3B-Instruct-FP8-dynamic` |

**说明**：这两种情况下，都涉及到 FP8 或 INT8 的 quant Kernel 和 GEMM+Dequant 融合 Kernel 的调用。

### 4.3 测试方法与 CLI 命令

#### 4.3.1 Prefill/Decode 吞吐测试（vllm bench）

**入口文件**：`vllm/entrypoints/cli/benchmark/throughput.py`（CLI 入口）、`benchmarks/benchmark_throughput.py`（底层脚本）

**测试命令**：
```bash
# INT8 模型吞吐测试
vllm bench throughput \
    --model RedHat/Qwen2.5-7B-Instruct-quantized.w8a8 \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 100 \
    --dtype auto \
    --tensor-parallel-size 1

# FP8 模型吞吐测试
vllm bench throughput \
    --model Qwen2.5-7B-Instruct-FP8-dynamic \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 100 \
    --dtype auto \
    --tensor-parallel-size 1
```

**输出指标**：
- Prefill 阶段：tokens/s
- Decode 阶段：tokens/s
- 端到端吞吐量：requests/s

#### 4.3.2 对话精度测试（vllm chat）

**入口文件**：`vllm/entrypoints/cli/main.py`（通过 `vllm chat` 子命令）

**测试代码**：
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="RedHat/Qwen2.5-7B-Instruct-quantized.w8a8",
    dtype="auto",
    tensor_parallel_size=1,
)

messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

outputs = llm.chat(
    messages,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=256)
)
print(outputs[0].outputs[0].text)
```

#### 4.3.3 精度评估（lm_eval）

**使用 lm-evaluation-harness 进行 PPL 评估**：
```bash
# INT8 模型 PPL 评估
lm_eval --model vllm \
    --model_args pretrained=RedHat/Qwen2.5-7B-Instruct-quantized.w8a8 \
    --tasks wikitext --num_fewshot 0

# FP8 模型 PPL 评估
lm_eval --model vllm \
    --model_args pretrained=Qwen2.5-7B-Instruct-FP8-dynamic \
    --tasks wikitext --num_fewshot 0
```

### 4.4 分析当前 GEMM Kernel 后端

#### 4.4.1 通过埋点分析

在 `vllm/model_executor/layers/quantization/fp8.py` 的 `Fp8LinearMethod.apply()` 方法中添加打印语句，确认实际调用的 GEMM 函数。

#### 4.4.2 通过 nsys Profiling

```bash
nsys profile -o vllm_baseline python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen2.5-7B-Instruct-FP8-dynamic')
outputs = llm.generate(['Hello'], SamplingParams(max_tokens=10))
"
```

在 Nsight Systems 中分析 CUDA Kernel 调用，确认是 CUTLASS、cuBLAS 还是 Triton 实现。

#### 4.4.3 csrc 目录 Kernel 分析

**关键 Kernel 文件路径**：

| Kernel 类型 | 文件路径 |
|------------|---------|
| FP8 量化 | `csrc/quantization/w8a8/fp8/` |
| INT8 量化 | `csrc/quantization/w8a8/int8/` |
| CUTLASS GEMM | `csrc/cutlass_extensions/` |
| 稀疏 GEMM | `csrc/sparse/cutlass/` |
| PyTorch 绑定 | `csrc/torch_bindings.cpp` |

**FP8 量化的关键函数**（`vllm/_custom_ops.py`）：
- `scaled_fp8_quant(input, scale, ...)`：FP16/BF16 → FP8 量化
- `cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)`：FP8 GEMM + Dequant

**INT8 量化**：
- 通过 `compressed-tensors` 量化配置加载，内部可能调用 CUTLASS INT8 GEMM 或 cuBLAS INT8

---

## 5. Phase 3：Dense 基线搭建和 Kernel 替换（cuBLASLt）

### 5.1 目标

使用 cuBLASLt 高性能库替换 vLLM 默认的 GEMM 实现，作为后续 cuSPARSELt 稀疏 GEMM 的试水。这个阶段的目的是：

1. 熟悉 LinearMethod 的替换流程
2. 验证 cuBLASLt API 的正确调用方式
3. 确保替换后的 Dense GEMM 性能和精度与原生接近
4. 为 cuSPARSELt 的 Sparse GEMM 实现积累经验

### 5.2 实现方案

#### 5.2.1 新建 cuBLASLtLinearMethod

**文件位置**：`slidesparse/core/cublaslt_linear_method.py`（外挂模块）

**类设计**：
```
cuBLASLtLinearMethod 继承 LinearMethodBase
├── create_weights()    # 创建权重参数，与原方法一致
├── apply()             # 替换 GEMM Kernel
│   ├── quant 保持不变（调用原有的 scaled_fp8_quant 或 scaled_int8_quant）
│   └── GEMM+Dequant 替换为 cuBLASLt 实现
└── process_weights_after_loading()  # 权重后处理
```

#### 5.2.2 cuBLASLt API 调用

**参考文档**：https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api

**关键 API 调用流程**：
1. `cublasLtCreate(&lightHandle)` — 创建 cuBLASLt 句柄
2. `cublasLtMatrixLayoutCreate(&Adesc, ...)` — 创建矩阵 A 的布局描述符
3. `cublasLtMatrixLayoutCreate(&Bdesc, ...)` — 创建矩阵 B 的布局描述符
4. `cublasLtMatrixLayoutCreate(&Cdesc, ...)` — 创建矩阵 C 的布局描述符
5. `cublasLtMatmulDescCreate(&operationDesc, ...)` — 创建矩阵乘法描述符
6. `cublasLtMatmulPreferenceCreate(&preference)` — 创建算法选择偏好
7. `cublasLtMatmulAlgoGetHeuristic(...)` — 获取最优算法
8. `cublasLtMatmul(...)` — 执行矩阵乘法

#### 5.2.3 INT8 版本实现

**数据类型配置**：
- A：INT8（量化后的激活）
- B：INT8（量化后的权重）
- C：INT32（累加结果）
- D：FP16/BF16（Dequant 后的输出）

**Dequant 融合**：cuBLASLt 支持 Epilogue 操作，可以将 scale 乘法融合到 GEMM 中：
- `CUBLASLT_EPILOGUE_DEFAULT`：无融合
- `CUBLASLT_EPILOGUE_RELU`：融合 ReLU
- `CUBLASLT_EPILOGUE_BIAS`：融合 Bias

对于 Dequant 操作（乘以 scale），需要设置 `scaleType` 和 `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER`。

#### 5.2.4 FP8 版本实现

**数据类型配置**：
- A：FP8 E4M3（量化后的激活）
- B：FP8 E4M3（量化后的权重）
- C/D：FP16/BF16（Dequant 后的输出）

**注意**：FP8 GEMM 需要 CUDA Compute Capability >= 8.9（H100/Ada）或特殊配置。

### 5.3 apply 方法设计

**伪代码**：
```python
def apply(self, layer, x, bias=None):
    # 1. Quant（保持不变）
    qinput, scale_a = ops.scaled_fp8_quant(x, ...)  # 或 int8 版本
    
    # 2. GEMM + Dequant（替换为 cuBLASLt）
    output = cublaslt_gemm_dequant(
        qinput,           # [M, K], FP8/INT8
        layer.weight,     # [N, K], FP8/INT8
        scale_a,          # 激活 scale
        layer.weight_scale,  # 权重 scale
        out_dtype=torch.bfloat16,
    )
    
    # 3. Bias
    if bias is not None:
        output = output + bias
    
    return output
```

### 5.4 验证标准

1. **正确性**：Dense cuBLASLt GEMM 的输出与原 CUTLASS GEMM 在数值上接近（允许量化误差）
2. **性能**：吞吐量与原生实现接近或更优
3. **精度**：PPL 与原生实现接近

### 5.5 CLI 启用方式

通过环境变量或命令行参数启用：
```bash
vllm serve model_name --enable-cublaslt
```

或在代码中：
```python
llm = LLM(model="...", quantization="fp8", extra_args={"enable_cublaslt": True})
```

---

## 6. Phase 4：离线工具链

### 6.1 工具概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              离线工具链 (Offline)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐ │
│   │ 权重 Prune   │ ──► │ 权重 Slide   │ ──► │ 权重 Compress│ ──► │ 算法 Search││
│   │   (Python)   │     │   (Python)   │     │   (CUDA)     │     │ (Python)  │ │
│   │              │     │              │     │              │     │           │ │
│   │ Dense [N,K]  │     │ Sparse [N,K] │     │ Slided [N,K']│     │ json 配置 │ │
│   │    ↓         │     │    ↓         │     │    ↓         │     │           │ │
│   │ Sparse [N,K] │     │ Slided [N,K']│     │ Compressed   │     │           │ │
│   └──────────────┘     └──────────────┘     │  [N, K'/2]   │     └───────────┘ │
│                                             └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**文件存放位置**：`slidesparse/offline/`

### 6.2 权重剪枝工具 (prune.py)

**功能**：将 Dense 权重剪枝为指定稀疏度的结构化稀疏权重

**输入**：
- Dense 权重张量：`[N, K]` 维度（N 为输出维度，K 为输入维度）
- 稀疏粒度参数：`Z:L`（每 L 个连续元素中有 Z 个零）
- 剪枝模式：`random` 或 `magnitude`

**输出**：
- 稀疏权重张量：`[N, K]` 维度，满足指定的结构化稀疏约束

**剪枝算法**：

**magnitude 模式（按绝对值大小剪枝）**：
1. 将权重张量 `[N, K]` 重塑为 `[N, K/group_size, group_size]`，形成多个稀疏组
2. 在每个组内，计算每个元素的绝对值
3. 使用 `torch.topk` 选出绝对值最大的 `num_nonzeros` 个位置
4. 生成二值掩码（被选中位置为1，其余为0）
5. 将掩码与原权重相乘，得到剪枝后的稀疏权重

**random 模式（随机剪枝）**：
1. 同样将权重重塑为组
2. 在每个组内随机选择 `num_nonzeros` 个位置保留
3. 其余位置置零

**关键参数计算**（其中 Z 为每组零元素数，L 为稀疏窗口大小）：
- `num_zeros_per_group = Z` — 每组需要置零的元素数
- `num_nonzeros = L - Z` — 每组保留的非零元素数

**需要处理的线性层**：
| 线性层名称 | 模型中的位置 | 典型维度 |
|-----------|------------|---------|
| Wqkv | `self_attn.qkv_proj` | `[3*H, H]` 或 `[(Q+2*KV), H]` |
| Wo | `self_attn.o_proj` | `[H, H]` |
| W13 | `mlp.gate_up_proj` | `[2*I, H]` |
| W2 | `mlp.down_proj` | `[H, I]` |

### 6.3 权重滑动工具 (slide.py)

**功能**：将稀疏权重通过滑动窗口机制拓展为符合 2:4 硬件格式的权重

#### 6.3.1 关键概念：Slide 导致 K 维度增大

**核心原理**：SlideSparse 的 slide 操作会将权重的 K 维度扩展。原始权重形状为 `[N, K]`，经过 slide 后变为 `[N, K']`，其中：

```
K' = K × expand_ratio
expand_ratio = (num_windows × L_tgt) / L_src
```

**示例计算（2:8 稀疏 → 2:4 硬件）**：
- 源稀疏格式：2:8（每 8 个元素有 2 个零，6 个非零）
- 目标硬件格式：2:4（每 4 个元素有 2 个零，2 个非零）
- 滑动步长：stride = L_tgt - Z_tgt = 4 - 2 = 2
- 窗口数：num_windows = (L_src - Z) / stride = (8 - 2) / 2 = 3
- 扩展比例：expand_ratio = (3 × 4) / 8 = 1.5

因此，原始 K=4096 的权重，经过 slide 后变为 K'=6144。

#### 6.3.2 Padding 策略：必须整除 4L

**cuSPARSELt 对齐要求**：K 必须被 padding 到能够被 `4 × L` 整除。

**Padding 公式**：
```
K_padded = ceil(K / (4 × L)) × (4 × L)
```

**示例**（L=8，原始 K=4090）：
- 4L = 32
- K_padded = ceil(4090 / 32) × 32 = 128 × 32 = 4096

**Padding 在 slide 之前执行**，确保每个稀疏组都完整。

#### 6.3.3 滑动拓展算法

**滑动拓展流程**：
1. **Padding 预处理**：确保 K 能被 `4 × L_src` 整除，必要时在末尾补零
2. **参数计算**：
   - `stride = L_tgt - Z_tgt`（滑动步长 = 目标窗口中非零元素个数 = 2）
   - `num_windows = (L_src - Z) // stride`（窗口数 = 源窗口非零元素数 / 每窗口非零数）
3. **权重分组**：将权重重塑为 `[N, num_groups, L_src]`
4. **滑动提取**：对每个窗口 `i`，提取位置 `[i*stride, i*stride+L_tgt)` 的元素
5. **拼接输出**：将所有窗口的结果拼接，形成 `[N, K']` 的扩展权重

### 6.4 权重压缩工具 (compress.py)

**功能**：调用 cuSPARSELt 库将 2:4 稀疏权重压缩为硬件可识别的格式

**输入**：
- 滑动后的权重：`[N, K']`，符合 2:4 稀疏格式
- 数据类型：FP8 / INT8

**输出**：
- 压缩后的权重：形状为 `[N, K'/2]`（2:4 稀疏压缩减半）
- 元数据：形状为 `[N, K'/8]`（每 4 个非零元素对应 1 字节元数据）

**压缩流程**：
1. 将权重转换为目标数据类型（FP8/INT8），并确保内存连续
2. 调用 cuSPARSELt 的压缩函数，指定稀疏类型为 `'2:4'`
3. 返回压缩后的权重张量和元数据

**vLLM 已有参考**：`vllm/_custom_ops.py` 中的 `cutlass_sparse_compress()` 函数

### 6.5 算法搜索工具 (search.py)

**功能**：离线搜索 cuSPARSELt GEMM 的最优算法 ID

**输入**：
- 模型线性层的 `[N, K]` 尺寸信息
- 不同的 batch size `M` 范围

**输出**：
- JSON 配置文件：记录每个 `(M, N, K)` 组合的最优算法 ID

**搜索流程**：
1. **遍历模型线性层**：获取每个线性层的 `(N, K)` 尺寸
2. **遍历 M 值范围**：针对不同的 batch size（如 1, 4, 8, 16, 32, ...）
3. **算法搜索**：
   - 创建测试输入张量 A `[M, K]` 和权重张量 B `[N, K]`
   - 遍历 cuSPARSELt 支持的所有算法 ID
   - 对每个算法执行多次（如 100 次）计时测试
   - 记录最快算法的 ID
4. **保存配置**：将 `{(M,N,K): best_algo_id}` 字典序列化为 JSON 文件

**JSON 文件格式示例**：
```json
{
  "1,4096,4096": 0,
  "4,4096,4096": 2,
  "16,4096,4096": 1,
  "32,4096,4096": 3
}
```

### 6.6 完整预处理脚本 (preprocess_weights.py)

**功能**：将原始 HuggingFace 模型的权重进行完整的 Prune → Slide → Compress 流程

**命令行使用方式**：
```bash
python slidesparse/offline/preprocess_weights.py \
    --input-model meta-llama/Llama-3.2-1B-Instruct \
    --output-dir ./checkpoints/llama-3.2-1b-slidesparse \
    --sparsity 2:8 \
    --prune-mode magnitude
```

**输出文件结构**：
```
./checkpoints/llama-3.2-1b-slidesparse/
├── model.safetensors          # 处理后的权重（key 名称保持不变）
├── config.json                # 原模型配置（复制）
├── tokenizer.json             # 原 tokenizer（复制）
├── tokenizer_config.json      # 原 tokenizer 配置（复制）
└── slidesparse_config.json    # SlideSparse 元数据
    ├── sparsity: "2:8"
    ├── processed_layers: [...]
    ├── original_shapes: {...}
    └── compressed_shapes: {...}
```

**关键说明**：
- 权重文件中的 key 名称保持不变（如 `model.layers.0.self_attn.q_proj.weight`）
- 但 shape 从原始的 `[N, K]` 变为 `[N, K'/2]`（经过 slide 和 compress 后）
- 这样设计是为了兼容 vLLM 的 DefaultModelLoader

---

## 7. Phase 5：模型加载（在线 Model Loader 修改）

### 7.1 核心思路：最小侵入修改

**关键发现**：vLLM 的 `DefaultModelLoader`（定义于 `vllm/model_executor/model_loader/default_loader.py`）实际上非常"宽容"。它只是把 `.safetensors` 文件里的 Tensor 读出来，塞给 Layer。

**核心原则**：
- 不需要重写整个 Loader
- 利用 `SlideSparseLinearMethod.create_weights()` 方法定义正确的 Parameter shape
- 只要 `.safetensors` 里的 weight shape 与 `create_weights()` 定义的一致，vLLM 就能正常加载

### 7.2 离线权重处理策略

**预期做法**：

1. **离线脚本处理**：把权重 Prune → Slide → Compress
2. **保存为 `.safetensors`**：Key 的名字保持不变（如 `model.layers.0.self_attn.q_proj.weight`）
3. **Shape 变化**：原来是 `[N, K]`，现在是 `[N, K'/2]`（经过 slide 扩展后再被 2:4 压缩减半）

### 7.3 SlideSparseLinearMethod.create_weights() 设计

**关键点**：在 `create_weights()` 方法里，定义的 Parameter shape 必须和 `.safetensors` 里的 shape 一致。

**伪代码**：
```python
def create_weights(
    self,
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    # 计算 slide 后的扩展尺寸
    K_expanded = calculate_expanded_k(input_size, self.sparsity)
    # 2:4 压缩后减半
    K_compressed = K_expanded // 2
    
    # 创建压缩后的权重参数，shape 为 [N, K'/2]
    weight = Parameter(
        torch.empty(
            sum(output_partition_sizes),
            K_compressed,
            dtype=params_dtype,
        ),
        requires_grad=False,
    )
    layer.register_parameter("weight", weight)
    
    # 创建稀疏元数据参数，shape 为 [N, K'/8]
    weight_meta = Parameter(
        torch.empty(
            sum(output_partition_sizes),
            K_expanded // 8,
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.register_parameter("weight_meta", weight_meta)
    
    # 创建权重 scale
    weight_scale = create_fp8_scale_parameter(output_partition_sizes)
    layer.register_parameter("weight_scale", weight_scale)
```

### 7.4 关于 CompressedTensor 的说明

**当前理解**：vLLM 中的 `CompressedTensor`（位于 `vllm/model_executor/layers/quantization/compressed_tensors/`）主要用于加载已经量化压缩的权重格式（如 INT4/INT8 的 GPTQ、AWQ 等），它与我们需要的"经过离线处理后的 2:4 稀疏压缩权重格式"是不同的概念。

**我们的处理方式**：
- 不需要依赖 vLLM 现有的 `CompressedTensor` 机制
- 直接在 `SlideSparseLinearMethod.create_weights()` 中定义正确的参数形状
- 离线脚本保存的 `.safetensors` 文件只需包含与之匹配的 Tensor

### 7.5 vLLM 官方稀疏示例参考

vLLM 已有 2:4 稀疏的基础支持（位于 `vllm/_custom_ops.py`）：

```python
# 稀疏压缩函数
def cutlass_sparse_compress(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    输入: a [M, K]，满足 2:4 稀疏
    输出:
        - a_nzs [M, K/2]: 非零元素
        - a_meta [M, K/8]: 稀疏元数据
    """
    return torch.ops._C.cutlass_sparse_compress(a)

# 稀疏 GEMM 函数
def cutlass_scaled_sparse_mm(
    a: torch.Tensor,        # [M, K] 稠密激活
    bt_nzs: torch.Tensor,   # [N, K/2] 压缩后的权重非零元素
    bt_meta: torch.Tensor,  # [N, K/8] 权重稀疏元数据
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """输出: [M, N]"""
    ...
```

这些 API 可以直接复用，简化我们的实现。

### 7.6 不需要修改 Loader 核心逻辑

**最终方案**：
1. 离线脚本生成 `.safetensors`，key 名称不变，shape 改变
2. `SlideSparseLinearMethod.create_weights()` 定义与之匹配的 Parameter shape
3. 使用 vLLM 默认的 `DefaultModelLoader` 加载
4. 在 `process_weights_after_loading()` 中进行必要的后处理（如设置 sparsity 标记）

这种方式的修改量最小，只需要：
- 新增 `SlideSparseLinearMethod` 类
- 在量化配置注册表中注册
- 不需要修改任何 Loader 代码

---

## 8. Phase 6：Sparse 搭建和 Kernel 替换（cuSPARSELt）

### 8.1 核心目标

这是最关键的工作阶段。需要：
1. 新建 `SlideSparseLinearMethod` 类，继承自 `LinearMethodBase`
2. 开发 Quant+Slide 融合的 Triton kernel（FP8 和 INT8 两个版本）
3. 开发 cuSPARSELt GEMM+Dequant 的封装（FP8 和 INT8 两个版本）
4. 处理 A100 的 Row-Major 输出性能问题

### 8.2 SlideSparseLinearMethod 类设计

**文件位置**：`slidesparse/core/slidesparse_linear_method.py`

**类结构**：
```
SlideSparseLinearMethod 继承 LinearMethodBase
├── __init__(self, quant_config)
├── create_weights()           # 创建压缩后的权重参数
├── apply()                    # 核心：quant+slide → sparse_gemm → dequant
└── process_weights_after_loading()  # 权重后处理
```

### 8.3 在线 Kernel 工具链

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              在线工具链 (Online)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────────────┐     ┌──────────────────┐     ┌────────────────────┐ │
│   │ Fused Quant + Slide  │ ──► │ Sparse GEMM      │ ──► │     Dequant        │ │
│   │      (Triton)        │     │   (cuSPARSELt)   │     │    (Triton)        │ │
│   │                      │     │                  │     │                    │ │
│   │ BF16 [M,K]           │     │ FP8/INT8 [M,K']  │     │ FP32/INT32 [M,N]   │ │
│   │    ↓                 │     │    ↓             │     │    ↓               │ │
│   │ FP8/INT8 [M,K']      │     │ FP32/INT32 [M,N] │     │ BF16 [M,N]         │ │
│   └──────────────────────┘     └──────────────────┘     └────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 8.3.1 Fused Quant + Slide Kernel (Triton)

**功能**：将 BF16 输入激活进行在线量化（FP8/INT8）并执行 Slide 扩展

**文件位置**：`slidesparse/kernels/fused_quant_slide.py`

**输入**：
- 输入激活：`[M, K]`，BF16 精度
- 稀疏格式参数：`Z:L`

**输出**：
- 量化+滑动后的激活：`[M, K']`，FP8/INT8 精度
- Scale：`[M]`（per-token）或 `[1]`（per-tensor）

**Triton Kernel 设计**：
1. **块级并行**：按 `(BLOCK_M, BLOCK_K)` 划分输入
2. **动态量化**：
   - 计算 `x_max = max(abs(x))`
   - 计算 `scale = x_max / 448.0`（FP8 E4M3）或 `/ 127.0`（INT8）
   - 量化：`x_quant = clamp(x / scale, -max_val, max_val)`
3. **滑动重排**：
   - 对每个滑动窗口 `w`，从量化结果中提取 `L_tgt` 个元素
   - 写入到扩展后的目标位置 `dst_group_start + w * L_tgt`

**需要开发两个版本**：
- `fused_quant_slide_fp8()` — FP8 E4M3 量化
- `fused_quant_slide_int8()` — INT8 量化

#### 8.3.2 Sparse GEMM (cuSPARSELt)

**参考文档**：https://docs.nvidia.com/cuda/cusparselt/functions.html

**功能**：执行 2:4 结构化稀疏矩阵乘法

**文件位置**：`slidesparse/kernels/cusparselt_gemm.py`（Python 封装）、`slidesparse/csrc/cusparselt_gemm.cu`（CUDA 实现）

**输入**：
- 量化+滑动后的激活：`[M, K']`，FP8/INT8
- 压缩后的权重：`[N, K'/2]`
- 稀疏元数据：`[N, K'/8]`
- 算法 ID：从离线搜索的 JSON 配置获取

**输出**：
- GEMM 结果：`[M, N]`，FP32/INT32

**cuSPARSELt 关键 API 调用**：
1. `cusparseLtInit()` — 初始化库
2. `cusparseLtStructuredDescriptorInit()` — 初始化稀疏矩阵描述符
3. `cusparseLtDenseDescriptorInit()` — 初始化稠密矩阵描述符
4. `cusparseLtMatmulDescriptorInit()` — 初始化乘法描述符
5. `cusparseLtMatmulAlgSelectionInit()` — 选择算法
6. `cusparseLtMatmulPlanInit()` — 初始化执行计划
7. `cusparseLtMatmul()` — 执行稀疏矩阵乘法

#### 8.3.3 Dequant Kernel (Triton)

**功能**：将 GEMM 输出反量化为 BF16

**文件位置**：`slidesparse/kernels/dequant.py`

**输入**：
- GEMM 输出：`[M, N]`，FP32/INT32
- scale_a：激活 scale
- scale_b：权重 scale

**输出**：
- 反量化结果：`[M, N]`，BF16

**计算公式**：
```
output_bf16 = input_fp32 * scale_a * scale_b
```

### 8.4 A100 Row-Major 输出性能问题及解决方案

#### 8.4.1 问题描述

通过测试发现，在 A100 GPU 上，cuSPARSELt 存在一个性能问题：

**假设 A×B=C，其中 A 和 B 都是行主序（Row-Major）**：
- 如果输出 C 是**行主序（Row-Major）**：性能比列主序慢约 30%，无法获得 2x 的稀疏加速收益
- 如果输出 C 是**列主序（Column-Major）**：可以获得完整的 2x 稀疏加速收益

**在 H100 和 B200 上**：C 是行主序或列主序的速度都是一样的，没有这个问题。

#### 8.4.2 影响分析

这意味着如果要在 A100 上跑出 2x 的速度，输出 C 必须是列主序的（即经过转置的）。而我们不希望这个列主序继续传递到后续的计算中，需要在 `apply()` 方法内部解决它。

**曾经考虑的方案**：
1. GEMM 输出列主序 C → Dequant + Transpose 融合 → 行主序 BF16 输出
2. 但现在 GEMM 如果要和 Dequant 融合，Transpose 就要单独做，很尴尬

#### 8.4.3 最终决定：放弃 Transpose

**决定**：关于 A100 的这个问题，我们选择放弃 Transpose。

**理由**：
1. 这是 NVIDIA cuSPARSELt 库的 Implementation Limitation，不是我们引入的问题
2. Transpose 找不到合适的融合点
3. 这个问题仅在 A100 上存在，H100 和 B200 没有这个问题
4. 单独做 Transpose 会引入额外的内存读写开销

**最终方案**：
- 让 cuSPARSELt 始终输出 **Row-Major** 格式
- 在 A100 上接受"诚实的、有效的提升"（虽然不是 2x，但仍有显著提升）
- 在 H100/B200 上可以获得完整的稀疏加速收益

**在文档和论文中需要说明**：
- A100 上 cuSPARSELt Row-Major 输出慢 30% 是 NVIDIA 库的已知问题
- 属于库的实现限制（Implementation Limitation），非本方案引入的问题

### 8.5 apply() 方法完整设计

**伪代码**：
```python
def apply(self, layer, x, bias=None):
    # 1. Fused Quant + Slide
    # BF16 [M, K] → FP8/INT8 [M, K']
    x_quant, scale_a = fused_quant_slide(
        x,
        src_sparsity=self.quant_config.sparsity,
        tgt_sparsity="2:4",
        dtype=self.activation_dtype,  # "fp8" or "int8"
    )
    
    # 2. Sparse GEMM (cuSPARSELt)
    # FP8/INT8 [M, K'] × Compressed [N, K'/2] → FP32/INT32 [M, N]
    output_raw = cusparselt_gemm(
        x_quant,           # [M, K']
        layer.weight,      # [N, K'/2] (compressed)
        layer.weight_meta, # [N, K'/8] (metadata)
        algo_id=self.get_algo_id(x.shape[0], layer.weight.shape[0], x_quant.shape[1]),
    )
    
    # 3. Dequant
    # FP32/INT32 [M, N] → BF16 [M, N]
    output = dequant(
        output_raw,
        scale_a=scale_a,
        scale_b=layer.weight_scale,
        out_dtype=torch.bfloat16,
    )
    
    # 4. Bias (如有)
    if bias is not None:
        output = output + bias
    
    return output
```

### 8.6 Kernel 文件组织

```
slidesparse/
├── kernels/
│   ├── __init__.py
│   ├── fused_quant_slide.py      # Triton: Quant+Slide 融合
│   ├── cusparselt_gemm.py        # Python: cuSPARSELt GEMM 封装
│   └── dequant.py                # Triton: 反量化
├── csrc/
│   ├── cusparselt_gemm.cu        # CUDA: cuSPARSELt C++ 实现
│   └── torch_bindings.cpp        # PyTorch C++ 绑定
└── core/
    └── slidesparse_linear_method.py  # SlideSparseLinearMethod 类
```

---

## 9. Phase 7：新模型引入（BitNet-1.58b）

### 9.1 BitNet 模型概述

**BitNet-1.58b** 是 Microsoft 提出的三元量化（Ternary Quantization）模型，权重和激活都被量化为 `+1/0/-1` 三个值。

**参考模型**：https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16

### 9.2 BitNet 与标准 Transformer 的区别

| 组件 | 标准 Transformer | BitNet |
|------|-----------------|--------|
| 线性层 | `nn.Linear` | `BitLinear`（三元权重） |
| 激活函数 | SiLU/GELU | ReLU² |
| LayerNorm | RMSNorm | 特殊 Norm |
| 权重格式 | FP16/BF16 | Ternary (+1/0/-1) |
| 激活量化 | 无或 FP8/INT8 | Ternary |

### 9.3 在 vLLM 中实现 BitNet

#### 9.3.1 新建模型文件

**文件位置**：`vllm/model_executor/models/bitnet.py`

**实现策略**：复用 Llama 的大部分代码，替换以下部分：

1. **MLP 层**：
   - 替换激活函数为 `ReLU²`（`x * relu(x)`）
   - 使用 `BitLinear` 替换标准线性层

2. **Attention 层**：
   - 使用 `BitLinear` 替换 QKV 投影和 O 投影

3. **Norm 层**：
   - 根据 BitNet 规范调整 Normalization

#### 9.3.2 BitLinear 实现

`BitLinear` 需要实现三元量化：

**权重量化（离线）**：
```python
def ternary_quantize_weight(weight):
    """将 BF16 权重量化为 +1/0/-1"""
    scale = weight.abs().mean()
    weight_ternary = torch.sign(weight) * (weight.abs() > 0.5 * scale)
    return weight_ternary, scale
```

**激活量化（在线）**：
```python
def ternary_quantize_activation(x):
    """将 BF16 激活量化为 +1/0/-1"""
    scale = x.abs().max()
    x_ternary = torch.sign(x) * (x.abs() > 0.5 * scale)
    return x_ternary, scale
```

#### 9.3.3 与 SlideSparseLinearMethod 的兼容性

**目标**：确保 BitNet 模型可以使用 `cuBLASLtLinearMethod` 和 `SlideSparseLinearMethod` 进行计算。

**实现方式**：
1. BitNet 的 `BitLinear` 继承自 `LinearBase`
2. 通过 `quant_config` 参数传递量化配置
3. `BitLinear` 的 forward 方法调用 `self.quant_method.apply()`

这样，BitNet 模型就可以：
- 使用原生 vLLM 的 Kernel 进行基线测试
- 使用 `cuBLASLtLinearMethod` 进行 Dense 基线测试
- 使用 `SlideSparseLinearMethod` 进行稀疏加速测试

### 9.4 离线权重处理

对于 BitNet 模型，离线处理流程为：

1. **加载 BF16 权重**：从 HuggingFace 下载
2. **三元量化**：将权重量化为 +1/0/-1
3. **Prune**：在三元权重上进行结构化剪枝
4. **Slide**：执行滑动扩展
5. **Compress**：2:4 稀疏压缩
6. **保存**：生成 `.safetensors` 文件

### 9.5 注册 BitNet 模型

**修改文件**：`vllm/model_executor/models/registry.py`

在模型注册表中添加 BitNet：
```python
# 在 _MODELS 字典中添加
"BitnetForCausalLM": ("bitnet", "BitnetForCausalLM"),
```

---

## 10. 代码组织结构

### 10.1 slidesparse/ 外挂模块

所有新增代码都放在仓库根目录的 `slidesparse/` 文件夹中，与 `/vllm`、`/csrc` 同级：

```
vllmbench/                           # 仓库根目录
├── vllm/                            # vLLM 官方源码（最小侵入修改）
├── csrc/                            # vLLM CUDA 源码
├── slidesparse/                     # SlideSparse 新增模块（外挂）
│   ├── __init__.py
│   │
│   ├── offline/                     # 离线工具链
│   │   ├── __init__.py
│   │   ├── prune.py                # 权重剪枝
│   │   ├── slide.py                # 权重滑动
│   │   ├── compress.py             # 权重压缩
│   │   ├── search.py               # 算法搜索
│   │   ├── autotune.py             # AutoTune
│   │   └── preprocess_weights.py   # 完整预处理脚本
│   │
│   ├── kernels/                     # Kernel 实现
│   │   ├── __init__.py
│   │   ├── fused_quant_slide.py    # Triton: Quant+Slide 融合
│   │   ├── cusparselt_gemm.py      # Python: cuSPARSELt 封装
│   │   ├── cublaslt_gemm.py        # Python: cuBLASLt 封装
│   │   └── dequant.py              # Triton: 反量化
│   │
│   ├── csrc/                        # CUDA 源码
│   │   ├── cusparselt_gemm.cu      # cuSPARSELt C++ 实现
│   │   ├── cublaslt_gemm.cu        # cuBLASLt C++ 实现
│   │   └── torch_bindings.cpp      # PyTorch C++ 绑定
│   │
│   ├── core/                        # 核心逻辑
│   │   ├── __init__.py
│   │   ├── slidesparse_config.py   # SlideSparseConfig
│   │   ├── slidesparse_linear_method.py  # SlideSparseLinearMethod
│   │   └── cublaslt_linear_method.py     # cuBLASLtLinearMethod
│   │
│   ├── test/                        # 测试文件
│   │   ├── test_kernels.py
│   │   ├── test_offline_tools.py
│   │   └── test_e2e.py
│   │
│   └── bench/                       # 性能测试
│       ├── bench_kernels.py
│       └── bench_e2e.py
│
└── ...
```

### 10.2 vLLM 源码最小侵入修改

只在 vLLM 的几个 `__init__.py` 或 registry 文件中添加必要的注册代码。

#### 10.2.1 量化配置注册

**文件**：`vllm/model_executor/layers/quantization/slidesparse.py`（新建空壳文件）

```python
"""SlideSparse 量化配置的 vLLM 集成入口"""

import sys
import os

# 将 slidesparse 模块添加到 Python 路径
# 仓库根目录: vllm/model_executor/layers/quantization/ -> 回退 4 层到仓库根目录
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
SLIDESPARSE_PATH = os.path.join(REPO_ROOT, "slidesparse")
if SLIDESPARSE_PATH not in sys.path:
    sys.path.insert(0, SLIDESPARSE_PATH)

# 从外挂模块导入
from slidesparse.core.slidesparse_config import SlideSparseConfig
from slidesparse.core.slidesparse_linear_method import SlideSparseLinearMethod

__all__ = ["SlideSparseConfig", "SlideSparseLinearMethod"]
```

#### 10.2.2 注册量化配置

**文件**：`vllm/model_executor/layers/quantization/__init__.py`

在 `QuantizationMethods` 类型定义中添加：
```python
"slidesparse",
"cublaslt",
```

在 `method_to_config` 字典中添加：
```python
from .slidesparse import SlideSparseConfig
# ...
"slidesparse": SlideSparseConfig,
```

### 10.3 CLI 参数扩展

通过 CLI 参数决定使用什么 LinearMethod 和权重加载方式：

```bash
# 使用 cuBLASLt Dense 基线
vllm serve model_name --enable-cublaslt

# 使用 SlideSparse 稀疏加速
vllm serve ./checkpoints/model-slidesparse --enable-slidesparse

# 或通过 quantization 参数
vllm serve ./checkpoints/model-slidesparse --quantization slidesparse
```

### 10.4 原仓库代码保持简洁

**修改原则**：
- 只在 `__init__.py` 或 registry 字典里添加几行注册代码
- 不修改核心逻辑代码
- 所有新功能通过外挂模块实现
- 通过 `sys.path` 机制实现模块导入

---

## 11. 测试与验证

### 11.1 单元测试

#### 11.1.1 Kernel 测试

**测试文件**：`slidesparse/test/test_kernels.py`

**测试类 `TestFusedQuantSlide`**：
- `test_output_shape`：验证不同 M、K、sparsity 组合的输出形状正确性
- `test_quantization_accuracy`：验证量化精度在可接受范围内
- `test_slide_correctness`：验证 slide 操作的正确性

**测试类 `TestSparseGemm`**：
- `test_correctness`：验证稀疏 GEMM 的计算正确性（对比 Dense GEMM）
- `test_performance`：验证稀疏 GEMM 的性能收益

**测试命令**：
```bash
pytest slidesparse/test/test_kernels.py -v
```

#### 11.1.2 离线工具测试

**测试文件**：`slidesparse/test/test_offline_tools.py`

- `test_prune`：验证剪枝后的权重满足稀疏约束
- `test_slide`：验证 slide 后的 K 维度扩展正确
- `test_compress`：验证压缩后的权重形状正确

### 11.2 端到端测试

#### 11.2.1 吞吐量测试

**测试流程**：
1. 使用预处理脚本生成 SlideSparse 权重
2. 运行 SlideSparse 模型的吞吐测试
3. 运行 baseline 模型的吞吐测试
4. 对比加速比

**测试命令**：
```bash
# 预处理权重
python slidesparse/offline/preprocess_weights.py \
    --input-model meta-llama/Llama-3.2-1B-Instruct \
    --output-dir ./checkpoints/llama-3.2-1b-slidesparse \
    --sparsity 2:8

# SlideSparse 吞吐测试
vllm bench throughput \
    --model ./checkpoints/llama-3.2-1b-slidesparse \
    --quantization slidesparse \
    --input-len 128 --output-len 128 --num-prompts 100

# Baseline 吞吐测试（FP8）
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --quantization fp8 \
    --input-len 128 --output-len 128 --num-prompts 100
```

### 11.3 精度评估

#### 11.3.1 PPL 测试

**使用 lm-eval 进行 PPL 评估**：
```bash
# Baseline PPL
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,quantization=fp8 \
    --tasks wikitext --num_fewshot 0

# SlideSparse PPL
lm_eval --model vllm \
    --model_args pretrained=./checkpoints/llama-3.2-1b-slidesparse,quantization=slidesparse \
    --tasks wikitext --num_fewshot 0
```

**预期结果**：
- PPL 增加应在可接受范围内（如 < 5%）
- 不同稀疏度（2:4, 2:6, 2:8, 2:10, 2:12）会有不同的精度损失

### 11.4 完整实验矩阵

| 维度 | 变量 |
|------|------|
| **GPU** | A100, H100, B200 |
| **模型** | Qwen2.5-0.5B/1.5B/3B/7B/14B, Llama3.2-1B/3B, BitNet-2B |
| **稀疏度** | 2:4, 2:6, 2:8, 2:10, 2:12 |
| **量化类型** | FP8, INT8 |
| **阶段** | Prefill, Decode |
| **指标** | 吞吐量 (tokens/s), PPL |

---

## 12. 附录：关键文件路径速查表

### 12.1 vLLM 核心文件

| 功能 | 文件路径 |
|------|---------|
| **入口点** | |
| LLM 类 | `vllm/entrypoints/llm.py` |
| CLI 入口 | `vllm/entrypoints/cli/main.py` |
| 吞吐测试 | `vllm/entrypoints/cli/benchmark/throughput.py` |
| **模型加载** | |
| 加载器入口 | `vllm/model_executor/model_loader/__init__.py` |
| 基类 | `vllm/model_executor/model_loader/base_loader.py` |
| 默认加载器 | `vllm/model_executor/model_loader/default_loader.py` |
| **模型定义** | |
| Llama | `vllm/model_executor/models/llama.py` |
| Qwen2 | `vllm/model_executor/models/qwen2.py` |
| 模型注册 | `vllm/model_executor/models/registry.py` |
| **线性层** | |
| 线性层定义 | `vllm/model_executor/layers/linear.py` |
| LinearMethodBase | `vllm/model_executor/layers/linear.py` |
| **量化** | |
| 量化配置入口 | `vllm/model_executor/layers/quantization/__init__.py` |
| 基类 | `vllm/model_executor/layers/quantization/base_config.py` |
| FP8 量化 | `vllm/model_executor/layers/quantization/fp8.py` |
| **自定义算子** | |
| 算子绑定 | `vllm/_custom_ops.py` |
| **CUDA 源码** | |
| 量化 kernel | `csrc/quantization/` |
| 稀疏 kernel | `csrc/sparse/cutlass/` |

### 12.2 SlideSparse 新增文件

| 功能 | 文件路径 |
|------|---------|
| **离线工具** | |
| 权重预处理 | `slidesparse/offline/preprocess_weights.py` |
| 权重剪枝 | `slidesparse/offline/prune.py` |
| 权重滑动 | `slidesparse/offline/slide.py` |
| 权重压缩 | `slidesparse/offline/compress.py` |
| 算法搜索 | `slidesparse/offline/search.py` |
| **Kernel** | |
| Quant+Slide | `slidesparse/kernels/fused_quant_slide.py` |
| cuSPARSELt GEMM | `slidesparse/kernels/cusparselt_gemm.py` |
| cuBLASLt GEMM | `slidesparse/kernels/cublaslt_gemm.py` |
| Dequant | `slidesparse/kernels/dequant.py` |
| **核心逻辑** | |
| SlideSparseConfig | `slidesparse/core/slidesparse_config.py` |
| SlideSparseLinearMethod | `slidesparse/core/slidesparse_linear_method.py` |
| cuBLASLtLinearMethod | `slidesparse/core/cublaslt_linear_method.py` |
| **vLLM 集成** | |
| 空壳转发 | `vllm/model_executor/layers/quantization/slidesparse.py` |
| BitNet 模型 | `vllm/model_executor/models/bitnet.py` |

### 12.3 关键函数签名速查

**vllm/model_executor/layers/linear.py**：
- `LinearMethodBase.create_weights()` — 创建权重参数
- `LinearMethodBase.apply(layer, x, bias)` — 执行线性变换
- `ColumnParallelLinear.forward(input_)` — 列并行前向
- `RowParallelLinear.forward(input_)` — 行并行前向

**vllm/model_executor/layers/quantization/base_config.py**：
- `QuantizationConfig.get_quant_method(layer, prefix)` — 获取量化方法
- `QuantizeMethodBase.process_weights_after_loading(layer)` — 权重后处理

**vllm/_custom_ops.py**：
- `scaled_fp8_quant(input, scale, ...)` — FP8 量化
- `cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)` — CUTLASS GEMM
- `cutlass_sparse_compress(a)` — 稀疏压缩
- `cutlass_scaled_sparse_mm(...)` — 稀疏 GEMM

**vllm/model_executor/layers/quantization/__init__.py**：
- `register_quantization_config(quantization)` — 注册量化配置（装饰器）
- `get_quantization_config(quantization)` — 获取量化配置

**vllm/model_executor/model_loader/__init__.py**：
- `register_model_loader(load_format)` — 注册模型加载器（装饰器）
- `get_model_loader(load_config)` — 获取模型加载器

---

## 结语

本文档详细描述了 SlideSparse 在 vLLM 框架中的完整集成方案，包括：

1. **理论基础**：SlideSparse 的核心原理和创新点
2. **七阶段工程流程**：从环境准备到最终实验的完整路径
3. **离线工具链**：权重 Prune、Slide、Compress、Search
4. **在线 Kernel**：Quant+Slide、Sparse GEMM、Dequant
5. **模型加载**：利用 vLLM 宽容的 DefaultModelLoader
6. **A100 兼容性**：放弃 Transpose，接受诚实的性能提升
7. **新模型支持**：BitNet-1.58b 的集成方案
8. **代码组织**：slidesparse/ 外挂模块 + 最小侵入修改

**关键设计决策**：
- 使用 cuSPARSELt 而非 CUTLASS Sparse（性能更优）
- 放弃 A100 上的 Transpose 优化（库限制）
- 采用外挂模块设计（保持原仓库简洁）
- 利用 create_weights 机制（无需重写 Loader）


