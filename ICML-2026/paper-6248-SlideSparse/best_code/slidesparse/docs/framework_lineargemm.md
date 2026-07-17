# vLLM 线性层与 GEMM 调用链详解 (Framework Linear & GEMM)

本文档专门分析 vLLM 中线性层的实现，包括四个关键投影层（Wqkv、Wo、W13、W2）的位置，以及量化、反量化和 GEMM 乘法的具体调用位置。本文档面向需要深入理解或修改 vLLM 底层 GEMM 计算的开发者。

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [Transformer 中的线性层](#2-transformer-中的线性层)
3. [四个关键线性层的位置](#3-四个关键线性层的位置)
4. [线性层类的详细定义](#4-线性层类的详细定义)
5. [量化方法与 GEMM 实现](#5-量化方法与-gemm-实现)
6. [FP8 量化详解](#6-fp8-量化详解)
7. [AWQ 量化详解](#7-awq-量化详解)
8. [GPTQ 量化详解](#8-gptq-量化详解)
9. [完整调用链图](#9-完整调用链图)
10. [CUDA/Triton Kernel 详解](#10-cudatriton-kernel-详解)
11. [如何替换自定义 Kernel](#11-如何替换自定义-kernel)
12. [实战示例](#12-实战示例)
13. [总结与最佳实践](#13-总结与最佳实践)

---

## 1. 背景与目标

### 1.1 什么是 GEMM？

GEMM（General Matrix Multiply）是深度学习中最核心的操作之一，尤其在 Transformer 模型中。每个线性层（Linear Layer）本质上就是一个 GEMM 操作：

```
Output = Input × Weight^T + Bias
```

数学表示：
```
Y[M, N] = X[M, K] × W[K, N]^T + b[N]

其中：
- M: batch_size × seq_len (token 数量)
- K: 输入维度 (hidden_size)
- N: 输出维度 (output_size)
```

### 1.2 为什么要关注 GEMM？

在 LLM 推理中，**GEMM 占据了约 70-80% 的计算时间**。优化 GEMM 是提升推理性能的关键：

```
典型 7B 模型每个 token 的计算分布：
┌─────────────────────────────────────────────────────────────────┐
│  GEMM (线性层)     ████████████████████████████████████  ~75%  │
│  Attention         ██████████                           ~15%  │
│  LayerNorm/Act     ████                                 ~7%   │
│  其他              ██                                   ~3%   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 你的目标

如果你计划替换 vLLM 推理过程中的 GEMM，通常需要处理以下三个步骤：

1. **Quant（量化）**: 将 FP16/BF16 激活值量化为低精度（如 FP8、INT8、INT4）
2. **GEMM（矩阵乘法）**: 执行量化后的矩阵乘法
3. **Dequant（反量化）**: 将结果反量化回 FP16/BF16（通常融合在 GEMM kernel 中）

```
FP16 Input                FP8 Input             FP8 Output           FP16 Output
    │                         │                     │                     │
    │    ┌───────────┐        │    ┌──────────┐     │    ┌───────────┐    │
    └───►│   Quant   │───────►│───►│   GEMM   │────►│───►│  Dequant  │───►│
         └───────────┘             └──────────┘          └───────────┘
             
         (scale_a)                 (FP8 × FP8)         (× scale_a × scale_b)
```

---

## 2. Transformer 中的线性层

### 2.1 Decoder-only Transformer 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Decoder Block (× N 层)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Multi-Head Self-Attention                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Input ──► QKV Projection (Wqkv) ──► Q, K, V                │  │  │
│  │  │                    │                                        │  │  │
│  │  │              (GEMM #1)                                      │  │  │
│  │  │  Q, K, V ──► Attention Scores ──► Attention Output         │  │  │
│  │  │                                                             │  │  │
│  │  │  Attention Output ──► Output Projection (Wo) ──► Output    │  │  │
│  │  │                              │                              │  │  │
│  │  │                        (GEMM #2)                           │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                           MLP (FFN)                                │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │  Input ──► Gate+Up Projection (W13) ──► Gate, Up           │  │  │
│  │  │                      │                                      │  │  │
│  │  │                (GEMM #3)                                    │  │  │
│  │  │  Gate, Up ──► SiLU(Gate) × Up ──► Intermediate             │  │  │
│  │  │                                                             │  │  │
│  │  │  Intermediate ──► Down Projection (W2) ──► Output          │  │  │
│  │  │                           │                                 │  │  │
│  │  │                     (GEMM #4)                               │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 四个关键 GEMM 操作

| GEMM | 名称 | 数学表达式 | 维度 |
|------|------|-----------|------|
| **#1 Wqkv** | QKV 投影 | `[Q, K, V] = X × W_qkv` | `[M, H] → [M, 3*H]` 或 `[M, H+2*KV]` |
| **#2 Wo** | 输出投影 | `O = Attn(Q,K,V) × W_o` | `[M, H] → [M, H]` |
| **#3 W13** | Gate+Up 投影 | `[Gate, Up] = X × W_13` | `[M, H] → [M, 2*I]` |
| **#4 W2** | Down 投影 | `Y = Act(Gate) * Up × W_2` | `[M, I] → [M, H]` |

其中：
- `M`: 序列长度（token 数）
- `H`: hidden_size（隐藏层维度）
- `I`: intermediate_size（中间层维度，通常 `I = 4*H` 或 `I = 8/3*H`）
- `KV`: KV head 维度（用于 GQA/MQA）

### 2.3 典型模型参数

| 模型 | hidden_size | intermediate_size | num_heads | num_kv_heads |
|------|-------------|-------------------|-----------|--------------|
| Llama-7B | 4096 | 11008 | 32 | 32 |
| Llama-70B | 8192 | 28672 | 64 | 8 |
| Qwen2.5-7B | 3584 | 18944 | 28 | 4 |
| Mistral-7B | 4096 | 14336 | 32 | 8 |

---

## 3. 四个关键线性层的位置

### 3.1 线性层定义总览

| 线性层 | 对应模型层 | 类名 | 文件位置 |
|-------|----------|------|---------|
| **Wqkv** | `self_attn.qkv_proj` | `QKVParallelLinear` | `vllm/model_executor/layers/linear.py` |
| **Wo** | `self_attn.o_proj` | `RowParallelLinear` | `vllm/model_executor/layers/linear.py` |
| **W13** | `mlp.gate_up_proj` | `MergedColumnParallelLinear` | `vllm/model_executor/layers/linear.py` |
| **W2** | `mlp.down_proj` | `RowParallelLinear` | `vllm/model_executor/layers/linear.py` |

### 3.2 在 Qwen2 模型中的使用位置

**文件**: `vllm/model_executor/models/qwen2.py`

```python
# =====================================================
# Attention 层中的线性层
# =====================================================
class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,  # ← 量化配置
        prefix: str = "",
        ...
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        
        # 计算每个 GPU 上的 head 数量（张量并行）
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # =====================================================
        # Wqkv - QKV 合并投影 (GEMM #1)
        # 输入: [batch, seq, hidden_size]
        # 输出: [batch, seq, q_size + 2 * kv_size]
        # =====================================================
        self.qkv_proj = QKVParallelLinear(
            hidden_size,                     # input_size
            self.head_dim,                   # head_size
            self.total_num_heads,            # total_num_heads
            self.total_num_kv_heads,         # total_num_kv_heads (用于 GQA)
            bias=True,                       # Qwen2 有 bias
            quant_config=quant_config,       # 传递量化配置
            prefix=f"{prefix}.qkv_proj",     # 用于量化层匹配
        )
        
        # =====================================================
        # Wo - 输出投影 (GEMM #2)
        # 输入: [batch, seq, num_heads * head_dim]
        # 输出: [batch, seq, hidden_size]
        # =====================================================
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,  # input_size
            hidden_size,                           # output_size
            bias=False,                            # Qwen2 o_proj 无 bias
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        
        # RoPE 和 Attention
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ...
    ) -> torch.Tensor:
        # ⭐ GEMM #1: QKV 投影
        qkv, _ = self.qkv_proj(hidden_states)
        
        # 分割 Q, K, V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 应用 RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Attention 计算
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        
        # ⭐ GEMM #2: 输出投影
        output, _ = self.o_proj(attn_output)
        return output


# =====================================================
# MLP 层中的线性层
# =====================================================
class Qwen2MLP(nn.Module):
    """MLP with SwiGLU activation."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        
        # =====================================================
        # W13 - Gate 和 Up 合并投影 (GEMM #3)
        # 输入: [batch, seq, hidden_size]
        # 输出: [batch, seq, 2 * intermediate_size]
        # Gate 和 Up 合并为一次 GEMM 以提高效率
        # =====================================================
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,                     # input_size
            [intermediate_size] * 2,         # [gate_size, up_size]
            bias=False,                      # SwiGLU 通常无 bias
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        
        # =====================================================
        # W2 - Down 投影 (GEMM #4)
        # 输入: [batch, seq, intermediate_size]
        # 输出: [batch, seq, hidden_size]
        # =====================================================
        self.down_proj = RowParallelLinear(
            intermediate_size,               # input_size
            hidden_size,                     # output_size
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        
        # SwiGLU 激活函数
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported.")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ⭐ GEMM #3: Gate + Up 投影
        gate_up, _ = self.gate_up_proj(x)
        
        # SiLU(gate) * up
        x = self.act_fn(gate_up)
        
        # ⭐ GEMM #4: Down 投影
        x, _ = self.down_proj(x)
        return x
```

### 3.3 在 Llama 模型中的使用位置

**文件**: `vllm/model_executor/models/llama.py`

结构与 Qwen2 几乎相同，只是类名不同。vLLM 中许多模型（Mistral、Yi、InternLM 等）都复用 Llama 的实现：

```python
# LlamaAttention 使用：
self.qkv_proj = QKVParallelLinear(...)  # Wqkv
self.o_proj = RowParallelLinear(...)     # Wo

# LlamaMLP 使用：
self.gate_up_proj = MergedColumnParallelLinear(...)  # W13
self.down_proj = RowParallelLinear(...)              # W2
```

---

## 4. 线性层类的详细定义

**文件位置**: `vllm/model_executor/layers/linear.py`

### 4.1 类继承结构

```
LinearBase (CustomOp)                    # 基类，定义通用接口
    │
    ├── ReplicatedLinear                 # 复制线性层（无并行）
    │       └── 用于不需要并行的场景
    │
    ├── ColumnParallelLinear             # 列并行线性层
    │   │   └── 输出维度沿 GPU 切分
    │   │   └── 每个 GPU 存储 weight 的一部分列
    │   │   └── 前向需要 all-gather 输出
    │   │
    │   ├── MergedColumnParallelLinear   # 合并列并行 (W13)
    │   │       └── 合并多个列并行层（Gate + Up）
    │   │
    │   └── QKVParallelLinear            # QKV 并行 (Wqkv)
    │           └── 专门为 Q/K/V 投影优化
    │           └── 支持 GQA (num_kv_heads < num_heads)
    │
    └── RowParallelLinear                # 行并行线性层 (Wo, W2)
            └── 输入维度沿 GPU 切分
            └── 每个 GPU 存储 weight 的一部分行
            └── 前向需要 all-reduce 输出
```

### 4.2 张量并行可视化

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ColumnParallelLinear (如 Wqkv)                       │
│                                                                         │
│  Input X [M, K]          Weight W [K, N]           Output Y [M, N]      │
│      │                   (全部权重)                    │                │
│      │                       ↓                        │                │
│      │                   切分为 N 份                   │                │
│      │                       ↓                        │                │
│  ┌───┴───┐         ┌────────────────────┐         ┌───┴───┐            │
│  │ GPU 0 │    ×    │ W[:, 0:N/tp]       │    =    │ Y_0   │            │
│  │ GPU 1 │    ×    │ W[:, N/tp:2N/tp]   │    =    │ Y_1   │            │
│  │  ...  │    ×    │      ...           │    =    │ ...   │            │
│  └───────┘         └────────────────────┘         └───────┘            │
│                                                       ↓                │
│                                              All-Gather (如需要)        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      RowParallelLinear (如 Wo, W2)                       │
│                                                                         │
│  Input X [M, K]          Weight W [K, N]           Output Y [M, N]      │
│  (从上层获得切分)         (全部权重)                    │                │
│      │                       ↓                        │                │
│  切分为 K 份             切分为 K 份                   │                │
│      ↓                       ↓                        │                │
│  ┌───────┐         ┌────────────────────┐         ┌───────┐            │
│  │ X_0   │    ×    │ W[0:K/tp, :]       │    =    │ Y_0   │            │
│  │ X_1   │    ×    │ W[K/tp:2K/tp, :]   │    =    │ Y_1   │            │
│  │  ...  │    ×    │      ...           │    =    │ ...   │            │
│  └───────┘         └────────────────────┘         └───────┘            │
│                                                       ↓                │
│                                                 All-Reduce              │
│                                                       ↓                │
│                                                   Final Y               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 核心代码：forward 方法中的 GEMM 调用

所有线性层的 forward 方法最终都调用 `self.quant_method.apply()`：

```python
# vllm/model_executor/layers/linear.py

# =====================================================
# ColumnParallelLinear.forward() (行 557-575)
# 用于 Wqkv, W13 等输出需要切分的层
# =====================================================
def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward of ColumnParallelLinear.
    
    Args:
        input_: [batch, seq, input_size]
        
    Returns:
        output: [batch, seq, output_size_per_partition]
        output_bias: bias if skip_bias_add else None
    """
    bias = self.bias if not self.skip_bias_add else None
    
    # ⭐ 核心 GEMM 调用 - 通过量化方法
    # 这里是实际的矩阵乘法发生的地方
    assert self.quant_method is not None
    output_parallel = self.quant_method.apply(self, input_, bias)
    
    # 如果需要收集所有 GPU 的输出
    if self.gather_output and self.tp_size > 1:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


# =====================================================
# RowParallelLinear.forward() (行 1388-1416)
# 用于 Wo, W2 等需要 all-reduce 的层
# =====================================================
def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward of RowParallelLinear.
    
    Args:
        input_: [batch, seq, input_size_per_partition]
        
    Returns:
        output: [batch, seq, output_size]
        output_bias: bias if skip_bias_add else None
    """
    # 如果输入是完整的，需要切分
    if self.input_is_parallel:
        input_parallel = input_
    else:
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.tp_size
        )
        input_parallel = splitted_input[self.tp_rank]
    
    # ⭐ 核心 GEMM 调用 - 通过量化方法
    assert self.quant_method is not None
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    output_parallel = self.quant_method.apply(self, input_parallel, bias_)
    
    # All-reduce 收集所有 GPU 的部分结果
    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel
    
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias
```

### 4.4 LinearBase 基类详解

```python
# vllm/model_executor/layers/linear.py

class LinearBase(CustomOp):
    """Base linear layer.
    
    这是所有线性层的基类，定义了通用的接口和量化方法选择逻辑。
    
    Args:
        input_size: 输入维度
        output_size: 输出维度
        skip_bias_add: 是否跳过 bias 添加（用于融合优化）
        params_dtype: 参数数据类型
        quant_config: 量化配置
        prefix: 参数名前缀（用于量化层匹配）
        return_bias: 是否返回 bias
        disable_tp: 是否禁用张量并行
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        super().__init__()
        
        # 保存参数
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        
        # 设置数据类型
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        
        self.quant_config = quant_config
        self.prefix = prefix
        
        # ⭐ 关键：根据 quant_config 选择量化方法
        # 这决定了 forward 时使用哪种 GEMM 实现
        if quant_config is None:
            # 无量化：使用标准 torch.nn.functional.linear
            self.quant_method: QuantizeMethodBase = UnquantizedLinearMethod()
        else:
            # 有量化：调用量化配置的 get_quant_method
            # 这会返回 Fp8LinearMethod, AWQLinearMethod 等
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        
        # 张量并行相关
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
```

---

## 5. 量化方法与 GEMM 实现

### 5.1 量化方法类型总览

根据 `quant_config` 的不同，`quant_method` 可能是以下类型：

| quant_method 类 | 量化类型 | 权重精度 | 激活精度 | 文件位置 |
|----------------|---------|---------|---------|---------|
| `UnquantizedLinearMethod` | 无量化 | FP16/BF16 | FP16/BF16 | `linear.py` |
| `Fp8LinearMethod` | FP8 | FP8 E4M3 | FP8 E4M3 | `quantization/fp8.py` |
| `AWQLinearMethod` | AWQ | INT4 | FP16 | `quantization/awq.py` |
| `AWQMarlinLinearMethod` | AWQ Marlin | INT4 | FP16 | `quantization/awq_marlin.py` |
| `GPTQLinearMethod` | GPTQ | INT4/INT8 | FP16 | `quantization/gptq.py` |
| `GPTQMarlinLinearMethod` | GPTQ Marlin | INT4 | FP16 | `quantization/gptq_marlin.py` |
| `BitsAndBytesLinearMethod` | BnB | INT4/INT8 | FP16 | `quantization/bitsandbytes.py` |

### 5.2 无量化情况 (UnquantizedLinearMethod)

**文件**: `vllm/model_executor/layers/linear.py` (行 196-240)

```python
class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization.
    
    这是最简单的情况，直接调用 PyTorch 的线性层实现。
    """
    
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
        """创建线性层权重。
        
        分配一个 [output_size, input_size] 的权重矩阵。
        """
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),     # 输出维度（可能被切分）
                input_size_per_partition,        # 输入维度（可能被切分）
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行线性变换。
        
        这是实际的 GEMM 调用位置。
        """
        # ⭐ GEMM 调用 - 最终调用 torch.nn.functional.linear
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

### 5.3 dispatch_unquantized_gemm 详解

**文件**: `vllm/model_executor/layers/utils.py`

```python
def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    """根据平台选择最佳的 GEMM 实现。
    
    这是 vLLM 的 GEMM 调度器，会根据当前运行的硬件平台
    选择最优的 GEMM 实现。
    """
    if current_platform.is_rocm():
        # AMD GPU：使用 ROCm 优化的 GEMM
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        # CPU：使用 CPU 优化的 GEMM
        return cpu_unquantized_gemm
    else:
        # NVIDIA GPU：使用默认 CUDA GEMM
        return default_unquantized_gemm


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """默认的 GEMM 实现（NVIDIA CUDA）。
    
    使用 PyTorch 的 F.linear，底层调用 cuBLAS GEMM。
    
    Args:
        layer: 线性层模块
        x: 输入张量 [batch, seq, input_size]
        weight: 权重张量 [output_size, input_size]
        bias: 偏置张量 [output_size]
        
    Returns:
        输出张量 [batch, seq, output_size]
    """
    # ⭐ 最终的 GEMM 调用
    # F.linear 内部调用 torch.mm 或 torch.addmm
    # 对于 CUDA，最终调用 cuBLAS GEMM
    return torch.nn.functional.linear(x, weight, bias)
```

---

## 6. FP8 量化详解 ⭐⭐⭐

FP8 是目前最先进的量化方法之一，在 NVIDIA H100 和 Ada 架构 GPU 上有原生硬件支持。

### 6.1 FP8 数据格式

```
FP8 E4M3 (权重和激活的默认格式):
┌────┬────────┬────────┐
│Sign│Exponent│Mantissa│
│ 1  │   4    │   3    │
└────┴────────┴────────┘
- 范围: ±448
- 精度: 相对 FP16 损失较小

FP8 E5M2 (一般用于梯度):
┌────┬────────┬────────┐
│Sign│Exponent│Mantissa│
│ 1  │   5    │   2    │
└────┴────────┴────────┘
- 范围: ±57344
- 精度: 较低
```

### 6.2 FP8 量化配置

**文件**: `vllm/model_executor/layers/quantization/fp8.py`

```python
class Fp8Config(QuantizationConfig):
    """Config class for FP8.
    
    FP8 量化的配置类，支持多种量化模式。
    """

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,  # 权重是否已预量化
        activation_scheme: str = "dynamic",          # 激活量化方案
        ignored_layers: list[str] | None = None,     # 忽略的层
        weight_block_size: list[int] | None = None,  # 块量化大小
    ):
        super().__init__()
        
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        
        # 激活量化方案
        # - "dynamic": 运行时计算 scale（更灵活）
        # - "static": 使用预计算的 scale（更快）
        if activation_scheme not in ["static", "dynamic"]:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        
        self.ignored_layers = ignored_layers or []
        
        # 块量化配置
        # 例如 [128, 128] 表示每 128x128 的块使用一个 scale
        self.weight_block_size = weight_block_size
    
    @classmethod
    def get_name(cls) -> str:
        return "fp8"
    
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]
    
    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # 需要 Ampere 或更新架构
    
    def get_quant_method(self, layer, prefix):
        """获取该层的量化方法。"""
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        return None
```

### 6.3 Fp8LinearMethod 类

**文件**: `vllm/model_executor/layers/quantization/fp8.py`

```python
class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    
    这是 FP8 量化的核心实现类，负责：
    1. 创建 FP8 量化的权重
    2. 在 forward 时执行 quant -> gemm -> dequant
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = quant_config.weight_block_size is not None
        
        # 根据配置选择不同的 GEMM 实现
        if self.block_quant:
            # 块量化：每个块有独立的 scale
            self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
                quant_config.weight_block_size
            )
        else:
            # Per-tensor 量化：整个张量一个 scale
            self.fp8_linear = Fp8LinearOp(
                act_quant_static=(quant_config.activation_scheme == "static"),
                act_quant_group_shape=None,
            )

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
        """创建 FP8 量化的权重和 scale 参数。"""
        output_size_per_partition = sum(output_partition_sizes)
        
        # 权重参数（FP8 或 FP16，取决于是否预量化）
        weight_dtype = (
            torch.float8_e4m3fn 
            if self.quant_config.is_checkpoint_fp8_serialized 
            else params_dtype
        )
        weight = create_fp8_weight_parameter(
            output_size_per_partition,
            input_size_per_partition,
            weight_dtype,
        )
        layer.register_parameter("weight", weight)
        
        # 权重 scale 参数
        if self.block_quant:
            # 块量化：需要 [num_blocks_out, num_blocks_in] 的 scale
            weight_scale = create_block_scale_parameter(...)
        else:
            # Per-tensor：需要 [1] 或 [output_size] 的 scale
            weight_scale = create_fp8_scale_parameter(output_partition_sizes)
        layer.register_parameter("weight_scale", weight_scale)
        
        # 输入 scale（如果是静态量化）
        if self.quant_config.activation_scheme == "static":
            input_scale = create_fp8_input_scale(output_partition_sizes)
            layer.register_parameter("input_scale", input_scale)
        else:
            layer.input_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行 FP8 量化的线性变换。
        
        这是 FP8 GEMM 的核心入口，执行：
        1. 激活量化 (FP16 -> FP8)
        2. FP8 GEMM
        3. 反量化 (FP8 -> FP16，通常融合在 GEMM 中)
        
        Args:
            layer: 包含权重的层
            x: 输入激活 [batch, seq, input_size]
            bias: 可选的偏置
            
        Returns:
            输出张量 [batch, seq, output_size]
        """
        # ⭐ FP8 GEMM 的核心入口
        
        # 路径 1: Marlin FP8 kernel（用于老GPU）
        if self.use_marlin:
            return apply_fp8_marlin_linear(
                x,
                layer.weight,
                layer.weight_scale,
                layer.input_scale,
                bias,
            )
        
        # 路径 2: 块量化 FP8 GEMM
        if self.block_quant:
            return self.w8a8_block_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
            )
        
        # 路径 3: Per-tensor FP8 GEMM（最常用）
        return self.fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=layer.input_scale,
            bias=bias,
        )
```

### 6.4 Fp8LinearOp 类（Per-tensor 量化）

**文件**: `vllm/model_executor/layers/quantization/utils/w8a8_utils.py`

```python
class Fp8LinearOp:
    """Per-tensor FP8 linear operation.
    
    实现 per-tensor 量化的 FP8 GEMM，包括：
    1. 动态或静态激活量化
    2. CUTLASS 或 Triton GEMM
    3. 融合的反量化
    """

    def __init__(
        self,
        act_quant_static: bool,            # 是否静态量化
        act_quant_group_shape: GroupShape,  # 量化粒度
    ):
        self.act_quant_static = act_quant_static
        self.act_quant_group_shape = act_quant_group_shape
        self.quant_fp8 = QuantFP8()  # 激活量化器

    def apply(
        self,
        input: torch.Tensor,           # 输入激活 [M, K]
        weight: torch.Tensor,          # 量化权重 [N, K] (FP8)
        weight_scale: torch.Tensor,    # 权重 scale
        out_dtype: torch.dtype,        # 输出数据类型
        input_scale: torch.Tensor | None = None,  # 输入 scale（静态）
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行 FP8 线性变换。
        
        流程:
        1. 量化激活: FP16 input -> FP8 qinput + scale_a
        2. GEMM: qinput @ weight.T
        3. 反量化: output * scale_a * scale_b -> FP16 output
        
        步骤 2 和 3 通常融合在一个 kernel 中。
        """
        
        # =====================================================
        # ⭐ 步骤 1: 激活量化 (Quant)
        # FP16 -> FP8 E4M3
        # =====================================================
        if self.act_quant_static:
            # 静态量化：使用预计算的 scale
            # scale 在校准阶段确定，推理时固定
            qinput, scale_a = ops.scaled_fp8_quant(input, input_scale)
        else:
            # 动态量化：运行时计算 scale
            # scale = max(|input|) / max_fp8_value
            qinput, scale_a = self.quant_fp8.apply(input)
        
        # =====================================================
        # ⭐ 步骤 2+3: GEMM + 反量化 (Dequant)
        # output = (qinput @ weight.T) * scale_a * scale_b
        # =====================================================
        output = self.gemm_op(
            qinput=qinput,           # FP8 激活
            weight=weight,           # FP8 权重
            out_dtype=out_dtype,     # 输出类型 (FP16)
            scale_a=scale_a,         # 激活 scale
            scale_b=weight_scale,    # 权重 scale
            bias=bias,
            output_shape=[input.shape[0], weight.shape[0]],
        )
        
        return output
    
    def gemm_op(self, qinput, weight, out_dtype, scale_a, scale_b, bias, output_shape):
        """选择并调用最佳的 GEMM kernel。"""
        # 优先使用 CUTLASS
        if cutlass_fp8_supported():
            return cutlass_scaled_mm(
                qinput, weight,
                scale_a, scale_b,
                out_dtype, bias
            )
        # 回退到 Triton
        return triton_scaled_mm(
            qinput, weight,
            scale_a, scale_b,
            out_dtype, bias
        )
```

### 6.5 量化函数详解 (scaled_fp8_quant)

**文件**: `vllm/_custom_ops.py` (行 1678-1735)

```python
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.
    
    量化公式:
        output = clamp(input / scale, -max_fp8, max_fp8)
        
    动态量化时 scale 计算:
        scale = max(|input|) / max_fp8_value
    
    Args:
        input: 输入张量 [M, K]，FP16/BF16
        scale: 预计算的 scale（静态量化用）
        use_per_token_if_dynamic: 是否 per-token 动态量化
        
    Returns:
        output: 量化后的张量 [M, K]，FP8 E4M3
        scale: 使用的 scale
    """
    shape = input.shape  # 获取输入形状
    out_dtype = current_platform.fp8_dtype()  # Platform-specific FP8 dtype
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    
    if scale is None:
        if use_per_token_if_dynamic:
            # ⭐ 动态 per-token 量化
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(output, input, scale, scale_ub)
        else:
            # ⭐ 动态 per-tensor 量化
            scale = torch.empty(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # ⭐ 静态量化
        torch.ops._C.static_scaled_fp8_quant(output, input, scale)

    return output, scale
```

### 6.6 CUTLASS GEMM 函数

**文件**: `vllm/_custom_ops.py` (行 828-876)

```python
def cutlass_scaled_mm(
    a: torch.Tensor,           # 量化后的激活 (FP8) [M, K]
    b: torch.Tensor,           # 量化后的权重 (FP8) [N, K] (需转置)
    scale_a: torch.Tensor,     # 激活 scale [M, 1] 或 [1]
    scale_b: torch.Tensor,     # 权重 scale [1, N] 或 [1]
    out_dtype: torch.dtype,    # 输出数据类型 (FP16/BF16)
    bias: torch.Tensor | None = None,  # 可选偏置
) -> torch.Tensor:
    """
    Fused GEMM with dequantization.
    
    数学公式:
        output = (scale_a * a) @ (scale_b * b).T + bias
        
    实际实现中，scale 在 GEMM 后应用以减少精度损失:
        output = (a @ b.T) * scale_a * scale_b + bias
    
    使用 CUTLASS 库实现的高效 FP8 GEMM kernel。
    """
    # 维度处理
    m = a.shape[0]
    n = b.shape[0]  # b 是 [N, K]，需要转置
    
    # 检查是否可以使用 CUTLASS
    if not cutlass_compatible(b):
        # 回退到 Triton 实现
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
            triton_scaled_mm,
        )
        return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    
    # 分配输出张量
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    
    # ⭐ 调用 CUTLASS kernel
    # 这是 C++ 实现的高效 FP8 GEMM
    torch.ops._C.cutlass_scaled_mm(
        out,        # 输出 [M, N]
        a,          # 激活 [M, K]
        b,          # 权重 [N, K]
        scale_a,    # 激活 scale
        scale_b,    # 权重 scale
        bias,       # 偏置
    )
    
    return out
```

---

## 7. AWQ 量化详解

AWQ (Activation-aware Weight Quantization) 是一种 4-bit 权重量化方法，保持激活为 FP16。

### 7.1 AWQ 原理

```
AWQ 的核心思想：
1. 权重量化为 INT4（4-bit）
2. 激活保持 FP16（不量化）
3. 保护"重要"权重通道（通过激活感知）

量化公式：
W_quant = round(W / scale) + zero_point
W_dequant = (W_quant - zero_point) * scale
```

### 7.2 AWQLinearMethod

**文件**: `vllm/model_executor/layers/quantization/awq.py`

```python
class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.
    
    AWQ 使用 group-wise 量化，每 group_size 个输入通道共享一个 scale。
    """
    
    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config
        self.group_size = quant_config.group_size  # 通常 128
    
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行 AWQ 量化的线性变换。
        
        AWQ 的 GEMM 流程：
        1. 激活保持 FP16
        2. 权重从 INT4 解包
        3. 权重反量化: W = (W_int4 - zero) * scale
        4. FP16 GEMM: Y = X @ W.T
        """
        # 获取量化权重和参数
        qweight = layer.qweight        # INT4 打包的权重
        scales = layer.scales          # 量化 scale [N, K/group_size]
        qzeros = layer.qzeros          # 量化零点
        
        # ⭐ AWQ GEMM（内部处理反量化）
        return ops.awq_gemm(
            x,              # FP16 激活
            qweight,        # INT4 权重
            scales,         # FP16 scale
            qzeros,         # INT4 零点
            layer.pack_factor,  # 打包因子（8 for INT4）
            bias,
        )
```

---

## 8. GPTQ 量化详解

GPTQ (Gradient-based Post-Training Quantization) 是另一种流行的 4-bit 量化方法。

### 8.1 GPTQ 原理

```
GPTQ 的核心思想：
1. 使用 Hessian 矩阵指导量化
2. 逐层量化，最小化重建误差
3. 支持 INT4/INT8 权重
```

### 8.2 GPTQLinearMethod

**文件**: `vllm/model_executor/layers/quantization/gptq.py`

```python
class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ."""
    
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行 GPTQ 量化的线性变换。"""
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        g_idx = layer.g_idx  # 组索引
        
        # GPTQ GEMM
        return ops.gptq_gemm(
            x, qweight, scales, qzeros, g_idx, bias
        )
```

### 8.3 Marlin 格式（高效 GPTQ）

Marlin 是 GPTQ 的高效变体，专门针对 GPU 优化。

**文件**: `vllm/model_executor/layers/quantization/gptq_marlin.py`

```python
class GPTQMarlinLinearMethod(LinearMethodBase):
    """High-performance GPTQ using Marlin format.
    
    Marlin 的优势：
    1. 权重重排以优化内存访问
    2. 专用 kernel 减少开销
    3. 性能接近 FP16
    """
    
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Marlin GEMM
        return ops.marlin_gemm(
            x,
            layer.B_q,      # Marlin 格式权重
            layer.s,        # scale
            layer.workspace,
            layer.size_m,
            layer.size_n,
            layer.size_k,
        )
```

---

## 9. 完整调用链图

### 9.1 FP8 量化的完整调用链

```
Model Forward
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  QKVParallelLinear/MergedColumnParallelLinear/RowParallelLinear.forward()
│  文件: vllm/model_executor/layers/linear.py
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  self.quant_method.apply(self, input_, bias)
│  文件: vllm/model_executor/layers/linear.py (行 565, 1405)
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Fp8LinearMethod.apply(layer, x, bias)
│  文件: vllm/model_executor/layers/quantization/fp8.py (行 610-687)
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  Marlin Path      │   │  Block Quant Path │   │  Per-tensor Path  │
│  (老GPU回退)       │   │  (块量化)          │   │  (标准FP8)        │
│                   │   │                   │   │                   │
│  apply_fp8_       │   │  W8A8BlockFp8     │   │  Fp8LinearOp      │
│  marlin_linear()  │   │  LinearOp.apply() │   │  .apply()         │
└───────────────────┘   └─────────┬─────────┘   └─────────┬─────────┘
                                  │                       │
                                  └───────────┬───────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         量化步骤 (Quant)                                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  动态量化: quant_fp8.apply(input)                                  │  │
│  │  或                                                               │  │
│  │  静态量化: ops.scaled_fp8_quant(input, input_scale)              │  │
│  │                                                                   │  │
│  │  底层调用:                                                        │  │
│  │  - torch.ops._C.dynamic_scaled_fp8_quant()                       │  │
│  │  - torch.ops._C.static_scaled_fp8_quant()                        │  │
│  │  - torch.ops._C.dynamic_per_token_scaled_fp8_quant()             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  输出: qinput (FP8), scale_a (FP32)                                     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      GEMM + Dequant (融合)                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  CUTLASS 路径:                                                    │  │
│  │  cutlass_scaled_mm(qinput, weight, scale_a, scale_b, ...)        │  │
│  │  → torch.ops._C.cutlass_scaled_mm(...)                           │  │
│  │                                                                   │  │
│  │  Triton 路径 (回退):                                               │  │
│  │  triton_scaled_mm(qinput, weight, scale_a, scale_b, ...)         │  │
│  │                                                                   │  │
│  │  DeepGEMM 路径 (块量化):                                          │  │
│  │  deepgemm_fp8_gemm(...)                                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  输出: output (FP16/BF16)                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 关键函数和文件对照表

| 步骤 | 函数名 | 文件路径 | 说明 |
|-----|-------|---------|------|
| **线性层入口** | | | |
| Column 前向 | `ColumnParallelLinear.forward()` | `layers/linear.py:557-575` | Wqkv, W13 |
| Row 前向 | `RowParallelLinear.forward()` | `layers/linear.py:1388-1416` | Wo, W2 |
| **量化方法** | | | |
| 无量化 | `UnquantizedLinearMethod.apply()` | `layers/linear.py:234-240` | FP16 GEMM |
| FP8 | `Fp8LinearMethod.apply()` | `quantization/fp8.py:610-687` | FP8 GEMM |
| AWQ | `AWQLinearMethod.apply()` | `quantization/awq.py` | INT4 GEMM |
| GPTQ | `GPTQLinearMethod.apply()` | `quantization/gptq.py` | INT4 GEMM |
| **量化操作** | | | |
| FP8 量化 | `scaled_fp8_quant()` | `_custom_ops.py:1678-1735` | FP16→FP8 |
| 动态量化 kernel | `dynamic_scaled_fp8_quant` | `torch.ops._C` | CUDA kernel |
| 静态量化 kernel | `static_scaled_fp8_quant` | `torch.ops._C` | CUDA kernel |
| **GEMM 操作** | | | |
| CUTLASS FP8 | `cutlass_scaled_mm()` | `_custom_ops.py:828-876` | 高效 FP8 GEMM |
| Triton FP8 | `triton_scaled_mm()` | `triton_scaled_mm.py` | Triton 实现 |
| AWQ GEMM | `awq_gemm()` | `_custom_ops.py` | AWQ 专用 |
| Marlin GEMM | `marlin_gemm()` | `_custom_ops.py` | 高效 INT4 |

---

## 10. CUDA/Triton Kernel 详解

### 10.1 CUDA Kernel 位置

```
csrc/
├── quantization/
│   ├── w8a8/                         # W8A8 量化
│   │   ├── fp8/                      # FP8 量化 kernel
│   │   │   ├── nvidia/               # NVIDIA 特定实现
│   │   │   ├── amd/                  # AMD 特定实现
│   │   │   └── common.cu             # 通用实现
│   │   └── int8/                     # INT8 量化
│   │
│   ├── awq/                          # AWQ kernel
│   │
│   ├── gptq/                         # GPTQ kernel
│   │
│   ├── gptq_marlin/                  # GPTQ Marlin kernel
│   │
│   ├── marlin/                       # Marlin kernel
│   │
│   └── fp4/                          # FP4 量化 kernel
│
├── cutlass_extensions/               # CUTLASS GEMM 扩展
│
└── torch_bindings.cpp                # PyTorch C++ 绑定
    └── 所有 torch.ops._C.xxx() 的定义
```

### 10.2 Triton Kernel 位置

```
vllm/model_executor/layers/quantization/
├── compressed_tensors/
│   └── triton_scaled_mm.py           # Triton FP8 GEMM
│
└── utils/
    ├── fp8_utils.py                  # FP8 工具和 Triton kernel
    └── w8a8_utils.py                 # W8A8 工具
```

### 10.3 Triton FP8 GEMM 示例

```python
# vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py

@triton.jit
def _scaled_mm_kernel(
    A, B, C,
    scale_a, scale_b,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Triton FP8 scaled matrix multiplication kernel.
    
    C = (A @ B) * scale_a * scale_b
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算块偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 加载 scale
    s_a = tl.load(scale_a + offs_m[:, None])
    s_b = tl.load(scale_b + offs_n[None, :])
    
    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K 维度循环
    for k in range(0, K, BLOCK_K):
        # 加载 A 块 (FP8)
        a = tl.load(A + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        # 加载 B 块 (FP8)
        b = tl.load(B + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        # GEMM
        acc += tl.dot(a, b)
    
    # 应用 scale（反量化）
    acc = acc * s_a * s_b
    
    # 存储结果
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)


def triton_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton implementation of scaled matrix multiplication."""
    M, K = a.shape
    N = b.shape[1]
    
    # 分配输出
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    
    # 网格配置
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    # 调用 kernel
    _scaled_mm_kernel[grid](
        a, b, c,
        scale_a, scale_b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=64,
    )
    
    if bias is not None:
        c += bias
    
    return c
```

---

## 11. 如何替换自定义 Kernel

### 11.1 方法一：替换量化方法类（推荐）

创建自定义的 `LinearMethod` 类，实现自己的 GEMM 逻辑：

```python
# my_custom_linear_method.py

from vllm.model_executor.layers.linear import LinearMethodBase

class MyCustomFp8LinearMethod(LinearMethodBase):
    """Custom FP8 linear method with custom kernels.
    
    自定义 FP8 线性方法，可以插入任意 kernel。
    """
    
    def __init__(self, quant_config):
        self.quant_config = quant_config
    
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
        """创建权重参数。"""
        # 创建 FP8 权重
        weight = torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=torch.float8_e4m3fn,
        )
        layer.register_parameter("weight", torch.nn.Parameter(weight))
        
        # 创建 scale
        weight_scale = torch.empty(1, dtype=torch.float32)
        layer.register_parameter("weight_scale", torch.nn.Parameter(weight_scale))
    
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行自定义的 quant-gemm-dequant。"""
        
        # ⭐ 1. 自定义量化
        qinput, scale_a = my_custom_quant(x)
        
        # ⭐ 2. 自定义 GEMM
        output = my_custom_gemm(
            qinput, 
            layer.weight, 
            scale_a, 
            layer.weight_scale
        )
        
        # ⭐ 3. 添加 bias
        if bias is not None:
            output = output + bias
        
        return output


# 自定义量化函数
def my_custom_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """自定义 FP8 量化。"""
    # 计算 scale
    max_val = torch.max(torch.abs(x))
    scale = max_val / 448.0  # FP8 E4M3 最大值
    
    # 量化
    qx = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    
    return qx, scale


# 自定义 GEMM（可以使用 Triton）
@triton.jit
def my_custom_gemm_kernel(...):
    """自定义 Triton GEMM kernel。"""
    ...


def my_custom_gemm(a, b, scale_a, scale_b):
    """调用自定义 GEMM kernel。"""
    ...
```

然后在量化配置中注册：

```python
# 修改 vllm/model_executor/layers/quantization/fp8.py

class Fp8Config(QuantizationConfig):
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            # 返回自定义方法
            return MyCustomFp8LinearMethod(self)
        ...
```

### 11.2 方法二：直接替换底层函数

在 `vllm/_custom_ops.py` 中替换函数：

```python
# 在应用启动时替换

import vllm._custom_ops as ops

# 保存原始函数
_original_scaled_fp8_quant = ops.scaled_fp8_quant
_original_cutlass_scaled_mm = ops.cutlass_scaled_mm

# 替换量化函数
def my_scaled_fp8_quant(input, scale=None, ...):
    """自定义 FP8 量化。"""
    # 你的实现
    ...

# 替换 GEMM 函数
def my_cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
    """自定义 GEMM。"""
    # 你的实现
    ...

# 应用替换
ops.scaled_fp8_quant = my_scaled_fp8_quant
ops.cutlass_scaled_mm = my_cutlass_scaled_mm
```

### 11.3 方法三：使用 Monkey Patching

```python
# 在使用 vLLM 之前执行

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

# 保存原始方法
_original_apply = Fp8LinearMethod.apply

# 自定义 apply 方法
def custom_apply(self, layer, x, bias=None):
    """自定义的 apply 方法。"""
    print(f"Custom GEMM called: input shape = {x.shape}")
    
    # 调用自定义 kernel
    # ...
    
    # 或调用原始方法
    return _original_apply(self, layer, x, bias)

# 替换
Fp8LinearMethod.apply = custom_apply
```

### 11.4 方法四：添加新的量化配置

```python
# 创建新的量化配置类

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class MyCustomQuantConfig(QuantizationConfig):
    """自定义量化配置。"""
    
    @classmethod
    def get_name(cls) -> str:
        return "my_custom_quant"
    
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]
    
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return MyCustomLinearMethod(self)
        return None


# 注册到 vLLM
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
QUANTIZATION_METHODS["my_custom_quant"] = MyCustomQuantConfig
```

使用：
```python
llm = LLM(model="...", quantization="my_custom_quant")
```

---

## 12. 实战示例

### 12.1 添加自定义 FP8 量化 Kernel

**步骤 1**: 创建 Triton Kernel

```python
# my_kernels/fp8_quant.py

import triton
import triton.language as tl

@triton.jit
def my_fp8_quant_kernel(
    input_ptr, output_ptr, scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """自定义 FP8 量化 kernel。"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载输入
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # 计算 scale（简化版，实际应该用 reduce）
    scale = tl.load(scale_ptr)
    
    # 量化
    qx = x / scale
    qx = tl.clamp(qx, -448.0, 448.0)
    
    # 存储（注意：Triton 目前不直接支持 FP8，需要位操作）
    tl.store(output_ptr + offsets, qx, mask=mask)


def my_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Python 接口。"""
    n_elements = x.numel()
    
    # 计算 scale
    scale = torch.max(torch.abs(x)) / 448.0
    
    # 分配输出
    output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    
    # 配置
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 调用 kernel
    my_fp8_quant_kernel[grid](
        x, output, scale.unsqueeze(0),
        n_elements,
        BLOCK_SIZE,
    )
    
    return output, scale
```

**步骤 2**: 创建 Triton GEMM Kernel

```python
# my_kernels/fp8_gemm.py

@triton.jit
def my_fp8_gemm_kernel(
    # 指针
    A, B, C,
    scale_a, scale_b,
    # 维度
    M, N, K,
    # 步长
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 编译时常量
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """自定义 FP8 GEMM kernel。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 块内偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K 维度循环
    for k in range(0, K, BLOCK_K):
        # 加载 A 和 B 块
        a = tl.load(A + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b = tl.load(B + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        
        # 矩阵乘法
        acc += tl.dot(a, b)
    
    # 加载 scale
    s_a = tl.load(scale_a)
    s_b = tl.load(scale_b)
    
    # 反量化
    acc = acc * s_a * s_b
    
    # 存储
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)
```

**步骤 3**: 创建 LinearMethod

```python
# my_kernels/linear_method.py

from vllm.model_executor.layers.linear import LinearMethodBase
from my_kernels.fp8_quant import my_fp8_quant
from my_kernels.fp8_gemm import my_fp8_gemm_kernel

class MyFp8LinearMethod(LinearMethodBase):
    """使用自定义 kernel 的 FP8 线性方法。"""
    
    def apply(self, layer, x, bias=None):
        # 1. 量化
        qx, scale_a = my_fp8_quant(x)
        
        # 2. GEMM
        M, K = x.shape[0], x.shape[-1]
        N = layer.weight.shape[0]
        
        output = torch.empty(M, N, device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
        
        my_fp8_gemm_kernel[grid](
            qx, layer.weight, output,
            scale_a, layer.weight_scale,
            M, N, K,
            qx.stride(0), qx.stride(1),
            layer.weight.stride(0), layer.weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        )
        
        # 3. Bias
        if bias is not None:
            output += bias
        
        return output
```

**步骤 4**: 集成到 vLLM

```python
# 使用 Monkey Patching
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from my_kernels.linear_method import MyFp8LinearMethod

# 替换
Fp8LinearMethod.apply = MyFp8LinearMethod.apply

# 或创建新配置并注册
```

---

## 13. 总结与最佳实践

### 13.1 替换 GEMM 的检查清单

要替换 vLLM 中的 FP8 quant-gemm-dequant，需要关注以下位置：

| 组件 | 文件 | 函数/类 |
|------|------|--------|
| **量化函数** | `vllm/_custom_ops.py` | `scaled_fp8_quant()` |
| **量化 kernel** | `torch.ops._C` | `dynamic_scaled_fp8_quant`, `static_scaled_fp8_quant` |
| **GEMM 函数** | `vllm/_custom_ops.py` | `cutlass_scaled_mm()` |
| **GEMM kernel** | `torch.ops._C` | `cutlass_scaled_mm` |
| **Triton 备选** | `triton_scaled_mm.py` | `triton_scaled_mm()` |
| **量化方法类** | `quantization/fp8.py` | `Fp8LinearMethod.apply()` |
| **工具类** | `w8a8_utils.py` | `Fp8LinearOp` |
| **线性层** | `linear.py` | `forward()` 方法 |

### 13.2 最佳实践

1. **选择合适的替换层级**：
   - 如果只改 kernel：修改 `_custom_ops.py` 中的函数
   - 如果改整个流程：创建新的 `LinearMethod`
   - 如果要完全自定义：创建新的 `QuantizationConfig`

2. **测试策略**：
   ```bash
   # 单元测试
   pytest tests/kernels/test_fp8_quant.py -v
   pytest tests/quantization/test_fp8.py -v
   
   # 集成测试
   python examples/offline_inference/basic/generate.py \
       --model meta-llama/Llama-3.2-1B-Instruct \
       --quantization fp8
   ```

3. **性能调优**：
   - 使用 `triton.testing.Benchmark` 测试 kernel 性能
   - 使用 `torch.cuda.Event` 测量端到端延迟
   - 使用 `nsys` 分析 kernel launch 开销

4. **兼容性考虑**：
   - 确保支持不同的输入形状（prefill vs decode）
   - 确保支持张量并行
   - 确保支持多种数据类型（FP16、BF16）

### 13.3 相关文档

- **项目整体结构** → [framework_overview.md](./framework_overview.md)
- **核心框架详解** → [framework_vllmcore.md](./framework_vllmcore.md)
- **官方文档** → https://docs.vllm.ai/en/stable/
- **CUTLASS 文档** → https://github.com/NVIDIA/cutlass
- **Triton 文档** → https://triton-lang.org/
