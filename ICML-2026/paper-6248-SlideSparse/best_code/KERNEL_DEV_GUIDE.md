# vLLM Kernel 开发指南

## 一、核心问题解答

### Q: vllm/vllm-openai 镜像里面是什么？

```
/usr/local/lib/python3.12/dist-packages/vllm/
├── _C.abi3.so                      # 编译好的 C++/CUDA 扩展
├── model_executor/
│   └── layers/
│       ├── linear.py               # 👈 线性层实现（你要劫持的地方）
│       ├── utils.py                # 👈 GEMM 函数派发
│       └── quantization/           # 👈 量化方法
└── ...
```

**关键点**：
- vLLM 是通过 `pip install` 安装的，不是源码
- `.so` 文件是编译好的 C++/CUDA kernel
- `.py` 文件可以通过 `pip install -e .` 覆盖

### Q: 如何让修改的代码生效？

1. `pip uninstall vllm` - 移除预装的 vLLM
2. 把你的 vLLM 源码挂载进容器
3. `pip install -e .` - 可编辑模式安装（改代码立即生效）

---

## 二、完整开发流程

### Step 1: 构建开发镜像

```bash
cd /home/v-hanshao/vllmbench
docker build -t vllm-dev:v0.13.0 -f Dockerfile.dev .
```

### Step 2: 启动开发容器

```bash
docker run --gpus all -it --rm --ipc=host \
    --name vllm-kernel-dev \
    -v /home/v-hanshao/vllmbench:/root/vllmbench \
    -v /home/v-hanshao/GPU:/root/GPU \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=${HF_TOKEN} \
    vllm-dev:v0.13.0 /bin/bash
```

### Step 3: 安装 vLLM 源码（进入容器后执行一次）

```bash
cd /root/vllmbench

# 方案A: 仅修改 Python 代码（推荐，最快）
VLLM_USE_PRECOMPILED=1 pip install -e .

# 方案B: 需要修改 C++/CUDA 代码（首次约 15-30 分钟）
pip install -e .
```

### Step 4: 验证安装

```bash
# 检查 vLLM 版本
python -c "import vllm; print(vllm.__version__)"

# 运行简单测试
vllm bench throughput --model Qwen/Qwen2.5-0.5B --input-len 128 --output-len 64 --num-prompts 10
```

---

## 三、你要劫持的代码位置

### 1. GEMM 替换点（最直接）

**文件**: `vllm/model_executor/layers/utils.py`

```python
# 第 96-103 行
def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)  # 👈 替换这里！
```

### 2. Linear 层 apply 方法

**文件**: `vllm/model_executor/layers/linear.py`

```python
# UnquantizedLinearMethod.apply() 第 237 行
def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

### 3. 自定义量化方法

**目录**: `vllm/model_executor/layers/quantization/`

参考 `fp8.py` 的实现，创建你自己的量化类。

---

## 四、代码修改示例

### 示例1: 在 utils.py 中添加你的 Kernel 开关

```python
# vllm/model_executor/layers/utils.py

import os

# 你的自定义 Kernel 开关
USE_CUSTOM_GEMM = os.environ.get("VLLM_USE_CUSTOM_GEMM", "0") == "1"

def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    if USE_CUSTOM_GEMM:
        # 调用你的自定义 GEMM
        from your_custom_kernels import custom_gemm
        return custom_gemm(x, weight, bias)
    
    return torch.nn.functional.linear(x, weight, bias)
```

### 示例2: 创建完整的自定义量化方法

见 `custom_kernels/README.md` 中的完整示例。

---

## 五、运行 Benchmark

```bash
# 基础吞吐量测试
vllm bench throughput \
    --model Qwen/Qwen2.5-0.5B \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 100

# 延迟测试
vllm bench latency \
    --model Qwen/Qwen2.5-0.5B \
    --input-len 512 \
    --output-len 128

# 使用你的自定义 Kernel
VLLM_USE_CUSTOM_GEMM=1 vllm bench throughput \
    --model Qwen/Qwen2.5-0.5B \
    --input-len 512 \
    --output-len 128
```

---

## 六、开发技巧

### 1. 快速调试（不重启容器）

因为使用了 `-e` 可编辑模式，修改 Python 代码后直接运行新命令即可生效。

### 2. 编译自定义 CUDA Kernel

```bash
cd /root/vllmbench/custom_kernels
nvcc -shared -o libcustom_gemm.so \
    -lcublas -lcusparselt \
    -I/usr/local/cuda/include \
    custom_gemm.cu
```

### 3. 保存开发环境（避免重复编译）

```bash
# 首次 pip install -e . 完成后
docker commit vllm-kernel-dev vllm-dev:v0.13.0-compiled
```

---

## 七、目录结构建议

```
/root/vllmbench/              # vLLM 源码 (pip install -e .)
├── vllm/
│   └── model_executor/
│       └── layers/
│           ├── linear.py     # 修改这里劫持 Linear 层
│           └── utils.py      # 修改这里替换 GEMM
│
└── custom_kernels/           # 你的自定义 Kernel
    ├── triton/
    │   └── quant_expand.py   # Triton Quant+Expand 实现
    ├── cuda/
    │   └── custom_gemm.cu    # CUDA GEMM 实现
    └── __init__.py           # Python 封装
```
