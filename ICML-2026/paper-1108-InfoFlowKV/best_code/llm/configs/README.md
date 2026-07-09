# KV Cache Recomputation - 配置说明

## 参数传入方式

所有的 recompute 参数都通过 **YAML 配置文件**传入，类似 vlm-dev 的设计。

## 配置文件结构

```yaml
# Model settings
model: "/path/to/model"  # 模型路径
device: "auto"            # 设备: auto, cuda, cuda:0, cpu

# Dataset settings
dataset: "2wikimqa"       # 数据集名称
num_samples: 200          # 样本数量

# KV cache recomputation settings
recompute_ratio: 0.15     # 重计算比例 (0.0-1.0)
method: "norm"            # 重要性评分方法
layer_start: 1            # 开始重计算的层 (0=所有层)
layer_end: null           # 结束层 (null=最后一层)

# Generation settings
max_new_tokens: 128       # 最大生成 token 数
do_sample: false          # 是否采样
temperature: 1.0          # 采样温度
top_p: 1.0               # Top-p 采样

# Output settings
output_dir: "results/"    # 结果输出目录
```

## 重要性评分方法 (method)

支持以下方法：
- `norm`: L2 范数
- `attn`: 注意力权重
- `entropy`: 熵
- `mass`: 质量
- `combined`: 组合方法

## 使用示例

### 1. Qwen 模型

```bash
# 使用配置文件运行
python scripts/qwen_inference_with_recompute.py --config configs/2wikimqa_recompute.yaml

# 使用不同的 recompute 参数
python scripts/qwen_inference_with_recompute.py --config configs/hotpotqa_recompute.yaml
```

### 2. ChatGLM 模型

```bash
python scripts/chatglm_inference_with_recompute.py --config configs/chatglm_2wikimqa.yaml
```

### 3. 调整参数

修改配置文件中的参数：

```yaml
# 提高 recompute 比例
recompute_ratio: 0.25

# 更换评分方法
method: "attn"

# 限制重计算层范围
layer_start: 10
layer_end: 20
```

## 配置文件示例

已创建的配置文件：
- `configs/2wikimqa_recompute.yaml` - Qwen + 2WikiMQA
- `configs/hotpotqa_recompute.yaml` - Qwen + HotpotQA
- `configs/chatglm_2wikimqa.yaml` - ChatGLM + 2WikiMQA

## 参数优先级

1. 配置文件中的参数
2. 脚本中的默认值

例如，如果配置文件中没有 `recompute_ratio`，则使用默认值 `0.15`。
