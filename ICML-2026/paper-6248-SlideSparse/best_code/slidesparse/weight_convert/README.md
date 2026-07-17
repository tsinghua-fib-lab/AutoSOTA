# SlideSparse Weight Convert 工具链

## 概述

本目录包含 SlideSparse Phase 4 的离线权重转换工具链，用于将 HuggingFace 的 compressed-tensor 格式模型（FP8/INT8）转换为支持 cuSPARSELt 2:4 稀疏加速的格式。

## 核心流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         离线处理流程 (Phase 4)                           │
└─────────────────────────────────────────────────────────────────────────┘

HuggingFace 模型 (compressed-tensor FP8/INT8)
    │
    ▼
┌──────────────────┐
│  1. 模型检查     │  检查模型是否已下载，获取路径
│     (entry.py)   │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  2. 剪枝 Prune   │  应用 Z:L 稀疏约束 (如 2:8)
│     (prune.py)   │  - FP8/INT8: 仅剪枝
│                  │  - BF16+BitNet: quant+prune 融合
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  3. 滑动 Slide   │  2:L → 2:4 滑动窗口映射
│     (slide.py)   │  K 维度扩展 (如 2:8 → ×1.5)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  4. 压缩 Compress│  cuSPARSELt 2:4 压缩
│     (compress.py)│  K 维度减半
└──────────────────┘
    │
    ▼
checkpoints_slidesparse/ (新格式模型)
```

## 文件结构

```
weight_convert/
├── README.md           # 本文档
├── entry.py            # 主入口脚本，串联完整流程
├── prune.py            # 剪枝脚本 (Z:L 稀疏)
├── slide.py            # 滑动脚本 (2:L → 2:4)
├── compress.py         # cuSPARSELt 压缩脚本
├── utils.py            # 共享工具函数
├── build/              # 编译后的 .so 文件
│   └── libbitnet_compress.so
├── test_correctness/   # 正确性测试
│   ├── test_prune.py
│   ├── test_slide.py
│   └── test_k_dimension.py
└── _archived/          # 旧版本参考代码
```

## 维度变化说明

以 2:8 稀疏为例 (`Z=2, L=8, expand_ratio=1.5`):

| 阶段 | 权重 Shape | 说明 |
|------|-----------|------|
| 原始 Dense | `[N, K]` | FP8/INT8 |
| Prune 后 | `[N, K]` | 满足 2:8 (每 8 个至少 2 个零) |
| Slide 后 | `[N, K×1.5]` | 满足 2:4 (每 4 个至少 2 个零) |
| Compress 后 | `[N, K×0.75]` | 2:4 压缩 (K×1.5/2) |

## 使用方法

### 完整流程

```bash
# 处理单个模型
python entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8

# 处理所有已下载的 FP8 模型
python entry.py --all --quant fp8 --Z 2 --L 8

# 仅执行 prune+slide（不压缩）
python entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8 --skip-compress
```

### 单独执行各步骤

```bash
# 仅剪枝
python prune.py --input /path/to/model --output /path/to/pruned --Z 2 --L 8

# 仅滑动
python slide.py --input /path/to/pruned --output /path/to/slided --Z 2 --L 8

# 仅压缩
python compress.py --input /path/to/slided --output /path/to/compressed
```

## 输出目录结构

```
checkpoints_slidesparse/
├── Qwen2.5-0.5B-FP8-SlideSparse/
│   ├── model.safetensors           # 压缩后的权重
│   ├── config.json                 # 原始配置（复制）
│   ├── tokenizer.json              # 分词器（复制）
│   └── slidesparse_config.json     # SlideSparse 元数据
└── ...
```

## 依赖

- Python 3.10+
- PyTorch 2.0+
- safetensors
- numpy
- (可选) numba - 用于加速 slide 计算
- cuSPARSELt - 通过 build/libbitnet_compress.so

## 注意事项

1. **FP8 支持**：FP8 E4M3 可以表示 ±1 和 0，因此 ternary 权重可以存储为 FP8
2. **INT8 输入**：INT8 模型跳过量化步骤，仅执行剪枝
3. **cuSPARSELt SO**：确保 `build/libbitnet_compress.so` 已正确编译
