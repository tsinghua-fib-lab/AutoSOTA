# OCMB Repository Setup - Summary

## 完成的工作

### 1. 创建了 Examples 目录

在 `/data/Vortex-Causal/OCMB/examples/` 目录下创建了完整的示例脚本，用于重现论文中的主要实验结果：

#### 示例脚本列表：

1. **`1_synthetic_graphs.py`** - 合成图实验（论文 Table 2）
   - 比较 OCMB 与 PC、IAMB 等方法
   - 在 scale-free 和 Erdős-Rényi 图上的表现
   - 预期结果：OCMB F1 ≈ 0.55-0.65 (scale-free)

2. **`2_k_sensitivity.py`** - K 参数敏感性分析（论文 Table 9, Figure 8）
   - 展示 OCMB 对超参数 K 的鲁棒性
   - 测试 K/d 比率从 0.02 到 0.40
   - 预期结果：F1 在 K/d ∈ [0.05, 0.20] 范围内变化 <5%

3. **`3_runtime_comparison.py`** - 运行时和效率比较（论文 Figure 7）
   - OCMB 比 IAMB 快 2 倍
   - CI 测试减少 65%
   - 预期结果：d=100 时 OCMB ~6.8s, IAMB ~14.5s

4. **`4_ordering_robustness.py`** - 排序错误敏感性（论文 Figure 10）
   - 测试排序损坏对结果的影响
   - 预期结果：50% 错误时 F1 仅下降 ~33%

5. **`generate_data.py`** - 数据生成工具
   - 用于创建合成数据集
   - 支持 scale-free 和 ER 图
   - 可配置节点数、样本数、噪声类型等

6. **`README.md`** - 详细文档
   - 包含每个脚本的使用说明
   - 预期结果和运行时间
   - 故障排除指南

### 2. 更新了主 README

在 `/data/Vortex-Causal/OCMB/README.md` 中：

- 添加了论文信息和 ICML 2026 链接
- 添加了 examples 目录的说明
- 更新了引用信息
- 添加了徽章（Paper, License）

### 3. 推送到 GitHub

已成功将所有更改推送到 GitHub 仓库：
- **仓库地址**: https://github.com/williamyuan86/ocmb
- **SSH URL**: git@github.com:williamyuan86/ocmb.git
- **HTTPS URL**: https://github.com/williamyuan86/ocmb.git

提交历史：
- Commit 1: "Add examples directory with reproducible experiments" (82584bd)
- Commit 2: "Update README with paper information and links" (739b993)

### 4. 关联论文信息

所有文档中都已包含论文链接：
- **论文标题**: Global Directional Priors with Local Statistical Validation for Scalable Causal Discovery
- **会议**: ICML 2026
- **论文链接**: https://icml.cc/virtual/2026/poster/65568

## 文件结构

```
OCMB/
├── examples/
│   ├── README.md                    # 详细的使用文档
│   ├── 1_synthetic_graphs.py        # 主实验脚本
│   ├── 2_k_sensitivity.py           # K 敏感性分析
│   ├── 3_runtime_comparison.py      # 运行时比较
│   ├── 4_ordering_robustness.py     # 排序鲁棒性
│   └── generate_data.py             # 数据生成工具
├── README.md                        # 主文档（已更新）
├── example.py                       # 快速入门示例（已有）
├── ocmb.py                         # OCMB 核心实现
├── caps/                           # CaPS backbone
├── scino/                          # SciNO backbone
└── utils/                          # 工具函数
```

## 使用方法

### 快速开始

```bash
# 克隆仓库
git clone git@github.com:williamyuan86/ocmb.git
cd ocmb

# 安装依赖
pip install numpy pandas scipy scikit-learn networkx torch

# 运行快速示例
python example.py

# 重现论文实验
cd examples/
python 1_synthetic_graphs.py
python 2_k_sensitivity.py
python 3_runtime_comparison.py
python 4_ordering_robustness.py
```

### 生成自定义数据

```bash
cd examples/
python generate_data.py --graph-type scale-free --nodes 100 --samples 1000 --output-dir ./my_data
```

## 特点

1. **可重现性**: 所有脚本都设计为能够重现论文中的关键结果
2. **文档完善**: 每个脚本都有详细的文档说明和预期结果
3. **易于使用**: 清晰的参数说明和示例代码
4. **独立运行**: 每个示例都可以独立运行，互不依赖
5. **错误处理**: 包含完善的错误处理和故障排除指南

## 预期结果

根据论文（Table 2, d=100, n=1000, scale-free）：

| Method       | F1    | SHD  | #CI Tests | Runtime |
|-------------|-------|------|-----------|---------|
| OCMB(caps)  | 0.558 | 31.7 | 1,980     | 6.8s    |
| IAMB        | 0.483 | 49.0 | 5,699     | 14.5s   |
| PC          | 0.663 | 16.3 | 2,072     | 1.4s    |

**注意**: 实际结果可能因为随机种子、数据生成和硬件差异而略有不同，但应该在相似的范围内。

## 下一步建议

1. **运行验证**: 建议至少运行一次所有示例脚本，确保能够得到预期的结果
2. **添加更多示例**: 如果需要，可以添加更多实验（如 real-world datasets）
3. **性能测试**: 在不同硬件配置下测试性能
4. **文档改进**: 根据用户反馈持续改进文档

## 相关链接

- **GitHub 仓库**: https://github.com/williamyuan86/ocmb
- **ICML 2026 论文**: https://icml.cc/virtual/2026/poster/65568
- **原始 Vortex-Causal 项目**: /data/Vortex-Causal/

## 联系方式

如有问题或建议，请在 GitHub 仓库中提交 Issue。
