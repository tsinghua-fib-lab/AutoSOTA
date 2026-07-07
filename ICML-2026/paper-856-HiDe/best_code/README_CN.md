# 🙈 HiDe

> **[HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling](https://arxiv.org/abs/2510.00054) 官方实现**

<div align="center">

[![论文](https://img.shields.io/badge/📜_论文-ArXiv-red)](https://arxiv.org/abs/2510.00054)
[![许可证](https://img.shields.io/badge/📄_许可证-Apache_2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/🐍_Python-3.11+-blue)]()
[![English](https://img.shields.io/badge/🇺🇸_English-README-blue)](README.md)

</div>

---

## 📰 最新动态

| 日期 | 更新内容 |
|:----:|:-------|
| 🎉 **2026/05/01** | 论文被 ICML 2026 接收！恭喜！ |
| 🔥 **2026/03/07** | 新增 Qwen3-VL 支持！ |
| 💻 **2025/12/15** | 代码正式开源！有问题欢迎提 Issue~ |
| 📕 **2025/10/28** | 论文发布于 ArXiv！ |

---

## 🎯 什么是 HiDe？

多模态大语言模型（MLLMs）在视觉理解任务上取得了显著进展，但它们在**高分辨率图像**上的表现仍然不尽如人意。

**等等...问题真的出在"目标太小"吗？** 🤔

我们的分析揭示了一个不同的真相：主要问题**不是目标尺寸**，而是**复杂的背景干扰**！🎭

<div align="center">
<img src="https://arxiv.org/html/2510.00054/pics/method.png" width="90%" alt="HiDe Framework"/>
<p><em>图：HiDe 框架概览 (TAD + LPD)</em></p>
</div>

### 💡 我们的解决方案

我们提出了 **HiDe（层次化解耦框架）**—— 一个**无需训练**的框架，包含：

| 组件 | 功能 |
|:-----|:-----|
| 🔍 **Token-wise Attention Decoupling (TAD)** | 解耦问题词元并识别关键信息词元，利用注意力权重实现与目标视觉区域的精确对齐 |
| ✂️ **Layout-Preserving Decoupling (LPD)** | 将目标区域从背景中解耦出来，重构紧凑表示的同时保留重要的空间布局 |

---

---

## 🗂️ 仓库结构

```
HiDe/
├── Hide/
│   ├── Qwen2.5/              # 🤖 Qwen2.5-VL 实现
│   │   ├── inference.py          # 核心推理逻辑
│   │   ├── cycle_infer.py        # 多 GPU 推理入口
│   │   ├── Get_box.py            # 基于注意力的边界框提取
│   │   ├── Vstar_Metric.py       # 评估指标
│   │   └── utiles.py             # 工具函数
│   │
│   ├── Qwen3/                # 🔥 Qwen3-VL 实现
│   │   ├── inference.py
│   │   ├── cycle_infer.py
│   │   ├── Get_box.py
│   │   └── utiles.py
│   │
│   └── Internvl/             # 👁️ InternVL3 实现
│       ├── cycle_inference_internvl.py
│       └── utiles_internvl.py
│
├── requirements.txt          # 📦 依赖项
├── LICENSE
└── README.md
```

---

## 🛠️ 安装

```bash
# 创建 conda 环境
conda create -n HiDe python=3.11.4
conda activate HiDe

# 安装依赖
pip install -r requirements.txt
```

---

## 📊 数据集准备

将数据集准备为以下 JSON 格式：

```json
[
    {
        "id": "0",
        "question": "手套的材质是什么？\n(A) 橡胶\n(B) 棉花\n(C) 凯夫拉\n(D) 皮革",
        "labels": "A",
        "image_path": "vstar_bench/direct_attributes/sa_4690.jpg",
        "category": "direct_attributes"
    },
    {
        "id": "1",
        "question": "簸箕的颜色是什么？\n(A) 紫色\n(B) 红色\n(C) 蓝色\n(D) 白色",
        "labels": "C",
        "image_path": "vstar_bench/direct_attributes/sa_86101.jpg",
        "category": "direct_attributes"
    }
]
```

---

## 🚀 快速开始

### Qwen2.5-VL

```bash
cd Hide/Qwen2.5

# 运行推理
python cycle_infer.py

# 计算评估指标
python Vstar_Metric.py
```

### Qwen3-VL

```bash
cd Hide/Qwen3
python cycle_infer.py
```

### InternVL3

```bash
cd Hide/Internvl
python cycle_inference_internvl.py
```

---

## ⚙️ 配置说明

在 `cycle_infer.py` 中可调整的关键超参数：

| 参数 | 描述 | 默认值 |
|:-----|:-----|:------|
| `sigma` | 高斯滤波器平滑注意力的 sigma 值 | `[3]` |
| `threshold` | 注意力二值化阈值 | `[0.7]` |
| `max_pixels` | 图像处理的最大像素数 | `16384` |
| `Parallels` | 是否启用多 GPU 并行处理 | `True/False` |

---

## 📝 引用

如果 HiDe 对你的研究有所帮助，请考虑引用：

```bibtex
@misc{liu2025hiderethinkingzoominmethod,
      title={HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling}, 
      author={Xianjie Liu and Yiman Hu and Yixiong Zou and Liang Wu and Jian Xu and Bo Zheng},
      year={2025},
      eprint={2510.00054},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.00054}, 
}
```

---

## 📄 许可证

本项目采用 Apache 2.0 许可证 —— 详情请参阅 [LICENSE](LICENSE) 文件。

---

## 🤝 致谢

感谢所有优秀的开源项目，让这项工作成为可能：
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [InternVL](https://github.com/OpenGVLab/InternVL)

---

<div align="center">

**HiDe 团队用 ❤️ 制作**

⭐ 如果觉得这个项目有用，请给我们一个 Star！ ⭐

</div>
