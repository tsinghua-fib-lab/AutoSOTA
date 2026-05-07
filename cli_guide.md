# autosota cli

自动化 SOTA 代码优化流水线。给定一篇 ML 论文的**本地已克隆代码仓库**（及可跑通评测的环境）和优化目标，由 AI 自主探索、研究、修改代码，迭代提升性能指标。现已轻量化至 CLI 0.1.0版本。

---

## 一分钟上手

> 前置：Node 18+、Python 3.10+、`bash`（Linux/macOS/WSL）、`git`。Claude Code CLI 会随 autosota 自动安装，无需单独安装。

```bash
# 1. 全局安装 autosota（同时自动安装 Claude Code CLI）
#    将 autosota-0.1.0.tgz 下载到本地后，cd 进文件所在目录，然后：
npm install -g ./autosota-0.1.0.tgz

# 2. 第一次运行 claude，接受服务条款（只需一次）
claude
# 按提示同意条款后退出（/exit 或 Ctrl+C）

# 3. 新建（或进入）你的工作目录，生成脚手架
#    工作目录可以放在任意位置，例如 ~/my-project 或 /data/my_exp
#    它只用来存放配置和输出，和代码仓库是两个独立的文件夹
mkdir my-project
cd my-project
autosota init                      # 生成 ./config.yaml + ./paper/ 模板

# 4. 编辑两处：
#    - ./config.yaml               填入 openrouter_api_key + 模型名
#    - ./paper/target.md           写清楚主指标、基线值、优化方向

# 5. 自检环境（无红叉才继续；黄色 ~ 表示 venv 未创建，首次运行会自动建好）
autosota doctor

# 6. 跑优化
#    --repo 指向你已 clone 到本地、且依赖 / 评测脚本已能正常运行的代码仓库
#    首次运行会在工作目录下自动创建 Python venv，无需提前手动安装
autosota --repo /path/to/your-clone --devices 0,1
```

输入 / 输出结构：

```
~/my-project/                        ← 工作目录（autosota init 在这里初始化）
├── config.yaml                      ← 你填的 API key + 模型名
├── paper/
│   ├── target.md                    ← 你填的优化目标（指标、基线值、方向）
│   └── paper.pdf                    ← 可选，论文 PDF
├── .autosota/                       ← 运行时自动生成，无需手动创建
│   ├── venv/                        ← Python 虚拟环境（首次运行自动创建）
│   └── papers/<name>/               ← 每篇论文的内部状态
│       ├── config.yaml              ← onboard 阶段自动生成（含 repo_path、eval 命令等）
│       └── runs/<timestamp>/        ← 每轮优化的详细记录
├── logs/sota/<name>.log             ← 完整运行日志
└── optimized_code/<name>/           ← 最佳代码副本（含 final_patch.diff）

/path/to/your-clone/                 ← 代码仓库（独立于工作目录，需提前 clone 并配好环境）
```

全部数据都落在**工作目录**下，**不会污染 npm 全局安装路径**。需要在 CI / 共享机器上换位置时，设置 `AUTOSOTA_WORKSPACE=/some/dir` 或 `AUTOSOTA_DATA_DIR=/some/dir` 即可。

其他常用子命令：

```bash
autosota --version          # 版本
autosota --help             # 完整用法与所有选项
autosota init --force       # 覆盖已有 config.yaml / paper/
autosota doctor             # 环境自检
```

---

## 工作原理

```
paper/target.md          ← 你填写：论文名、指标、基线值、优化方向
paper/paper.pdf          ← 你提供（可选）：论文 PDF
paper/priors/            ← 你提供（可选）：先验知识注入（见下文）
本地代码仓库              ← 你已 clone，且本机环境能跑通评测（GPU/依赖自备）

         ↓  autosota [--repo /path/to/clone]

┌─────────────────────────────────────────────────────────┐
│ 1. Onboard  — Claude Code 在本机探路，自动发现 eval 命令、 │
│              基线指标，生成 config.yaml（含 repo_path）   │
├─────────────────────────────────────────────────────────┤
│ 2. Research — 调用 deep research 模型调研最新 SOTA 方案   │
├─────────────────────────────────────────────────────────┤
│ 3. [可选] Ideas Review                                    │
│     Claude 生成 idea library → 人工审核编辑 → 继续        │
├─────────────────────────────────────────────────────────┤
│ 4. Optimize — Claude Code 迭代优化循环（最多 N 轮）：      │
│     选 idea → 修改代码 → 本机评估 → 记录分数 → 滚动回退    │
└─────────────────────────────────────────────────────────┘

         ↓  优化完成

optimized_code/<paper>/      ← 最佳代码副本（从本机 repo_path 复制）
optimized_code/<paper>/final_patch.diff  ← `_baseline` → `_best` 的 diff（依赖仓库内 tag）
logs/sota/<paper>.log        ← 完整运行日志
```

编排器会将 **Claude Code 子进程的工作目录（cwd）设为 `config.yaml` 中的 `repo_path`**，与「在论文仓库里改代码、跑评估」一致；终端里 `[bash] $ ...` 等行来自 `optimizer/scripts/run.py` 对流式输出的过滤，表示模型正在调用工具。

---

## 快速开始

### 0. 前置准备（首次使用新机器时）

#### 0.1 安装代理（国内机器访问 GitHub / Anthropic 需要）

```bash
git clone --branch master --depth 1 \
  https://gh-proxy.org/https://github.com/nelvko/clash-for-linux-install.git \
  && cd clash-for-linux-install \
  && bash install.sh
```

#### 0.2 配置 OpenRouter API Key

将以下内容追加到 `~/.bashrc`，然后刷新：

```bash
cat <<EOF >> ~/.bashrc
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="你的key"
export ANTHROPIC_API_KEY=""
EOF

source ~/.bashrc
```

验证 Claude Code 可用：

```bash
claude
/model anthropic/claude-sonnet-4.6
# 随便说一句话，看是否正常回复
```

---

### 1. 安装环境

> `npm` 命令由 Node.js 自带，所以必须先装 Node.js（无法通过 npm 反过来装自己）。  
> 已经有 `node -v` ≥ 18 的用户可以直接跳到第 (2) 步。

```bash
# (1) 安装 Node.js 18+ —— 任选一种方式
#     ① 用 nvm（推荐，无需 sudo）：
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20 && nvm use 20
#     ② 或直接用系统包管理器：
#        Ubuntu/Debian:  sudo apt install -y nodejs npm
#        macOS:          brew install node

# (2) 全局安装 autosota（同时自动安装 Claude Code CLI）
#     将 autosota-0.1.0.tgz 下载到本地，cd 进文件所在目录，然后：
cd ~/Downloads                   # 换成你实际放 tgz 的目录
npm install -g ./autosota-0.1.0.tgz

# (3) 首次运行 claude 接受服务条款（只需一次）
claude       # 按提示同意条款后，/exit 退出
```

### 2. 初始化工作目录

**在你想存放配置和输出的目录下运行一次 `init`**（可以是任意位置，和代码仓库无关）：

```bash
cd ~/my-project                  # 新建或进入工作目录
autosota init                    # 生成 config.yaml 和 paper/ 模板
```

### 3. 填写配置

`autosota init` 已自动生成 `config.yaml`，直接编辑它：

```yaml
openrouter_api_key: "sk-or-v1-..."          # OpenRouter API Key
claude_model: "anthropic/claude-sonnet-4.6"  # 优化用模型
research_model: "openai/o4-mini-deep-research"  # 调研用模型
```

> `config.yaml` 已加入 `.gitignore`，不会被提交到远程仓库。

### 4. 准备输入文件

`autosota init` 已自动在工作目录下生成 `paper/` 目录，其中包含示例文件（`target.md` 是占位范例，`paper.pdf` 是示意占位，**都需要替换成你自己的内容**）。

**`paper/target.md`** — 描述优化目标（把范例内容改成你的论文信息）：

```markdown
**论文**：TS-RAG: Retrieval-Augmented Generation based Time Series ...

**任务**：使用 Chronos-Bolt 在 ETTh1 数据集上进行零样本预测，降低预测误差。

## 主要指标

| 指标 | 说明 | 优化方向 | 当前基线值 |
|------|------|----------|------------|
| `ETTh1_MSE` | ETTh1 均方误差（**主指标**） | 越低越好 ↓ | 0.3616 |
| `ETTh1_MAE` | ETTh1 平均绝对误差 | 越低越好 ↓ | 0.3650 |

## 目标

在主指标 `ETTh1_MSE` 上相比基线提升 5% 以上。
```

**`paper/paper.pdf`** — 论文 PDF（可选，把范例占位文件替换成真实 PDF，帮助 Claude 理解背景）

**`paper/priors/`** — 先验知识目录（可选，见[先验知识注入](#先验知识注入可选)）

### 5. 运行

```bash
# 最简方式：在工作目录下运行，--repo 指向已 clone 且能跑通评测的代码仓库
autosota --repo /path/to/your-clone

# 手动指定论文名称（默认从 target.md 自动推断）
autosota ts-rag --repo /path/to/your-clone

# 指定 GPU
autosota --repo /path/to/your-clone --devices 2,3

# 已有 onboard 结果，跳过 onboard 直接重跑优化
autosota --skip-onboard --skip-research

# 强制重新 onboard
autosota --repo /path/to/your-clone --force-onboard

# 先生成 idea，人工审核后再优化（见 Ideas Review 一节）
autosota --repo /path/to/your-clone --review-ideas
```

### Mock 仓库冒烟（可选）

仓库内自带最小假「论文代码」与一键脚本，用于验证本机路径、`record_score.sh`、dry-run / 真跑 Claude 等流程，**不消耗真实训练**：

```bash
bash examples/setup_mock_demo.sh
autosota mock-demo --skip-research --dry-run               # 只组 prompt，不启动 Claude
autosota mock-demo --skip-research --max-total-minutes 15  # 真跑 Claude，短时限（分钟）
```

说明见 `examples/README.md`。

---

## 命令行参数

```
autosota [paper_name] [选项]

可选 — 基础控制：
  [paper_name]              结果目录名（默认从 paper/target.md 自动推断）
  --repo <path_or_url>      本地克隆路径或仓库提示（传给 onboard；无 config 时建议提供）
  --devices <gpu_ids>       GPU 设备（默认: 0,1）
  --paper-dir <path>        target.md 所在目录（默认: ./paper/）
  --api-key <key>           OpenRouter API Key（覆盖 config.yaml）
  --skip-onboard            跳过 onboard，使用已有 config.yaml
  --force-onboard           强制重新 onboard
  --skip-research           跳过文献调研阶段
  --skip-export             跳过优化后的代码导出
  --export-on-failure       optimize 非 0 退出时仍导出 optimized_code/（默认仅成功时导出）
  --dry-run                 只生成 prompt，不实际运行

可选 — 优化参数覆盖（覆盖 config.yaml 中的对应值）：
  --max-iter N              最大迭代轮数
  --target-pct N            目标提升百分比（如: 10 表示 10%）
  --max-debug N             每轮最大调试次数
  --max-debug-min N         每次调试超时分钟数
  --research-timeout N      文献调研超时分钟数
  --max-total-minutes M     Claude 进程墙钟上限（分钟；冒烟测试可设 10～15）

可选 — 先验知识注入：
  --priors-dir <path>       先验知识目录（默认自动检测 ./paper/priors/）

可选 — Ideas 人工审核：
  --review-ideas            两阶段模式：先生成 idea library，暂停等待人工审核，再继续优化
  --ideas-file <path>       直接注入已有 idea_library.md（跳过 PHASE 2 idea 生成）
```

---

## 输出结构

所有输出都落在**工作目录**（`autosota init` 所在目录）下：

```
~/my-project/                                    ← 工作目录
├── logs/
│   ├── sota/<paper>.log                         # 整体运行日志（同时输出到终端）
│   └── optimizer_detail/
│       ├── <paper>_onboard.log                  # onboard 阶段 Claude 详细日志
│       └── <paper>.log                          # optimize 阶段 Claude 详细日志
├── .autosota/papers/<paper>/
│   ├── config.yaml                              # onboard 生成的论文配置
│   └── runs/latest/
│       ├── logs/
│       │   ├── master_prompt.md                 # 发给 Claude 的完整 prompt
│       │   ├── effective_config.yaml            # 含 CLI 覆盖后的实际配置快照
│       │   └── ideas_prompt.md                  # ideas review 模式使用的 prompt（如有）
│       ├── memory/
│       │   ├── research_report.md               # 文献调研报告
│       │   ├── code_analysis.md                 # 代码分析
│       │   └── idea_library.md                  # 优化 idea 库（可人工编辑后注入）
│       └── results/
│           ├── scores.jsonl                     # 每轮评估分数记录
│           └── optimization_curve.png           # 优化曲线图
└── optimized_code/<paper>/                      # 最佳代码快照（从 repo_path 复制）
    └── final_patch.diff                         # git diff _baseline _best
```

`autosota` 结束时退出码与 Claude Code 进程一致，便于 CI 判断。默认仅在优化成功（退出码 0）时导出代码；失败时仍要快照可加 `--export-on-failure`。

跟踪优化进度：

```bash
# 实时查看 Claude 在做什么
tail -f ~/my-project/logs/optimizer_detail/ts-rag.log

# 查看分数变化
cat ~/my-project/.autosota/papers/ts-rag/runs/latest/results/scores.jsonl
```

---

## 论文配置（.autosota/papers/\<paper\>/config.yaml）

此 `config.yaml` 由 onboard 步骤自动生成（不是工作目录根下的 `config.yaml`），主要字段：

```yaml
paper_title: "TS-RAG: ..."
repo_path: /path/to/TS-RAG          # 本机绝对路径
venv_path: ""                       # 可选：.venv/bin/activate
env_vars: "CUDA_VISIBLE_DEVICES=0"   # 可选
eval_command: bash script/zeroshot_chronos.sh
eval_output_format: "..."      # 告诉 Claude 如何解析评测输出
primary_metric: ETTh1_MSE      # 主优化指标
metric_direction: lower         # lower / higher
baseline_metrics:
  ETTh1_MSE: 0.3616
  ETTh1_MAE: 0.3650
target_improvement_pct: 5.0    # 目标提升百分比（可用 --target-pct 覆盖）
max_iterations: 24             # 最大迭代轮数（可用 --max-iter 覆盖）
max_debug_attempts: 3          # 每轮最大调试次数（可用 --max-debug 覆盖）
max_debug_minutes: 15          # 每次调试超时分钟（可用 --max-debug-min 覆盖）
research_timeout_minutes: 60   # 文献调研超时分钟（可用 --research-timeout 覆盖）
max_total_hours: 15.0          # Claude 进程墙钟上限（小时；可用 --max-total-minutes 覆盖）
gpu_devices: "0,1"             # 由 autosota 根据 --devices 写入 config
```

`autosota` 每次运行会把 **`gpu_devices` 覆盖为当前 `--devices`**，请与评测脚本、`env_vars` 中的 GPU 设置一致。

---

## 先验知识注入（可选）

在 `paper/priors/` 目录（或 `--priors-dir` 指定的任意路径）下放置以下文件，Claude 会在生成 idea 时优先参考：

| 文件 | 作用 |
|------|------|
| `references.md` | 背景文献与已知技术，Claude 生成 idea 时参考 |
| `ideas.md` | 用户直接指定的 idea，Claude **必须**全部纳入 idea library |
| `directions.md` | 探索方向约束（PREFERRED 优先 / FORBIDDEN 禁止） |

三个文件均为可选，放哪个用哪个。`paper_template/priors/` 目录下有带注释的模板可以参考。

```bash
# 使用 paper/priors/ 下的先验文件（自动检测，无需额外参数）
autosota --repo /path/to/your-clone

# 或手动指定任意目录
autosota --repo /path/to/your-clone --priors-dir /path/to/my_priors/
```

---

## Ideas 人工审核（可选）

使用 `--review-ideas` 开启两阶段模式，在优化开始前先让 Claude 生成完整的 idea library，由人工审核、筛选、编辑后再继续：

```
阶段一：Idea 生成
  Claude 完成 PHASE 0（本地环境）+ PHASE 1（代码分析）+ PHASE 2（idea 生成）
  → 写入 <工作目录>/.autosota/papers/<paper>/runs/latest/memory/idea_library.md
  → 终端打印文件路径，等待你按 Enter

         ↓  你打开文件，审核 idea，可以：删除低质量 idea、修改描述、
            调整优先级、添加自己的 idea

阶段二：正式优化
  按 Enter 确认后，Claude 以审核后的 idea library 直接进入 PHASE 3
  （跳过重新生成 idea，节省时间）
```

```bash
# 方式 A：两阶段自动流程
autosota --repo /path/to/your-clone --review-ideas

# 方式 B：已有审核好的 idea 文件，直接注入（跳过生成阶段）
autosota --skip-onboard --ideas-file ./paper/priors/ideas.md
```

---

## 崩溃恢复

优化中断后，直接加 `--skip-onboard` 重启（无需重新 onboard，会创建新 run）：

```bash
autosota --skip-onboard --skip-research
```

---

## 依赖

- 本机已配置好的 GPU 驱动 / CUDA（若评测需要 GPU）
- Python 3.10+（用于 venv 创建）
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)（随 `npm install -g ./autosota-0.1.0.tgz` 自动安装）
- [OpenRouter](https://openrouter.ai/) API Key
