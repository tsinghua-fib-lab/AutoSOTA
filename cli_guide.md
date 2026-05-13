# autosota

自动化 SOTA 代码优化流水线。给定一篇 ML 论文的**本地已克隆代码仓库**（及可跑通评测的环境）和优化目标，由 AI 自主探索、研究、修改代码，迭代提升性能指标。当前 CLI 0.2.0 版本。

---

## 📣 News

- **2026-05-13** 📝 补充了 DeepSeek 官方直连和经 OpenRouter 调 DeepSeek 两种配置方式的详细说明。
- **2026-05-12** 🚀 发布 **v0.2.0** — 支持 DeepSeek 等国产模型直接接入（不用再绑死 OpenRouter），代码优化和文献调研可以分开用不同的 API key，新增评测脚本防篡改保护。
- **2026-04-29** 🔧 发布 **v0.1.1** — 新增轮间暂停功能（`-i`），跑着跑着想看看进度、加点指示不用杀进程了；新增 `ask / steer / sessions` 等交互命令。
- **2026-04-22** 🎉 发布 **v0.1.0** — 首版发布，实现论文代码自动优化闭环：自动探索仓库配置、查阅相关文献、多轮迭代改代码，结束后自动导出最优代码和 patch。
- **2026-04-21** 💻 在实验室服务器初始化 AutoSOTA CLI 项目。

---

## 一分钟上手（离线包全局安装）

> 前置：Node 18+、Python 3.10+、`bash`（Linux/macOS/WSL）、`git`。Claude Code CLI 会随 autosota 自动安装，无需单独安装。

```bash
# 1. 用开发者发的 .tgz 离线包全局安装（同时自动安装 Claude Code CLI）
npm install -g ./autosota-X.Y.Z.tgz
# 升级到新版本时同样用这条命令覆盖装即可（详见 1.1 升级到新版本）

# 2. 第一次运行 claude，接受服务条款（只需一次）
claude
# 按提示同意条款后退出（/exit 或 Ctrl+C）

# 3. 新建（或进入）你的工作目录，生成脚手架
#    工作目录可以放在任意位置，例如 ~/my-project 或 /data/my_exp
#    它只用来存放配置和输出，和代码仓库是两个独立的文件夹
cd ~/my-project
autosota init               # 生成 ./config.yaml + ./paper/ 模板

# 4. 编辑两处：
#    - ./config.yaml               填入 openrouter_api_key + 模型名
#                                  （或 claude_api_key + research_api_key 拆账单到不同渠道）
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
autosota sessions           # 列出当前工作目录下所有历史 run
autosota inspect <ref>      # 查看单个 run 的详细结果（ref 可以是 paper 名 / run_id / latest）
autosota inspect <ref> --logs   # 实时 tail -f 某个 run 的主日志（Ctrl+C 退出）
autosota ask <问题...>          # 针对最新 run 问 Claude（当前跑到第几轮、哪个 idea 最好等）
autosota steer <指示...>        # 向正在运行的 run 注入一条"下一轮必须照做"的用户指示
autosota -i <paper> ...         # 启用交互模式：每轮迭代结束后自动暂停
autosota pause   [<run>]        # 中途打开/关闭暂停模式（--off 关闭）
autosota continue [<run>]       # 放行一个正在暂停的 run，让它进入下一轮
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

#### 0.2 配置 API Key（用于初次跑通 `claude` CLI）

> 注意：autosota 自身的运行**只读 `config.yaml`**，并在内部为 `claude` 子进程
> 注入正确的 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`。这一节只是为了让你
> 第一次手动跑 `claude` 接受服务条款时能跑通。

任选一种渠道，把对应 env 加到 `~/.bashrc`：

```bash
# 选项 A：OpenRouter（一个 key 同时跑 Claude 和 Research）
cat <<EOF >> ~/.bashrc
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="你的openrouter_key"
export ANTHROPIC_API_KEY=""
EOF

# 选项 B：Anthropic 官方 / Coding Plan（更便宜，仅 Claude；调研另配 research_api_key）
cat <<EOF >> ~/.bashrc
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
export ANTHROPIC_AUTH_TOKEN="sk-ant-你的key"
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

# (2) 用开发者发的 .tgz 离线包全局安装（同时自动安装 Claude Code CLI）
npm install -g ./autosota-X.Y.Z.tgz

# (3) 首次运行 claude 接受服务条款（只需一次）
claude       # 按提示同意条款后，/exit 退出
```

### 1.1 升级到新版本

拿到开发者发的新版 `.tgz` 离线包之后，直接覆盖装：

```bash
npm install -g ./autosota-X.Y.Z.tgz

# 验证
autosota --version          # 显示新版本号即成功
```

如果新功能没生效，多半是 npm 有缓存——按下面任一方式强装：

```bash
# 方式一：跳过缓存
npm install -g --force ./autosota-X.Y.Z.tgz

# 方式二：彻底清缓存再装
npm cache clean --force
npm install -g ./autosota-X.Y.Z.tgz
```

> **当前装的什么版本？**
> ```bash
> autosota --version
> ```

### 2. 初始化工作目录

**在你想存放配置和输出的目录下运行一次 `init`**（可以是任意位置，和代码仓库无关）：

```bash
cd ~/my-project                  # 新建或进入工作目录
autosota init                    # 生成 config.yaml 和 paper/ 模板
```

> git 直接检出的用户：`autosota init` 同样适用；也可以直接在检出目录下手动复制 `config.yaml.example → config.yaml`。

### 3. 填写配置

`autosota init` 已自动生成 `config.yaml`，直接编辑它：

```yaml
# ── 默认 / 兼容旧版（一个 key 走天下） ──
openrouter_api_key: "sk-or-v1-..."           # 同时给 Claude + Research 用

# ── 代码优化（Claude Code）──
claude_model: "anthropic/claude-sonnet-4.6"
claude_api_key: ""                            # 留空回退到 openrouter_api_key
claude_base_url: ""                           # 留空回退到 https://openrouter.ai/api

# ── 文献调研（deep_research.py）──
research_model: "openai/o4-mini-deep-research"
research_api_key: ""                          # 留空回退到 openrouter_api_key
research_base_url: ""                         # 留空回退到 https://openrouter.ai/api/v1
```

**为什么要分开 key？** OpenRouter 没有 Claude Coding Plan，跑长时间优化的费用消耗较大。
把 `claude_*` 单独配置成更便宜的渠道（DeepSeek / Anthropic 官方 / Moonshot / 智谱 / 自建中转……），
再让 `research_*` 选适合的调研模型，可以显著降低成本。留空时所有字段都会回退到
`openrouter_api_key`，行为完全等价于旧版本。

**Claude 端能直接接哪些？** 凡是提供 `/v1/messages`（Anthropic 兼容协议）的端点都能直填，
目前国内/国际主流厂商基本都已上：

| 提供商 | `claude_base_url` | `claude_model` 示例 |
|---|---|---|
| **DeepSeek** ⭐ 推荐 | `https://api.deepseek.com/anthropic` | `deepseek-v4-pro` / `deepseek-v4-flash` |
| Anthropic 官方 | `https://api.anthropic.com` | `claude-sonnet-4-5-20250929` |
| OpenRouter | `https://openrouter.ai/api` | `anthropic/claude-sonnet-4.6` |
| Moonshot Kimi | `https://api.moonshot.cn/anthropic` | `kimi-k2-0905-preview` |
| 智谱 GLM | `https://open.bigmodel.cn/api/anthropic` | `glm-4.6` |
| OpenAI / Gemini | ❌ 无原生 Anthropic 端 | 需经 [claude-code-router](https://github.com/musistudio/claude-code-router) 中转 |

**关于调研模型**：只有以下几个是「专用深度调研模型」——它们会真正联网搜索 arXiv / Web，生成带真实引用的报告（经 OpenRouter 调用）：
- `openai/o4-mini-deep-research`（最常用，速度与质量均衡）
- `openai/o3-deep-research`（更强但更慢更贵）
- `perplexity/sonar-deep-research`

普通 chat 模型（`deepseek-v4-pro`、`gpt-4o` 等）也能填，但不会联网，只凭训练数据作答，适合 2024 年以前的经典方向或快速冒烟。

常见组合示例：

```yaml
# ── 示例 1：全走 OpenRouter（最省事，一把 key 搞定）──
openrouter_api_key: "sk-or-v1-你的-key"
claude_model:       "anthropic/claude-sonnet-4.6"
research_model:     "openai/o4-mini-deep-research"
# claude_api_key / claude_base_url / research_api_key / research_base_url 留空即可

# ── 示例 2：DeepSeek V4 Pro 跑代码优化 + OpenAI o4-mini 真·深度调研（推荐）──
# 代码优化走 DeepSeek（便宜，迭代几十轮也不心疼）
# 调研只跑一次，走专用深度调研模型（真联网搜文献，质量更高）
# 模型名带 [1m] → 走 DeepSeek 官方 anthropic 端点的 1M 上下文档位，长轮迭代不撞 window；
# 1M 档位计费略高（>32K 输入按高一档单价），短任务可改回 deepseek-v4-pro。
openrouter_api_key: ""
claude_api_key:    "sk-你的-deepseek-key"
claude_base_url:   "https://api.deepseek.com/anthropic"
claude_model:      "deepseek-v4-pro[1m]"
research_api_key:  "sk-or-v1-你的-openrouter-key"
research_base_url: "https://openrouter.ai/api/v1"
research_model:    "openai/o4-mini-deep-research"

# ── 示例 3：只用 OpenRouter 一个 key 调 DeepSeek（不推荐，但能跑）──
# 适用人群：只想维护一个 OpenRouter 账号，不想再去 DeepSeek 官网注册。
# 代价：拿不到 1M 窗口（OpenRouter 不认 [1m] 后缀）、拿不到 prompt caching、
#       不能 extended thinking，长轮迭代容易撞 context 超长。
#       建议同时砍迭代轮数（--max-iter 8 --max-debug 2）并 --skip-research。
openrouter_api_key: ""
claude_api_key:    "sk-or-v1-你的-openrouter-key"
claude_base_url:   "https://openrouter.ai/api"
claude_model:      "deepseek/deepseek-v4-pro"      # OpenRouter slug 格式，别带 [1m]（OpenRouter 不认）
research_api_key:  "sk-or-v1-你的-openrouter-key"
research_base_url: "https://openrouter.ai/api/v1"
research_model:    "openai/o4-mini-deep-research"
claude_env:
  CLAUDE_CODE_MAX_OUTPUT_TOKENS: 16000              # 砍输出，给输入腾窗口
  CLAUDE_CODE_EFFORT_LEVEL: low                     # 关 thinking，避免 thinking 块膨胀
  ANTHROPIC_DEFAULT_OPUS_MODEL:   deepseek/deepseek-v4-pro
  ANTHROPIC_DEFAULT_SONNET_MODEL: deepseek/deepseek-v4-pro
  ANTHROPIC_DEFAULT_HAIKU_MODEL:  deepseek/deepseek-chat    # 子任务用便宜 chat，省窗口
  CLAUDE_CODE_SUBAGENT_MODEL:     deepseek/deepseek-chat
```

> `config.yaml` 已加入 `.gitignore`，不会被提交到远程仓库。

#### 示例 3 的注意事项（走 OpenRouter+DeepSeek 必看）

**光改 `config.yaml` 还不够，这条线有一个坑要处理。**

OpenRouter 会把你的请求随机转给旗下多家服务商（DeepInfra / Fireworks / DeepSeek 官方 …），
这些服务商能接受的最大上下文长度不一样，有的只有 64K。autosota 每跑一轮，
prompt 里的对话历史就会增长一点，通常跑到第 8–10 轮就会超过某些服务商的上限，
直接报错 `400 prompt is too long`，而且**你不知道哪一轮会炸**，很难排查。

**步骤一：在 OpenRouter 控制台固定服务商**

先把请求锁定到支持长上下文（128K）的服务商，否则后面怎么调都没用：

- 打开 https://openrouter.ai/settings/preferences → **Provider Preferences**
- 找到 `deepseek/deepseek-v4-pro`，Sort 改成 **Throughput / Context**
- **Allowed providers** 只勾 `DeepSeek`（官方）和 `DeepInfra`，其他全关掉

**步骤二：别一上来就跑满 24 轮**

即使锁了服务商，OpenRouter 这条线最多也只有 128K 窗口，没有 prompt caching（每轮都按全价收），
跑太多轮一样会炸，而且费用也比走 DeepSeek 官方贵。建议从 12 轮起步，看情况再调：

```bash
# 先跑 12 轮试水
autosota --repo /path/to/repo --max-iter 12 --max-debug 2

# 跑完看日志：logs/optimizer_detail/<paper>.log
# - 12 轮都正常 → 下次可以加到 16 轮
# - 某轮报 400 → 加上 --skip-research 再试，或直接降到 --max-iter 8
```

> **效果 trade-off**：轮数砍一半大约损失 30% 的提升幅度（autosota 通常前 8 轮拿走大部分收益，
> 后期是细调阶段）。加 `--skip-research` 会让优化方向变窄，模型只能靠读代码和论文想 idea，
> 不能参考其他相关工作。**`paper/target.md` 和 priors 不要动**，这俩是优化方向感的来源。

**步骤三：还是撞了怎么办**

看 `logs/optimizer_detail/<paper>.log` 里的报错，如果确认是 `400 prompt is too long`，
三步都做了还撞，说明这条线真的到极限了。最省心的解法是把 `claude_*` 换回[示例 2](#3-填写配置)
的 DeepSeek 官方端点——1M 窗口 + prompt caching（重复部分按 1/10 价计费）+ extended thinking
只在官方端点可用，OpenRouter 转一手就全没了，长任务跑下来官方端点其实更便宜。

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

**npm CLI 方式（推荐）：**

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

**git 直接检出方式（在项目根目录下）：**

```bash
bash run.sh --repo /path/to/your-clone
bash run.sh ts-rag --repo /path/to/your-clone
bash run.sh --repo /path/to/your-clone --devices 2,3
bash run.sh --skip-onboard --skip-research
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

## 查看历史记录（sessions / inspect）

在工作目录下随时查看过往的优化结果，不用手动去翻 `.autosota/papers/...` 目录。

> 以下所有示例中的 `savvy` 只是**示例论文名**（来自 `target.md` 里填的 `paper_name`）。实际使用时替换成你自己的论文名即可，例如 `autosota inspect <你的论文名>`。

### `autosota sessions` — 列出所有 run

```bash
$ autosota sessions
autosota sessions  (/path/to/my-project)

  Paper     Run                           Started            Status        Iters   Best (Δ vs baseline)
  ────────────────────────────────────────────────────────────────────────────────────────────────────
  savvy     run_20260422_131540 (latest)  2026-04-22 13:15   ✓ complete    3/3     59.400  +2.41%
  savvy     run_20260422_125019           2026-04-22 12:50   ~ aborted     0/3     58.000
  ts-rag    run_20260420_174437 (latest)  2026-04-20 17:44   ✓ complete    12/24   0.2901  +19.9%
```

- `Status`：`✓ complete`（正常结束，有 `final_report.md`）/ `⚡ running`（进行中）/ `~ aborted`（中断）/ `○ empty`（从未跑过）
- `Iters`：已完成优化轮数 / `max_iterations`（不含 baseline）
- `Best`：此次 run 的最佳主指标 + 相对 baseline 的百分比变化

常用选项：

```bash
autosota sessions --paper savvy   # 只看某一篇论文
autosota sessions --json          # 机器可读 JSON，方便 jq / 脚本化
```

### `autosota inspect <ref>` — 查看单个 run 的详情

`ref` 可以是：
- `latest`（最新一次 run）
- 论文名：如 `savvy`（自动选该论文的 latest）
- 完整 `run_id`：`run_20260422_131540`
- 短后缀：`131540`（匹配 run_id 结尾）

```bash
$ autosota inspect savvy
savvy — SAVVY: Spatial Awareness via Audio-Visual LLMs ...

  Run       : run_20260422_131540  (latest)
  Dir       : .../runs/run_20260422_131540
  Started   : 2026-04-22 13:15:40   (~1h 59m)
  Status    : ✓ complete
  Target    : overall_accuracy ↑ higher is better  •  baseline 58.000  •  target +6.9%

  Iterations (3/3)
    Iter   Idea                                                 Metric    Δ vs base    Status
    ───────────────────────────────────────────────────────────────────────────────────────
    0      Paper baseline                                       58.000    +0.00%       ✓ success  (baseline)
    1      Fuse SD and Seg tracks for static object localiz…    57.100    -1.55%       ✓ success
    2      Cluster centroid for sounding position + best K…     58.900    +1.55%       ✓ success
    3      Tune direction angle thresholds for 3-choice an…     59.400    +2.41%       ✓ success  ← best

  Best      : 59.400 (iter 3, commit ad44b9f)   +2.41% vs baseline   (target +6.9%)

  Files
    ✓  effective config   .../logs/effective_config.yaml
    ✓  master prompt      .../logs/master_prompt.md
    ✓  idea library       .../memory/idea_library.md
    ✓  research report    .../memory/research_report.md
    ✓  scores.jsonl       .../results/scores.jsonl
    ✓  optim curve        .../results/optimization_curve.png
    ✓  final report       .../final_report.md
    ...
```

也可以直接把单个文件 dump 到 stdout（方便 `| less` / 重定向 / 喂给其他工具）：

```bash
autosota inspect savvy --report    # 打印 final_report.md
autosota inspect savvy --prompt    # 打印 master_prompt.md
autosota inspect savvy --scores    # 打印 scores.jsonl
autosota inspect savvy --json      # 结构化 JSON 供脚本使用
```

---

## 实时跟进（inspect --logs / ask / steer）

这三条命令让你在 run "跑起来之后还能说话"，不用停掉进程、不用手动翻日志。

### `autosota inspect <ref> --logs` — 实时尾随日志

等价于 `tail -F .../logs/sota/<paper>.log`，按 **Ctrl+C** 退出。

```bash
autosota inspect savvy --logs            # 默认：主 SOTA 日志（高信噪比）
autosota inspect savvy --logs detail     # Claude 详细 JSON 流日志（调试用）
autosota inspect savvy --logs onboard    # onboard 阶段的日志
```

### `autosota ask <问题...>` — 针对当前 run 问 Claude

脚本会自动打包上下文（`scores.jsonl`、`idea_library.md` 前 4000 字符、主日志末尾 120 行、基本指标与状态），然后调用 `claude -p` 回答你的问题。

```bash
autosota ask 现在优化到第几轮了？效果怎么样？
autosota ask 下一步最该尝试哪个 idea？为什么？
autosota ask --run savvy 为什么第 2 轮精度反而下降了？
autosota ask --run latest summarize what ideas have been tried so far
```

- 默认问的是工作空间中**最新一次 run**；用 `--run <paper|run_id|latest>` 可指定其他 run。
- 不会中断正在跑的优化进程，只是起一个新的 claude 子进程读取只读上下文。
- 需要 `claude` CLI 可用，以及 workspace 根目录下 `config.yaml` 中的 `claude_model`（fallback: `anthropic/claude-sonnet-4.6`）。

### `autosota steer <指示...>` — 把用户指示注入下一轮

向**正在运行**的 run 目录写一个 `steer.md`，优化器的 prompt 会让 Claude 在**下一轮开头**优先读它、照做、然后删掉这个文件。

```bash
autosota steer 下一轮试试混合精度训练
autosota steer 注意不要再改 model.py 里的优化器，先聚焦在数据侧
autosota steer --run savvy focus on reducing memory, not accuracy
```

- 写完就返回，不会阻塞（run 还在后台正常跑）。
- 只有**最近一次**的 steer 有效，重复调用会覆盖上一条未被消费的指示。
- 若当前 run 已结束，命令会给出警告但依然写入文件（此时不会生效，除非你后续 resume）。
- 若 steer 内容与安全红线（R1–R6）冲突，Claude 会在 `idea_library.md` 里记录冲突并跳过，继续正常流程。

---

## 交互暂停模式（-i / pause / continue）

默认情况下 `autosota` 一旦启动会一口气跑完所有迭代。如果你想**每轮结束后停一下**，看看效果再决定下一轮怎么走（比如配合 `steer`），可以用交互模式。

### 启动时启用

```bash
autosota -i savvy --repo ~/savvy/SAVVY --max-iter 5
```

加 `-i` (或 `--interactive`) 即可。每轮 `record_score.sh` 写完、`idea_library.md` 更新完之后，优化器会**主动暂停**，等你显式放行才继续。

### 典型工作流

终端 A — 跑优化（前台或 nohup 都行）：
```bash
$ autosota -i savvy --repo ~/savvy/SAVVY
# ... iter 1 跑完 ...
# Claude 输出: [Interactive] Paused after iter 1. Run 'autosota continue' to resume.
```

终端 B — 检视与干预：
```bash
$ autosota sessions
  savvy   run_20260425_120500 (latest)   2026-04-25 12:05   ⏸ paused (iter 1)   1/5   ...
$ autosota inspect savvy --scores       # 看这轮指标
$ autosota ask 这轮为什么没怎么涨？      # 让 Claude 分析一下
$ autosota steer 下一轮试试调 angle threshold 而不是再改 fusion   # 给指示
$ autosota continue                      # 放行
                                         # → Claude 进 iter 2，先读 steer.md 然后照做
```

### 中途切换

不必启动时就决定，你可以**任何时候**打开/关闭暂停模式：

```bash
autosota pause             # 当前 run 从下一轮开始暂停
autosota pause --off       # 之后不再暂停（如果当前正在 paused，仍需 continue 放行）
autosota continue          # 仅放行当前那一次暂停
```

### 实现要点

- 通过 `run_dir/{interactive,paused,continue}.flag` 三个空文件协同，对优化主流程零侵入：
  - `interactive.flag` 是开关，`-i` 启动时创建；`autosota pause` 也写它；`pause --off` 删它
  - `paused.flag` 由 Claude 在每轮末尾创建，记录当前停在哪一轮
  - `continue.flag` 由 `autosota continue` 写入，Claude 看到后会同时删掉前两个 flag
- `sessions` / `inspect` 看到 `paused.flag` 就显示 `⏸ paused (iter N)`
- 底层等待用 `sleep + flag 检测` 轮询，每次最多 9 分钟（受 Claude Code 工具超时限制），到点优化器会自动重新等——所以挂着一晚上没问题，token 开销可忽略

> 不加 `-i` 的 run 完全不受影响——优化器只是在每轮末尾做个 `if [ -f interactive.flag ]` 检查，几乎零开销。

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

# ── 评测脚本保护（可选，默认关闭）──
# 不写这一段 → 行为跟以前完全一样（Claude 可以改 repo 里任何文件）
# 写了这一段 → 启用 SHA256 指纹校验，详见下文「评测脚本保护」小节
# protected_paths:
#   - "eval.py"                  # 评测入口脚本
#   - "src/metrics/"             # metric 计算模块
#   - "data/test/"               # 测试集 / ground truth
```

`autosota` 每次运行会把 **`gpu_devices` 覆盖为当前 `--devices`**，请与评测脚本、`env_vars` 中的 GPU 设置一致。

---

## 评测脚本保护（`protected_paths`，可选）

> **完全 opt-in 的功能**：默认关闭，行为跟没这功能时一模一样。当你发现模型在偷偷改评测脚本刷分（比如把 `accuracy = correct / total` 改成 `accuracy = max(correct, 0.95)`），或者怀疑它篡改了测试集时，再来开它。

### 何时考虑启用

- ✅ 跑过几轮发现 `_best` 涨得不正常，diff 出来发现动了 metric 代码
- ✅ paper 的评测协议特别敏感（比赛复现 / 论文 reproducibility 实验）
- ✅ 用 deepseek / GLM 这种比较"自由发挥"的模型
- ❌ 第一次跑、还没观察到问题——可以先不开

### 工作机制

1. **iter 0（baseline）跑完** → `record_score.sh` 自动给 `protected_paths` 里每个文件 / 目录算 SHA256，存到 `<repo_path>/.autosota_protected_hashes.json`
2. **每一轮迭代结束**前 `record_score.sh` 重新算一遍指纹，跟基线对比
3. **指纹不匹配** → exit code 9（PROTOCOL VIOLATION），这一轮：
   - **不会** `git commit`
   - **不会**写入 `scores.jsonl`
   - **不会**更新 `_best` 标签
   - 终端打印被改动的文件清单
4. Claude 读到 prompt 里关于 exit 9 的处理说明（仅当启用时才注入到 prompt），会自动 `git checkout` 还原并把这个 idea 标记为 `REJECTED (red line)`，跳到下一个 idea

模型本质是奖励驱动的，"动评测脚本拿不到分"试一两次它就放弃了——这是比红线提示靠谱得多的硬约束。

### 启用方法

编辑对应 paper 的 `config.yaml`（`<workspace>/.autosota/papers/<paper名>/config.yaml`），把示例那几行的注释取消、填上你要锁的路径：

```yaml
protected_paths:
  - "eval_savvy_qa.py"
  - "data/test_json/savvy_bench.json"
```

存量 paper 第一次启用时记得删掉旧指纹（如果之前跑过）：
```bash
rm -f <repo_path>/.autosota_protected_hashes.json
```

下次 `autosota <paper>` 跑到 iter=0 会自动重建基线指纹。

### 关闭 / 撤销

- 想**临时关掉**：注释掉 `config.yaml` 里的 `protected_paths` 块即可，下次跑就完全跳过校验，回到老行为
- 评测协议**确实需要修改**（比如修评测里的 bug）：改完之后 `rm <repo_path>/.autosota_protected_hashes.json`，下一次跑会以新版为基线重建

### 怎么填（启用时）

| 该锁 | 不该锁 |
|------|--------|
| ✅ 评测主入口脚本（`eval_command` 实际执行的那个） | ❌ 主流程 / 优化目标代码（`main.py`、`src/model/`） |
| ✅ Metric 计算模块（如果与主入口分离） | ❌ 每轮 main.py 重新生成的中间产物（如 `data/output/predavmap.json`） |
| ✅ 测试集 / benchmark 数据 / ground truth labels | ❌ 训练集（如果你确实允许调训练超参） |
| ✅ Held-out gold answers | ❌ 配置文件（除非配置就是测试协议本身） |

路径都是**相对 `repo_path`** 的；列文件比列目录便宜（少算 hash）。**宁可多锁不能少锁**——锁错了顶多让 Claude 多 reject 一次 idea，少锁了等于没设防。

> ⚠️ **小心目录命名陷阱**：有的 repo 把"算法预测代码"和"算分数代码"都塞在 `evaluation/` 之类的目录里。**不能光看目录名整个锁**——要跟着 `eval_command` 入口的 import 链 trace 才准。建议启用前先 `git log` 看下历史 commit 改过哪些文件，确认你想锁的那些过去没被合法 idea 修改过。

### 与"软约束"配合（启用时）

`protected_paths` 是最后一道**硬防线**。前面还有两层（默认就生效）：

1. **`paper/target.md`** 里写"评测协议（不可改动）"小节 — 让 Claude 在**生成 idea 阶段**就过滤掉违规方向（前置约束）
2. **prompt 红线**（`optimize_prompt.md` 已内置）— 在 idea 自审 / 实施环节再挡一道
3. **`protected_paths` SHA256 校验**（本节，opt-in） — 万一前两层都没拦住，奖励信号层兜底

三层叠加，模型基本不会再去碰评测脚本。

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
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)（随 `npm install -g ./autosota-X.Y.Z.tgz` 自动安装）
- [OpenRouter](https://openrouter.ai/) API Key

