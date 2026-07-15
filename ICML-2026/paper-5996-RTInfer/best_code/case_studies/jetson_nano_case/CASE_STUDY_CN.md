# Modern Mixed Workload Case（推荐版本）

这个版本是根据 rebuttal 之后的定位重做的解释用 case：不再只用原文里的小 CNN，而是混合了具体的现代/大模型任务：Traffic light detection、YOLOv8-L 高分辨率检测、MobileViT-S、ViT-L、GPT-2-small KV cache、ResNet152 wildfire detection。它的目标是说明：在端侧内存紧张、模型更大且异构、任务周期性到达的情况下，SOTA/Pantheon 即使允许 `RT + BE` 并发，仍然会因为缺少跨 RT task 的全局 `Time x Address` packing 和 load-aware variant switching，被迫让后续 RT task 浅早退或 miss deadline；RTInfer 则通过 ALC、memory-layout-aware scheduling、Delta-Graph 三个技术点把更多 RT task 同时塞进 memory budget 内。

## 运行方式

```bash
cd RTInfer
./case_studies/jetson_nano_case/run_case.sh
```

推荐输出文件：

- `outputs/modern_mixed_case.svg`：推荐使用的新图。
- `outputs/modern_variant_table.csv`：每个任务的 SOTA/RTInfer 选择，包括 exit、pruning、latency、memory、accuracy、missing chunks。
- `outputs/modern_pantheon_trace.csv`：SOTA 的 `Time x Address` trace。
- `outputs/modern_rtinfer_trace.csv`：RTInfer 的 `Time x Address` trace。
- `outputs/modern_online_decisions.md`：逐 arrival/DDL 的 online 决策说明。
- `outputs/modern_summary.txt`：DMR、deadline-weighted accuracy、completed-only accuracy、makespan。

## Workload

- 模拟设备：Jetson Nano。
- Memory budget：`4096 MiB`。
- 不再画全局常驻 Reserve IF。图中的斜线块只表示被抢占 RT job 的局部 `IF hold`：中间特征保留在该任务原先的低地址 allocation 里，之后该 job 可以从断点继续。
- 高压窗口：`0-760 ms`，重点是 `0-160 ms` 的首个 burst 和 `180/255/310 ms` 的第二周期到达。

RT tasks：

| Task | Model | Period | Arrival | DDL |
| --- | --- | --- | --- | --- |
| Task I | Traffic light detection / MobileNetv2-SSDLite-300 | 180ms | 0ms | 260ms |
| Task II | High-res object detection / YOLOv8-L-1080p | 220ms | 35ms | 320ms |
| Task III | UAV scene recognition / MobileViT-S | 240ms | 70ms | 340ms |
| Task IV | Large scene recognition / ViT-L-1024 | 300ms | 105ms | 420ms |
| Task V | Edge command generation / GPT-2-small KV | 360ms | 130ms | 500ms |
| Task VI | Wildfire detection / ResNet152-512 | 260ms | 160ms | 380ms |

第二周期 arrivals：`A_I'=180ms`、`A_II'=255ms`、`A_III'=310ms`。

## SOTA / Pantheon 语义

公平性约束：SOTA 允许 `RT + BE` 并发。图中 BE analytics 被画成多个绿色分段，而不是连续长条：它会在低压区间与 RT task 同时 live，在 RT burst 高压时被 preempt / paused，之后再 resume。因此这个 case 没有通过“抢死 BE”来制造 RTInfer 的优势。

为什么 SOTA 先运行 Task I：`A_I=0ms`，此时还没有其他 RT task 到达，所以 Pantheon 的 arrival/urgency queue 自然把 Task I 作为队头运行。这不是我们人为让 SOTA 先执行某个有利或不利的任务。

Reserve IF 的正确语义：SOTA 图里的斜线块不是固定扣掉的一块公共 reserve，而是 `T1-IF` 这样的局部中间特征保留。这个 case 里 Task I 先执行 `T1-C1`，随后被 YOLOv8-L 抢占；`T1-IF` 留在 Task I 原本的 base address 上，YOLOv8-L 的 `T2-C1/C2` 堆叠在它上方，等 YOLOv8-L 完成后 Task I 回到原地址区间继续执行 `T1-C2/C3`，不用从头重算。

但 SOTA 不做 RTInfer 的三件事：

- 不从 ALC variant atlas 里为多个 RT task 联合选择轻量高精度变体。
- 不把多个 RT task 的 chunks/KV/IF 作为统一矩形做全局 `Time x Address` packing。
- 不使用 Delta-Graph 对 variant switching 做 load-aware missing chunk 加载。

因此 SOTA 的在线行为是：

- Task I 首次使用前需要 `Full load`，之后作为队头可以跑到 `E3`，同时 BE0 在高地址区间并发运行。
- YOLOv8-L 到达后触发抢占，Task I 的 IF 在原地址保留；YOLOv8-L 从理想的 `E4` 降到 `E2`，并被分配在 held IF 上方。
- MobileViT-S 和 ViT-L 因为等待更久，分别降到 `E1`。
- GPT-2 使用 `T5-C1, KV=max` worst-case KV cache rectangle，一整块大内存持续占用。
- 第二周期 Task I 在 SOTA 中仍然是 `Full reload`，加载条明显长于 RTInfer 的 Delta load。
- Task VI 和第二周期 Task I miss deadline，图中用红色 `MISS` 标注。

最终结果：

- `DMR = 2/7 = 0.2857`
- `deadline_weighted_accuracy = 0.4943`
- `completed_only_accuracy = 0.6920`
- `makespan = 760ms`

## RTInfer 语义

RTInfer 的 online scheduler 在每次 arrival 时重新规划 active set：

- ALC 提供现代模型的 variant atlas，例如 `P25-E4`、`P50-E3`、`P25-E3-stepped-KV`。图中每个 RT 矩形右上角都有 `p=...,E...`，其中 `p` 是剪枝率，`E` 是 early-exit point。
- Memory-layout-aware scheduler 把 RT chunks、BE、局部 IF hold 作为 `Time x Address` 里的 memory 对象布局；H2D load 画在时间线上方独立 lane，不占 address plane。
- Delta-Graph 的优势主要体现在第二轮同模型 variant switching：首次进入 GPU 仍然是 `Full load`，之后 `T1'`、`T2'`、`T3'` 只加载 missing chunks，显示为更短的 `Delta load` strip。约束是 `load_end <= first_use_start`。
- GPT-2 KV cache 用 `T5-C1/C2/C3, KV step` stepped footprint，而不是 SOTA 的 `KV=max` rectangle。

关键区间：

- `t=170-210ms`，RTInfer 同时 live：MobileNetv2-SSDLite-300、YOLOv8-L-1080p、MobileViT-S、ViT-L-1024、GPT-2-small KV step、ResNet152 wildfire detection，以及 BE。
- 所有对象地址互不重叠，顶部低于 `4096 MiB` memory budget。

最终结果：

- `DMR = 0/9 = 0`
- `deadline_weighted_accuracy = 0.8656`
- `completed_only_accuracy = 0.8656`
- `makespan = 500ms`

## 三个技术点在图中的对应关系

- ALC：RTInfer 选择 `P25-E4`、`P50-E3`、`P50-E2`、`P25-E3-stepped-KV`，不是简单更浅早退；图中右上角 `p,E` 标签直接体现剪枝率和早退点。同一 task 第二周期的高剪枝 variant 横向宽度更短，表示运行时间更短。
- Memory-layout-aware scheduler：RTInfer 下半图在 `170-210ms` 形成多 RT task 并发，但 address 错开。
- Delta-Graph：首次加载为 `Full load`，第二轮同模型切换为更短 `Delta load`；异构模型之间不强行共享权重，但同一模型 variant switching 避免 full reload。

## 图中标号如何读

- 左上角 `T1-C1` 表示第 1 个任务的第 1 个执行 chunk；第二周期用 `T1'-C1`，不会和第一个 job 混淆。
- 中央 `C1/C2/C3` 表示当前矩形对应的执行 chunk。GPT-2 也统一用 `T5-C*`，KV 信息只作为 `KV=max` 或 `KV step` 的 footprint 注释。
- 右上角 `p=0.25,E4` 表示该矩形属于剪枝率 `0.25`、第 4 个 early-exit point 的变体。
- 矩形宽度表示该 chunk/variant 的运行时间，矩形高度表示 memory footprint。
- 红色边框和 `MISS` 表示该 job 超过 DDL。

---

# Historical Note：旧版 3-Task Case

下面内容是旧版最小 case 的说明，保留作历史对照；当前推荐使用上面的 modern mixed workload 输出。

# Jetson Nano 端侧内存受限最小 Case

这个 case 用一个很小的例子从头到尾说明 RTInfer 为什么比 Pantheon/SOTA 好。它不是追求复杂，而是让图、时间轴、变体选择和 online 过程都能一眼看清。

## 设备与预算

- 模拟设备：Jetson Nano。
- 共享内存预算：4096 MiB。
- Reserve IF：512 MiB，始终保留给中间特征/接口缓存。
- RT 任务：3 个。
- BE 任务：1 个，低优先级，有 RT burst 时可被抢占。

## Offline 阶段

RTInfer 对每个任务离线生成 Variant Atlas：

- 插入 early exits。
- 对 backbone 和 exits 做结构化剪枝。
- 对每个 `(pruning ratio, exit point)` 变体做 profiling，记录 latency、memory、accuracy。
- 把同一模型的变体切成 Delta-Graph chunks，记录每次切换时需要 H2D 加载的 missing chunks。

本 case 里最终关键变体如下：

| Task | Pantheon 选择 | RTInfer 选择 |
| --- | --- | --- |
| Task I / Traffic detection | `E2`, pruning `0`, latency `105ms`, memory `1200MiB`, acc `0.86` | `P25-E3`, pruning `0.25`, latency `135ms`, memory `1150MiB`, acc `0.91` |
| Task II / Sign classification | `E1`, pruning `0`, latency `52ms`, memory `700MiB`, acc `0.80` | `P25-E3`, pruning `0.25`, latency `95ms`, memory `760MiB`, acc `0.94` |
| Task III / Scene recognition | `E1`, pruning `0`, latency `54ms`, memory `850MiB`, acc `0.58` | `P25-E2`, pruning `0.25`, latency `120ms`, memory `900MiB`, acc `0.86` |

## SOTA / Pantheon Online 过程

Pantheon 的核心限制是 RT 串行队列。即使有 early exit，后到的任务也要等待前面的 RT task 结束。

| Time | Online action |
| --- | --- |
| `t=0ms` | Task I 到达，开始运行，选择 `E2`。 |
| `t=20ms` | Task II 到达，但 Task I 还在执行，只能排队。 |
| `t=40ms` | Task III 到达，也只能排队。 |
| `t=105ms` | Task I 结束。Task II 剩余 slack 只有 `105ms`，选择浅层 `E1`。 |
| `t=157ms` | Task II 结束。Task III 剩余 slack 只有 `103ms`，选择浅层 `E1`。 |
| `t=211ms` | RT 队列清空，BE task 才恢复。 |

结果：

- DMR：`0/3`。
- Deadline-weighted accuracy：`0.7467`。
- RT makespan：`211ms`。
- 问题：虽然没有 miss deadline，但 Task II 和 Task III 因为排队被迫使用浅早退点，精度明显下降。

## RTInfer Online 过程

RTInfer 每次 RT arrival 都重新处理 active set：

1. 按 deadline urgency 排序。
2. 从 Variant Atlas 里选择满足 slack、memory、accuracy 的候选。
3. 用 Memory-Layout-Aware Scheduler 把活跃任务放进 `Time x Address` 平面。
4. 用 Delta-Graph 只加载 missing chunks，并保证 first-use 前加载完成。
5. BE task 放在低优先级 stream，RT burst 时抢占。

| Time | Online action |
| --- | --- |
| `t=0ms` | Task I 到达，选择 `P25-E3`，accuracy `0.91`。 |
| `t=0-8ms` | Delta-Graph 只加载 Task I missing chunks：`220MiB`。 |
| `t=8-143ms` | Task I 开始执行，占用 address `[512, 1662)`。 |
| `t=20ms` | Task II 到达，scheduler 重新布局 active set，选择 `P25-E3`，accuracy `0.94`。 |
| `t=20-26ms` | 加载 Task II missing chunks：`140MiB`。 |
| `t=26-121ms` | Task II 执行，占用 address `[1662, 2422)`。 |
| `t=40ms` | Task III 到达。Full unpruned 三任务并发会超过 `4096MiB`，因此选择 `P25-E2`。 |
| `t=40-48ms` | 加载 Task III missing chunks：`180MiB`。 |
| `t=48-121ms` | Task I、Task II、Task III 三个 RT 任务同时并发执行。 |
| `t=168ms` | 最后一个 RT task 完成，BE task 恢复。 |

结果：

- DMR：`0/3`。
- Deadline-weighted accuracy：`0.9033`。
- RT makespan：`168ms`。
- 关键优势：同样满足 deadline，但 RTInfer 因为并发执行，保住了更深 exit 的精度。

## 三个技术点在图里的体现

- Accuracy-Calibrated Variant Co-Optimization：图中 RTInfer 不是选择普通浅早退，而是选择 `P25-E3`、`P25-E2` 这类“剪枝 + 更深早退点”的高精度轻量变体。
- Memory-Layout-Aware Scheduler：图中 `t=48-121ms` 同时有三个 RT rectangles，且 address 互不重叠，总和低于 4096 MiB budget。
- Delta-Graph Load-Aware Pipeline：图中顶部的 `Delta load I/II/III` 小块表示只加载 missing chunks，不做 full model reload，并且都在 first-use 前完成。

## 输出文件

- `outputs/jetson_nano_case.svg`：图。
- `outputs/variant_table.csv`：所有任务变体。
- `outputs/pantheon_trace.csv`：Pantheon trace。
- `outputs/rtinfer_trace.csv`：RTInfer trace。
- `outputs/online_decisions.md`：online 决策过程。
- `outputs/summary.txt`：最终指标。
