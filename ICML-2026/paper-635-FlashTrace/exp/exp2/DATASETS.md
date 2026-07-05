# exp/exp2 数据集与样本流说明

本文件说明 Experiment 2 中支持的数据集、样本结构，以及在「采样阶段」与「归因阶段」的处理方式。

## 支持的数据集
- `morehopqa`（`data/with_human_verification.json`）
- RULER 系列 JSONL：`hotpotqa_long`、`niah_*`、`vt_*`（自动在 `data/ruler_multihop/<len>/.../validation.jsonl` 搜索），或直接传入任意 RULER JSONL 路径
- 其余数据集（如 math）被显式跳过
- 归因阶段同样优先使用缓存文件 `exp/exp2/data/<name>.jsonl`，否则按上述规则解析；传入存在的 JSONL 路径也会按 RULER 结构加载

### 共同的样本字段定义
```json
{
  "prompt": "<上下文+问题>",
  "target": "<答案或生成>",
  "indices_to_explain": [start_tok, end_tok] | null, // token-level：需要解释的 generation token span（闭区间）
  "attr_mask_indices": [...],       // legacy：覆盖率金标句子索引（当前 exp2 不再使用），可能为 null
  "sink_span": [start, end] | null, // 生成 token 中的答案片段
 "thinking_span": [start, end] | null, // 生成 token 中的 CoT 片段
  "metadata": { ... }               // 数据集特定元信息
}
```
- **`CachedExample`**：`dataset_utils.py` 统一的内存态结构，字段与上述 JSON 完全一致，用于采样阶段（加载原始数据）与归因阶段（加载缓存或原始）。
- **缓存行（JSONL）**：`sample_and_filter.py` 写入的每行 JSON，与 `CachedExample` 字段一一对应。
- **采样阶段处理流（通用）**：
  1. 加载原始数据集样本（`prompt`/`indices_to_explain` 等保持一致）。
  2. 按模板调用生成模型，要求「思考文本 + 末尾 \\box{} 答案」。
  3. 若生成不符合「思考 + 单个 \\box{} 且无尾巴」的格式，直接丢弃该样本。
  4. 提取思考片段与 `\\box{}` 内文本，仅用 `\\box{}` 内文调用判定模型。
  5. 判定为 True 时，重新拼接「思考片段 + 去除 box 包裹的答案文本」作为 `target`，并据此记录 `sink_span`/`thinking_span`。
  6. 写入缓存：只保留 `reference_answer`、`judge_response`（可选 `boxed_answer`），不再存储 `candidate_answer`。

### 生成切分与 span 解析
- `split_boxed_generation`（`dataset_utils.py`）校验格式：必须是「非空思考文本 + 单个末尾 \\box{}」且箱体之后无其他字符，否则直接跳过。
- `target` 由「思考片段 + 换行 + 最终答案文本（无 box）」重组。
- `attach_spans_from_answer` 使用 tokenizer 的 offset mapping 将最终答案在 `target` 中的字符区间映射到 token 级索引，得到 `sink_span`；`thinking_span` 取从开头到 `sink_span` 前一 token 的闭区间。两者均为 token 级 span，满足后续多跳 IFR 的调用约定。
- `indices_to_explain` 在采样写缓存时统一设置为 `sink_span`（boxed 内文在 `target` 中对应的 generation token span）。

---

## MoreHopQA
- **原始样本结构（`MoreHopQAAttributionDataset` → `CachedExample`）**
  ```json
  {
    "prompt": "<context 拼接>\\n<question>",
    "target": null,
    "indices_to_explain": null,
    "attr_mask_indices": null,
    "sink_span": null,
    "thinking_span": null,
    "metadata": {
      "answer": "<gold answer>",
      "_id": "<example id>",
      "original_context": <原始上下文结构>
    }
  }
  ```
  - 加载时机：`DatasetLoader.load_raw("morehopqa")` 在采样阶段、归因阶段（无缓存时）都会产出 `CachedExample`。
  - 说明：exp2 的 token-level row/rec 需要 `target` + 可定位的答案 token span；建议先跑 `sample_and_filter.py` 产出缓存后再做归因评估。

- **采样阶段（生成 & 过滤后写缓存）**
  ```json
  {
    "prompt": "<同上>",
    "target": "<生成的 CoT + 最终答案文本（已去掉 box 包裹）>",
    "indices_to_explain": [start_tok, end_tok],
    "attr_mask_indices": null,
    "sink_span": [start_tok, end_tok] | null,
    "thinking_span": [start_tok, end_tok] | null,
    "metadata": {
      "answer": "<gold answer>",
      "_id": "<example id>",
      "original_context": <原始上下文结构>,
      "reference_answer": "<gold answer>",
      "judge_response": "<True/False 文本>",
      "boxed_answer": "<可选，boxed 解析结果>"
    }
  }
  ```
  - `sink_span`/`thinking_span`：仅在成功解析 `\\box{}` 时填充；`target` 为「思考 + 最终答案文本」的裁剪版。
  - 写入：`exp/exp2/data/morehopqa.jsonl`。

- **归因阶段（加载缓存优先）**
  - 加载：`run_exp.py` 优先 `load_cached`（JSONL → `CachedExample`），否则回退原始结构并在线生成 `target`。
  - 使用：忠实度（token-level RISE/MAS）直接用缓存的 `target`；`ifr_multi_hop` 在有 `sink_span`/`thinking_span` 时限定答案/CoT，否则视整个生成为 sink。

---

## RULER 热点问答（`hotpotqa_long`）
- **原始样本结构（`RulerAttributionDataset` → `CachedExample`）**
  ```json
  {
    "prompt": "<input> + <answer_prefix>",
    "target": "<answer_prefix + sep + ', '.join(outputs)>",
    "indices_to_explain": [0],
    "attr_mask_indices": [<句子索引>...] | null,
    "sink_span": null,
    "thinking_span": null,
    "metadata": {
      "dataset": "ruler",
      "length": <int>,
      "length_w_model_temp": <any>,
      "outputs": [...],
      "answer_prefix": "<str>",
      "token_position_answer": <any>,
      "needle_spans": [
        {
          "title": "<str>",
          "doc_index": <int>,
          "document_number": <int>,
          "sentence_index": <int>,
          "sentence": "<str>",
          "context_span": [start, end],
          "span": [start, end],
          "snippet": "<str>"
        },
        ...
      ],
      "prompt_sentence_count": <int>,
      "reference_answer": "<在 loader 中补充，来自 outputs 或 target>"
    }
  }
  ```
  - 加载时机：`DatasetLoader.load_raw("hotpotqa_long")` 在采样阶段、归因阶段（无缓存时）都会产出 `CachedExample`。

- **采样阶段（生成 & 过滤后写缓存）**
  ```json
  {
    "prompt": "<同上>",
    "target": "<生成的 CoT + 最终答案文本（已去掉 box 包裹）>",
    "indices_to_explain": [-2],
    "attr_mask_indices": [<句子索引>...] | null,
    "sink_span": [start_tok, end_tok] | null,
    "thinking_span": [start_tok, end_tok] | null,
    "metadata": {
      "dataset": "ruler",
      "length": <int>,
      "length_w_model_temp": <any>,
      "outputs": [...],
      "answer_prefix": "<str>",
      "token_position_answer": <any>,
      "needle_spans": [...],
      "prompt_sentence_count": <int>,
      "reference_answer": "<outputs 拼接或 target>",
      "judge_response": "<True/False 文本>",
      "boxed_answer": "<可选>"
    }
  }
  ```
  - `attr_mask_indices` 保留原值；`indices_to_explain` 统一为末句 `[-2]`（最后一个非 EOS 生成句）；`sink_span`/`thinking_span` 仅在成功解析 `\\box{}` 时填充；`target` 为「思考 + 最终答案文本」的裁剪版。
  - 写入：`exp/exp2/data/hotpotqa_long.jsonl`。

- **归因阶段（加载缓存优先）**
  - 加载：优先 `load_cached`（JSONL → `CachedExample`），否则回退原始解析。
  - 使用：覆盖率使用 `attr_mask_indices`；忠实度与 `ifr_multi_hop` 利用缓存的 `sink_span`/`thinking_span` 定位答案/CoT，若缺失则视整个生成为 sink。

---

## RULER NIAH / Variable Tracking（`niah_*`, `vt_*`）
- **原始样本结构（同 RULER 通用）**
  ```json
  {
    "prompt": "<input> + <answer_prefix>",
    "target": "<answer_prefix + sep + ', '.join(outputs)>",
    "indices_to_explain": [0],
    "attr_mask_indices": [<句子索引>...] | null,
    "sink_span": null,
    "thinking_span": null,
    "metadata": {
      "dataset": "ruler",
      "length": <int>,
      "length_w_model_temp": <any>,
      "outputs": [...],
      "answer_prefix": "<str>",
      "token_position_answer": <any>,
      "needle_spans": [...],
      "prompt_sentence_count": <int>,
      "reference_answer": "<在 loader 中补充>"
    }
  }
  ```
  - 加载时机：`DatasetLoader.load_raw("<niah_* 或 vt_*>")` 在采样阶段、归因阶段（无缓存时）使用。

- **采样阶段（生成 & 过滤后写缓存）**
  ```json
  {
    "prompt": "<同上>",
    "target": "<思考 + 最终答案文本（无 box），无其他尾巴>",
    "indices_to_explain": [start_tok, end_tok],
    "attr_mask_indices": [<句子索引>...] | null,
    "sink_span": [start_tok, end_tok] | null,
    "thinking_span": [start_tok, end_tok] | null,
    "metadata": {
      "dataset": "ruler",
      "length": <int>,
      "length_w_model_temp": <any>,
      "outputs": [...],
      "answer_prefix": "<str>",
      "token_position_answer": <any>,
      "needle_spans": [...],
      "prompt_sentence_count": <int>,
      "reference_answer": "<outputs 拼接或 target>",
      "judge_response": "<True/False 文本>",
      "boxed_answer": "<可选>"
    }
  }
  ```
  - 生成/判定流程与 `hotpotqa_long` 相同；`target` 是裁剪后的「思考 + 最终答案文本」。
  - 写入：`exp/exp2/data/<dataset>.jsonl`（例如 `niah_mq_q2.jsonl`, `vt_h6_c1.jsonl`）。

- **归因阶段（加载缓存优先）**
  - 与 `hotpotqa_long` 相同：优先缓存，否则原始；恢复率（`recovery_ruler`）使用 `metadata.needle_spans`（映射到 prompt tokens）；多跳 IFR 在有 `sink_span`/`thinking_span` 时作用于答案/CoT。

---

## `indices_to_explain` 约定
- token-level：`indices_to_explain = [start_tok, end_tok]`（闭区间），坐标系为 `tokenizer(target, add_special_tokens=False)` 的 generation token indices。
- exp2 推荐：`indices_to_explain == sink_span`，即 boxed 内文（最终答案）在 `target` 中对应的 token span。

---

## 自定义 RULER JSONL 路径
- 若 `--dataset` 传入存在的 JSONL 路径，`dataset_from_name` 按 RULER 文件解析，字段与流程同 RULER 系列。
- 采样、归因阶段行为与上文 RULER 描述一致，只是文件名由显式路径决定。

---

## 归因阶段加载优先级与效果
- `run_exp.py` 加载顺序：`exp/exp2/data/<name>.jsonl` 缓存 > 显式给定的 JSONL 路径 > 原始解析（MoreHopQA 或 RULER）
- 恢复率 (`mode=recovery_ruler`) 仅支持 RULER（要求 `metadata.needle_spans`），否则拒绝
- 忠实度 (`mode=faithfulness_gen`) 使用生成文本；`ifr_multi_hop` 在有 `sink_span`/`thinking_span` 时才对答案/CoT 做多跳，否则退化为整段生成
