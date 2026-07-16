# README

This repository contains **Python-only** scripts to:
1) generate zero-width (ZW) watermark datasets,  
2) fine-tune a base LLM with **LoRA** (single GPU, reproducible), and  
3) probe watermark detectability by batched generation.

---

## Files

- `watermark.py` — dataset generation (ZW watermark insertion + deterministic user/text assignment)  
- `alphabet.txt` — zero-width alphabet (one `U+...` code point per line) used by both generation and probing  
- `train_fast_full.py` — offline LoRA fine-tuning (local model path only)  
- `probe_watermarks_batch_fast.py` — batched probing (generation + ZW extraction/matching)
- `result-analysis` folder — visualization scripts for experiment 1-3 **(fig.2, 3, 4, and 6 in the paper)**

---

## 1) Dataset generation — `watermark.py`

### What it does
- Loads one of the supported sources: `blog1k`, `poems`, `cnn_news`.
- Selects a training set of size `--train-size-total`.
- For each user `user_0001`, `user_0002`, … it generates **two ZW halves**:
  - `wm_first` (inserted into the first half of the document)
  - `wm_second` (inserted into the second half of the document)
- Inserts ZW symbols into text using cluster insertion with spacing, with **inclusion-preserving** allocation:
  - increasing `--num-users` or `--wm-per-user` preserves earlier assignments (prefix property),
  - watermark halves are enforced to be **globally unique across users**.
- For `cnn_news`, additionally builds per-chunk prompts (`cnn_news_prompts.jsonl`) matching the probe logic.

### Outputs
For a given `--output-dir-root <ROOT>` and `--version <VERSION>`, the script writes to:
```
<ROOT>/<VERSION>/
  ├── train.jsonl
  ├── cnn_news_prompts.jsonl        (only when --source cnn_news)
  ├── users/
  │   └── user_0001_watermarks.jsonl ...
  └── users_manifest.json
```
**Note:** the script defines `test.jsonl` internally, but the current version **does not write** it.

### CLI arguments (exact)
- `--source {blog1k,Gutenberg,poems,cnn_news}` (default: `blog1k`)
- `--train-size-total INT` (default: `1000`)
- `--num-users INT` (default: `1`)
- `--wm-per-user INT` (default: `50`)
- `--test-wm-ratio FLOAT` (default: `0.5`) *(currently unused for output)*
- `--seed INT` (default: `42`)
- `--wm-offset INT` (default: `0`)  
  Global offset applied to per-user watermark RNG seeds.  
  Changes watermark strings **without changing which texts are watermarked**.
- `--version STR` (default: `xp2-2K`)
- `--blog1k-csv PATH` (default: `dataset/raw/blog1000.csv` relative to repo)
- `--gutenberg-csv PATH` (default: `dataset/raw/Gutenberg.csv`)
- `--poems-csv PATH` (default: `dataset/raw/PoetryFoundationData.csv`)
- `--cnn-news-csv PATH` (default: `dataset/raw/cnn_dm_3.0.0_train_article_8k_10k_chars.csv`)
- `--output-dir-root PATH` (default: `./`)
- `--max-wm-per-user INT` (default: `200`)  
  Reserved capacity per user to guarantee inclusion.
- `--max-tokens INT` (default: `3800`)  
  Filters overly long texts by token count **only if > 0**.

### Important note about `--max-tokens`
If `--max-tokens > 0`, the script currently loads a tokenizer from `"meta-llama/Llama-2-7b-hf"` for token counting.
This may require internet access unless that tokenizer is cached locally. If you want fully offline generation, set:
- `--max-tokens 0` (disables token filtering)

### Important note about `--wm-offset`

The `--wm-offset` argument controls the **per-user watermark generation only**.

Concretely, it adds a global integer offset to the RNG seed used to generate
each user's `(wm_first, wm_second)` pair.

This allows changing watermark content **without changing dataset sampling** for each experiment.

Specifically:

**What changes**
- The actual zero-width watermark strings inserted into text.
- Each user receives a different watermark pattern for different offsets.

**What does NOT change**
- Which texts are selected for training.
- Which texts are assigned to each user.
- The total number of watermarked / non-watermarked samples.

In other words, `--wm-offset` guarantees:
> *the set of watermarked documents is identical, but the embedded watermarks differ.*

### Example

```bash
# Same watermarked rows, but different watermark strings
python3 watermark.py \
  --source cnn_news \
  --train-size-total 1000 \
  --num-users 1 \
  --wm-per-user 100 \
  --seed 123 \
  --wm-offset 1 \
  --version XP1-T1000_U1_P100_seed123_offset1 \
  --output-dir-root ./data \
  --max-wm-per-user 200 \
  --max-tokens 0
```


### Example (recommended, offline-friendly)
Generate a dataset version under `./data/<VERSION>/...`:
```bash
python3 watermark.py   --source cnn_news   --train-size-total 1000   --num-users 1   --wm-per-user 100   --seed 123   --version XP1-T1000_U1_P100_seed123   --output-dir-root ./data   --max-wm-per-user 200   --max-tokens 0
```

### Supported datasets & expected input format

`watermark.py` supports multiple text sources.  
Each source expects a **CSV file with a specific schema**, which is enforced by the code.

#### blog1k
- **CLI**: `--source blog1k`
- **Default path**: `dataset/raw/blog1000.csv`
- **Required columns**:
  - `text` — full document text
- **Filtering**:
  - documents with fewer than **200 words** are discarded
- **Notes**:
  - only the `text` column is used
  - one row corresponds to one document

#### poems (Poetry Foundation)
- **CLI**: `--source poems`
- **Default path**: `dataset/raw/PoetryFoundationData.csv`
- **Required columns**:
  - `Title`
  - `Poem`
  - `Poet`
  - `Tags`
- **Text construction**:
  Each row is converted into a single text field as:
  - `Title` `Poem` `Poet` `Tags`
- **Filtering**:
- documents with fewer than **200 words** are discarded
- **Notes**:
- metadata is intentionally embedded to mimic natural documents

#### cnn_news
- **CLI**: `--source cnn_news`
- **Default path**:  
`dataset/raw/cnn_dm_3.0.0_train_article_8k_10k_chars.csv`
- **Required columns**:
- `article` — full news article text
- **Filtering**:
- documents with fewer than **200 words** are discarded
- **Special handling**:
- each article is split into **400-word chunks**
- **each chunk is watermarked independently**
- chunk-level prompts are generated for probing
- **Additional output**:
- `cnn_news_prompts.jsonl`, with entries of the form:
  ```json
  {
    "row_id": <int>,
    "user_id": "user_0001",
    "chunk_id": <int>,
    "prompt": "<first half of the watermarked chunk>",
    "wm_first": "<ZW string>",
    "wm_second": "<ZW string>"
  }
  ```
- **Notes**:
- prompt construction exactly matches the probing logic
- enables chunk-level watermark detection on long articles

---

## 2) Training — `train_fast_full.py`

### What it does
- Loads a **locally available** base model via `--model_path`.
- Loads the dataset from `train.jsonl` for `--wm_version` under `--data_base`.
- Deterministically splits into train/eval (90/10) with `--seed`.
- Tokenizes **without truncation** and then chunks into:
  - `seq_len = 4096`
  - **non-overlapping** chunks (`stride = 0`)
- Fine-tunes with LoRA on target modules: `q_proj,k_proj,v_proj,o_proj`.
- Saves:
  - `final_model/` (Trainer save)
  - `final_adapter/` (LoRA adapter)
  - `adapter_meta.json` (anonymized metadata)

### Expected dataset location
The script searches (in order):
1) `<data_base>/<wm_version>/train.jsonl`
2) `<data_base>/data/processed/<wm_version>/train.jsonl`

If you use the recommended generator command above (`--output-dir-root ./data`), then (1) works.

### Outputs
```
<data_base>/outputs/<model_tag>/
  lora-single-<wm_version>-seed<seed>/
    ├── final_model/
    ├── final_adapter/
    │   └── adapter_meta.json
    ├── train_raw.jsonl
    └── eval_raw.jsonl
```

### CLI arguments (exact)
Required:
- `--model_path PATH`  
  Local path to a downloaded base model folder.

Optional (common):
- `--data_base PATH` (default: `./data`)
- `--wm_version STR` (default: `xp2-2K-seed42`)
- `--model_tag STR` (default: `model_A`) — neutral output tag (recommended for anonymization)
- `--seed INT` (default: `42`)
- `--deterministic` — enables strict deterministic algorithms (may raise if an op is non-deterministic)
- `--use_fa2` — enables FlashAttention2 (`attn_implementation="flash_attention_2"`), may reduce determinism

LoRA / optimization:
- `--lora_r INT` (default: `12`)
- `--lora_alpha INT` (default: `32`)
- `--lora_dropout FLOAT` (default: `0.05`)
- `--lr FLOAT` (default: `2e-4`)
- `--warmup_ratio FLOAT` (default: `0.03`)

### Example
```bash
python3 train_fast_full.py   --model_path ./models/base_model   --data_base ./data   --wm_version XP1-T1000_U1_P100_seed123   --model_tag model_A   --seed 42   --deterministic
```

---

## 3) Probing — `probe_watermarks_batch_fast.py`

### What it does
- Loads a LoRA adapter (merged or via AutoPEFT).
- Builds prompts and generates continuations in batches.
- Extracts zero-width symbols using `alphabet.txt` and checks **substring matches**:
  - `wm_first` must appear in ZW extracted from the **prompt**
  - `wm_second` must appear in ZW extracted from the **generated continuation**
- Runs multiple probe seeds: `[11, 22, 33]`
- Saves detailed JSON + summary JSON/CSV.

### Which prompts are used?
- If `<data_base>/<wm_version>/cnn_news_prompts.jsonl` exists, the probe uses it (CNN chunk prompts).
- Otherwise it loads `<data_base>/<wm_version>/train.jsonl` and builds prompts as:
  - `prompt = first_half_of(watermarked_text)` with `split_idx = len(words)//2 + 4`

### Outputs
The probe writes under:
```
<data_base>/probe_outputs/<MODEL_TAG>/<adapter_relative_path>/
  ├── watermark_probe_<wm_version>.json
  ├── summary_<wm_version>.json
  └── summary_<wm_version>.csv
```

### CLI arguments (exact)
Required:
- `--data_base PATH`
- `--wm_version STR`
- `--model_id STR`  
  Used to derive `MODEL_TAG` (by replacing `/` with `_`), and as fallback base model identifier.

Optional:
- `--adapter_path PATH` (default: `None`)  
  **Strongly recommended** to provide explicitly when using `train_fast_full.py` outputs (see below).
- `--skip_if_exists {0,1}` (default: `1`)
- `--batch-size INT` (default: `4`)
- `--max-length INT` (default: `4096`) tokenizer truncation limit for prompt encoding
- `--compile` enables `torch.compile`

### IMPORTANT: adapter path and model_id with this repository
`train_fast_full.py` saves adapters under:
`<data_base>/outputs/<model_tag>/lora-single-<wm_version>-seed<seed>/final_adapter`

However, the probe script's *auto-discovery* logic primarily checks `outputs_full/...` patterns.
So, when reproducing results from this artifact, you should pass `--adapter_path` explicitly.

Also, if your adapter config contains a local `base_model_name_or_path`, the probe will load the tokenizer from it.
Otherwise, set `--model_id` to a valid local or hub identifier that can load a tokenizer.

### Example (recommended: explicit adapter path)
```bash
python3 probe_watermarks_batch_fast.py   --data_base ./data   --wm_version XP1-T1000_U1_P100_seed123   --model_id ./models/base_model   --adapter_path ./data/outputs/model_A/lora-single-XP1-T1000_U1_P100_seed123-seed42/final_adapter   --batch-size 4   --skip_if_exists 0
```

### Hugging Face token (optional)

If you need to download a model or tokenizer from the Hugging Face Hub, set the token in your shell:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
---

## End-to-end minimal reproduction (suggested)

```bash
# 1) Generate data
python3 watermark.py   --source cnn_news   --train-size-total 1000   --num-users 1   --wm-per-user 100   --seed 123   --version XP1-T1000_U1_P100_seed123   --output-dir-root ./data   --max-wm-per-user 200   --max-tokens 0

# 2) Train (offline)
python3 train_fast_full.py   --model_path ./models/base_model   --data_base ./data   --wm_version XP1-T1000_U1_P100_seed123   --model_tag model_A   --seed 42   --deterministic  --use_fa2

# 3) Probe
python3 probe_watermarks_batch_fast.py   --data_base ./data   --wm_version XP1-T1000_U1_P100_seed123   --model_id ./models/base_model   --adapter_path ./data/outputs/model_A/lora-single-XP1-T1000_U1_P100_seed123-seed42/final_adapter   --batch-size 4   --skip_if_exists 0
```

---

## Notes for Reviewers

- The artifact is anonymized and avoids hard-coded usernames / machine paths.
- Training is offline-friendly **if the base model is available locally**.
- For fully offline dataset generation, use `--max-tokens 0` to disable tokenizer-based filtering.
---

## License

For research artifact evaluation only. Replace with your preferred license if needed.
