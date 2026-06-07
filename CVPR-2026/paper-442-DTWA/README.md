# VRGA

## [*Deeper Thought, Weaker Aim: Understanding and Mitigating Perceptual Impairment during Reasoning in Multimodal Large Language Models*](https://arxiv.org/pdf/2603.14184)


🎉 **Accepted to CVPR 2026**

---

# Overview

VRGA investigates an important phenomenon in Multimodal Large Language Models (MLLMs):

> **deeper reasoning can sometimes impair visual perception performance.**

We analyze how extended reasoning affects visual grounding and attention allocation in models such as Qwen2.5-VL and Qwen3-VL, and propose an attention intervention method to mitigate perceptual degradation during reasoning.

This repository provides:

* Evaluation code for Qwen2.5-VL / Qwen3-VL
* Attention intervention implementation
* Dataset interface for custom benchmarks
* Reproducible evaluation pipeline

---

# Installation

## Environment

Recommended environment:

* Python >= 3.10
* CUDA >= 12.1
* PyTorch >= 2.4

Install the required Transformers version:

```bash
pip install -U transformers==4.52.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

# Important Modification

Replace the original Qwen modeling file with the customized implementation provided in this repository:

```text
models/modeling_qwen2_5_vl.py
```

Specifically, overwrite the original file inside the installed Transformers package.

---

# Model Preparation

Download the desired Qwen model locally, for example:

```text
models/Qwen2.5-VL-3B-Instruct/
```

If your model path differs, modify the following line in `eval_qwen.py`:

```python
model_id = f"models/{model_name}"
```


# Dataset Preparation

Datasets should be placed in the corresponding paths expected by `EvalDataset`.

Alternatively, you can modify the dataset paths directly inside the `EvalDataset` class.

---

# Dataset Format

Each dataset loader should return:

```python
list[dict]
```

where every sample contains the following fields:

| Field      | Type                                   | Description                 |
| ---------- | -------------------------------------- | --------------------------- |
| `id`       | `int` / `str`                          | Unique sample identifier    |
| `image`    | `str` / `bytes` / `dict` / `PIL.Image` | Input image                 |
| `question` | `str`                                  | Question text               |
| `answer`   | `str`                                  | Ground-truth answer         |
| `type`     | `str`                                  | *(Optional)* category label |

Example:

```python
{
    "id": 0,
    "image": "/data/images/sample.jpg",
    "question": "Is there a cat in the image?",
    "answer": "Yes",
    "type": "existence"
}
```

---

# Adding a New Dataset

Add a new method inside the `EvalDataset` class:

```python
class EvalDataset:
    def _load_my_dataset(self):
        data_path = "path/to/your/data.json"
        raw = json.load(open(data_path))

        data = []
        for i, item in enumerate(raw):
            data.append({
                "id": i,
                "image": item["image_path"],
                "question": item["query"],
                "answer": item["ground_truth"],
                "type": item.get("category", ""),
            })

        return data
```

Then register it inside the `mapping` dictionary in `get_data()`:

```python
def get_data(self):
    mapping = {
        # existing datasets...
        "my_dataset": self._load_my_dataset,
    }
```

---

# Evaluation

## Supported Reasoning Modes

The evaluation script supports two reasoning modes:

| Mode         | Description                                           |
| ------------ | ----------------------------------------------------- |
| `base`       | Original model inference                              |
| `modify_att` | Attention intervention during reasoning (VRGA method) |

---

# Usage

## Basic Command

```bash
python eval_qwen.py \
    --model_name Qwen2.5-VL-3B-Instruct \
    --device 0 \
    --data_name POPE \
    --modify modify_att \
    --max_new_tokens 2000
```

---

# Arguments

| Argument           | Type  | Default                  | Description                           |
| ------------------ | ----- | ------------------------ | ------------------------------------- |
| `--model_name`     | `str` | `Qwen2.5-VL-3B-Instruct` | Model name                            |
| `--device`         | `int` | `0`                      | GPU index (`cuda:0`)                  |
| `--data_name`      | `str` | `POPE`                   | Dataset name                          |
| `--modify`         | `str` | `modify_att`             | Enable intervention with `modify_att` |
| `--max_new_tokens` | `int` | `2000`                   | Maximum generation length             |

---

# Output Format

Results are saved to:

```text
results/{data_name}_{model_name}_{modify}_maxNew{max_new_tokens}.jsonl
```

Each line contains one JSON record:

```json
{
    "id": 0,
    "question": "Is there a cat in the image?",
    "ori_response": ["The answer is No.\n<answer>No</answer>"],
    "think_response": ["Let's think step by step. ... \n<answer>No</answer>"],
    "answer": "Yes",
    "type": "existence"
}
```

---


# Evaluate with Attention Intervention

```bash
python eval_qwen.py \
    --data_name POPE \
    --modify modify_att
```


# Acknowledgements

This project is built upon:

* Qwen2.5-VL
* Qwen3-VL
* HuggingFace Transformers

We thank the open-source community for their valuable contributions.
