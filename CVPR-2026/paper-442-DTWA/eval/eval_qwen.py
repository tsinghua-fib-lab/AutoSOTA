import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Vision-Language Model Evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2.5-VL-3B-Instruct",
        help="Model name (e.g., Qwen2.5-VL-3B-Instruct, Qwen3-VL-30B-A3B-Instruct)",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device index (default: 0)"
    )
    parser.add_argument(
        "--modify", type=str, default="modify_att", help="Modification type (modify_att or base)"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="POPE",
        help="Dataset name (HallusionBench, POPE, mmstar, etc.)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2000, help="Maximum new tokens to generate"
    )
    return parser.parse_args()


def load_model(model_name, device):
    """加载模型和处理器"""
    os.environ["HF_ENDPOINT"] = "http://mirrors.tools.huawei.com/huggingface"
    model_id = f"/data/guiyang/xywu/ivine/models/{model_name}"

    if "Qwen3-VL" in model_name:
        from transformers import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval().to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


class EvalDataset:
    """统一的数据集加载类"""

    def __init__(self, data_name):
        self.data_name = data_name

    def _load_parquet(self, path):
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _load_mmhal(self):
        path = "/data/guiyang/xywu/ivine/datasets/MMHal-Bench/data/train-00000-of-00001.parquet"
        records = self._load_parquet(path)
        data = []
        for i, e in enumerate(records):
            e["id"] = i
            e["type"] = e["question_type"]
            e["answer"] = e["gt_answer"]
            data.append(e)
        return data

    def _load_hallusionbench(self):
        path = "/data/guiyang/xywu/ivine/datasets/HallusionBench/data/image-00000-of-00001_with_index.parquet"
        records = self._load_parquet(path)
        data = []
        for e in records:
            e["id"] = e.pop("index")
            e["answer"] = e.pop("gt_answer_details")
            e["type"] = f"{e['category']}_{e['subcategory']}"
            data.append(e)
        return data

    def _load_rh_halu(self):
        image_root = "/data/guiyang/xywu/ivine/datasets/RH-Bench"
        ds = self._load_json(os.path.join(image_root, "halu_data.json"))
        data = []
        for e in ds:
            e["type"] = e.pop("question_type")
            e["image"] = os.path.join(image_root, e["image"])
            data.append(e)
        return data

    def _load_rh_reason(self):
        image_root = "/data/guiyang/xywu/ivine/datasets/RH-Bench"
        ds = self._load_json(os.path.join(image_root, "reason_data.json"))
        data = []
        for e in ds:
            e["type"] = e.pop("question_type")
            e["image"] = os.path.join(image_root, e["image"])
            data.append(e)
        return data

    def _load_mmbench(self):
        path = "/data/guiyang/xywu/ivine/datasets/MMBench_EN/data/dev-00000-of-00001-75b6649fb044d38b.parquet"
        records = self._load_parquet(path)
        data = []
        for e in records:
            e["id"] = e.pop("index")
            hint = e["hint"]
            query = e["question"]
            e["question"] = f"Hint:{hint}Question{query}\nOptions:A.{e['A']}\nB.{e['B']}\nC.{e['C']}\nD.{e['D']}"
            e["type"] = e.pop("category")
            data.append(e)
        return data

    def _load_haloquest(self):
        data_path = "/data/guiyang/xywu/ivine/datasets/haloquest/haloquest-eval.json"
        image_dir = "/data/guiyang/xywu/ivine/datasets/haloquest/images"
        ds = self._load_json(data_path)
        data = []
        for e in ds:
            e["image"] = os.path.join(image_dir, e["image_name"])
            if not os.path.exists(e["image"]):
                continue
            e["answer"] = e["groundtruth responses"]
            e["type"] = e["hallucination type"]
            data.append(e)
        return data

    def _load_mmstar(self):
        path = "/data/guiyang/xywu/ivine/datasets/MMStar/mmstar.parquet"
        records = self._load_parquet(path)
        data = []
        for e in records:
            e["id"] = e.pop("index")
            e["type"] = f"{e['category']}_{e['l2_category']}"
            data.append(e)
        return data

    def _load_mathvista_mini(self):
        path = "/data/guiyang/xywu/ivine/datasets/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
        records = self._load_parquet(path)
        data = []
        for e in records:
            e["id"] = e.pop("pid")
            e["image"] = e["decoded_image"]
            e["question"] = e["query"]
            e["type"] = e["metadata"]["task"]
            data.append(e)
        return data

    def _load_pope(self):
        folder_path = "/data/guiyang/xywu/ivine/datasets/POPE"
        parquet_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".parquet")
        ]
        df_list = [pd.read_parquet(f) for f in parquet_files]
        combined_df = pd.concat(df_list, ignore_index=True)
        records = combined_df.to_dict(orient="records")
        data = []
        for e in records:
            e["type"] = e["category"]
            data.append(e)
        return data

    def _load_amber(self):
        folder_path = "/data/guiyang/xywu/ivine/datasets/amber/discriminative"
        parquet_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".parquet")
        ]
        df_list = [pd.read_parquet(f) for f in parquet_files]
        combined_df = pd.concat(df_list, ignore_index=True)
        records = combined_df.to_dict(orient="records")
        data = []
        for e in records:
            e["question"] = e["query"]
            e["answer"] = e["truth"]
            data.append(e)
        return data

    def get_data(self):
        mapping = {
            "HallusionBench": self._load_hallusionbench,
            "RH_halu": self._load_rh_halu,
            "RH_reason": self._load_rh_reason,
            "mmbench": self._load_mmbench,
            "haloquest": self._load_haloquest,
            "mmstar": self._load_mmstar,
            "MathVistaMini": self._load_mathvista_mini,
            "POPE": self._load_pope,
            "amber": self._load_amber,
            "mmhal": self._load_mmhal,
        }
        if self.data_name not in mapping:
            raise ValueError(f"Unknown dataset: {self.data_name}")
        return mapping[self.data_name]()


def load_image(image_input):
    """统一加载图像为 PIL.Image"""
    if isinstance(image_input, dict) and "bytes" in image_input:
        return Image.open(BytesIO(image_input["bytes"])).convert("RGB")
    elif isinstance(image_input, bytes):
        return Image.open(BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    else:
        return image_input


def generate_inputs(processor, model, image_path, question, mode="ori", modify=""):
    """生成模型输入"""
    instruct_map = {
        "ori": "The final answer MUST BE in <answer> </answer> tags.",
        "think": "Let's think step by step.",
    }
    instruct = instruct_map.get(mode, instruct_map["ori"])

    content = []
    if image_path is not None:
        raw_image = load_image(image_path)
        content.append({"type": "image", "image": raw_image, "max_pixels": 1024 * 28 * 28 * 2})
    content.append({"type": "text", "text": f"{question}\n{instruct}"})

    messages_query = [{"role": "user", "content": content}]
    image_inputs, _ = process_vision_info(messages_query) if image_path is not None else (None, None)

    text_query = processor.apply_chat_template(
        messages_query, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text_query],
        images=image_inputs if image_path is not None else None,
        padding=True,
        return_tensors="pt",
    )

    if image_path is not None:
        input_ids = inputs["input_ids"][0]
        vision_start = (input_ids == processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")).nonzero(as_tuple=True)[0].item()
        vision_end = (input_ids == processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")).nonzero(as_tuple=True)[0].item()
        q_ids = processor.tokenizer(question, add_special_tokens=False).input_ids
        q_end_pos = vision_end + len(q_ids)

        if modify:
            inputs[modify] = True
            inputs["q_end_pos"] = q_end_pos
            inputs["vision_start"] = vision_start
            inputs["vision_end"] = vision_end
            inputs["grid_w"] = inputs["image_grid_thw"][0][2].item() // 2

    inputs = inputs.to(model.device)
    return inputs


def make_jsonable(obj):
    """递归地将 numpy/torch 类型转换为纯 Python 类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    return obj


def get_output_text(processor, output, inputs):
    """从模型输出中解码文本"""
    generated_ids = output["sequences"]
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def evaluate(model, processor, data, data_name, model_name, modify="modify_att", max_new_tokens=2000, with_cot=False):
    """评估函数"""
    tag = f"{data_name}_{model_name}_{modify}_maxNew{max_new_tokens}"
    print(f"Evaluation tag: {tag}")

    save_root = "/data/guiyang/xywu/ivine/rebuttal"
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, f"{tag}.jsonl")

    have_judged = []
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            for line in f:
                e = json.loads(line)
                have_judged.append(e["id"])

    total = len(data)
    with tqdm(total=total, desc=f"Evaluating {data_name}") as pbar:
        i = 0
        while i < total:
            sample = data[i]

            if sample.get("image") is None:
                i += 1
                pbar.update(1)
                continue

            dataset_id = sample["id"]
            if dataset_id in have_judged:
                i += 1
                pbar.update(1)
                continue

            question = sample["question"]
            gt_answer = sample["answer"]

            with torch.no_grad():
                ori_inputs = generate_inputs(processor, model, sample["image"], question, mode="ori", modify=modify)
                ori_output = model.generate(
                    **ori_inputs,
                    max_new_tokens=max_new_tokens,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict_in_generate=True,
                )
                ori_output_text = get_output_text(processor, ori_output, ori_inputs)

                think_output_text = None
                if with_cot:
                    think_inputs = generate_inputs(processor, model, sample["image"], question, mode="think", modify=modify)
                    think_output = model.generate(
                        **think_inputs,
                        max_new_tokens=max_new_tokens,
                        output_attentions=True,
                        output_hidden_states=False,
                        return_dict_in_generate=True,
                    )
                    think_output_text = get_output_text(processor, think_output, think_inputs)

            result = {
                "id": dataset_id,
                "question": question,
                "ori_response": ori_output_text,
                "answer": gt_answer,
                "type": sample.get("type", ""),
            }
            if with_cot and think_output_text is not None:
                result["think_response"] = think_output_text

            result_safe = make_jsonable(result)
            with open(save_path, "a") as f:
                json.dump(result_safe, f)
                f.write("\n")

            i += 1
            pbar.update(1)


if __name__ == "__main__":
    args = parse_args()

    device = f"cuda:{args.device}"
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_name}")
    print(f"Modify: {args.modify}")

    model, processor = load_model(args.model_name, device)

    eval_dataset = EvalDataset(args.data_name)
    data = eval_dataset.get_data()

    evaluate(
        model=model,
        processor=processor,
        data=data,
        data_name=args.data_name,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        with_cot=False,
    )


    # python eval_qwen.py --model_name Qwen2.5-VL-3B-Instruct --device 0 --data_name POPE --modify modify_att --max_new_tokens 2000
