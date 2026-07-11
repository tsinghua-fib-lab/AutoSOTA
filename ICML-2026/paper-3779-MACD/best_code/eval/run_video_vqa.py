from __future__ import annotations

import argparse
import os
import re
from typing import Any

import torch

from macd.decoding import enable_macd_decoding
from macd.io import ensure_dir, masked_video_name, read_jsonl
from macd.models import get_model_device, load_video_llm

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


def extract_option_letters(options: list[str] | None) -> list[str]:
    if not options:
        return ["A", "B", "C", "D"]
    letters: list[str] = []
    for option in options:
        match = re.match(r"\s*([A-Z])[\.\)]", str(option))
        if match and match.group(1) not in letters:
            letters.append(match.group(1))
    return letters or ["A", "B", "C", "D"]


def normalize_choice(text: str, allowed_letters: list[str]) -> str:
    allowed = set(allowed_letters)
    for char in str(text).strip().upper():
        if char in allowed:
            return char
    return str(text).strip()


def build_messages(video_path: str, question: dict[str, Any], task: str, total_pixels: int, min_pixels: int, max_frames: int) -> list[dict[str, Any]]:
    prompt = str(question.get("text", ""))
    if task == "yesno":
        system = "Answer with exactly Yes or No. Output only one word, no punctuation, no extra text."
        user_text = f"Answer with exactly Yes or No: {prompt}"
    else:
        options = question.get("options") or []
        letters = extract_option_letters(options)
        allowed = "/".join(letters)
        option_text = "\n".join(options)
        system = f"Output ONLY the option letter, one of {allowed}, with no extra text."
        user_text = f"{prompt}\n{option_text}\nAnswer with the letter of the correct option ({allowed}):"

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    "max_frames": max_frames,
                },
            ],
        },
    ]


def prepare_inputs(processor: Any, messages: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fix fps list -> scalar for huggingface_hub strict validation
    for k in list(video_kwargs.keys()):
        if isinstance(video_kwargs[k], (list, tuple)):
            video_kwargs[k] = video_kwargs[k][0] if video_kwargs[k] else 1.0
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    # Don't set cache_position manually — transformers handles it since v4.45
    return inputs


def apply_pixel_mask(inputs: dict[str, Any], mask_level: float) -> None:
    if mask_level <= 0:
        return
    alpha = max(0.0, min(1.0, float(mask_level)))
    for key in ("pixel_values_videos", "pixel_values_images", "pixel_values"):
        if key in inputs and isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key] * (1.0 - alpha)


def decode_output(processor: Any, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def build_original_path_map(orig_question_file: str | None, orig_video_dir: str) -> dict[str, str]:
    if not orig_question_file:
        return {}
    mapping: dict[str, str] = {}
    for item in read_jsonl(orig_question_file):
        video_id = str(item.get("video_id", ""))
        rel_path = item.get("image")
        if video_id and rel_path:
            mapping[video_id] = os.path.join(orig_video_dir, rel_path)
    return mapping


def resolve_original_video(question: dict[str, Any], args: argparse.Namespace, original_map: dict[str, str]) -> str:
    video_id = str(question.get("video_id", ""))
    if video_id in original_map:
        return original_map[video_id]
    return os.path.join(args.orig_video_dir, str(question.get("image", "")))


def resolve_counterfactual_video(question: dict[str, Any], args: argparse.Namespace) -> str:
    if args.dist_use_question_image:
        return os.path.join(args.dist_video_dir, str(question.get("image", "")))
    video_id = str(question.get("video_id", ""))
    rel = masked_video_name(video_id, args.counterfactual_suffix)
    if args.counterfactual_subdir:
        rel = os.path.join(args.counterfactual_subdir, rel)
    return os.path.join(args.dist_video_dir, rel)


def generate_baseline(model: Any, processor: Any, video_path: str, question: dict[str, Any], args: argparse.Namespace) -> str:
    messages = build_messages(video_path, question, args.task, args.video_total_pixels, args.video_min_pixels, args.max_frames)
    inputs = prepare_inputs(processor, messages, get_model_device(model))
    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    return decode_output(processor, inputs["input_ids"], generated)


def generate_cd(model: Any, processor: Any, original_video: str, counterfactual_video: str, question: dict[str, Any], args: argparse.Namespace) -> str:
    device = get_model_device(model)
    orig_messages = build_messages(original_video, question, args.task, args.video_total_pixels, args.video_min_pixels, args.max_frames)
    cf_messages = build_messages(counterfactual_video, question, args.task, args.video_total_pixels, args.video_min_pixels, args.max_frames)
    inputs_orig = prepare_inputs(processor, orig_messages, device)
    inputs_cf = prepare_inputs(processor, cf_messages, device)
    apply_pixel_mask(inputs_cf, args.image_mask_level)

    model_inputs = {
        "use_cd": True,
        "input_ids": inputs_orig["input_ids"],
        "input_ids_cd": inputs_cf["input_ids"],
        "attention_mask": inputs_orig.get("attention_mask"),
        "attention_mask_cd": inputs_cf.get("attention_mask"),
        "pixel_values_videos": inputs_orig.get("pixel_values_videos"),
        "pixel_values_videos_cd": inputs_cf.get("pixel_values_videos"),
        "image_grid_thw": inputs_orig.get("image_grid_thw"),
        "video_grid_thw": inputs_orig.get("video_grid_thw"),
        "image_grid_thw_cd": inputs_cf.get("image_grid_thw"),
        "video_grid_thw_cd": inputs_cf.get("video_grid_thw"),
        "second_per_grid_ts": inputs_orig.get("second_per_grid_ts"),
        "second_per_grid_ts_cd": inputs_cf.get("second_per_grid_ts"),
        "mm_token_type_ids": inputs_orig.get("mm_token_type_ids"),
        "mm_token_type_ids_cd": inputs_cf.get("mm_token_type_ids"),
        "max_new_tokens": args.max_new_tokens,
        "cd_alpha": args.cd_alpha,
        "cd_beta": args.cd_beta,
    }
    # Don't pass cache_position to generate() in transformers >= 4.45
    for key in list(model_inputs.keys()):
        if model_inputs[key] is None:
            model_inputs.pop(key)
    generated = model.generate(
        **model_inputs,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    return decode_output(processor, inputs_orig["input_ids"], generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline or MACD contrastive decoding on a video QA JSONL.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--orig-video-dir", required=True)
    parser.add_argument("--orig-question-file", default=None, help="Optional *_orig.jsonl used to map video_id to original videos.")
    parser.add_argument("--method", choices=["baseline", "macd", "vcd"], default="macd")
    parser.add_argument("--task", choices=["yesno", "mcqa"], default="yesno")
    parser.add_argument("--dist-video-dir", default=None, help="Counterfactual/distorted video root for MACD/VCD.")
    parser.add_argument("--dist-use-question-image", action="store_true", help="Resolve distorted video from question['image'].")
    parser.add_argument("--counterfactual-subdir", default=None)
    parser.add_argument("--counterfactual-suffix", default="_merged_max.mp4")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "auto"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--video-total-pixels", type=int, default=20480 * 28 * 28)
    parser.add_argument("--video-min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--image-mask-level", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    if args.method != "baseline" and not args.dist_video_dir:
        raise ValueError("--dist-video-dir is required for MACD/VCD")

    model, processor = load_video_llm(args.model_path, args.torch_dtype)
    if args.method != "baseline":
        enable_macd_decoding(model)

    questions = read_jsonl(args.question_file)
    original_map = build_original_path_map(args.orig_question_file, args.orig_video_dir)
    ensure_dir(os.path.dirname(args.answers_file))
    with open(args.answers_file, "w", encoding="utf-8") as output:
        for question in tqdm(questions, desc=args.method):
            try:
                original_video = resolve_original_video(question, args, original_map)
                if args.method == "baseline":
                    answer = generate_baseline(model, processor, original_video, question, args)
                else:
                    counterfactual_video = resolve_counterfactual_video(question, args)
                    answer = generate_cd(model, processor, original_video, counterfactual_video, question, args)
                if args.task == "mcqa":
                    answer = normalize_choice(answer, extract_option_letters(question.get("options")))
                record = {
                    "question_id": question.get("question_id"),
                    "video_id": question.get("video_id"),
                    "prompt": question.get("text", ""),
                    "text": answer,
                    "model_id": args.model_path,
                    "metadata": {"method": args.method, "alpha": args.cd_alpha, "beta": args.cd_beta},
                }
            except Exception as exc:
                # Fallback: try baseline if MACD fails (e.g., missing counterfactual video)
                if args.method != "baseline":
                    try:
                        answer = generate_baseline(model, processor, original_video, question, args)
                        record = {
                            "question_id": question.get("question_id"),
                            "video_id": question.get("video_id"),
                            "prompt": question.get("text", ""),
                            "text": answer,
                            "model_id": args.model_path,
                            "metadata": {"method": args.method + "_fallback_baseline"},
                        }
                        output.write(__import__("json").dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        continue
                    except Exception:
                        pass
                record = {
                    "question_id": question.get("question_id"),
                    "video_id": question.get("video_id"),
                    "prompt": question.get("text", ""),
                    "text": "ERROR",
                    "metadata": {"error": str(exc), "method": args.method},
                }
                print(f"[ERROR] {question.get('question_id')}: {exc}")
            output.write(__import__("json").dumps(record, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
