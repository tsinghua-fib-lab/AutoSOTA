"""Simplified MACD inference without monkey-patching transformer internals."""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import torch
import torch.nn.functional as F
from macd.io import ensure_dir, read_jsonl
from macd.models import get_model_device, load_video_llm

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def build_messages(video_path, question, task, total_pixels, min_pixels, max_frames):
    prompt = str(question.get("text", ""))
    if task == "yesno":
        system = "Answer with exactly Yes or No. Output only one word, no punctuation, no extra text."
        user_text = f"Answer with exactly Yes or No: {prompt}"
    else:
        options = question.get("options") or []
        letters = []
        for option in options:
            match = re.match(r"\s*([A-Z])[\.\)]", str(option))
            if match and match.group(1) not in letters:
                letters.append(match.group(1))
        letters = letters or ["A", "B", "C", "D"]
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
                {"type": "video", "video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels, "max_frames": max_frames},
            ],
        },
    ]


def prepare_inputs(processor, messages, device):
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    for k in list(video_kwargs.keys()):
        if isinstance(video_kwargs[k], (list, tuple)):
            video_kwargs[k] = video_kwargs[k][0] if video_kwargs[k] else 1.0
    inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, **video_kwargs, padding=True, return_tensors="pt")
    return inputs.to(device)


def macd_generate(model, processor, orig_video, cf_video, question, args):
    """MACD contrastive decoding implemented without monkey-patching."""
    device = get_model_device(model)

    orig_messages = build_messages(orig_video, question, args.task, args.video_total_pixels, args.video_min_pixels, args.max_frames)
    cf_messages = build_messages(cf_video, question, args.task, args.video_total_pixels, args.video_min_pixels, args.max_frames)

    inputs_orig = prepare_inputs(processor, orig_messages, device)
    inputs_cf = prepare_inputs(processor, cf_messages, device)

    input_ids = inputs_orig["input_ids"]
    input_ids_cf = inputs_cf["input_ids"]

    alpha = args.cd_alpha
    beta = max(args.cd_beta, 1e-12)

    # First forward pass: process full video+prompt for both original and CF
    with torch.no_grad():
        output_orig = model(
            input_ids=input_ids,
            attention_mask=inputs_orig.get("attention_mask"),
            pixel_values_videos=inputs_orig.get("pixel_values_videos"),
            video_grid_thw=inputs_orig.get("video_grid_thw"),
            mm_token_type_ids=inputs_orig.get("mm_token_type_ids"),
            use_cache=True,
            return_dict=True,
        )
        output_cf = model(
            input_ids=input_ids_cf,
            attention_mask=inputs_cf.get("attention_mask"),
            pixel_values_videos=inputs_cf.get("pixel_values_videos"),
            video_grid_thw=inputs_cf.get("video_grid_thw"),
            mm_token_type_ids=inputs_cf.get("mm_token_type_ids"),
            use_cache=True,
            return_dict=True,
        )

    # Generate token by token with CD
    generated_ids = input_ids.clone()
    cf_ids = input_ids_cf.clone()
    pkv_orig = output_orig.past_key_values
    pkv_cf = output_cf.past_key_values

    # Get EOS token IDs
    eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_ids is None:
        eos_token_ids = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    elif eos_token_ids is None:
        eos_token_ids = []

    # Also stop on newline tokens
    nl_tokens = processor.tokenizer.encode("\n", add_special_tokens=False)

    for step in range(args.max_new_tokens):
        # Logits from KV-cached forward (no visual inputs needed after first pass)
        with torch.no_grad():
            out_orig = model(
                input_ids=generated_ids[:, -1:],
                past_key_values=pkv_orig,
                use_cache=True,
                return_dict=True,
            )
            out_cf = model(
                input_ids=cf_ids[:, -1:],
                past_key_values=pkv_cf,
                use_cache=True,
                return_dict=True,
            )

        pkv_orig = out_orig.past_key_values
        pkv_cf = out_cf.past_key_values

        logits_orig = out_orig.logits[:, -1, :]
        logits_cf = out_cf.logits[:, -1, :]

        # Contrastive decoding formula from the paper
        cutoff = torch.log(torch.tensor(beta, device=device, dtype=logits_orig.dtype)) + logits_orig.max(dim=-1, keepdim=True).values
        cd_logits = ((1.0 + alpha) * logits_orig - alpha * logits_cf).masked_fill(logits_orig < cutoff, -float("inf"))

        # Apply temperature
        if args.temperature > 0:
            cd_logits = cd_logits / args.temperature

        # Apply top-k
        if args.top_k > 0:
            k = min(args.top_k, cd_logits.shape[-1])
            topk_vals, _ = torch.topk(cd_logits, k)
            cd_logits = cd_logits.masked_fill(cd_logits < topk_vals[:, -1:], -float("inf"))

        # Apply top-p
        if args.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(cd_logits, descending=True)
            cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = torch.zeros_like(cd_logits, dtype=torch.bool)
            mask.scatter_(-1, sorted_idx, cumsum >= args.top_p)
            # Keep the first token that exceeds the threshold
            first_exceed = mask.long().argmax(dim=-1, keepdim=True)
            keep_mask = torch.arange(mask.shape[-1], device=device).unsqueeze(0) < first_exceed + 1
            cd_logits = cd_logits.masked_fill(~keep_mask, -float("inf"))

        probs = F.softmax(cd_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        if probs.sum() == 0:
            probs = F.softmax(logits_orig, dim=-1)  # fallback to original
        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        cf_ids = torch.cat([cf_ids, next_token], dim=-1)

        # Check for EOS
        if next_token.item() in eos_token_ids:
            break
        if next_token.item() in nl_tokens:
            break

    # Decode the generated tokens
    trimmed = generated_ids[:, input_ids.shape[1]:]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    # Extract just Yes/No from the response
    import re as _re
    match = _re.search(r"(?i)\b(yes|no)\b", raw)
    if match:
        return match.group(1).capitalize()
    return raw.split()[0] if raw.split() else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--orig-video-dir", required=True)
    parser.add_argument("--orig-question-file", default=None)
    parser.add_argument("--task", choices=["yesno", "mcqa"], default="yesno")
    parser.add_argument("--dist-video-dir", default=None)
    parser.add_argument("--counterfactual-subdir", default=None)
    parser.add_argument("--counterfactual-suffix", default="_merged_max.mp4")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "auto"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--video-total-pixels", type=int, default=20480 * 28 * 28)
    parser.add_argument("--video-min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    model, processor = load_video_llm(args.model_path, args.torch_dtype)

    questions = read_jsonl(args.question_file)
    original_map = {}
    if args.orig_question_file:
        for item in read_jsonl(args.orig_question_file):
            video_id = str(item.get("video_id", ""))
            rel_path = item.get("image")
            if video_id and rel_path:
                original_map[video_id] = os.path.join(args.orig_video_dir, rel_path)

    ensure_dir(os.path.dirname(args.answers_file))
    with open(args.answers_file, "w", encoding="utf-8") as output:
        for question in tqdm(questions, desc="macd-simple"):
            try:
                video_id = str(question.get("video_id", ""))
                original_video = original_map.get(video_id) or os.path.join(args.orig_video_dir, str(question.get("image", "")))
                cf_video = os.path.join(args.dist_video_dir, args.counterfactual_subdir or "", video_id + args.counterfactual_suffix)
                if not os.path.isfile(cf_video):
                    cf_video = original_video
                answer = macd_generate(model, processor, original_video, cf_video, question, args)
                record = {
                    "question_id": question.get("question_id"),
                    "video_id": video_id,
                    "prompt": question.get("text", ""),
                    "text": answer,
                    "model_id": args.model_path,
                    "metadata": {"method": "macd-simple", "alpha": args.cd_alpha, "beta": args.cd_beta},
                }
            except Exception as exc:
                record = {
                    "question_id": question.get("question_id"),
                    "video_id": question.get("video_id"),
                    "prompt": question.get("text", ""),
                    "text": "ERROR",
                    "metadata": {"error": str(exc), "method": "macd-simple"},
                }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
