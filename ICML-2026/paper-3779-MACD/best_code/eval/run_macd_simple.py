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
    """MACD contrastive decoding - CF processed from scratch each step."""
    device = get_model_device(model)

    orig_messages = build_messages(orig_video, question, args.task, args.video_total_pixels, args.video_min_pixels, args.max_frames)
    cf_messages = build_messages(cf_video, question, args.task, args.video_total_pixels, args.video_min_pixels, args.max_frames)

    inputs_orig = prepare_inputs(processor, orig_messages, device)
    inputs_cf = prepare_inputs(processor, cf_messages, device)

    alpha = args.cd_alpha
    beta = max(args.cd_beta, 1e-12)
    eps = 1e-12

    # Get EOS token IDs
    eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_ids is None:
        eos_token_ids = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    elif eos_token_ids is None:
        eos_token_ids = []
    # Also stop on im_end token
    eos_token_ids.append(151645)

    input_ids = inputs_orig["input_ids"]
    generated_ids = input_ids.clone()
    cf_ids_base = inputs_cf["input_ids"]

    pkv_orig = None

    for step in range(args.max_new_tokens):
        # Forward pass on original video (KV-cached after first pass)
        with torch.no_grad():
            if pkv_orig is None:
                out_orig = model(
                    input_ids=input_ids,
                    attention_mask=inputs_orig.get("attention_mask"),
                    pixel_values_videos=inputs_orig.get("pixel_values_videos"),
                    video_grid_thw=inputs_orig.get("video_grid_thw"),
                    mm_token_type_ids=inputs_orig.get("mm_token_type_ids"),
                    use_cache=True,
                    return_dict=True,
                )
            else:
                out_orig = model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=pkv_orig,
                    use_cache=True,
                    return_dict=True,
                )
            pkv_orig = out_orig.past_key_values
            logits_orig = out_orig.logits[:, -1, :]

        # Entropy-gated CD: compute expert entropy to decide whether to apply CD
        cd_entropy_threshold = getattr(args, 'cd_entropy_threshold', None)
        alpha_entropy_scale = getattr(args, 'alpha_entropy_scale', 0.0)
        if cd_entropy_threshold is not None and cd_entropy_threshold > 0:
            probs_orig = F.softmax(logits_orig, dim=-1)
            expert_entropy = -(probs_orig * torch.log(probs_orig + 1e-12)).sum(dim=-1)
            skip_cd = expert_entropy.item() < cd_entropy_threshold
        else:
            skip_cd = False

        if skip_cd:
            # Expert is confident — skip CD, use original logits directly
            cd_logits = logits_orig.clone()
        else:
            # Forward pass on CF video - full sequence each step (handles different lengths)
            generated_so_far = generated_ids[:, input_ids.shape[1]:]
            cf_full_ids = torch.cat([cf_ids_base, generated_so_far], dim=-1)
            # Extend mm_token_type_ids for the generated text tokens (text-only = 0)
            cf_mm_type = inputs_cf.get("mm_token_type_ids")
            if cf_mm_type is not None and generated_so_far.shape[1] > 0:
                # Append zeros for generated text tokens
                cf_mm_ext = torch.zeros(1, generated_so_far.shape[1], dtype=cf_mm_type.dtype, device=device)
                cf_mm_full = torch.cat([cf_mm_type, cf_mm_ext], dim=-1)
            else:
                cf_mm_full = cf_mm_type
            cf_attn = torch.ones(1, cf_full_ids.shape[1], device=device, dtype=torch.long)
            with torch.no_grad():
                out_cf = model(
                    input_ids=cf_full_ids,
                    attention_mask=cf_attn,
                    pixel_values_videos=inputs_cf.get("pixel_values_videos"),
                    video_grid_thw=inputs_cf.get("video_grid_thw"),
                    mm_token_type_ids=cf_mm_full,
                    use_cache=False,
                    return_dict=True,
                )
                logits_cf = out_cf.logits[:, -1, :]

            # Separate temperature calibration for expert and amateur before CD
            T_expert = getattr(args, 'cd_temp_expert', 1.0)
            T_amateur = getattr(args, 'cd_temp_amateur', 1.0)
            logits_orig_scaled = logits_orig / max(T_expert, 1e-8)
            logits_cf_scaled = logits_cf / max(T_amateur, 1e-8)

            # Adaptive alpha: scale alpha by expert entropy
            if alpha_entropy_scale > 0:
                probs_orig = F.softmax(logits_orig_scaled, dim=-1)
                expert_entropy = -(probs_orig * torch.log(probs_orig + 1e-12)).sum(dim=-1)
                adaptive_alpha = alpha * (1.0 + alpha_entropy_scale * expert_entropy)
            else:
                adaptive_alpha = alpha

            # Contrastive decoding formula from the paper
            cutoff = torch.log(torch.tensor(beta, device=device, dtype=logits_orig_scaled.dtype)) + logits_orig_scaled.max(dim=-1, keepdim=True).values
            cd_logits = ((1.0 + adaptive_alpha) * logits_orig_scaled - adaptive_alpha * logits_cf_scaled).masked_fill(logits_orig_scaled < cutoff, -float("inf"))

        # Apply temperature
        if args.temperature > 0:
            cd_logits = cd_logits / args.temperature

        # Apply top-k
        if args.top_k > 0:
            k = min(args.top_k, cd_logits.shape[-1])
            topk_vals, _ = torch.topk(cd_logits, k)
            cd_logits = cd_logits.masked_fill(cd_logits < topk_vals[:, -1:], -float("inf"))

        # Apply top-p (nucleus sampling)
        if args.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(cd_logits, descending=True)
            cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumsum >= args.top_p
            # Keep first token that reaches threshold
            first_exceed = mask.long().argmax(dim=-1, keepdim=True)
            keep_mask = torch.arange(mask.shape[-1], device=device).unsqueeze(0) <= first_exceed
            cd_logits = cd_logits.masked_fill(~keep_mask.scatter(-1, sorted_idx, keep_mask), -float("inf"))

        # Use greedy decoding (no sampling) for stability with CD
        next_token = cd_logits.argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Check for stop after single token (we only need Yes/No)
        nt = next_token.item()
        if nt in eos_token_ids:
            break
        break  # Always stop after first token - we only need Yes/No

    # Decode the generated tokens
    trimmed = generated_ids[:, input_ids.shape[1]:]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    # Extract just Yes/No
    match = re.search(r"(?i)\b(yes|no)\b", raw)
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
    parser.add_argument("--cd-entropy-threshold", type=float, default=None,
                        help="Skip CD when expert entropy below this threshold (e.g., 0.3).")
    parser.add_argument("--alpha-entropy-scale", type=float, default=0.0,
                        help="Per-token adaptive alpha scale: alpha = base * (1 + scale * entropy). 0=disabled.")
    parser.add_argument("--cd-temp-expert", type=float, default=1.0,
                        help="Temperature for expert logits before CD (default 1.0 = no scaling).")
    parser.add_argument("--cd-temp-amateur", type=float, default=1.0,
                        help="Temperature for amateur logits before CD (default 1.0 = no scaling).")
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
        for question in tqdm(questions, desc="macd v2"):
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
                    "metadata": {"method": "macd-v2", "alpha": args.cd_alpha, "beta": args.cd_beta},
                }
            except Exception as exc:
                record = {
                    "question_id": question.get("question_id"),
                    "video_id": question.get("video_id"),
                    "prompt": question.get("text", ""),
                    "text": "ERROR",
                    "metadata": {"error": str(exc), "method": "macd-v2"},
                }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
