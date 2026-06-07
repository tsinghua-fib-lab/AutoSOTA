import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import argparse
import base64
import csv
import io
import json
import os
import re
from datetime import datetime

import numpy as np
import requests
from PIL import Image
from hispatial.inference import MoGeProcessor, HiSpatialPredictor

LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)

MAPPING = {
    "location": ["location_above", "location_closer_to_camera", "location_next_to"],
    "height": ["height_higher"],
    "orientation": ["orientation_in_front_of", "orientation_on_the_left", "orientation_viewpoint"],
    "multi_object": [
        "multi_object_closer_to", "multi_object_facing",
        "multi_object_viewpoint_towards_object", "multi_object_parallel",
        "multi_object_same_direction",
    ],
}
TYPES = ["height", "location", "orientation", "multi_object"]
SUBTYPES = sum([MAPPING[k] for k in TYPES], [])


def _normalize_to_letter(model_out, opt_texts):
    """Extract A/B/C/D from model output via regex, or fall back to option text matching."""
    if not isinstance(model_out, str):
        return None
    s = model_out.strip()
    s = re.sub(r"\s*<eos>\s*$", "", s).strip()
    m = LETTER_RE.search(s)
    if m:
        return m.group(1).upper()
    for k, v in opt_texts.items():
        if v and v.strip() and v.strip().lower() in s.lower():
            return k
    return None


def _is_correct_regex(model_out, gt_letter, opt_texts):
    pred = _normalize_to_letter(model_out, opt_texts)
    return 1 if (pred is not None and pred.upper() == gt_letter.upper()) else 0


def _is_correct_truncate(model_out, gt_letter):
    return 1 if gt_letter in model_out[:2] else 0


def _id_to_group(variant_id):
    parts = variant_id.split("-")
    return parts[0] if parts else variant_id


def _needs_flip(variant_id):
    return "-flip" in variant_id


def _build_prompt(question, A, B, C, D):
    opts = []
    if A: opts.append(f"A. {A}")
    if B: opts.append(f"B. {B}")
    if C: opts.append(f"C. {C}")
    if D: opts.append(f"D. {D}")
    return (question or "").strip() + ("\n" + "\n".join(opts) if opts else "")


def _load_image_from_row(b64_str, url, timeout=30):
    """Load image from base64 string or URL."""
    if b64_str and len(b64_str) > 10:
        try:
            raw = base64.b64decode(b64_str, validate=False)
            return Image.open(io.BytesIO(raw)).convert("RGB"), True
        except Exception:
            pass
    if url and url.startswith(("http://", "https://")):
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB"), False
    raise ValueError("No valid base64 or URL found for image.")


def test(vlm_model_path, tsv_path, save_path, gpu_rank=0):
    """Run evaluation on the 3DSRBench TSV dataset."""
    processor = MoGeProcessor(device_name=f"cuda:{gpu_rank}")
    model_wrapper = HiSpatialPredictor(model_load_path=vlm_model_path, gpu_rank=gpu_rank)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results_agg_truncate = {}
    results_agg_regex = {}
    total_num = 0

    csv.field_size_limit(1_000_000_000)

    print(f"Reading TSV: {tsv_path}")
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header

        for i, row in enumerate(reader):
            if len(row) < 12:
                print(f"[WARN] Row {i} has insufficient columns ({len(row)}), skipping.")
                continue

            total_num += 1
            variant_id = (row[1] or "").strip()
            question_text = (row[2] or "").strip()
            optA, optB = (row[3] or "").strip(), (row[4] or "").strip()
            optC, optD = (row[5] or "").strip(), (row[6] or "").strip()
            gt_letter = (row[7] or "").strip()
            subtype = (row[8] or "").strip()
            img_b64 = (row[9] or "").strip()
            image_url = (row[11] or "").strip()

            group_id = _id_to_group(variant_id)
            prompt = _build_prompt(question_text, optA, optB, optC, optD)
            opt_texts = {"A": optA, "B": optB, "C": optC, "D": optD}

            try:
                image, is_local_image = _load_image_from_row(img_b64, image_url)
            except Exception as e:
                for agg in (results_agg_truncate, results_agg_regex):
                    if group_id in agg:
                        agg[group_id][0] *= 0
                    else:
                        agg[group_id] = [0, subtype]
                continue

            if _needs_flip(variant_id) and not is_local_image:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            try:
                xyz_values = processor.apply_transform(image)
            except Exception:
                for agg in (results_agg_truncate, results_agg_regex):
                    if group_id in agg:
                        agg[group_id][0] *= 0
                    else:
                        agg[group_id] = [0, subtype]
                continue

            try:
                image_np = np.array(image.convert("RGB"))
                decoded_str = str(model_wrapper.query(
                    image=image_np, prompt=prompt, xyz_values=xyz_values,
                ))
            except Exception:
                for agg in (results_agg_truncate, results_agg_regex):
                    if group_id in agg:
                        agg[group_id][0] *= 0
                    else:
                        agg[group_id] = [0, subtype]
                continue

            ok_truncate = _is_correct_truncate(decoded_str, gt_letter)
            ok_regex = _is_correct_regex(decoded_str, gt_letter, opt_texts)

            for agg, ok in ((results_agg_truncate, ok_truncate), (results_agg_regex, ok_regex)):
                if group_id in agg:
                    agg[group_id][0] *= ok
                else:
                    agg[group_id] = [ok, subtype]

            if (i + 1) % 50 == 0:
                print(f"Processed: {i + 1} ...", end="\r")

    print(f"\nInference complete. Total processed: {total_num}. Calculating metrics...")

    def _calc_metrics(results_agg):
        all_scores = [score for score, _ in results_agg.values()]
        overall_acc = np.mean(all_scores) if all_scores else 0.0
        type_acc = {}
        for t in TYPES:
            scores = [s for s, sub_t in results_agg.values() if sub_t in MAPPING[t]]
            type_acc[t] = np.mean(scores) if scores else float("nan")
        subtype_acc = {}
        for t in SUBTYPES:
            scores = [s for s, sub_t in results_agg.values() if sub_t == t]
            subtype_acc[t] = np.mean(scores) if scores else float("nan")
        return overall_acc, type_acc, subtype_acc

    acc_trunc, type_trunc, sub_trunc = _calc_metrics(results_agg_truncate)
    acc_regex, type_regex, sub_regex = _calc_metrics(results_agg_regex)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"--- 3DSRBench Metrics Report (grouped accuracy) ---\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input TSV: {tsv_path}\n")
        f.write(f"Groups: {len(results_agg_truncate)}\n")
        f.write("=" * 60 + "\n\n")

        f.write("[TRUNCATE] model_out[:2] matching\n\n")
        f.write(f"Overall Accuracy: {acc_trunc * 100:.2f}%\n\n")
        f.write("--- By Category ---\n")
        for cat, acc in type_trunc.items():
            f.write(f"  {cat:<40}: {f'{acc * 100:.2f}%' if not np.isnan(acc) else 'N/A'}\n")
        f.write("\n--- By Sub-category ---\n")
        for cat, acc in sub_trunc.items():
            f.write(f"  {cat:<40}: {f'{acc * 100:.2f}%' if not np.isnan(acc) else 'N/A'}\n")

        f.write("\n" + "=" * 60 + "\n\n")

        f.write("[REGEX] letter extraction matching\n\n")
        f.write(f"Overall Accuracy: {acc_regex * 100:.2f}%\n\n")
        f.write("--- By Category ---\n")
        for cat, acc in type_regex.items():
            f.write(f"  {cat:<40}: {f'{acc * 100:.2f}%' if not np.isnan(acc) else 'N/A'}\n")
        f.write("\n--- By Sub-category ---\n")
        for cat, acc in sub_regex.items():
            f.write(f"  {cat:<40}: {f'{acc * 100:.2f}%' if not np.isnan(acc) else 'N/A'}\n")

    with open(f"{save_path}_accuracy.json", "w") as f:
        json.dump({
            "truncate": {
                "accuracy": acc_trunc,
                "by_category": type_trunc,
                "by_subcategory": sub_trunc,
            },
            "regex": {
                "accuracy": acc_regex,
                "by_category": type_regex,
                "by_subcategory": sub_regex,
            },
        }, f, indent=4)

    print("Done.")
    with open(save_path, "r", encoding="utf-8") as f:
        print(f.read())


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HiSpatial on 3DSRBench v1")
    parser.add_argument("--vlm_model_path", type=str, required=True, help="Path to VLM checkpoint (weights.pt)")
    parser.add_argument("--tsv_path", type=str, required=True, help="Path to 3DSRBenchv1.tsv")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the metrics report")
    parser.add_argument("--gpu_rank", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_args()
    test(
        vlm_model_path=args.vlm_model_path,
        tsv_path=args.tsv_path,
        save_path=args.save_path,
        gpu_rank=args.gpu_rank,
    )


if __name__ == "__main__":
    main()
