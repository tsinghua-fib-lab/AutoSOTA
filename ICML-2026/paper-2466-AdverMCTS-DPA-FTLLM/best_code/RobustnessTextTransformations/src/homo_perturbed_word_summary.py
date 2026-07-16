#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
from collections import defaultdict, Counter

# =========================
# path configuration
# =========================
INPUT_JSONL = "output/homo_perturbed.jsonl"
DETAIL_OUTPUT_JSONL = "output/homoglyph_wordlevel_details.jsonl"
SUMMARY_OUTPUT_CSV = "output/homoglyph_wordlevel_summary.csv"

# =========================
# Homoglyph mapping
# =========================
UNICODE_PAIRS = [
    ("abcdefghijklmnopqrstuvwxyz", "аbϲdеfɡhіϳklmnοрqrѕtuvwхуz"),
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ΑΒϹDΕFGΗΙЈΚLΜΝΟΡQRЅΤUVWΧΥΖ"),
]

CHAR_DICT = {}
for s1, s2 in UNICODE_PAIRS:
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            CHAR_DICT[c1] = c2

HOMOGLYPH_CHARS = set(CHAR_DICT.values())


def normalize_text(text):
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    return text.replace("\r\n", "\n")


def contains_homoglyph(token: str) -> bool:
    return any(ch in HOMOGLYPH_CHARS for ch in token)


def split_words(text: str):
    text = normalize_text(text) or ""
    return text.split()


def extract_homoglyph_words(text: str):
    words = split_words(text)
    return [w for w in words if contains_homoglyph(w)]


def safe_ratio(num, den):
    return num / den if den not in (0, None) else None


def load_grouped_records(path: str):
    groups = defaultdict(lambda: {"origin": None, "attacks": []})
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] line {line_no} JSON 解析失败，跳过")
                continue

            sid = item.get("_sample_id")
            if sid is None:
                print(f"[WARN] line {line_no} 缺少 _sample_id，跳过")
                continue

            if item.get("attack") is None:
                groups[sid]["origin"] = item
            else:
                groups[sid]["attacks"].append(item)
    return groups


def analyze_origin_vs_attack_wordlevel(origin_text: str, perturbed_text: str):
    origin_text = normalize_text(origin_text) or ""
    perturbed_text = normalize_text(perturbed_text) or ""

    origin_homo_words = extract_homoglyph_words(origin_text)
    perturbed_homo_words = extract_homoglyph_words(perturbed_text)

    # 顺序可变，但单词内容和重复次数必须完全一致
    origin_homo_word_counter = Counter(origin_homo_words)
    perturbed_homo_word_counter = Counter(perturbed_homo_words)

    word_exact_preserved = (origin_homo_word_counter == perturbed_homo_word_counter)
    changed = not word_exact_preserved

    lost_words = origin_homo_word_counter - perturbed_homo_word_counter
    gained_words = perturbed_homo_word_counter - origin_homo_word_counter

    return {
        "origin_homo_words": origin_homo_words,
        "perturbed_homo_words": perturbed_homo_words,
        "origin_homo_word_count": len(origin_homo_words),
        "perturbed_homo_word_count": len(perturbed_homo_words),
        "origin_homo_word_counter": dict(origin_homo_word_counter),
        "perturbed_homo_word_counter": dict(perturbed_homo_word_counter),
        "lost_homo_words": dict(lost_words),
        "gained_homo_words": dict(gained_words),
        "word_exact_preserved": word_exact_preserved,
        "changed": changed,
    }


def init_attack_summary():
    return {
        "num_records": 0,
        "num_ok_records": 0,
        "word_exact_preserved_count": 0,
        "changed_count": 0,
        "sum_origin_homo_word_count": 0,
        "sum_perturbed_homo_word_count": 0,
    }


def update_attack_summary(summary_dict, attack_name, compare_item):
    s = summary_dict[attack_name]
    s["num_records"] += 1

    if compare_item.get("compare_status") != "ok":
        return

    s["num_ok_records"] += 1
    s["sum_origin_homo_word_count"] += compare_item.get("origin_homo_word_count", 0) or 0
    s["sum_perturbed_homo_word_count"] += compare_item.get("perturbed_homo_word_count", 0) or 0

    if compare_item.get("word_exact_preserved") is True:
        s["word_exact_preserved_count"] += 1
    if compare_item.get("changed") is True:
        s["changed_count"] += 1


SUMMARY_FIELDNAMES = [
    "attack",
    "num_records",
    "word_exact_preserved_count",
    "word_exact_preserved_rate",
]


def finalize_summary_rows(summary_dict):
    rows = []
    for attack_name in sorted(summary_dict.keys()):
        s = summary_dict[attack_name]

        rows.append({
            "attack": attack_name,
            "num_records": s["num_records"],
            "word_exact_preserved_count": s["word_exact_preserved_count"],
            "word_exact_preserved_rate": safe_ratio(
                s["word_exact_preserved_count"],
                s["num_records"],
            ),
        })
    return rows


def write_summary_csv(rows, path):
    if not rows:
        print(f"[WARN] 没有汇总行可写: {path}")
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    groups = load_grouped_records(INPUT_JSONL)
    total_written = 0
    attack_summary = defaultdict(init_attack_summary)

    with open(DETAIL_OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for sid in sorted(groups.keys()):
            origin = groups[sid]["origin"]
            attacks = groups[sid]["attacks"]

            if origin is None:
                rec = {
                    "_sample_id": sid,
                    "status": "missing_origin",
                    "comparisons": [],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_written += 1
                continue

            if not origin.get("is_watermarked", False):
                continue

            origin_text = origin.get("perturbed_watermarked")
            if origin_text is None:
                origin_text = origin.get("watermarked")

            if origin_text is None:
                rec = {
                    "_sample_id": sid,
                    "status": "missing_origin_text",
                    "comparisons": [],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_written += 1
                continue

            comparisons = []
            for attack_item in attacks:
                attack_name = attack_item.get("attack")
                perturbed_text = attack_item.get("perturbed_watermarked")

                if perturbed_text is None:
                    compare_rec = {
                        "attack": attack_name,
                        "_status": attack_item.get("_status"),
                        "compare_status": "missing_perturbed_text",
                    }
                    comparisons.append(compare_rec)
                    update_attack_summary(attack_summary, attack_name, compare_rec)
                    continue

                result = analyze_origin_vs_attack_wordlevel(origin_text, perturbed_text)

                compare_rec = {
                    "attack": attack_name,
                    "_status": attack_item.get("_status"),
                    "compare_status": "ok",
                    **result,
                }
                comparisons.append(compare_rec)
                update_attack_summary(attack_summary, attack_name, compare_rec)

            rec = {
                "_sample_id": sid,
                "status": "ok",
                "is_watermarked": origin.get("is_watermarked"),
                "user_id": origin.get("user_id"),
                "origin_status": origin.get("_status"),
                "num_attacks_found": len(attacks),
                "comparisons": comparisons,
            }

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_written += 1

    summary_rows = finalize_summary_rows(attack_summary)
    write_summary_csv(summary_rows, SUMMARY_OUTPUT_CSV)

    print(f"Done. Wrote {total_written} grouped records to:")
    print(DETAIL_OUTPUT_JSONL)
    print("Summary CSV written to:")
    print(SUMMARY_OUTPUT_CSV)


if __name__ == "__main__":
    main()