#!/usr/bin/env python3
"""Compute CHAIR metrics from generated captions."""
import json, os, sys
sys.path.insert(0, "/repo")
from utils.chair import CHAIR, read_jsonl

def compute_chair(response_file, coco_path="/datasets/coco"):
    annotation_dir = os.path.join(coco_path, "annotations")
    raw = read_jsonl(response_file)
    data = [{"image_id": int(item.get("image_id", item.get("question_id", -1))),
             "caption": item.get("text", item.get("caption", ""))} for item in raw]
    unique = list({item["image_id"]: item for item in data}.values())
    img_ids = sorted([item["image_id"] for item in unique])
    evaluator = CHAIR(imids=img_ids, coco_annotation_path=annotation_dir)
    evaluator.get_annotations()
    scores = evaluator.compute_chair(unique)
    results = {"CHAIRs": round(scores["CHAIRs"] * 100, 2),
               "CHAIRi": round(scores["CHAIRi"] * 100, 2)}
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", required=True)
    parser.add_argument("--coco_path", default="/datasets/coco")
    args = parser.parse_args()
    results = compute_chair(args.response_file, args.coco_path)
    print(json.dumps(results, indent=2))
