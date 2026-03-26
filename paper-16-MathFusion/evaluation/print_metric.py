import argparse
import json
import os

parser = argparse.ArgumentParser(description="dart math metric")

parser.add_argument(
    "--gen_save_path",
    type=str,
    default="",
)

args = parser.parse_args()

with open(args.gen_save_path, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

correct_num = 0
for d in data:
    if d["correct"]:
        correct_num += 1

print(f"Accuracy: {correct_num / len(data)}")

with open(args.gen_save_path.replace('.jsonl', 'metric.json'), "w", encoding='utf-8') as f:
    json.dump({"accuracy": correct_num / len(data)}, f, ensure_ascii=False, indent=4)