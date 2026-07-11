import json
from pathlib import Path

# Find all JSON files in narratives subfolders
narratives_dir = Path("narratives")
json_files = list(narratives_dir.glob("*/clean_task_instructions_*.json"))

total_tasks = 0
codes_with_no_tasks = 0

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        num_tasks = len(data["clean_task_instructions"])
        total_tasks += num_tasks
        if num_tasks == 0:
            codes_with_no_tasks += 1

total_codes = len(json_files)
avg_tasks = total_tasks / total_codes if total_codes > 0 else 0
rate_no_tasks = codes_with_no_tasks / total_codes if total_codes > 0 else 0

print(f"Total codes: {total_codes}")
print(f"Total clean task instructions: {total_tasks}")
print(f"Average clean task instructions per code: {avg_tasks:.2f}")
print(f"Codes with no clean task instructions: {codes_with_no_tasks}")
print(f"Rate of codes with no clean task instructions: {rate_no_tasks:.2%}")
