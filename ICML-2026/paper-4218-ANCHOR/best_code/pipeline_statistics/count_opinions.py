import json
from pathlib import Path

# Find all JSON files in opinion_texts subfolders
opinion_dir = Path("opinion_texts")
json_files = list(opinion_dir.glob("*/opinions_*.json"))

total_opinions = 0
codes_with_no_opinions = 0

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        num_results = len(data["results"])
        total_opinions += num_results
        if num_results == 0:
            codes_with_no_opinions += 1

total_codes = len(json_files)
avg_opinions = total_opinions / total_codes if total_codes > 0 else 0
rate_no_opinions = codes_with_no_opinions / total_codes if total_codes > 0 else 0

print(f"Total codes: {total_codes}")
print(f"Total fetched opinions: {total_opinions}")
print(f"Average opinions per code: {avg_opinions:.2f}")
print(f"Codes with no opinions: {codes_with_no_opinions}")
print(f"Rate of codes with no opinions: {rate_no_opinions:.2%}")
