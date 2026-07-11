import json
from pathlib import Path

# Find all JSON files in courtlistener_search_results subfolders
results_dir = Path("courtlistener_search_results")
json_files = list(results_dir.glob("*/results_*.json"))

total_results = 0
codes_with_no_results = 0

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        num_results = len(data["results"])
        total_results += num_results
        if num_results == 0:
            codes_with_no_results += 1

total_codes = len(json_files)
avg_results = total_results / total_codes if total_codes > 0 else 0
rate_no_results = codes_with_no_results / total_codes if total_codes > 0 else 0

print(f"Total codes: {total_codes}")
print(f"Total fetched results: {total_results}")
print(f"Average results per code: {avg_results:.2f}")
print(f"Codes with no results: {codes_with_no_results}")
print(f"Rate of codes with no results: {rate_no_results:.2%}")
