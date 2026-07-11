import json
from pathlib import Path

# Find all JSON files in narratives subfolders
narratives_dir = Path("narratives")
json_files = list(narratives_dir.glob("*/classifications_*.json"))

total_classifications = 0
classification_yes_count = 0
violates_usc_yes_count = 0
both_yes_count = 0
codes_with_no_classifications = 0

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        classifications = data["classifications"]
        num_classifications = len(classifications)
        total_classifications += num_classifications

        for item in classifications:
            classification_yes = item.get("classification") == "yes"
            violates_usc_yes = item.get("violates_usc") == "yes"

            if classification_yes:
                classification_yes_count += 1
            if violates_usc_yes:
                violates_usc_yes_count += 1
            if classification_yes and violates_usc_yes:
                both_yes_count += 1

        if num_classifications == 0:
            codes_with_no_classifications += 1

total_codes = len(json_files)
avg_classifications = total_classifications / total_codes if total_codes > 0 else 0
rate_classification_yes = classification_yes_count / total_classifications if total_classifications > 0 else 0
rate_violates_usc_yes = violates_usc_yes_count / total_classifications if total_classifications > 0 else 0
rate_both_yes = both_yes_count / total_classifications if total_classifications > 0 else 0
rate_no_classifications = codes_with_no_classifications / total_codes if total_codes > 0 else 0

print(f"Total codes: {total_codes}")
print(f"Total classifications: {total_classifications}")
print(f"Classifications with 'yes' to computer use: {classification_yes_count}")
print(f"Rate of 'yes' to computer use: {rate_classification_yes:.2%}")
print(f"Classifications with 'yes' to USC violation question: {violates_usc_yes_count}")
print(f"Rate of 'yes' to USC violation question: {rate_violates_usc_yes:.2%}")
print(f"Classifications with 'yes' to both questions: {both_yes_count}")
print(f"Rate of 'yes' to both questions: {rate_both_yes:.2%}")
print(f"Average classifications per code: {avg_classifications:.2f}")
print(f"Codes with no classifications: {codes_with_no_classifications}")
print(f"Rate of codes with no classifications: {rate_no_classifications:.2%}")
