import json
from pathlib import Path

# Find all codes with verified_refinements subfolders
narratives_dir = Path("narratives")
all_code_dirs = list(narratives_dir.glob("usc_*"))

total_validations = 0
verified_true_count = 0
codes_with_validations = 0
codes_without_validations = 0

for code_dir in all_code_dirs:
    validation_file = code_dir / "verified_refinements" / "round_5_validation.json"
    if validation_file.exists():
        codes_with_validations += 1
        with open(validation_file) as f:
            data = json.load(f)
            validations = data["validations"]
            total_validations += len(validations)
            verified_true_count += sum(1 for v in validations if v.get("verified") == True)
    else:
        codes_without_validations += 1

total_codes = len(all_code_dirs)
avg_validations = total_validations / total_codes if total_codes > 0 else 0
rate_verified_true = verified_true_count / total_validations if total_validations > 0 else 0
rate_without_validations = codes_without_validations / total_codes if total_codes > 0 else 0

print(f"Total codes: {total_codes}")
print(f"Total round 5 validations: {total_validations}")
print(f"Validations verified as true: {verified_true_count}")
print(f"Rate of validations verified as true: {rate_verified_true:.2%}")
print(f"Average validations per code: {avg_validations:.2f}")
print(f"Codes without round 5 validations: {codes_without_validations}")
print(f"Rate of codes without round 5 validations: {rate_without_validations:.2%}")
