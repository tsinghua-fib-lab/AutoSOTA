import json
from pathlib import Path

# Find all codes with iterative_refinement subfolders
narratives_dir = Path("narratives")
all_code_dirs = list(narratives_dir.glob("usc_*"))

total_refinements = 0
codes_with_refinements = 0
codes_with_no_refinements = 0

for code_dir in all_code_dirs:
    refinement_dir = code_dir / "iterative_refinement" 
    if refinement_dir.exists():
        round_files = list(refinement_dir.glob("round_5*.json"))
        if round_files:
            codes_with_refinements += 1
            for json_file in round_files:
                with open(json_file) as f:
                    data = json.load(f)
                    total_refinements += len(data["refinements"])
        else:
            codes_with_no_refinements += 1
    else:
        codes_with_no_refinements += 1

total_codes = len(all_code_dirs)
avg_refinements = total_refinements / total_codes if total_codes > 0 else 0
rate_no_refinements = codes_with_no_refinements / total_codes if total_codes > 0 else 0

print(f"Total codes: {total_codes}")
print(f"Total iterative refinements: {total_refinements}")
print(f"Average refinements per code: {avg_refinements:.2f}")
print(f"Codes with no iterative refinements: {codes_with_no_refinements}")
print(f"Rate of codes with no iterative refinements: {rate_no_refinements:.2%}")
