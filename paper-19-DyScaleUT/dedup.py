import json
from tqdm import tqdm

# Load all results and deduplicate by (task_id, sol_id, ut_id)
seen = {}
dup_count = 0

print("Loading and deduplicating results...")
with open('/repo/output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/100_sol_100_ut_result.jsonl', 'r') as f:
    for line in tqdm(f):
        try:
            data = json.loads(line)
            key = (data['task_id'], data['sol_id'], data['ut_id'])
            if key not in seen:
                seen[key] = data
            else:
                dup_count += 1
        except:
            pass

print(f"Total unique entries: {len(seen)}")
print(f"Duplicates removed: {dup_count}")
print(f"Expected: {164 * 100 * 100} = 1,640,000")

# Check coverage
total_expected = 0
missing = 0
# We don't know exact tasks, just check if count is right
if len(seen) >= 1600000:
    print("Coverage looks OK!")
elif len(seen) < 1600000:
    print(f"WARNING: Missing {1640000 - len(seen)} entries")

# Save deduped results
print("Saving deduped results...")
with open('/repo/output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/100_sol_100_ut_result_dedup.jsonl', 'w') as f:
    for data in tqdm(seen.values()):
        f.write(json.dumps(data) + '\n')
print("Done!")
