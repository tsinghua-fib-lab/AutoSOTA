import json
from tqdm import tqdm

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

dataset = load_jsonl('/repo/data/benchmark/input_humaneval+_sol.jsonl')
sol_anno = load_jsonl('/repo/data/result/humaneval+/sol_llama3-8b_200_anno.jsonl')
task_sol_results = {}
for data in sol_anno:
    for sol_id in range(len(data['solutions'])):
        task_sol_results[f"{data['task_id']}-{sol_id}"] = data['solutions'][sol_id]['plus_status']

ut_result = load_jsonl('/repo/output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/100_sol_100_ut_result.jsonl')
task_sol_ut_results = {}
for data in tqdm(ut_result, desc="Loading ut results"):
    task_sol_ut_results[f"{data['task_id']}-{data['sol_id']}-{data['ut_id']}"] = data['result']

# Check the actual UT execution count per problem
print("Checking coverage per problem...")
problem_sol_ut_count = {}
for key in task_sol_ut_results:
    parts = key.rsplit('-', 2)
    task_id = parts[0]
    if task_id not in problem_sol_ut_count:
        problem_sol_ut_count[task_id] = 0
    problem_sol_ut_count[task_id] += 1

print(f"Total problems in UT results: {len(problem_sol_ut_count)}")
# Should be 164
# What's the count per problem?
counts = list(problem_sol_ut_count.values())
print(f"Min/Max/Avg entries per problem: {min(counts)}/{max(counts)}/{sum(counts)/len(counts):.0f}")
# 100 sol × 100 UT = 10000 per problem

# What are the actual sol_ids in the results?
sol_ids_seen = set()
for key in list(task_sol_ut_results.keys())[:10000]:
    parts = key.rsplit('-', 2)
    sol_ids_seen.add(int(parts[1]))
print(f"Sol IDs range: {min(sol_ids_seen)}-{max(sol_ids_seen)}")
