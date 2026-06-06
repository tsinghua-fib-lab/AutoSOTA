import json
from tqdm import tqdm

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

# Load the benchmark data (164 problems)
dataset = load_jsonl('/repo/data/benchmark/input_humaneval+_sol.jsonl')

# Load solution annotations (ground truth for each solution)
sol_anno = load_jsonl('/repo/data/result/humaneval+/sol_llama3-8b_200_anno.jsonl')
task_sol_results = {}
for data in sol_anno:
    for sol_id in range(len(data['solutions'])):
        task_sol_results[f"{data['task_id']}-{sol_id}"] = data['solutions'][sol_id]['plus_status']

# Load the unit test results
ut_result = load_jsonl('/repo/output/humaneval+/llama3-8b_sol_coderm-8b_ut/details/100_sol_100_ut_result.jsonl')
task_sol_ut_results = {}
for data in tqdm(ut_result, desc="Loading ut results"):
    task_sol_ut_results[f"{data['task_id']}-{data['sol_id']}-{data['ut_id']}"] = data['result']

print(f"\nData loaded: {len(dataset)} problems, {len(task_sol_results)} sol annotations")

# Method 2: Majority threshold - a solution is "accepted" by multiple UTs
# if it passes more than 50% of unit tests (absolute majority)
sol_num = 100
ut_num = 100

TP_total = 0
TN_total = 0
FP_total = 0
FN_total = 0

for data in tqdm(dataset, desc="Computing quality metrics"):
    task_id = data['task_id']
    
    for sol_id in range(sol_num):
        gt_correct = (task_sol_results.get(f"{task_id}-{sol_id}", 'fail') == 'pass')
        
        # Count passes for this solution across all 100 UTs
        pass_count = 0
        for ut_id in range(ut_num):
            key = f"{task_id}-{sol_id}-{ut_id}"
            if key in task_sol_ut_results and task_sol_ut_results[key] == 'pass':
                pass_count += 1
        
        # "Accepted" if majority of UTs say it passes (>= 50% threshold)
        ut_accepted = (pass_count >= ut_num / 2)
        
        if gt_correct and ut_accepted:
            TP_total += 1
        elif gt_correct and not ut_accepted:
            FN_total += 1
        elif not gt_correct and ut_accepted:
            FP_total += 1
        else:
            TN_total += 1

print(f"\n=== Quality Metrics (Majority Threshold >= 50%) ===")
print(f"TP={TP_total}, TN={TN_total}, FP={FP_total}, FN={FN_total}")
total = TP_total + TN_total + FP_total + FN_total
accuracy = (TP_total + TN_total) / total * 100
precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
far = FP_total / (FP_total + TP_total) * 100 if (FP_total + TP_total) > 0 else 0
frr = FN_total / (FN_total + TN_total) * 100 if (FN_total + TN_total) > 0 else 0

print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"FAR: {far:.2f}%")
print(f"FRR: {frr:.2f}%")

print()

# Method 3: Single best solution per problem is accepted, rest rejected
# But use random tiebreaking
# This is the same as method 1 but let me try with choosing ONE solution
import random
TP_total = 0
TN_total = 0
FP_total = 0
FN_total = 0

for data in tqdm(dataset, desc="Method 3: Single best selection"):
    task_id = data['task_id']
    
    # Count passes for all solutions
    sol_pass_counts = {}
    for sol_id in range(sol_num):
        pass_count = 0
        for ut_id in range(ut_num):
            key = f"{task_id}-{sol_id}-{ut_id}"
            if key in task_sol_ut_results and task_sol_ut_results[key] == 'pass':
                pass_count += 1
        sol_pass_counts[sol_id] = pass_count
    
    max_pass = max(sol_pass_counts.values())
    top_sol_list = [sid for sid, cnt in sol_pass_counts.items() if cnt == max_pass]
    # Pick one from top_sol_list
    selected_sol_id = random.choice(top_sol_list)
    
    for sol_id in range(sol_num):
        gt_correct = (task_sol_results.get(f"{task_id}-{sol_id}", 'fail') == 'pass')
        ut_accepted = (sol_id == selected_sol_id)
        
        if gt_correct and ut_accepted:
            TP_total += 1
        elif gt_correct and not ut_accepted:
            FN_total += 1
        elif not gt_correct and ut_accepted:
            FP_total += 1
        else:
            TN_total += 1

print(f"\n=== Quality Metrics (Single Best selected) ===")
print(f"TP={TP_total}, TN={TN_total}, FP={FP_total}, FN={FN_total}")
total = TP_total + TN_total + FP_total + FN_total
accuracy = (TP_total + TN_total) / total * 100
precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
far = FP_total / (FP_total + TP_total) * 100 if (FP_total + TP_total) > 0 else 0
frr = FN_total / (FN_total + TN_total) * 100 if (FN_total + TN_total) > 0 else 0

print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"FAR: {far:.2f}%")
print(f"FRR: {frr:.2f}%")
