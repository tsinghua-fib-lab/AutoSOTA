import json
import random
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

# Compute the quality metrics for MULTIPLE unit tests (100 UTs combined)
sol_num = 100
ut_num = 100

# For each problem, compute which solution is selected by majority voting
# then compare to ground truth
TP_total = 0
TN_total = 0
FP_total = 0
FN_total = 0

sol_ids = list(range(sol_num))
ut_ids = list(range(ut_num))

for data in tqdm(dataset, desc="Computing quality metrics"):
    task_id = data['task_id']
    
    # Count unit tests passed for each solution
    sol_pass_ut_set = {}
    for sol_id in sol_ids:
        pass_set = set()
        for ut_id in ut_ids:
            key = f"{task_id}-{sol_id}-{ut_id}"
            if key in task_sol_ut_results and task_sol_ut_results[key] == 'pass':
                pass_set.add(ut_id)
        sol_pass_ut_set[sol_id] = pass_set
    
    # MAJORITY VOTING: select the solution that passes the most tests
    max_pass = max(len(s) for s in sol_pass_ut_set.values())
    top_sol_list = [sid for sid, s in sol_pass_ut_set.items() if len(s) == max_pass]
    
    # For each solution:
    # The "selected" (accepted by UT) = in top_sol_list
    # But we need per-solution classification
    # Actually, for Table 2 "Multiple Unit Tests", looking at the formulas:
    # Each solution has a ground truth label (correct/incorrect)
    # Each solution is either accepted (in top_sol_list) or rejected by the combined UTs
    
    # But wait - re-reading this: majority voting selects ONE solution as best
    # The "multiple unit tests" evaluation must be about:
    # - A solution is classified as "correct" if it's the one selected by majority voting
    # - Ground truth is the actual plus_status
    #
    # But this seems odd for per-solution classification...
    # 
    # Let me think about it differently:
    # "Selected" solution = the one with most unit test passes
    # For that solution: TP if it's actually correct, FP if it's actually wrong
    # For other solutions: TN if they're wrong and not selected, FN if they're correct but not selected
    
    # Actually rethinking: looking at the paper's framing more carefully
    # The unit test is a classifier: for a given (problem, solution) pair,
    # the combined unit tests "accept" or "reject" the solution
    # 
    # For multiple UTs: the solution that gets the most passes is "accepted"
    # All others are "rejected"
    #
    # Let me also check if ALL top solutions are counted or just one
    # From calc_best_of_n: it picks one solution from top_sol_list randomly
    
    # For quality metrics, let's use: the selected solution is "accepted"
    # all others are "rejected"
    
    # Which solutions are the "top" (accepted)?
    # Using same logic as calc_best_of_n
    
    # For classification:
    # Accepted (predicted correct) = the selected solution
    # Rejected (predicted incorrect) = all others
    
    # Let's consider:
    # If there are ties, all tied solutions are "accepted"
    
    for sol_id in sol_ids:
        gt_correct = (task_sol_results.get(f"{task_id}-{sol_id}", 'fail') == 'pass')
        ut_accepted = (sol_id in top_sol_list)
        
        if gt_correct and ut_accepted:
            TP_total += 1
        elif gt_correct and not ut_accepted:
            FN_total += 1
        elif not gt_correct and ut_accepted:
            FP_total += 1
        else:  # not gt_correct and not ut_accepted
            TN_total += 1

print(f"\n=== Quality Metrics for Multiple Unit Tests (100 UTs) ===")
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
