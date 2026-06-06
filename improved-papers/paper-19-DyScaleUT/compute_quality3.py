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

print(f"\nData loaded: {len(dataset)} problems, {len(task_sol_results)} sol annotations")

# Try with first 100 solutions (indices 0-99) from 200 total
sol_num = 100

# Approach: for each problem, the "multiple UT" evaluates ALL 100 candidate solutions
# A solution is "accepted" by the combined UTs if it is top-ranked (most passes)
# similar to how bset-of-N selection works
# But for classification: 
#   - the "accepted" group = all solutions tied at the top pass count
#   - the "rejected" group = the rest

for THRESHOLD in [0, 1]:
    TP_total = 0
    TN_total = 0
    FP_total = 0
    FN_total = 0

    for data in tqdm(dataset, desc=f"Threshold={THRESHOLD}"):
        task_id = data['task_id']
        
        sol_pass_counts = {}
        for sol_id in range(sol_num):
            pass_count = sum(1 for ut_id in range(100)
                            if task_sol_ut_results.get(f"{task_id}-{sol_id}-{ut_id}", 'fail') == 'pass')
            sol_pass_counts[sol_id] = pass_count
        
        max_pass = max(sol_pass_counts.values())
        
        for sol_id in range(sol_num):
            gt_correct = (task_sol_results.get(f"{task_id}-{sol_id}", 'fail') == 'pass')
            
            if THRESHOLD == 0:
                # All tied at max are "accepted"
                ut_accepted = (sol_pass_counts[sol_id] == max_pass)
            else:
                # Only solutions with >= max are accepted (same as THRESHOLD==0)
                ut_accepted = (sol_pass_counts[sol_id] == max_pass)
            
            if gt_correct and ut_accepted:
                TP_total += 1
            elif gt_correct and not ut_accepted:
                FN_total += 1
            elif not gt_correct and ut_accepted:
                FP_total += 1
            else:
                TN_total += 1

    total = TP_total + TN_total + FP_total + FN_total
    accuracy = (TP_total + TN_total) / total * 100
    precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
    recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far = FP_total / (FP_total + TP_total) * 100 if (FP_total + TP_total) > 0 else 0
    frr = FN_total / (FN_total + TN_total) * 100 if (FN_total + TN_total) > 0 else 0

    print(f"\nTP={TP_total}, TN={TN_total}, FP={FP_total}, FN={FN_total}")
    print(f"Accuracy: {accuracy:.2f}%  (paper: 80.46%)")
    print(f"F1 Score: {f1*100:.2f}%  (paper: 81.27%)")
    print(f"FAR: {far:.2f}%  (paper: 16.48%)")
    print(f"FRR: {frr:.2f}%  (paper: 22.71%)")

# Let me also check with consistency-based selection (like in calc_best_of_n)
print("\n=== Consistency-based selection (like calc_best_of_n) ===")
TP_total = 0
TN_total = 0
FP_total = 0
FN_total = 0

for data in tqdm(dataset, desc="Consistency"):
    task_id = data['task_id']
    
    # Get pass sets
    sol_pass_ut_set = {}
    for sol_id in range(sol_num):
        pass_set = set()
        for ut_id in range(100):
            key = f"{task_id}-{sol_id}-{ut_id}"
            if task_sol_ut_results.get(key, 'fail') == 'pass':
                pass_set.add(ut_id)
        sol_pass_ut_set[sol_id] = pass_set
    
    # Rerank by passed UT count
    sorted_sols = sorted(sol_pass_ut_set.items(), key=lambda x: len(x[1]), reverse=True)
    top_pass_ut_num = len(sorted_sols[0][1])
    top_sol_list = [(sid, s) for sid, s in sorted_sols if len(s) == top_pass_ut_num]
    
    # Consistency selection from calc_best_of_n
    select_sol_ids = []
    max_consistency = 0
    for v1 in top_sol_list:
        consistency = sum(1 for v2 in top_sol_list if v1[1] == v2[1])
        if consistency > max_consistency:
            select_sol_ids = [v1[0]]
            max_consistency = consistency
        elif consistency == max_consistency:
            select_sol_ids.append(v1[0])
    
    for sol_id in range(sol_num):
        gt_correct = (task_sol_results.get(f"{task_id}-{sol_id}", 'fail') == 'pass')
        ut_accepted = (sol_id in select_sol_ids)
        
        if gt_correct and ut_accepted:
            TP_total += 1
        elif gt_correct and not ut_accepted:
            FN_total += 1
        elif not gt_correct and ut_accepted:
            FP_total += 1
        else:
            TN_total += 1

total = TP_total + TN_total + FP_total + FN_total
accuracy = (TP_total + TN_total) / total * 100
precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
far = FP_total / (FP_total + TP_total) * 100 if (FP_total + TP_total) > 0 else 0
frr = FN_total / (FN_total + TN_total) * 100 if (FN_total + TN_total) > 0 else 0

print(f"TP={TP_total}, TN={TN_total}, FP={FP_total}, FN={FN_total}")
print(f"Accuracy: {accuracy:.2f}%  (paper: 80.46%)")
print(f"F1 Score: {f1*100:.2f}%  (paper: 81.27%)")
print(f"FAR: {far:.2f}%  (paper: 16.48%)")
print(f"FRR: {frr:.2f}%  (paper: 22.71%)")
