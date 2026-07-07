from collections import defaultdict
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
def merge_json_files(num_gpus,cycle_times,savedir):
    merged_data = []
    for rank in range(num_gpus):
        ranksavedir = savedir.replace(".json",f"_rank-{rank}.json")
        with open(ranksavedir, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # 然后再处理
            objects = content.replace('\n}\n{\n', '\n}|-|-|{\n').split('|-|-|')
            data_list = [json.loads(obj) for obj in objects if obj.strip()]
            merged_data.extend(data_list)
    return merged_data

def load_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON 文件内容应为一个包含字典对象的列表")

    return data

def read_multi_line_json_objects(file_path):
    data = []
    buffer = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        # 然后再处理
        objects = content.replace('}\n{', '}|-|-|{').split('|-|-|')
        # print(objects[0])
        data_list = []
        for obj in objects:
            if obj.strip(): data_list.append(json.loads(obj))
            # print(obj.strip())
        # data_list = [json.loads(obj) for obj in objects if obj.strip()]
        data.extend(data_list)

    return data

def calculate_category_accuracy(data):
    
    inference_methods = list(data[0]["answer"].keys())

    stats = defaultdict(lambda: {
        method: {"correct": 0, "total": 0} for method in inference_methods
    })

    total_stats = {
        method: {"correct": 0, "total": 0} for method in inference_methods
    }

    # 新增：统计 CoT 帮正、帮反、都对、都错的情况
    correction_stats = {
        "cot_helped_positive": 0,  # ori 错，cot 对
        "cot_helped_negative": 0,  # ori 对，cot 错
        "cot_both_correct": 0,     # ori 和 cot 都对
        "cot_both_wrong": 0,       # ori 和 cot 都错
        "total": 0
    }

    for item in data:
        try:
            category = item["category"]
        except:
            category = item["Category"]
        gt = item["Ground truth"].strip().upper()

        answers = {method: item["answer"][method][-1].strip().upper() for method in inference_methods}

        correction_stats["total"] += 1

        if 'ori' in answers and 'cot_1' in answers:
            ori_correct = answers['ori'] == gt
            cot_correct = answers['cot_1'] == gt

        for method in inference_methods:
            ans = answers[method]
            stats[category][method]["total"] += 1
            total_stats[method]["total"] += 1
            if gt == ans:
                stats[category][method]["correct"] += 1
                total_stats[method]["correct"] += 1

    output = []

    for category in sorted(stats.keys()):
        row = {"category": category}
        for method in inference_methods:
            correct = stats[category][method]["correct"]
            total = stats[category][method]["total"]
            accuracy = correct / total if total > 0 else 0
            row[f"{method}_correct"] = correct
            row[f"{method}_total"] = total
            row[f"{method}_accuracy"] = f"{accuracy:.1%}"
        output.append(row)

    overall_row = {"category": "Overall"}
    for method in inference_methods:
        correct = total_stats[method]["correct"]
        total = total_stats[method]["total"]
        accuracy = correct / total if total > 0 else 0
        overall_row[f"{method}_correct"] = correct
        overall_row[f"{method}_total"] = total
        overall_row[f"{method}_accuracy"] = f"{accuracy:.1%}"

    # # 添加 CoT 分析字段
    total_count = correction_stats["total"]

    output = [overall_row] + output

    df = pd.DataFrame(output)

    return df

#Your results
print(calculate_category_accuracy(read_multi_line_json_objects(r"Vstar_results.json")).T)
