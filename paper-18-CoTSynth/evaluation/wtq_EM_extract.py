import json
import os
from tqdm import tqdm
import re
import argparse
from tableqa_util import normalize_answer, exact_match

def extract_answers(answer_list):
    """从模型答案中提取最终答案"""
    
    line = answer.split('answer is')[-1].strip()
    return line

def extract_em(data_path):
    """计算EM（Exact Match）分数"""
    with open(data_path, 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f.readlines()]
    ans = []
    cnt = 0
    
    for data in tqdm(datas, desc="Processing data"):
        data['extract_answer'] = []
        
        reference_answer = [normalize_answer(ref_data) for ref_data in data['answer']]
        assistant_answer = normalize_answer(data['summary_answer'][0].split('Therefore,')[-1])

        data['flag'] = exact_match(reference_answer, assistant_answer)
        if data['flag']:
            cnt += 1
        ans.append(data)
    
    # 计算准确率
    percentage = (cnt / len(datas)) * 100
    formatted_percentage = "{:.2f}%".format(percentage)
    print(f"{data_path.split('/')[-1]}:")
    print(f"percentage: {formatted_percentage}")

if __name__ == '__main__':


    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description="Evaluate Exact Match (EM) score.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input JSONL file.")
    
    args = parser.parse_args()

    # 调用函数
    extract_em(args.data_path)