import re
import json
import os
from tqdm import tqdm
from rouge_score import rouge_scorer
from tableqa_util import normalize_answer
import argparse

# 设置 HF-Mirror 加速环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def fetaqa_eval(input_file):
    """
    使用 ROUGE-L 评估模型答案与参考答案的相似度，并计算平均得分。
    """
    # 加载测试数据
    with open(input_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    datas = [json.loads(i) for i in lines]
    
    avg_score = 0
    scored_data = []

    for data in tqdm(datas, desc="Evaluating files"):
        # 获取参考答案和模型生成答案
        reference_answer = normalize_answer(data['answer'])
        assistant_answer = normalize_answer(data['summary_answer'][0])

        # 计算 ROUGE-L 分数
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_score = scorer.score(reference_answer, assistant_answer)['rougeL'].recall

        # 保存分数到数据中
        data['rouge_score'] = rouge_score
        scored_data.append(data)

        # 累积总得分
        avg_score += rouge_score

    # 计算平均分数并转换为百分比
    avg_score /= len(datas)
    avg_score *= 100
    formatted_percentage = "{:.2f}%".format(avg_score)

    # 打印评估结果
    print(f"Input file: {input_file}")
    print(f"Average ROUGE-L Recall Score: {formatted_percentage}")


if __name__ == '__main__':
    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description="Evaluate ROUGE-L Recall score for FETAQA dataset.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSONL file.")

    args = parser.parse_args()

    # 调用评估函数
    fetaqa_eval(args.input_file)