from typing import Optional, Callable, TypeVar
import os, sys, re
import json
import argparse
from tqdm import tqdm
sys.path.append('inference/code')
from math_judger import MathJudger

def extract_answer(is_chinese, model_output, is_deepseek=False):
    # deepseekmath has special answering format
    if is_deepseek:
        if is_chinese:
            matches = re.findall('## 解题答案(.*)', model_output)
        else:
            matches = re.findall('The answer is: (.*)', model_output)
    
        # 检测是否至少找到一个匹配，如果没有就直接整个送进去找\boxed{}
        if matches:
            # 如果找到多个匹配，取最后一个
            model_answer = matches[-1].strip()
            return model_answer
        else:
            return model_output
        
    if is_chinese:
        matches = re.findall('所以最终答案是(.*)', model_output)
    else:
        last_valid_expression = find_last_valid_expression(model_output, prefix="\\boxed{")
        matches = [last_valid_expression] if last_valid_expression is not None else []
    # 检测是否至少找到一个匹配，如果没有就直接整个送进去找\boxed{}
    if matches:
        # 如果找到多个匹配，取最后一个
        model_answer = matches[-1].strip()
        return model_answer
    else:
        return model_output


def find_last_valid_expression(
    response: str, prefix: str = "\\boxed{", extract_result: Callable[[str], str] = lambda x: x
) -> Optional[str]:
    """
    Find the last correct brace sequence that starts with prefix and passes extract_result; return it including prefix
    """
    while True:
        try:
            start = response.rindex(prefix)
            try:
                excerpt = parse_until_valid_brace_sequence(response[start:], keep_prefix=True)
                return extract_result(excerpt)
            except Exception:  # missing suffix or extract_result failed
                response = response[:start]
        except ValueError:
            return None


def parse_until_valid_brace_sequence(text: str, start: int = 0, end: Optional[int] = None, keep_prefix: bool = False) -> str:
    original_start = start
    start = text.index('{', start)
    balance = 1
    for i in range(start + 1, end if end is not None else len(text)):
        if text[i] == '{':
            balance += 1
        elif text[i] == '}':
            balance -= 1
        if balance == 0:
            return text[(original_start if keep_prefix else start): i + 1]
    raise ValueError("text does not have a correct bracket {/} ")

def judge_result(args):
    judger = MathJudger()
    print(os.listdir(args.input_dir))

    for dataset in os.listdir(args.input_dir):
        print('-'*10 + dataset + '-'*10)
        if "TP" in dataset:
            print("Warning: Theorem proving problems cannot currently be automatically assessed.")
            continue
        is_chinese = True if 'zh' in dataset else False # 也没有这个属性了
        for i in os.listdir(os.path.join(args.input_dir, dataset)):
            
            results_path = os.path.join(args.input_dir, dataset, i)
            print(results_path)
            if os.path.exists(results_path):
                for model in tqdm(os.listdir(results_path)):
                    full_num = 0
                    machine_scored_num = 0
                    correct_num = 0
                    available_id_list = []    # deduplication
                    merged_result = []
                    is_deepseek = True if 'deepseek' in model else False
                    res_i_path = os.path.join(results_path, model)
                    for single_result in tqdm(os.listdir(res_i_path)):
                        if single_result[-5:] != '.json':
                            continue
                        single_result_path = os.path.join(res_i_path, single_result) # 具体的某个结果文件
                        with open(single_result_path, 'r', encoding='utf-8') as f:
                            single_result_dict = json.load(f)
                            for id, question in enumerate(single_result_dict):
                                if (len(question['model_output'][model]['raw_output'])>0 and question['model_output'][model]['raw_output'] != '<Inappropriate content in response>' and question['model_output'][model]['raw_output']!='<No response>' and ('code:' not in question['model_output'][model]['raw_output'] or 'message:' not in question['model_output'][model]['raw_output'])):
                                    if question['id'] in available_id_list:    # 重复数据
                                        continue
                                    else:
                                        available_id_list.append(question['id'])
                                full_num += 1 # 这俩没用到
                                machine_scored_num += 1 # 这俩没用到
                                
                                model_answer = question['model_output'][model]['raw_output']
                                model_answer = extract_answer(is_chinese, model_answer, is_deepseek)

                                answer_type = question['answer_type']
                                if 'Tuple' in answer_type: # 目前可机评的数据中 没有 need_human_evaluate
                                    judge_result = judger.judge(model_answer, question['final_answer'][0])
                                    
                                else:
                                    if question['error']:
                                        print(question['error'])
                                        if ',' in question['error']:
                                            precisions = question['error'].split(',')
                                            precisions = [float(p) if p else 1e-8 for p in precisions]
                                            judge_result = judger.judge(model_answer, question['final_answer'][0], precisions)
                                        else:
                                            precision = float(question['error'])
                                            judge_result = judger.judge(model_answer, question['final_answer'][0], precision)
                                    else:
                                        judge_result = judger.judge(model_answer, question['final_answer'][0])
                                print(judge_result)

                                if judge_result:
                                    correct_num += 1 # 貌似也没用到
                                single_result_dict[id]['model_output'][model]['answer'] = model_answer
                                single_result_dict[id]['model_output'][model]['correctness'] = judge_result
                            merged_result += single_result_dict # 保留所有的处理结果

                    if not os.path.exists(os.path.join(args.output_dir, model)):
                        os.makedirs(os.path.join(args.output_dir, model))
                    with open(os.path.join(args.output_dir, model, f'{dataset}.json'), 'w', encoding='utf-8') as f:
                        json.dump(merged_result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='generated')
    parser.add_argument("--output_dir", type=str, default='merged')
    args = parser.parse_args()
    judge_result(args)
