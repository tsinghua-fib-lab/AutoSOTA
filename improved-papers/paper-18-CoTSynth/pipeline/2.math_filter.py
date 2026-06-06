import re
import json
from tqdm import tqdm
from prompts import GSM8K_FEW_SHOT_PROMPT, GSM8K_ZERO_SHOT_PROMPT
import os
import random
from collections import Counter, defaultdict
from math_equivalence import is_equiv, _strip_string
from dart_math.eval import *
import argparse

math_evaluator = EvaluatorMathBatch()

def get_model_answer(model_responses):
    model_answers = []
    for output_text in model_responses:
        try:
            model_answer = math_evaluator.extract_ans(output_text)
        except:
            model_answer = None
        model_answers.append({'text': output_text, 'numeric': model_answer})
    numeric_answers = [ma['numeric'] for ma in model_answers]
    filtered_answers = [num for num in numeric_answers if num is not None]
    majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None
    return majority_answer

def check(input_path, output_path):

    lines = open(input_path).readlines()
    results = []
    datasize = len(lines)
    cntt = 0

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(datasize), desc='Evaluating'):
            flag = 0
            data = json.loads(lines[i])
            data['correct_answer'] = []
            for item in data['model_answer']:
                data['model_answer'] = get_model_answer([item])
                correct = is_equiv(math_evaluator.extract_ans(data['solution']), data['model_answer'])
                if correct:
                    data['correct_answer'].append(item)
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            f_out.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter script for model responses")
    
    # Add command-line arguments
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file.")
    
    # Parse arguments
    args = parser.parse_args()

    check(args.input_path, args.output_path)