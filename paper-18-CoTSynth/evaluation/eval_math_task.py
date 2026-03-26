import re
import json
from tqdm import tqdm
import os
from transformers import AutoTokenizer
from collections import Counter
from math_equivalence import is_equiv, _strip_string
import torch
from dart_math.eval import *
import gc

math_evaluator = EvaluatorMathBatch()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore = [
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        return None

def get_model_answer(model_responses, name, assistant_responses=None):
    model_answers = []
    # Include all responses: summary + assistant answers
    all_responses = list(model_responses)
    if assistant_responses:
        all_responses.extend(assistant_responses)
    for output_text in all_responses:
        if 'gsm' in name or 'GSM' in name:
            try:
                model_answer = extract_predicted_answer(output_text)
                if len(model_answer) == 0:
                    model_answer = math_evaluator.extract_ans(output_text)
            except:
                model_answer = None
        else:
            try:
                model_answer = _strip_string(math_evaluator.extract_ans(output_text))
            except:
                model_answer = None
        model_answers.append({'text': output_text, 'numeric': model_answer})
    numeric_answers = [ma['numeric'] for ma in model_answers]
    filtered_answers = [num for num in numeric_answers if num is not None]
    majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None
    return majority_answer

def eval(dataset, input_file, dataset_name):
    # Load the data from the generated JSONL file
    input_path = f"../outputs/{dataset}/{input_file}.jsonl"
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    results = []
    datasize = len(lines)
    
    for i in tqdm(range(datasize), desc='Evaluating'):
        data = lines[i]
        data['model_answer'] = get_model_answer(
            data['summary_answer'], dataset_name,
            assistant_responses=data.get('assistant_answer', [])
        )
        if 'gsm' in dataset_name or 'GSM' in dataset_name:
            correct = is_equiv(data['gold_answer'], data['model_answer'])
        else:
            correct = is_equiv(data.get('answer', math_evaluator.extract_ans(str(data['solution']))), data['model_answer'])
        results.append({
            'correct': correct
        })
        data['correct'] = correct
    
    cnt = 0
    for result in results:
        if result['correct']:
            cnt += 1
    total = len(results)
    accuracy = cnt / total if total > 0 else 0
    print(f"{dataset_name}\nAccuracy: {cnt} / {total} = {accuracy:.4f}")
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate model outputs.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset being processed.', default='MATH')
    parser.add_argument('--input_file', type=str,  help='Base name of the input file to evaluate.' , default='math_llama3-8B-5ans')
    parser.add_argument('--dataset_name', type=str,  help='Dataset name for evaluation (e.g., gsm8k).' , default='MATH')
    
    args = parser.parse_args()

    eval(args.dataset, args.input_file, args.dataset_name)
