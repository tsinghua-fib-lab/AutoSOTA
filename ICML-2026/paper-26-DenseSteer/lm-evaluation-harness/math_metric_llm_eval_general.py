import json
import os
from collections import Counter
import numpy as np
import re
import json
import os
import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List
import datasets
from vllm import LLM, SamplingParams
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--directory_path', type=str, required=True, help='The directory path to be checked')
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-32B-Instruct')
parser.add_argument('--tensor_parallel_size', type=int, default=4)

args = parser.parse_args()
directory_path = args.directory_path
task = args.task
check_model_name = args.model_name
tensor_parallel_size = args.tensor_parallel_size

def check_answers_with_llm(problematic_batch):

    llm = LLM(model=check_model_name, dtype="float16", tensor_parallel_size=tensor_parallel_size)  
    
    input_texts = []
    for index, problem, resp_answer, ground_answer in problematic_batch:
        prompt = (
            f"Given a math problem, its correct answer, and the model's generated answer, "
            f"determine if the model's generated answer is correct. Respond with 'True' if the answer is correct and "
            f"'False' if it is incorrect. Directly provide your judgement 'True' or 'False' without any other description.\n"
            f"Problem: {problem}\nCorrect Answer: {ground_answer}\nModel's Generated Answer: {resp_answer}\nYour judgement:"
        )
        input_text = (
            f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        input_texts.append(input_text)
    

    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.0, 
        top_p=1.0,
        top_k=1,
    )
    
    outputs = llm.generate(input_texts, sampling_params)
    
    results = []
    decoded_output = []
    for i, output in enumerate(outputs):
        full_text = output.outputs[0].text.strip()
        generated_part = full_text.replace(input_texts[i], "").strip()
        decoded_output.append(generated_part)
    
    for output in decoded_output:
        if "True" in output:
            results.append(1)
        elif "False" in output:
            results.append(0)
        else:
            results.append(None)
    
    return decoded_output, results






# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\fbox{" in s:  
        left = "\\fbox{"
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]

    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string






output_data = []

index = 0
problematic_batch = []

for filename in os.listdir(directory_path):
    if filename.startswith(f"samples_{task}"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                if 'Olympiad' in filename:
                    problem = data['doc']['question']
                    answer = data['doc']['final_answer'][0]
                elif 'Numina' in filename:
                    problem = data['doc']['problem']
                    answer = remove_boxed(last_boxed_only_string(data['doc']['solution'])) if last_boxed_only_string(data['doc']['solution']) is not None else None
                else:
                    problem = data['doc']['problem']
                    answer = data['doc']['answer']
                resp = data['resps'][0][0]
                
                resp_answer = remove_boxed(last_boxed_only_string(resp)) if last_boxed_only_string(resp) is not None else None
                exact_match = 0
                llm_check_result = 0
                
                if resp_answer is not None: 
                    if is_equiv(resp_answer, answer):
                        exact_match = 1
                        llm_check_result = 1
                    if exact_match == 0:
                        problematic_batch.append((index, problem, resp_answer, answer))
                
                output_data.append({
                    'index': index,
                    'problem': problem,
                    'answer': answer,
                    'resp_answer': resp_answer,
                    'exact_match': exact_match,
                    'llm_check_result': llm_check_result,  
                    'llm_check_result_reason': "No need to check"
                })
                index += 1

decoded_output, batch_results = check_answers_with_llm(problematic_batch)

for i, (index, _, _, _) in enumerate(problematic_batch):
    output_data[index]['llm_check_result'] = batch_results[i]
    output_data[index]['llm_check_result_reason'] = "LLM check result: " + decoded_output[i]



problematic_output_data = [
    {
        'index': item['index'],
        'problem': item['problem'],
        'resp_answer': item['resp_answer'],
        'answer': item['answer'],
        'exact_match': item['exact_match'],
        'llm_check_result': item['llm_check_result'],
        'llm_check_result_reason': item['llm_check_result_reason']
    }
    for item in output_data if item['exact_match'] == 0 and item['resp_answer'] is not None
]

problematic_file_path = directory_path + f'/problematic_instances_{task}.json'
with open(problematic_file_path, 'w', encoding='utf-8') as problematic_file:
    json.dump(problematic_output_data, problematic_file, ensure_ascii=False, indent=4)


exact_match_sum = sum(item['exact_match'] for item in output_data)
llm_check_result_sum = sum(item['llm_check_result'] if item['llm_check_result'] is not None else 0 for item in output_data)
none_resp_answer_sum = sum(1 for item in output_data if item['resp_answer'] is None)
none_llm_check_result_sum = sum(1 for item in output_data if item['llm_check_result'] is None)


print("The number of none resp_answer is ", none_resp_answer_sum)
print("The accuracy of exact_match is ", exact_match_sum / len(output_data))
print("The accuracy of llm_check_result is ", (llm_check_result_sum) / len(output_data))
print("The number of none llm_check_result is ", none_llm_check_result_sum)


output_file_path = directory_path + f'/llm_answer_check_result_{task}.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_data, output_file, ensure_ascii=False, indent=4)

summary_results = {
    "none_resp_answer_count": none_resp_answer_sum,
    "exact_match_accuracy": exact_match_sum / len(output_data),
    "llm_check_result_accuracy": llm_check_result_sum / len(output_data),
    "none_llm_check_result_count": none_llm_check_result_sum
}

summary_file_path = directory_path + f'/summary_results_{task}.json'
with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
    json.dump(summary_results, summary_file, ensure_ascii=False, indent=4)