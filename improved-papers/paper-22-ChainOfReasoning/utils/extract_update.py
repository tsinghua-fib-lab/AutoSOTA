# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from itertools import islice
from utils.post_processing import normalize_answer
from utils.sr_func import execute_completions_math, execute_completions_prove

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def update_success_math(sample, success, blocks, outputs_data):
    if len(blocks) > 0:
        sample["complete"] = success
        sample["has_code"] = True
        if success:
            if not outputs_data.strip():
                sample["gen_texts"] = sample["gen_texts"] + f"\n```lean_output\nEmpty Output.\n```"
            else:
                truncation_limit = 200
                if len(outputs_data) > truncation_limit:
                    outputs_data = outputs_data[:truncation_limit] + " ... (output truncated)"
                sample["gen_texts"] = sample["gen_texts"] + f"\n```lean_output\n{outputs_data}\n```"
        else:
            sample["gen_texts"] = sample["gen_texts"] + f"\n```lean_output\nExecution Failed.\n```"
        sample["lean4_result"] = str(outputs_data)
    else:
        print("no code has ever been generated, STOP")
        sample["has_code"] = False
        sample["complete"] = False
    return sample

def process_batch_math(samples, verifier):
    completions = []
    for sample in samples:
        if sample["gen_texts"].endswith("```"):
            completions.append(sample["gen_texts"])
        else:
            completions.append(sample["gen_texts"] + "\n```")

    successes, all_blocks, outputs_data = execute_completions_math(verifier, completions)

    updated_samples = samples.map(
        lambda sample, idx: update_success_math(sample, successes[idx], all_blocks[idx], outputs_data[idx]),
        with_indices=True
    )
    
    return updated_samples

def update_success_prove(sample, success, blocks):
    if len(blocks) > 0:
        sample["complete"] = success
        sample["has_code"] = True
        sample["should_prune"] = not success
    else: 
        print("no code has ever been generated, STOP")
        sample["should_prune"] = True
        sample["has_code"] = False
        sample["complete"] = False
    return sample

def process_batch_prove(samples, verifier, formal_statement, header):
    completions = []
    for sample in samples:
        if sample["gen_texts"].endswith("```"):
            completions.append(sample["gen_texts"])
        else:
            completions.append(sample["gen_texts"] + "\n```")

    successes, all_blocks = execute_completions_prove(verifier, completions, formal_statement, header)

    updated_samples = samples.map(
        lambda sample, idx: update_success_prove(sample, successes[idx], all_blocks[idx]),
        with_indices=True
    )
    
    return updated_samples

def check_answer(sample):
    answer = sample.get("sample_answer", "")
    if isinstance(answer, float):
        answer = str(answer)

    def is_numeric_match(a, b, tolerance=1e-6):
        try:
            return abs(float(a) - float(b)) < tolerance
        except ValueError:
            return False

    correct_answer = normalize_answer(answer)
    python_answer = normalize_answer(sample.get("python_result", ""))
    lean4_answer = normalize_answer(sample.get("lean4_result", ""))
    model_answer = sample.get("model_answer", "")

    sample["python_correct"] = (
        python_answer == correct_answer
        or is_numeric_match(python_answer, correct_answer)
        or model_answer == correct_answer
    )

    sample["lean_correct"] = (
        lean4_answer == correct_answer or is_numeric_match(lean4_answer, correct_answer)
    )
    
    sample["final_correct"] = sample["python_correct"] or sample["lean_correct"]

    return sample

def check_answer_maj(model_answer, python_answer, lean4_answer, answer):

    def is_numeric_match(a, b, tolerance=1e-6):
        try:
            return abs(float(a) - float(b)) < tolerance
        except ValueError:
            return False

    if isinstance(answer, float):
        answer = str(answer)
    if isinstance(python_answer, float):
        python_answer = str(python_answer)
    
    if lean4_answer is None:
        lean4_answer = ""
        
    elif isinstance(lean4_answer, float):
        lean4_answer = str(lean4_answer)
    if isinstance(model_answer, float):
        model_answer = str(model_answer)
    
    correct_answer = normalize_answer(answer)
    python_answer = normalize_answer(python_answer)
    lean4_answer = normalize_answer(lean4_answer)
    model_answer = model_answer

    return python_answer == correct_answer \
        or is_numeric_match(python_answer, correct_answer) \
        or model_answer == correct_answer \
        or lean4_answer == correct_answer or is_numeric_match(lean4_answer, correct_answer)


def update_python_sample(sample):
    pattern = r"[^.\n]*formal proof[^.\n]*Lean(?:\s*[3-4]*)?[^.\n]*\."
    
    lines = sample["gen_texts"].splitlines()
    
    if len(lines) >= 3:
        last_lines = lines[-3:]
        last_lines = [re.sub(pattern, "", line) for line in last_lines]
        lines[-3:] = last_lines
        
    sample["gen_texts"] =  "\n".join(lines).strip() + "\nLet's use Python to perform these calculations.\n### Formal Code in Python: ```python\n"
    return sample

def update_lean_sample(sample):
    pattern = r"[^.\n]*formal proof[^.\n]*Lean(?:\s*[3-4]*)?[^.\n]*\."
    
    lines = sample["gen_texts"].splitlines()
    
    if len(lines) >= 3:
        last_lines = lines[-3:]
        last_lines = [re.sub(pattern, "", line) for line in last_lines]
        lines[-3:] = last_lines

    sample["gen_texts"] =  "\n".join(lines).strip() + "\nNext, let's write the corresponding formal proof in Lean 4 to prove this.\n### Formal proof in Lean 4: ```lean4\n"
    return sample

def update_summary_sample(sample):
    sample["gen_texts"] =  sample["gen_texts"] + "\n"
    return sample

def update_sample(sample, test):
    if sample["gen_texts"].endswith("```lean4"):
        sample["gen_texts"] += "\n" + test["formal_statement"]
    else:
        sample["gen_texts"] += "```lean4\n" + "\n" + test["formal_statement"]
    return sample
