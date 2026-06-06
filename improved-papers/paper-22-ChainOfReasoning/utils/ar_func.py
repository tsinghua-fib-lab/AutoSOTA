# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from utils.post_processing import extract_boxed_answer, normalize_answer

class PythonREPL:
    def __init__(self, timeout=5):
        self.timeout = timeout

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(*_):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    def __call__(self, query):
        query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
        query = query.strip().split("\n")
        if "print(" not in query[-1]:
            if "#" in query[-1]:
                query[-1] = query[-1].split("#")[0]
            query[-1] = "print(" + query[-1] + ")"
        query = "\n".join(query)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)
            with self.time_limit(self.timeout):
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode == 0:
                    output = result.stdout
                    return True, output.strip()
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif temp_file_path in m:
                        st = m.index('"/') + 1 if '"/' in m else 0
                        ed = m.index(temp_file_path) + 1 if temp_file_path in m else None
                        clr = m[st:ed] if not ed else m[st:]
                        m = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()

def execute_python_completion(executor, completion, return_status, last_code_block):
    executions = re.findall(r"```python(.*?)```", completion, re.DOTALL)
    
    if len(executions) == 0:
        return completion, False if return_status else completion
    if last_code_block:
        executions = [executions[-1]]
    outputs = []
    successes = []
    for code in executions:
        success = False
        for lib in ("subprocess", "venv"):
            if lib in code:
                output = f"{lib} is not allowed"
                outputs.append(output)
                successes.append(success)
                continue
        try:
            success, output = executor(code)
        except TimeoutError as e:
            print("Code timed out")
            output = e
        if not success and not return_status:
            output = ""
        outputs.append(output)
        successes.append(success)
    output = str(outputs[-1]).strip()
    success = successes[-1]
    if return_status:
        return output, success
    return output

def postprocess_completion(text, return_status=False, last_code_block=False):
    executor = PythonREPL()
    result = execute_python_completion(executor, text, return_status=return_status, last_code_block=last_code_block)
    del executor
    return result

def get_python_answer(sample, check_last_n_chars=500):
    gen_text = sample["gen_texts"]
    region_to_check = gen_text[-check_last_n_chars:]
    if ("answer is" in region_to_check or "\\boxed" in region_to_check):
        num_output_blocks = len(re.findall(r"```output(.*?)```", gen_text, re.DOTALL))
        if num_output_blocks == 0:
            print("The model hallucinated the code answer")
            return sample
        if "boxed" in region_to_check:
            try:
                answer = normalize_answer(extract_boxed_answer(region_to_check))
            except Exception:
                answer = "-1"
        else:
            answer = normalize_answer(region_to_check)
        sample["model_answer"] = answer
    return sample

def process_python_batch(sample):
    gen_text = sample["gen_texts"]
    num_python_blocks = len(re.findall(r"```python(.*?)```", gen_text, re.DOTALL))
    if num_python_blocks == 0:
        print("no code has ever been generated, STOP")
        sample["has_code"] = False
        return sample
    code_result = postprocess_completion(gen_text)
    
    truncation_limit = 200
    if len(code_result) > truncation_limit:
        code_result = code_result[:truncation_limit] + " ... (output truncated)"
    sample["gen_texts"] = gen_text + f"\n```output\n{code_result}\n```"
    sample["python_result"] = str(code_result)
    return sample
