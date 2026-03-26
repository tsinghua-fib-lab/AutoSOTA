# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import glob
import json
import argparse
import logging
from tqdm.auto import tqdm
from multiprocessing import Pool
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prover.lean.verifier import Lean4ServerScheduler

prompt = r'''Complete the following Lean 4 code:

```lean4
/--In a room, there are 4 chairs along each side of a square table. The length of the side of the table is 1 meter. What is the total length of all sides of the table?-/
def side_length := 1
def number_of_sides := 4
def total_length := side_length * number_of_sides
#reduce total_length   -- returns 4
```


Complete the following Lean 4 code:

```lean4
/--Dave breaks 2 guitar strings per night when playing live.  If he performs 6 shows a week for 12 weeks, how many guitar strings will he need to replace?-/
def strings_broken_per_night := 2
def shows_per_week := 6
def weeks := 12

def strings_broken_per_week := strings_broken_per_night * shows_per_week
def total_strings_broken := strings_broken_per_week * weeks

#reduce total_strings_broken  -- returns 144
```


Complete the following Lean 4 code:

```lean4
/--Convert the binary number $101_{(2)}$ to a decimal number.-/
def binary_to_decimal : List Nat → Nat
| []       => 0
| (b :: bs) => b + 2 * binary_to_decimal bs

def binary_101 := [1, 0, 1]

def decimal_value := binary_to_decimal binary_101.reverse

#reduce decimal_value  -- returns 5
```


Complete the following Lean 4 code:

```lean4
/--The sum of the first $n$ terms of an arithmetic sequence is given by $n^{2} + 5n$. Find the tenth term of the sequence.-/
def S (n : Nat) : Nat := n^2 + 5 * n

def a (n : Nat) : Nat := S n - S (n - 1)

def a_10 := a 10

#eval a_10  -- returns 24
```


Complete the following Lean 4 code:

```lean4
/--If \\( x \\) is a number less than \\(-2\\), which of the following expressions has the least value?\n(A) \\( x \\)\n(B) \\( x + 2 \\)\n(C) \\( \\frac{1}{2}x \\)\n(D) \\( x - 2 \\)\n(E) \\( 2x \\).-/
def expression_A (x : Int) : Int := x
def expression_B (x : Int) : Int := x + 2
def expression_C (x : Int) : Int := x / 2
def expression_D (x : Int) : Int := x - 2
def expression_E (x : Int) : Int := 2 * x

def least_value_expression (x : Int) : String :=
  let values : List (String × Int) := [
    ("A", expression_A x),
    ("B", expression_B x),
    ("C", expression_C x),
    ("D", expression_D x),
    ("E", expression_E x)
  ]
  values.foldl (λ acc val => if val.snd < acc.snd then val else acc) ("A", expression_A x)
  |>.fst

def x : Int := -3
def result := least_value_expression x
#eval result  -- returns \"E\"
```


Complete the following Lean 4 code:

```lean4
/--The integer 42 is:\n(D) divisible by 7.-/
def is_divisible_by_7 (n : Int) : Bool :=
  n % 7 == 0
  
  def result := is_divisible_by_7 42
  
  #eval result  -- returns true
```


Complete the following Lean 4 code:

```lean4
/--Definition: A triangle whose one side is twice the length of the other side is called a double-length triangle. If isosceles triangle $\\triangle ABC$ is a double-length triangle and the length of side $AB$ is $10$, then the length of the base $BC$ is ______.-/
def AB := 10
def AC := 10

def BC_case1 := 2 * AB
def BC_case2 := AB / 2

def triangle_inequality_case1 := (AB + AC > BC_case1) && (AB + BC_case1 > AC) && (AC + BC_case1 > AB)
def triangle_inequality_case2 := (AB + AC > BC_case2) && (AB + BC_case2 > AC) && (AC + BC_case2 > AB)

def valid_case := if triangle_inequality_case1 then BC_case1 else if triangle_inequality_case2 then BC_case2 else 0

#eval valid_case  -- returns 5
```
'''

code_prefix = "Complete the following Lean 4 code:\n\n```lean4\n"

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

def setup_argparse():
    parser = argparse.ArgumentParser(description="Process and verify Lean 4 code completions.")
    parser.add_argument('--input_dir', default="/workspace/cp_data/CoT-python-lean4_300k_results_verified_numina/verified", help='Input directory for JSONL files')
    parser.add_argument('--output_dir', default="/workspace/cp_data/CoT-python-lean4_300k_results_verified_numina/ds-verified", help='Output directory for processed files')
    parser.add_argument('--model_name', default="deepseek-ai/DeepSeek-Prover-V1.5-RL", help='Model name for tokenizer and LLM')
    parser.add_argument('--n', type=int, default=64, help='Number of completions to generate per prompt') 
    parser.add_argument('--start_idx', type=int, required=True, help='Start index for file processing')
    parser.add_argument('--end_idx', type=int, required=True, help='End index for file processing')
    return parser.parse_args()

def setup_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(model=model_name, max_num_batched_tokens=8192, seed=42, trust_remote_code=True, gpu_memory_utilization=0.45)
    return tokenizer, model

def setup_lean4_scheduler():
    return Lean4ServerScheduler(max_concurrent_requests=24, timeout=300, memory_limit=10, name='verifier')

def setup_sampling_params(n):
    return SamplingParams(
        temperature=1.0,
        max_tokens=2048,
        top_p=0.95,
        n=n,
        stop=["```lean4"],
    )

def read_jsonl(file):
    try:
        with open(file, 'r') as f:
            for line in f:
                yield json.loads(line)
    except IOError as e:
        logger.error(f"Error reading file {file}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON in file {file}: {e}")

def process_test(test, model, sampling_params, lean4_scheduler):
    informal_text = test["messages"][1]["content"]
    informal_text = informal_text.split("# Informal proof:")[0].split("# Problem:")[-1].strip()
    informal_text = f"import Mathlib\n/--{informal_text} -/"
    
    model_inputs = [prompt + code_prefix + informal_text]

    model_outputs = model.generate(
        model_inputs,
        sampling_params,
        use_tqdm=False,
    )
    
    all_lean4_blocks = []
    fail_number = 0
    
    for output in model_outputs[0].outputs:
        result = "```lean4\n" + informal_text + output.text
        lean4_blocks = re.findall(r'```lean4(.*?)```', result, re.DOTALL)
        if len(lean4_blocks) == 1:
            all_lean4_blocks.append(lean4_blocks[-1].strip())
        else:
            logger.info(f"Generate failed: {result}")
            fail_number += 1
    
    logger.info(f"Generate failed: {fail_number}")
    
    if all_lean4_blocks:
        request_id_list = lean4_scheduler.submit_all_request(all_lean4_blocks)
        outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
        for i, o in enumerate(outputs_list):
            if o['complete']:
                test["correction_prefix"] = code_prefix + informal_text
                test["correction"] = all_lean4_blocks[i]
                test["lean4_verify"] = 1
                return test
    return None

def process_file(file, model, sampling_params, lean4_scheduler, output_dir):
    logger.info(f"Processing file: {file}")
    infer_list = [item for item in read_jsonl(file) if not item.get("lean4_verify")]
    
    if not infer_list:
        logger.warning(f"No items to process in file: {file}")
        return 0, 0

    total_count = len(infer_list)
    success_count = 0
    
    file_name = file.split("/")[-1]
    output_file = f"{output_dir}/{file_name}"
    
    with open(output_file, "w") as fw:
        for test in tqdm(infer_list, desc="Processing tests"):
            result = process_test(test, model, sampling_params, lean4_scheduler)
            if result:
                json.dump(result, fw, ensure_ascii=False)
                fw.write("\n")
                success_count += 1

    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    logger.info(f"File: {file_name}")
    logger.info(f"Total processed: {total_count}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Success rate: {success_rate:.2f}%")

    return total_count, success_count

def main():
    args = setup_argparse()
    _, model = setup_model(args.model_name)
    lean4_scheduler = setup_lean4_scheduler()
    sampling_params = setup_sampling_params(args.n)
    
    files_list = glob.glob(f"{args.input_dir}/*.jsonl")
    files_list = files_list[args.start_idx:args.end_idx]
    
    total_processed = 0
    total_success = 0
    
    for file in tqdm(files_list, desc="Processing files"):
        file_total, file_success = process_file(file, model, sampling_params, lean4_scheduler, args.output_dir)
        total_processed += file_total
        total_success += file_success
    
    lean4_scheduler.close()

    overall_success_rate = (total_success / total_processed) * 100 if total_processed > 0 else 0
    logger.info("Overall Statistics:")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Total successfully processed: {total_success}")
    logger.info(f"Overall success rate: {overall_success_rate:.2f}%")

if __name__ == "__main__":
    main()
