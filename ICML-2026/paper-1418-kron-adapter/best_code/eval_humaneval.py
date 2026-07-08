try:
    from human_eval.data import write_jsonl, read_problems
except ImportError:
    write_jsonl = None
    read_problems = None

try:
    from human_eval.evaluation import evaluate_functional_correctness
except ImportError:
    evaluate_functional_correctness = None

from fire import Fire
from tqdm import trange, tqdm
from utils import initialize_text_to_text_model, model_inference
import re
import os
from peft import PeftModel

ALPACA_PREFIX_TEMPLATE_MD = """Below is an instruction that describes a task.\n Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code: 
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""

def post_process(text):
    text = text.replace("```", "")
    text = text.replace("\t", "    ")
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    lines = text.split("\n")
    spaces_for_each_line = []
    for line in lines:
        match = re.match(r'^( *)', line)
        if match:
            leading_spaces = len(match.group(1))
            spaces_for_each_line.append(leading_spaces)
    try:
        def_line = [i for i, line in enumerate(lines) if "def" in line][0]
        def_line_space = spaces_for_each_line[def_line]
    except:
        print("No def line found")
        print(text)
        def_line_space = 0
    rank_unique_spaces = sorted(list(set(spaces_for_each_line)))
    indentation_level = {}
    i = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            i += 1
            indentation_level[space] = i
    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_lines.append("    " * indentation_level[space] + line.lstrip())
    return "\n".join(new_lines)

def generate_one_completion(model, tokenizer, model_type, prompt, template=True):
    if template:
        prompt_in = ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=prompt)
    pred_text = model_inference(model, tokenizer, prompt_in, model_type, max_target_length=512)
    post_pred = post_process(pred_text)
    return post_pred

def humaneval(model, tokenizer, save_dir, model_type = "CausalLM", model_name="llama/llama-2-7b-hf"):
    if write_jsonl is None or evaluate_functional_correctness is None:
        print('human_eval not installed, skipping humaneval')
        return
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    problems = read_problems()
    num_samples_per_task = 1
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(model, tokenizer, model_type, problems[task_id]["prompt"]))
        for task_id in tqdm(problems, desc="Tasks")
        for _ in range(num_samples_per_task)
    ]
    target_name = os.path.join(save_dir, f"{model_name.replace('/', '_')}_humaneval_samples.jsonl")
    write_jsonl(target_name, samples)
    results = evaluate_functional_correctness(target_name, [1])
    print(results)
