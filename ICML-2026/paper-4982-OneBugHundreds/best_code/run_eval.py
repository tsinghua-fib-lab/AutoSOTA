#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
from tqdm import tqdm
from openai import OpenAI
from accuracy_checking import OverAllResultsAnalysis

def Calling_LLM(user_prompt,model_name,api_key,base_url):
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant with expertise in coding and security."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    # Create chat completion request
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return completion.choices[0].message.content


def load_prompt_configs(prompt_config):
    with open("./prompts/prompts.json", 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    return prompt_data[prompt_config]

def get_commit_messages(commit_id, linux_dir):
    """
    get the commit message of the commit
    """
    try:
        result = subprocess.run(
            ["git", "show", commit_id, "--no-patch", "--color=always","--pretty=format:%B"],
            cwd=linux_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        message = result.stdout
        return message
    except subprocess.CalledProcessError as e:
        exit(e)
        return ""

def get_commit_changes(commit_id, linux_dir):
    """
    get the the changed code of a commit in the whole function context 
    """
    cmd = [ "git", "diff",
            "-W",               # show entire function enclosing changes
            f"{commit_id}^",    # the parent commit
            commit_id           # the commit itself
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=linux_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        code_change = result.stdout
        return code_change
    except subprocess.CalledProcessError as e:
        exit(e)
        return ""

def get_unpatched_function_code(commit_id: str, linux_dir) -> str: 
    """
    Given a Git commit hash (commit_id), return a single string containing
    the full function(s) as they were *before* that commit modified them.

    This works by running:
        git diff -W <commit_id>^ <commit_id>

    and then stripping away all added lines ('+' ), diff‐metadata lines,
    and keeping only lines that were present before the commit (i.e. lines
    starting with ' ' or '-'). Leading '-' or ' ' are removed so you get
    exactly the code that existed before the changes.

    - You must call this from within a clone of the repo (or set GIT_DIR/GIT_WORK_TREE
      appropriately).
    - If the commit touches multiple functions (or multiple files), it will
      concatenate them all in one string in the order diff emits them.
    - If you only want one function but the commit changes more than one, 
      you could further split by file or by function signature. This version
      returns everything pre‐change in one blob.

    Raises:
        subprocess.CalledProcessError if the git command fails (e.g. invalid commit ID).
    """

    diff_output = get_commit_changes(commit_id, linux_dir)
    if diff_output == "":
        return ""
    
    original_lines = []
    for raw_line in diff_output.splitlines():
        # Skip any diff metadata:
        if raw_line.startswith("diff ") or raw_line.startswith("index ") \
           or raw_line.startswith("--- ") or raw_line.startswith("+++ ") \
           or raw_line.startswith("@@"):
            continue

        # Lines that were removed in the commit (“-” prefix) are part of the old code.
        if raw_line.startswith("-"):
            # drop the leading '-' so we recover exactly what was in the old file
            original_lines.append(raw_line[1:])
        # Lines that were unchanged in the hunk (“ ” prefix) were also present before the commit.
        elif raw_line.startswith(" "):
            original_lines.append(raw_line[1:])
        # Skip added lines (“+”) entirely and anything else
        # (new content isn’t part of the “before” code).
        else:
            continue

    # Join everything back into one string.
    return "\n".join(original_lines)

def get_patched_function_code(commit_id: str, linux_dir) -> str: 

    diff_output = get_commit_changes(commit_id, linux_dir)
    if diff_output == "":
        return ""
    
    original_lines = []
    for raw_line in diff_output.splitlines():
        # Skip any diff metadata:
        if raw_line.startswith("diff ") or raw_line.startswith("index ") \
           or raw_line.startswith("--- ") or raw_line.startswith("+++ ") \
           or raw_line.startswith("@@"):
            continue

        # Lines that were removed in the commit (“-” prefix) are part of the old code.
        if raw_line.startswith("+"):
            # drop the leading '+' so we recover exactly what was in the old file
            original_lines.append(raw_line[1:])
        # Lines that were unchanged in the hunk (“ ” prefix) were also present before the commit.
        elif raw_line.startswith(" "):
            original_lines.append(raw_line[1:])
        # Skip added lines (“-”) entirely and anything else
        # (new content isn’t part of the “before” code).
        else:
            continue

    # Join everything back into one string.
    return "\n".join(original_lines)

def load_gt_data(linux_dir,prompt_config):
    gt_file = "./data/ground_truth.json"
    if prompt_config == "HuRule":
        gt_file = "./data/ground_truth_human_rules.json"
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    cases_with_at_least_2_patches = {}
    count = 0
    for case in ground_truth_data:
        if len(ground_truth_data[case]) >= 2:
            count += len(ground_truth_data[case]) 
            cases_with_at_least_2_patches[case] = ground_truth_data[case]
    print("Loaded ",len(cases_with_at_least_2_patches), " recurring patterns with ",count, " patches.")
    gt_data = {}
    print("Prepare patch and target data....")
    count_sp = 0
    count_code = 0
    for rule in tqdm(cases_with_at_least_2_patches):
        commits = cases_with_at_least_2_patches[rule]
        seed_cmt =  commits[0]
        patch_message = get_commit_messages(seed_cmt,linux_dir)
        patch_code = get_commit_changes(seed_cmt,linux_dir)
        seed_patch = patch_message+"\n"+patch_code
        count_sp += 1 
        targets = []
        for i in range(1,len(commits)):
            cmti = commits[i]
            unpatched_target = get_unpatched_function_code(cmti,linux_dir)  # test unpatched version
            patched_target = get_patched_function_code(cmti,linux_dir)  # test patched version
            targets.append((cmti,unpatched_target,patched_target))
            count_code += 2
        gt_data[rule] = {}
        gt_data[rule]["seed_cmt"] = seed_cmt
        gt_data[rule]["seed_patch"] = seed_patch
        gt_data[rule]["targets"] = targets
    print("Load ",count_sp," seed patches and ", count_code, " target code (Including patched and unpatched)!" )
    return gt_data

def build_prompts(prompt_template,code_targets,rule,patch,prompt_config):
    prompts = []
    for case in code_targets:
        cmt = case[0]
        unpatched_code = case[1]
        patched_code = case[2]
        unpatched_prompt = ""
        patched_prompt = ""
        if prompt_config == "Basic":
            unpatched_prompt = prompt_template.format(unpatched_code)
            patched_prompt = prompt_template.format(patched_code)
        elif prompt_config == "Patch":
            unpatched_prompt = prompt_template.format(patch,unpatched_code)
            patched_prompt = prompt_template.format(patch,patched_code)
        elif prompt_config == "Rule+Patch":
            unpatched_prompt = prompt_template.format(rule,patch,unpatched_code)
            patched_prompt = prompt_template.format(rule,patch,patched_code)
        else:
            unpatched_prompt = prompt_template.format(rule,unpatched_code)
            patched_prompt = prompt_template.format(rule,patched_code)
        prompts.append((cmt,unpatched_prompt,patched_prompt))
    return prompts


def load_json_or_empty(filename):
    if os.path.isfile(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # Return empty list if file exists but is not valid JSON
                return []
    return []

def run_evaluation(prompt_config, model_name, base_url, api_key, linux_dir):

    prompt_template = load_prompt_configs(prompt_config)
    gt_data = load_gt_data(linux_dir,prompt_config)
    EvaluationData = []
    for sec_rule in gt_data:
        print("Working on rule: ", sec_rule)
        seed_patch = gt_data[sec_rule]["seed_patch"]
        seed_cmt = gt_data[sec_rule]["seed_cmt"]
        targets = gt_data[sec_rule]["targets"]
        prompts = build_prompts(prompt_template,targets,sec_rule,seed_patch,prompt_config)
        for case in tqdm(prompts):
            data_point = {}
            data_point["sec_rule"] = sec_rule
            data_point["seed_cmt"] = seed_cmt
            target_cmt = case[0]
            data_point["target_cmt"] = target_cmt
            unpatched_prompt = case[1]
            data_point["unpatched_prompt"] = unpatched_prompt
            patched_prompt = case[2]
            data_point["patched_prompt"] = patched_prompt
            unpatched_output = Calling_LLM(unpatched_prompt,model_name,api_key,base_url)
            data_point["unpatched_output"] = unpatched_output
            patched_output = Calling_LLM(patched_prompt,model_name,api_key,base_url)
            data_point["patched_output"] = patched_output
            EvaluationData.append(data_point)
        with open(output_file, "w") as f:
            json.dump(EvaluationData, f,indent=4)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pass config to your program")
    ap.add_argument("-c", "--prompt_config", default="Rule", help="Could be Basic, Patch, Rule, HuRule, Rule+Patch, or RuleNOCoT")
    ap.add_argument("-m", "--model_name", required=True, help="LLM to use")
    ap.add_argument("-u", "--base_url", required=True, help="Service endpoint")
    ap.add_argument("-k", "--api_key", required=True, help="API key")
    ap.add_argument("-l", "--linux_dir", required=True, help="the dir of the Linux kernel git repo")
    args = ap.parse_args()
    prompt_options = ['Basic', 'Patch','HuRule', 'Rule', 'Rule+Patch', 'RuleNOCoT']
    if args.prompt_config not in prompt_options:
        print("Select one prompt config from: ",prompt_options )
        exit(1)
    # print out parameters
    print("prompt_config:", args.prompt_config)
    print("model_name:", args.model_name)
    print("base_url:", args.base_url)
    print("Linux Dir:", args.linux_dir)
    # check if results exit
    output_file = "./results/EvaluationResults_"+args.prompt_config+"_"+args.model_name+".json"
    print(output_file)
    EvaluationData = load_json_or_empty(output_file)
    if len(EvaluationData) == 0:
        print("run evaluation")
        EvaluationData = run_evaluation(args.prompt_config, args.model_name, args.base_url, args.api_key, args.linux_dir)
    print("Overall results for ",args.model_name, " with ",args.prompt_config," configurations.")
    OverAllResultsAnalysis(output_file)
