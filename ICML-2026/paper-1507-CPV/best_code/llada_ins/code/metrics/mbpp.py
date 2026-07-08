import json
import os
import re
import sys
import glob
import ast
import traceback
from typing import Dict, List, Optional, Set, Tuple
import evaluate
import argparse

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def refine_text(text: str) -> str:
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > 100:
        lines = lines[:100]
    max_valid_lines = 0
    max_valid_snippet = ""
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            current_snippet = "\n".join(lines[i : j + 1])
            if syntax_check(current_snippet):
                valid_line_count = sum(1 for line in lines[i : j + 1] if line.strip())
                if valid_line_count > max_valid_lines:
                    max_valid_lines = valid_line_count
                    max_valid_snippet = current_snippet
    return max_valid_snippet


def get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
    name2deps = {}
    for name, node in nodes:
        deps = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    pass
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    visited = set()
    to_visit = [entrypoint]
    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)
    return visited


def get_definition_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    elif isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


def has_return_statement(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(node))


def sanitize(text: str, entrypoint: Optional[str] = None) -> str:
    text = refine_text(text)
    try:
        code = extract_longest_valid_code(text)
        if not code:
            return ""
        tree = ast.parse(code)
    except (SyntaxError, MemoryError):
        return ""
    definitions = {}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            name = node.name
            definitions[name] = ("class", node)
        elif isinstance(node, ast.FunctionDef):
            name = node.name
            definitions[name] = ("function", node)
        elif isinstance(node, ast.Assign):
            name = get_definition_name(node)
            if name:
                definitions[name] = ("variable", node)
    if entrypoint and entrypoint in definitions:
        name2deps = get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = get_function_dependency(entrypoint, name2deps)
    else:
        reachable = set(definitions.keys())
    sanitized_output = []
    for node in imports:
        sanitized_output.append(ast.unparse(node))
    for name, (_, node) in definitions.items():
        if name in reachable:
            sanitized_output.append(ast.unparse(node))
    return "\n".join(sanitized_output)


def extract_python_code(text: str) -> str:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match: return match.group(1).strip()
    return ""


def find_entry_point(code_block: str) -> Optional[str]:
    if not code_block:
        return None
    match = re.search(r"^\s*def\s+([a-zA-Z_]\w*)", code_block, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def evaluate_mbpp_results(directory):
    print("\n" + "="*50 + f"\nProcessing MBPP directory: {directory}\n" + "="*50)
    
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in directory '{directory}'.")
        return
    
    all_predictions, all_references = [], []
    agg_stats = {
        "processed": 0, "failed_extraction": 0, "failed_entry_point": 0
    }

    print(f"Found {len(jsonl_files)} files to process...")
    for file_path in jsonl_files:
        print(f"  -> Processing file: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    raw_generation = item['resps'][0][0]
                    reference_tests = item['target']
                    
                    
                    code_to_process = extract_python_code(raw_generation)
                    if not code_to_process:
                        agg_stats["failed_extraction"] += 1
                        continue
                    
                    entry_point = find_entry_point(code_to_process)
                    if not entry_point:
                        agg_stats["failed_entry_point"] += 1
                        continue
                    
                    sanitized_code = sanitize(code_to_process, entrypoint=entry_point)
                    all_predictions.append([sanitized_code])
                    all_references.append(reference_tests)
                    agg_stats["processed"] += 1
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"    Error processing file '{os.path.basename(file_path)}': {e}")
                    continue
    
    total_samples = agg_stats["processed"] + agg_stats["failed_extraction"] + agg_stats["failed_entry_point"]
    
    if agg_stats["processed"] > 0:
        print(f"\nLoading the code_eval evaluator and starting evaluation...")
        code_eval = evaluate.load("code_eval")
        pass_at_k_results, _ = code_eval.compute(
            references=all_references, 
            predictions=all_predictions, 
            k=[1],
            num_workers=max(1, os.cpu_count() // 2)
        )
        pass_1_score = pass_at_k_results.get("pass@1", 0.0)
        correct_answers = round(pass_1_score * agg_stats["processed"])
    else:
        print("No valid data processed. Cannot calculate results.")
        return
    
    accuracy = (correct_answers / total_samples) * 100 if total_samples > 0 else 0
    print("\n" + "-" * 80)
    print(f"Results for '{os.path.basename(directory)}'")
    print(f"  - Accuracy:               {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--res_path", 
        type=str, 
        required=True)
    args = parser.parse_args()
    
    evaluate_mbpp_results(directory=args.res_path)