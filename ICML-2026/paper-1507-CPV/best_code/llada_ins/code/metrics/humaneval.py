import os
import sys
import json
import ast
import traceback
import glob
from typing import Dict, List, Optional, Set, Tuple
import evaluate as hf_evaluate
import argparse
import re
import ast
import warnings


os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# 需要清理的特殊 token 模式
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|(?:eot_id|endoftext|end_of_text|pad|eos|bos|unk|sep|cls|mask|"
    r"im_start|im_end|assistant|user|system|begin_of_text|end_header_id|"
    r"start_header_id|reserved_special_token_\d+|finetune_right_pad_id)\|>"
)


def _clean_special_tokens(text: str) -> str:
    """清理模型输出中的特殊 token，如 <|eot_id|>, <|endoftext|> 等"""
    # 移除所有特殊 token
    text = SPECIAL_TOKENS_PATTERN.sub("", text)
    # 移除可能的连续空白
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def refine_text(text: str) -> str:
    # 0. 首先清理特殊 token（这是关键步骤！）
    text = _clean_special_tokens(text)

    # 1. 基础替换
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. 修复：数字后面直接跟了方法调用（如 1.isupper() -> x.isupper()）
    # 这是导致 "invalid decimal literal" 的核心原因
    text = re.sub(r'\b(\d+)\.([a-zA-Z_])', r'var_\1.\2', text)

    # 3. 修复：数字作为循环变量（如 for 1 in ... -> for var_1 in ...）
    text = re.sub(r'for\s+(\d+)\s+in', r'for var_\1 in', text)

    # 4. 修复：数字与关键字粘连（如 0if, 1else）
    text = re.sub(r'(\d+)(if|else|for|while|and|or|return)\b', r'\1 \2', text)

    return text.strip() + "\n"


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False

def extract_valid_code_top_down(text: str) -> str:
    """
    优化后的提取逻辑：从上往下解析，如果不合法，逐行从后往前删减。
    这通常比 O(N^2) 的全扫描快得多，且更符合代码补全的逻辑。
    """
    lines = text.splitlines()
    # 限制处理长度，防止极端情况
    if len(lines) > 500:
        lines = lines[:500]

    for i in range(len(lines), 0, -1):
        snippet = "\n".join(lines[:i])
        try:
            ast.parse(snippet)
            return snippet
        except (SyntaxError, MemoryError):
            continue
    return ""


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

    # 改进的提取逻辑
    code = extract_valid_code_top_down(text)
    if not code:
        return ""

    try:
        tree = ast.parse(code)
    except:
        return ""

    definitions = {}
    imports = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions[node.name] = ("class", node)
        elif isinstance(node, ast.FunctionDef):
            # 注意：删掉 "必须有 return" 的硬性限制。
            # 因为有些 HumanEval 题目（如原地排序）可能没有显式 return。
            definitions[node.name] = ("function", node)
        elif isinstance(node, ast.Assign):
            name = get_definition_name(node)
            if name:
                definitions[name] = ("variable", node)

    # 依赖分析逻辑保持不变
    if entrypoint:
        name2deps = get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = get_function_dependency(entrypoint, name2deps)
    else:
        reachable = set(definitions.keys())

    sanitized_output = []
    for node in imports:
        sanitized_output.append(ast.unparse(node))

    # 按照定义的顺序重新组织，确保 entrypoint 对应的函数被包含
    for name, (_, node) in definitions.items():
        if name in reachable:
            sanitized_output.append(ast.unparse(node))

    return "\n".join(sanitized_output)


def evaluate_humaneval_results(directory):
    print("\n" + "="*50 + f"\nProcessing HumanEval directory: {directory}\n" + "="*50)

    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in directory '{directory}'.")
        return

    all_predictions, all_references = [], []
    processed_count = 0

    print(f"Found {len(jsonl_files)} files to process...")

    for file_path in jsonl_files:
        print(f"  -> Processing file: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    task_id = item["doc"]["task_id"]
                    # 打印当前正在处理的任务
                    print(f"Processing Task: {task_id}", end='\r')
                    raw_generation = item['resps'][0][0]
                    prompt = item["doc"]["prompt"]
                    entry_point = item["doc"]["entry_point"]
                    reference = item["target"]

                    # code_to_sanitize = raw_generation.split("```python\n", 1)[-1].split("```")[0]
                    code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", raw_generation, re.DOTALL)
                    code_to_sanitize = code_blocks[0] if code_blocks else raw_generation

                    full_text = prompt + "\n" + code_to_sanitize
                    sanitized_code = sanitize(full_text, entry_point)

                    all_predictions.append([sanitized_code])
                    all_references.append(reference)
                    processed_count += 1
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"    Error processing file '{os.path.basename(file_path)}': {e}")
            continue

    print(f"\nLoading the code_eval evaluator and starting evaluation...")

    code_eval = hf_evaluate.load("code_eval")

    pass_at_k_results, _ = code_eval.compute(
        references=all_references,
        predictions=all_predictions,
        k=[1],
        num_workers=max(1, os.cpu_count() // 2)
    )

    pass_1_score = pass_at_k_results.get("pass@1", 0.0)

    if processed_count > 0:
        accuracy = pass_1_score * 100
    else:
        print("No valid data processed. Cannot calculate results.")
        return

    print("\n" + "-" * 80)
    print(f"Results for '{os.path.basename(directory)}'")

    print(f"  - Accuracy (pass@1):           {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--res_path",
        type=str,
        required=True,
        help="Path to the directory containing result .jsonl files"
    )
    args = parser.parse_args()

    evaluate_humaneval_results(directory=args.res_path)
