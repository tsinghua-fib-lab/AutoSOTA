import ast
import re
from typing import Dict, List, Optional, Set, Tuple

import evaluate as hf_evaluate


# ============================================================================
# Local sanitize implementation (替代 lm_eval.tasks.humaneval.sanitize_utils)
# ============================================================================

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


def _clean_trailing_comments(text: str) -> str:
    """清理代码后面的注释说明（通常以 ``` 开头的 markdown 块或解释性文字）

    例如输入:
        def func():
            return x
        ```

        This function does...

    输出:
        def func():
            return x
    """
    # 如果文本中有 ```，只保留第一个 ``` 之前的内容
    if "```" in text:
        text = text.split("```")[0]
    return text.rstrip()


def _refine_text(text: str) -> str:
    # 首先清理特殊 token
    text = _clean_special_tokens(text)
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\b(\d+)\.([a-zA-Z_])", r"var_\1.\2", text)
    text = re.sub(r"for\s+(\d+)\s+in", r"for var_\1 in", text)
    text = re.sub(r"(\d+)(if|else|for|while|and|or|return)\b", r"\1 \2", text)
    return text.strip() + "\n"


def _extract_valid_code_top_down(text: str) -> str:
    lines = text.splitlines()
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


def _get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
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


def _get_function_dependency(
    entrypoint: str, call_graph: Dict[str, Set[str]]
) -> Set[str]:
    visited = set()
    to_visit = [entrypoint]
    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)
    return visited


def _get_definition_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    elif isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


def sanitize(text: str, entrypoint: Optional[str] = None) -> str:
    text = _refine_text(text)
    code = _extract_valid_code_top_down(text)
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
            definitions[node.name] = ("function", node)
        elif isinstance(node, ast.Assign):
            name = _get_definition_name(node)
            if name:
                definitions[name] = ("variable", node)

    if entrypoint:
        name2deps = _get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = _get_function_dependency(entrypoint, name2deps)
    else:
        reachable = set(definitions.keys())

    sanitized_output = []
    for node in imports:
        sanitized_output.append(ast.unparse(node))

    for name, (_, node) in definitions.items():
        if name in reachable:
            sanitized_output.append(ast.unparse(node))

    return "\n".join(sanitized_output)


# ============================================================================

try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def clean_markdown_for_humaneval(text: str) -> str:
    """专门为HumanEval任务优化的清理函数"""
    # 首先清理特殊 token
    text = _clean_special_tokens(text)
    text = text.strip()

    # 优先匹配python代码块
    python_pattern = r"```python\s*\n(.*?)```"
    match = re.search(python_pattern, text, re.DOTALL)
    if match:
        return match.group(1).rstrip()

    # 其次匹配通用代码块
    generic_pattern = r"```\s*\n(.*?)```"
    match = re.search(generic_pattern, text, re.DOTALL)
    if match:
        return match.group(1).rstrip()

    # 如果没有代码块，返回原文本
    return text


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]

    processed_predictions = []
    for preds in predictions:
        processed_preds = []
        for p in preds:
            # 使用正确的清理函数
            cleaned = clean_markdown_for_humaneval(p)
            processed_preds.append(cleaned)
        processed_predictions.append(processed_preds)

    res = compute_.compute(
        references=references,
        predictions=processed_predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + _clean_trailing_comments(_clean_special_tokens(r)) for r in resp]
        for resp, doc in zip(resps, docs)
    ]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            sanitize(
                doc["prompt"] + "\n" + r.split("```python\n", 1)[-1].split("```")[0],
                doc["entry_point"],
            )
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
