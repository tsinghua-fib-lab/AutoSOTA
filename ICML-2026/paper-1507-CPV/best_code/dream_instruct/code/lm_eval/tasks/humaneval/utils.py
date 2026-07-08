import ast
import re
from typing import Dict, List, Optional, Set, Tuple

import evaluate as hf_evaluate


# ============================================================================
# Local sanitize implementation (替代 lm_eval.tasks.humaneval.sanitize_utils)
# ============================================================================
def _refine_text(text: str) -> str:
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


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]

    processed_predictions = []
    for preds in predictions:
        processed_preds = []
        for p in preds:
            processed_preds.append(p.strip("```")[0] if "```" in p else p)
        processed_predictions.append(processed_preds)

    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


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
