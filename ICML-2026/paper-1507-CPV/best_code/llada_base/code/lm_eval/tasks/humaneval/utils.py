import evaluate as hf_evaluate
from lm_eval.tasks.humaneval.sanitize_utils import sanitize
import re

try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


# def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
#     global compute_
#     assert k is not None
#     if isinstance(k, int):
#         k = [k]
#     res = compute_.compute(
#         references=references,
#         predictions=predictions,
#         k=k,
#     )
#     return res[0]

def clean_markdown_for_humaneval(text: str) -> str:
    """专门为HumanEval任务优化的清理函数"""
    text = text.strip()
    
    # 优先匹配python代码块
    python_pattern = r'```python\s*\n(.*?)```'
    match = re.search(python_pattern, text, re.DOTALL)
    if match:
        return match.group(1).rstrip()
    
    # 其次匹配通用代码块
    generic_pattern = r'```\s*\n(.*?)```'
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
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            sanitize(
                doc["prompt"] + "\n" + r.split('```python\n', 1)[-1].split('```')[0],
                doc["entry_point"]
            )
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]