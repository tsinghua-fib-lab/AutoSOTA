import ast

# CoT-SC code snippet from Figure 7a of the APPL paper
CODE_SNIPPET = """
@ppl
def cot_consistency(cot_examples, question, num_trials):
    cot_examples
    question
    return marginalize([gen() for _ in range(num_trials)])
"""

tree = ast.parse(CODE_SNIPPET.strip())
nodes = list(ast.walk(tree))
ast_size = len(nodes)
print(f"ast_size: {ast_size}")
