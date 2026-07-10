"""
Arena-Hard judgment prompts and configuration
Based on arena-hard-auto repository
"""

# Original Arena-Hard system prompt
OG_ARENA_HARD_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""

# Creative writing system prompt (without self-generation requirement)
CREATIVE_WRITING_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

When evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""

# Judge settings for different categories (baseline models and system prompts)
JUDGE_SETTINGS = {
    "hard_prompt": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "coding": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "math": {
        "baseline": "o3-mini-2025-01-31",
        "system_prompt": OG_ARENA_HARD_PROMPT,
    },
    "creative_writing": {
        "baseline": "gemini-2.0-flash-001",
        "system_prompt": CREATIVE_WRITING_PROMPT,
    },
}

# Prompt template for pairwise comparison
PROMPT_TEMPLATE = """<|User Prompt|>
{QUESTION}

<|The Start of Assistant A's Answer|>
{ANSWER_A}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{ANSWER_B}
<|The End of Assistant B's Answer|>"""

# Regex patterns to extract judgment labels
REGEX_PATTERNS = [
    r'\[\[([AB<>=]+)\]\]',  # [[A>>B]]
    r'\[([AB<>=]+)\]',       # [A>>B]
]
