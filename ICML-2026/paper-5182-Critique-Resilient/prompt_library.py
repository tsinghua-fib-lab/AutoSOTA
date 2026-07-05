from pathlib import Path
from typing import List, Optional


def read_guidance(path: str) -> str:
    return Path(path).read_text().strip()


def load_question_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "question_quality.md"))


def load_answer_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "answer_quality.md"))


def load_critique_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "critique_quality.md"))


def load_judgment_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "judgment_quality.md"))


def load_self_critique_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "self_critique_quality.md"))


def load_debate_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "debate_quality.md"))


def load_debate_illposed_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "debate_illposed_quality.md"))


def load_debate_critique_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "debate_critique_quality.md"))


def load_judgment_illposed_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "judgment_illposed_quality.md"))


def load_judgment_critique_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "judgment_critique_quality.md"))


def build_question_prompt(
    topic: str,
    guidance_text: str,
    previous_questions: Optional[List[str]],
    previous_context: Optional[str] = None,
) -> str:
    """
    Prompt for generating a challenging mathematics question with solution.
    This implements the tester/questioner role in the benchmarking framework.
    """
    extra = ""
    if previous_questions:
        attempts = "\n".join(f"- {q}" for q in previous_questions)
        context = previous_context or (
            "The following questions on this topic failed the self-solve gate or meaningfulness check.\n"
            "Generate a materially different, well-posed replacement that avoids these issues:"
        )
        extra = (
            "\n\n**Previous Attempts:**\n"
            f"{context}\n"
            f"{attempts}\n"
        )
    return (
        "# Task: Generate a Challenging Mathematics Problem\n\n"
        "You are acting as a **tester** in a benchmarking framework. Your goal is to create a single, "
        "challenging but solvable mathematics problem along with a complete, verifiable solution.\n\n"
        "## Quality Rubric\n\n"
        f"{guidance_text}\n\n"
        "## Topic\n\n"
        f"Generate a problem in the following domain: **{topic}**\n"
        f"{extra}\n"
        "## Output Format\n\n"
        "Use exactly this structure:\n\n"
        "[QUESTION]\n"
        "<Your problem statement here>\n\n"
        "[ANSWER]\n"
        "<Your complete solution here>\n\n"
        "**Important**: Your solution will be verified first (self-solve gate). If it fails verification, "
        "the question will be rejected without being used to test other models.\n"
        "Use standard LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$."
    )


def build_answer_prompt(question: str, guidance_text: str) -> str:
    """
    Prompt for answering a mathematics question.
    This implements the testee/answerer role in the benchmarking framework.
    """
    return (
        "# Task: Solve the Mathematics Question\n\n"
        "You are acting as a **testee** in a benchmarking framework. Provide a complete, rigorous answer "
        "to the question below.\n\n"
        "## Answer Quality Requirements\n\n"
        f"{guidance_text}\n\n"
        "## The Question\n\n"
        f"{question}\n\n"
        "## Your Response\n\n"
        "Provide your complete answer below. Use standard LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$.\n"
        "If the question is ill-posed, explicitly state this and explain why rather than attempting to answer."
    )


def build_self_check_prompt(
    question: str,
    answer: str,
    self_critique_guidance: str,
    answer_guidance: str,
) -> str:
    """
    Prompt for self-critique of an answer during the self-improvement loop.
    """
    answer_section = (
        "## Answer Quality Requirements\n\n"
        f"{answer_guidance}\n\n"
    )
    return (
        "# Task: Evaluate Your Own Answer\n\n"
        "Review the answer you provided to the question below and assess whether it meets the quality standards.\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "## Your Answer\n\n"
        f"{answer}\n\n"
        f"{answer_section}"
        "## Evaluation Rubric\n\n"
        f"{self_critique_guidance}\n\n"
        "## Required Output Format\n\n"
        "Return ONLY a JSON object that matches this JSON Schema (no additional text):\n\n"
        "```json\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["verdict", "ill_posed", "issues", "improvements"],\n'
        '  "properties": {\n'
        '    "verdict": { "type": "string", "enum": ["pass", "fail"] },\n'
        '    "ill_posed": { "type": "boolean" },\n'
        '    "issues": { "type": "array", "items": { "type": "string" } },\n'
        '    "improvements": { "type": "string" }\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "**Field Descriptions:**\n"
        '- `verdict`: "pass" if the answer is correct and complete, "fail" otherwise\n'
        '- `ill_posed`: true if the question itself is unanswerable as stated, false otherwise\n'
        '- `issues`: List of specific problems with the answer (empty list if none)\n'
        '- `improvements`: Short, concrete guidance on how to fix the answer (empty string if verdict is "pass")\n\n'
        'Use standard LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$.'
    )


def build_refine_prompt(question: str, answer: str, feedback: str, guidance: str) -> str:
    """
    Prompt for refining an answer based on self-critique feedback.
    """
    return (
        "# Task: Improve Your Answer\n\n"
        "Revise your answer to address the issues identified in the feedback below.\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "## Current Answer\n\n"
        f"{answer}\n\n"
        "## Feedback\n\n"
        f"{feedback}\n\n"
        "## Quality Standards\n\n"
        f"{guidance}\n\n"
        "## Your Revised Answer\n\n"
        "Provide only the improved answer below (no meta-commentary). "
        "Use standard LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$."
    )


def build_critique_prompt(question: str, author: str, answer: str, guidance: str) -> str:
    """
    Prompt for critiquing another model's answer.
    This implements the verification/critique role in the benchmarking framework.
    """
    return (
        "# Task: Verify Answer Correctness\n\n"
        "You are evaluating an answer provided by a model. Your role is **verification**, "
        "not independent solution generation.\n\n"
        "## Critique Guidelines\n\n"
        f"{guidance}\n\n"
        f"## Question\n\n"
        f"{question}\n\n"
        "## Answer to Evaluate\n\n"
        f"{answer}\n\n"
        "## Required Output Format\n\n"
        "Return ONLY a JSON object that matches this JSON Schema (no additional text):\n\n"
        "```json\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["verdict", "notes"],\n'
        '  "properties": {\n'
        '    "verdict": {\n'
        '      "type": "string",\n'
        '      "enum": ["correct", "incorrect", "insufficient", "obscure"]\n'
        "    },\n"
        '    "notes": { "type": "string" },\n'
        '    "suggestions": { "type": "string" }\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "**Verdict Definitions:**\n"
        '- `"correct"`: Mathematically sound, complete, and adequately rigorous\n'
        '- `"incorrect"`: Contains errors, invalid reasoning, or fundamental flaws\n'
        '- `"insufficient"`: Partially correct but meaningfully incomplete\n'
        '- `"obscure"`: Too unclear to verify\n\n'
        'Use standard LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$.\n\n'
    )


def build_critique_self_check(question: str, answer: str, critique: str, guidance: str) -> str:
    """
    Prompt for self-critique of a critique during refinement.
    """
    return (
        "# Task: Review Your Critique for Accuracy\n\n"
        "Assess whether your critique correctly identifies issues (if any) and provides accurate, evidence-based reasoning.\n"
        "Also verify that your critique follows the required critique JSON format.\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "## Answer Being Critiqued\n\n"
        f"{answer}\n\n"
        "## Your Critique\n\n"
        f"{critique}\n\n"
        "## Evaluation Standards\n\n"
        f"{guidance}\n\n"
        "## Critique Format Check\n\n"
        "Your critique must be valid JSON with `verdict` in {correct, incorrect, insufficient, obscure}, "
        "`notes` as a string, and optional `suggestions`. If the format is invalid, set verdict to \"fail\" "
        "and list the formatting problems in `issues`.\n\n"
        "## Required Output Format\n\n"
        "Return ONLY a JSON object that matches this JSON Schema (no additional text):\n\n"
        "```json\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["verdict", "issues", "improvements"],\n'
        '  "properties": {\n'
        '    "verdict": { "type": "string", "enum": ["pass", "fail"] },\n'
        '    "issues": { "type": "array", "items": { "type": "string" } },\n'
        '    "improvements": { "type": "string" }\n'
        "  }\n"
        "}\n"
        "```\n\n"
        '- `verdict`: "pass" if your critique is accurate and well-justified, "fail" if it needs revision\n'
        '- `issues`: Specific problems with your critique (e.g., incorrect claims, missing evidence)\n'
        "- `improvements`: Guidance on how to make the critique more accurate"
        "\n\n"
        "Use standard LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$."
    )


def build_critique_refine(question: str, answer: str, critique: str, feedback: str) -> str:
    """
    Prompt for refining a critique based on self-check feedback.
    """
    return (
        "# Task: Improve Your Critique\n\n"
        "Revise your critique to address the issues identified below.\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "## Answer Being Critiqued\n\n"
        f"{answer}\n\n"
        "## Your Current Critique\n\n"
        f"{critique}\n\n"
        "## Feedback on Your Critique\n\n"
        f"{feedback}\n\n"
        "## Required Output Format\n\n"
        "Return ONLY a JSON object that matches this JSON Schema (no additional text):\n\n"
        "```json\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["verdict", "notes"],\n'
        '  "properties": {\n'
        '    "verdict": {\n'
        '      "type": "string",\n'
        '      "enum": ["correct", "incorrect", "insufficient", "obscure"]\n'
        "    },\n"
        '    "notes": { "type": "string" },\n'
        '    "suggestions": { "type": "string" }\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "**Verdict Definitions:**\n"
        '- `"correct"`: Mathematically sound, complete, and adequately rigorous\n'
        '- `"incorrect"`: Contains errors, invalid reasoning, or fundamental flaws\n'
        '- `"insufficient"`: Partially correct but meaningfully incomplete\n'
        '- `"obscure"`: Too unclear to verify\n\n'
        "Provide your improved critique using the JSON format above.\n"
        "Use standard LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$."
    )
