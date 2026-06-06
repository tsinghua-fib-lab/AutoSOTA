# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

def extract_boxed_answer(text):
    def last_boxed_only_string(text):
        idx = text.rfind("\\boxed")
        if idx < 0:
            idx = text.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(text):
            if text[i] == "{":
                num_left_braces_open += 1
            if text[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is None:
            return None
        return text[idx : right_brace_idx + 1]

    def remove_boxed(boxed):
        left = "\\boxed{"
        try:
            assert boxed[: len(left)] == left
            assert boxed[-1] == "}"
            length = len(left)
            return boxed[length:-1]
        except Exception:
            return None

    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    answer = remove_boxed(boxed)
    return answer

def normalize_answer(answer):
    answer = answer.replace("dfrac", "frac")
    answer = answer.replace("cffrac", "frac")
    match = re.search(r"(.*?)Problem:", answer, flags=re.S)
    if match:
        answer = match.group(1)
    subs = [("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""), ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}"), ("\\le", "<")]
    remove = ["square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours", "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots", "\n", "\r", "\f", "\%"]
    sub_patterns = [r"(\\text\{)(.*?)(\})", r"(\\textbf\{)(.*?)(\})", r"(\\overline\{)(.*?)(\})", r"(\\boxed\{)(.*)(\})"]
    split_patterns = [r"finalansweris(.*)", r"answer?is:?(.*)", r"oxed\{(.*?)\}", r"\$(.*?)\$"]
    for before, after in subs:
        answer = answer.replace(before, after)
    for expr in remove:
        answer = answer.replace(expr, "")
    for pattern in sub_patterns:
        answer = re.sub(pattern, "\\2", answer)
    for pattern in split_patterns:
        if len(re.findall(pattern, answer)) > 0:
            answer = re.findall(pattern, answer)[-1]
    answer = answer.strip()
    if "rac" in answer and "\\frac" not in answer:
        answer = answer.replace("rac", "\\frac")
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer
