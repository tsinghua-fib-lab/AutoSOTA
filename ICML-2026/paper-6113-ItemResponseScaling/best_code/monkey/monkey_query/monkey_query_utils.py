import pickle
import re
import string
from huggingface_hub import snapshot_download
from typing import Dict, List, Optional
from datasets import load_dataset
SEED = 42

model_nickname2helm_model_name = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "meta/llama-3-8b",
    "meta-llama/Meta-Llama-3-70B-Instruct": "meta/llama-3-8b",
    "meta-llama/Meta-Llama-3-70B": "meta/llama-3-8b",
    "EleutherAI/pythia-6.9b": "eleutherai/pythia-6.9b",
    "EleutherAI/pythia-12b": "eleutherai/pythia-6.9b", # 12b and 6.9b have same context length
    "mistralai/Mistral-7B-v0.1": "mistralai/mistral-7b-v0.1",
    "Qwen/Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B": "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B": "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-V2-Lite-Chat": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # 163840 -> 131072 -> 128000
    "google/gemma-3-27b-it": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

def create_prompts_and_answers(model_nickname, dataset, num_prompts_to_use):
    cache_dir = snapshot_download(
        repo_id="stair-lab/monkey_query_pre", 
        repo_type="dataset",
    )
    helm_model_name = model_nickname2helm_model_name[model_nickname]
    file_path = f"{cache_dir}/{helm_model_name.replace('/', '_')}_{dataset}_pre_query.pkl"
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
        
    sampled_df = df.sample(n=num_prompts_to_use, random_state=SEED).reset_index(drop=True)
    data = {
        "problems":          sampled_df["instance.input.text"].tolist(),
        "problems_indices":  sampled_df["prompt_index"].tolist(),
        "prompts":           sampled_df["request.prompt"].tolist(),
        "solutions":         sampled_df["solution"].tolist()
    }
    return data

def create_prompts_and_answers_zero_shot(dataset: str, num_prompts_to_use: int):
    if dataset != "gsm":
        raise ValueError(f"Unsupported dataset '{dataset}'; only 'gsm' is supported.")

    platinum_dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
    filtered = platinum_dataset.filter(
        lambda item: item["cleaning_status"] in ["consensus", "verified"]
    )
    all_indices = list(range(len(filtered)))
    
    # Slice to the first num_prompts_to_use entries (keeping original indices)
    selected_indices = all_indices[:num_prompts_to_use]
    subset = filtered.select(selected_indices)
    problems = subset["question"]
    base_prompt = (
        "You are an expert in math. When given a question:\n"
        "- **Show your work.** Write calculations concisely.\n"
        "- **End with “The answer is X”.** Replace X with the final numeric answer.\n"
        "- **Do not include any currency symbols (e.g. “$”) or time suffixes (e.g. “:00 AM”/“:00 PM”) in X.** X must be purely numeric.\n"
        "- **Do not include commas in X.** Use “70000” instead of “70,000”.\n"
        "- **Do not output X with decimal places.** Do not use formats like 18.0 or 18.00; use “18” instead.\n"
        "- **Do not add any content after “The answer is X”.**\n\n"
        "Now solve this problem: "
    )
    prompts = [f"{base_prompt}{q}" for q in problems]
    solutions = subset["answer"]
    
    return {
        "problems": problems,
        "problems_indices": selected_indices,
        "prompts": prompts,
        "solutions": solutions,
    }

## exact_match
def exact_match(response: str, solution: str) -> float:
    if not response:
        return 0
    return 1 if solution.strip() == response.strip() else 0

## quasi_exact_match
def normalize_text(text: str, should_remove_articles: bool = True) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    normalized_text = remove_punc(lower(text))
    if should_remove_articles:
        normalized_text = remove_articles(normalized_text)
    return white_space_fix(normalized_text)

def quasi_exact_match(solution: str, response: str) -> float:
    if not response:
        return 0
    return 1 if normalize_text(solution) == normalize_text(response) else 0

# gsm final_number_exact_match
def final_number_exact_match(gold: str, pred: str) -> float:
    """
    Returns 1 iff the final number in gold and pred match.
    Similar to exact_match_indicator.
    Example:
    - gold = "The answer is 15."
    - pred = "The answer is 15 eggs."
    - Returns 1
    """

    def get_final_number(x: str) -> str:
        matches = re.findall(r"-?[\d,]+(?:.\d+)?", x)
        if not matches:
            return ""
        return matches[-1].replace(",", "")

    return exact_match(get_final_number(gold), get_final_number(pred))

def advanced_gsm_eval(gold: str, pred: str) -> float:
    """
    Returns 1 iff the final number in gold and the number in the last
    "The answer is X." in pred match.
    Example:
    - gold = "#### 15."
    - pred = "Calculation… The answer is 15."
    - Returns 1
    """

    def get_final_number(x: str) -> str:
        matches = re.findall(r"-?[\d,]+(?:.\d+)?", x)
        if not matches:
            return ""
        return matches[-1].replace(",", "")

    def get_pred_number(x: str) -> str:
        # Extract the number X from the last "The answer is X."
        matches = re.findall(r"The answer is (-?[\d,]+(?:.\d+)?)", x)
        return matches[-1].replace(",", "") if matches else ""

    gold_num = get_final_number(gold)
    pred_num = get_pred_number(pred)
    return 1.0 if exact_match(gold_num, pred_num) else 0.0


## is_equiv_chain_of_thought
def last_boxed_only_string(string: str) -> Optional[str]:
    """Source: https://github.com/hendrycks/math

    Extract the last \\boxed{...} or \\fbox{...} element from a string.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def get_answer(solution: Optional[str]) -> Optional[str]:
    if solution is None:
        return None
    last_boxed = last_boxed_only_string(solution)
    if last_boxed is None:
        return None
    answer = remove_boxed(last_boxed)
    if answer is None:
        return None
    return answer

def remove_boxed(string: str) -> Optional[str]:
    """Source: https://github.com/hendrycks/math

    Extract the text within a \\boxed{...} environment.

    Example:
    >>> remove_boxed(\\boxed{\\frac{2}{3}})
    \\frac{2}{3}
    """
    left = "\\boxed{"
    try:
        assert string[: len(left)] == left
        assert string[-1] == "}"
        return string[len(left) : -1]
    except Exception:
        return None

def is_equiv(str1: Optional[str], str2: Optional[str]) -> float:
    """Returns (as a float) whether two strings containing math are equivalent up to differences of formatting in
    - units
    - fractions
    - square roots
    - superfluous LaTeX.

    Source: https://github.com/hendrycks/math
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return 1.0
    if str1 is None or str2 is None:
        return 0.0

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        return float(ss1 == ss2)
    except Exception:
        return float(str1 == str2)


def _fix_fracs(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat fractions.

    Examples:
    >>> _fix_fracs("\\frac1b")
    \frac{1}{b}
    >>> _fix_fracs("\\frac12")
    \frac{1}{2}
    >>> _fix_fracs("\\frac1{72}")
    \frac{1}{72}
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat fractions formatted as a/b to \\frac{a}{b}.

    Example:
    >>> _fix_a_slash_b("2/3")
    \frac{2}{3}
    """
    if len(string.split("/")) != 2:
        return string
    a_str = string.split("/")[0]
    b_str = string.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Remove units (on the right).
    "\\text{ " only ever occurs (at least in the val set) when describing units.
    """
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Reformat square roots.

    Example:
    >>> _fix_sqrt("\\sqrt3")
    \sqrt{3}
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string: str) -> str:
    """Source: https://github.com/hendrycks/math

    Apply the reformatting helper functions above.
    """
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv_chain_of_thought(str1: str, str2: str) -> float:
    """Strips the solution first before calling `is_equiv`."""
    ans1 = get_answer(str1)
    ans2 = get_answer(str2)

    return is_equiv(ans1, ans2)