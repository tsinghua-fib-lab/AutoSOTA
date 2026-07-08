import json
import re
import os
import glob
import argparse


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
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


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left) :]
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except (AssertionError, IndexError):
        return s


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
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


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def fix_a_slash_b(string):
    if string is None:
        return None
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a_num = float(a)
        b_num = float(b)
        new_string = "\\frac{" + a + "}{" + b + "}"
        return new_string
    except (ValueError, TypeError):
        return string


def strip_string(string):
    if string is None:
        return None
    string = str(string).strip()
    while re.search(r"(\d),(\d{3})", string):
        string = re.sub(r"(\d),(\d{3})", r"\1\2", string)
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("\\%", "")
    string = remove_right_units(string)
    if string.startswith("."):
        string = "0" + string
    if " ." in string:
        string = string.replace(" .", " 0.")
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1].strip()
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None or str2 is None:
        return False
    str1_clean = strip_string(str1)
    str2_clean = strip_string(str2)
    if isinstance(str1_clean, (int, float)) and isinstance(str2_clean, (int, float)):
        return abs(str1_clean - str2_clean) < 1e-6
    try:
        if abs(float(str1_clean) - float(str2_clean)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(f"Comparing normalized strings: '{ss1}' vs '{ss2}'")
        return ss1 == ss2
    except Exception:
        return str(str1).strip() == str(str2).strip()


def parse_math500_answers_from_jsonl(json_path):
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    total_correct = 0
    total_processed = 0
    processed_items = []

    for item in data:
        total_processed += 1
        question = item.get("doc", {}).get("question", "")
        ground_truth_str = str(item.get("target"))
        raw_generation = ""
        resps = item.get("resps")
        if resps and isinstance(resps, list) and len(resps) > 0 and isinstance(resps[0], list) and len(resps[0]) > 0:
            raw_generation = resps[0][0]

        extracted_answer_str = None
        boxed_str = last_boxed_only_string(raw_generation)
        if boxed_str:
            extracted_answer_str = remove_boxed(boxed_str)

        if extracted_answer_str is None:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                extracted_answer_str = answer_match.group(1).strip()

        is_correct = is_equiv(extracted_answer_str, ground_truth_str)
        if is_correct:
            total_correct += 1

        processed_items.append({
            "question": question,
            "raw_generation": raw_generation,
            "extracted_answer": extracted_answer_str,
            "ground_truth": ground_truth_str,
            "is_correct": is_correct,
        })

    return (
        total_correct,
        total_processed,
        processed_items,
    )


def evaluate_math500_results(directory):
    print("\n" + "="*50 + f"\nProcessing directory: {directory}\n" + "="*50)

    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in directory '{directory}'.")
        return

    agg_results = {
        "correct": 0, "processed": 0,
        "all_items": []
    }

    print(f"Found {len(jsonl_files)} files to process...")
    for file_path in jsonl_files:
        print(f"  -> Processing file: {os.path.basename(file_path)}")
        try:
            (
                correct, processed, detailed_results
            ) = parse_math500_answers_from_jsonl(json_path=file_path)

            agg_results["correct"] += correct
            agg_results["processed"] += processed
            agg_results["all_items"].extend(detailed_results)
        except Exception as e:
            print(f"    Error processing file '{os.path.basename(file_path)}': {e}")
            continue

    total_processed = agg_results["processed"]
    if total_processed > 0:
        accuracy = (agg_results["correct"] / total_processed) * 100
    else:
        print("No valid data processed. Cannot calculate results.")
        return

    print("\n" + "-" * 80)
    print(f"Results for '{os.path.basename(directory)}'")
    print(f"  - Accuracy:               {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--res_path",
        type=str,
        required=True,
        help="Path to the directory containing result .jsonl files"
    )
    args = parser.parse_args()

    evaluate_math500_results(directory=args.res_path)
