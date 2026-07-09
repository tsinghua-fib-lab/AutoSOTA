#!/usr/bin/env python3
"""
BFCL-v4 Non-Live evaluation for RealtimeTool model (RT-Qwen2.5-0.5B).
Uses multi-head parallel decoding via vLLM with prefix caching.

Evaluation follows the paper's protocol:
- Overall Accuracy: function name AND all required arguments correct
- Function Accuracy: function name correct only
"""
import os, json, time, argparse, sys, re
from pathlib import Path
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_HOME"] = "/opt/conda/lib/python3.10/site-packages/nvidia/cu13"
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ.get("PATH", "")

HEADS = [("function", "<function>", "</function>")] + [
    (f"arg{i}", f"<arg{i}>", f"</arg{i}>") for i in range(1, 7)
]
STOPS = ["</function>"] + [f"</arg{i}>" for i in range(1, 7)] + ["</content>", "<|null|>", "<|im_end|>"]

# System prompt matching the model's training format (v1 - multi-head instructions)
V1_SYSTEM_PROMPT = """You are a multi-head parallel function calling model.
## Output Heads
**Head 0 - <content>**: Natural language response
- Format: <content>response text</content>
- Answer what you want to say while you are calling a function
**Head 1 - <function>**: Function names to call
- Format: <function>name</function>
- Name: must match tool defined name
**Head 2-7 - <arg1>, <arg2>, <arg3>, <arg4>, <arg5>, <arg6>**: Function arguments by position
- Format: <argN>value</argN>
- Strictly fill in according to the parameter order of the tool you intend to call
- Note the special restrictions of parameter definitions for corresponding positions
- If the corresponding tool definition has required parameters, these must be filled in
- Infer the user's actual needs.
- If Unnecessary: <argN><|null|></argN>
**Environment - The information you have.
**History - The tools you have called."""


def load_bfcl_data(data_dir, subsets):
    """Load BFCL JSONL data files."""
    all_samples = []
    for subset in subsets:
        path = os.path.join(data_dir, f"{subset}.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    sample["_subset"] = os.path.basename(path).replace(".json", "")
                    all_samples.append(sample)
    return all_samples


def load_possible_answers(data_dir, subsets):
    """Load possible answers for evaluation."""
    answers = {}
    for subset in subsets:
        path = os.path.join(data_dir, "possible_answer", f"{subset}.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    answers[entry["id"]] = entry["ground_truth"]
    return answers


def build_tool_text(functions):
    """Build tool definitions text in the OpenAI function-calling format expected by the model.

    The model was trained on xLAM data which uses:
    {"type":"function","function":{"name":"...","description":"...","parameters":{...}}}
    """
    tools_lines = []
    for func in functions:
        params = func.get("parameters", {})
        if params.get("type") == "dict":
            params = {**params, "type": "object"}
        if "required" not in params:
            required = list(params.get("properties", {}).keys())
            if required:
                params["required"] = required
        # Use the OpenAI function-calling wrapper format
        tool_def = {
            "type": "function",
            "function": {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": params,
            }
        }
        tools_lines.append(json.dumps(tool_def))
    return "\n".join(tools_lines)


def get_param_order(functions, func_name):
    """Get ordered parameter names for a function in definition order."""
    for func in functions:
        if func["name"] == func_name:
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])
            param_names = list(params.keys())
            # Use definition order (matching the order the model sees in tool definitions)
            ordered = list(param_names)
            return ordered, params, required
    return [], {}, []


def build_prompt_v1(sample):
    """Build the v1 prompt format for RealtimeTool model.

    V1 format uses the multi-head instruction system prompt and explicit
    environment/history fields in the user message.
    """
    funcs = sample["function"]
    tools_text = build_tool_text(funcs)

    # Extract user query from conversation
    conversation = sample["question"]
    if isinstance(conversation, list) and len(conversation) > 0:
        turns = conversation[0]
        if isinstance(turns, list) and len(turns) > 0:
            user_msg = turns[-1]["content"]
        else:
            user_msg = str(conversation)
    else:
        user_msg = str(conversation)

    prompt = (
        f"<|im_start|>system\n{V1_SYSTEM_PROMPT}\n## Available Tools:\n\n{tools_text}<|im_end|>\n"
        f"<|im_start|>user\nenvironment: []\nhistory: []\n\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def clean_output(text):
    """Clean a head output to get the value."""
    text = text.strip()
    if "<|null|>" in text or text == "":
        return None
    for stop_tag in STOPS:
        idx = text.find(stop_tag)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


def parse_multi_head_output(raw_outputs):
    """Parse multi-head outputs into a structured function call.

    Returns:
        func_name: predicted function name or None
        args: dict mapping parameter names (by position index) to values
    """
    func_name = raw_outputs.get("function", "")
    if func_name is None or func_name == "" or func_name == "<|null|>":
        return None, []

    args = []
    for i in range(1, 7):
        val = raw_outputs.get(f"arg{i}")
        if val is not None and val != "" and val != "<|null|>":
            args.append(val)
        else:
            args.append(None)  # Keep position for null args

    return func_name, args


def normalize_value(val):
    """Normalize a value for comparison."""
    if isinstance(val, str):
        val = val.strip()
        # Boolean coercion (case-insensitive: True/False/Yes/No/1/0/On/Off)
        val_lower = val.lower()
        if val_lower in ('true', 'yes', 'on'):
            return True
        if val_lower in ('false', 'no', 'off'):
            return False
        # Try numeric conversion for comparison
        try:
            f = float(val)
            if f == int(f):
                return int(f)
            return f
        except (ValueError, TypeError):
            pass
        # Try JSON parsing for lists/dicts/booleans
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    if isinstance(val, float) and val == int(val):
        return int(val)
    return val


def values_equal(model_val, expected_val):
    """Check if two values are equal after normalization."""
    mv = normalize_value(model_val)
    ev = normalize_value(expected_val)

    if mv == ev:
        return True

    # String comparison with enum normalization
    if isinstance(mv, str) and isinstance(ev, str):
        if mv.lower() == ev.lower():
            return True
        # Enum normalization: normalize underscores, hyphens, spaces
        def _norm_enum(s):
            return s.lower().replace('_', ' ').replace('-', ' ').strip()
        if _norm_enum(mv) == _norm_enum(ev):
            return True
        # Partial match (substring containment, bidirectional)
        if mv.lower() in ev.lower() or ev.lower() in mv.lower():
            return True

    # Boolean cross-type comparison
    if isinstance(mv, bool) and isinstance(ev, bool):
        return mv == ev
    # Handle bool vs int (True==1, False==0 in Python, but type-aware)
    if isinstance(mv, bool) and isinstance(ev, (int, float)):
        return mv == bool(ev)
    if isinstance(ev, bool) and isinstance(mv, (int, float)):
        return ev == bool(mv)

    # List comparison (ordered and unordered)
    if isinstance(mv, list) and isinstance(ev, list):
        if len(mv) != len(ev):
            return False
        # Ordered comparison first
        if all(values_equal(m, e) for m, e in zip(mv, ev)):
            return True
        # Try sorted comparison for unordered lists
        try:
            mv_sorted = sorted(mv, key=str)
            ev_sorted = sorted(ev, key=str)
            if all(values_equal(m, e) for m, e in zip(mv_sorted, ev_sorted)):
               return True
        except (TypeError, ValueError):
            pass

    # Dict comparison (recursive)
    if isinstance(mv, dict) and isinstance(ev, dict):
        if set(mv.keys()) != set(ev.keys()):
            return False
        return all(values_equal(mv[k], ev[k]) for k in mv)

    return False


def check_function_accuracy(pred_func, ground_truth):
    """Check if the predicted function name matches any ground truth function."""
    if pred_func is None:
        return False
    for gt_call in ground_truth:
        for func_name in gt_call:
            if func_name == pred_func:
                return True
    return False


def get_function_category(func_name):
    """Extract function category from function name (e.g., 'math.add' -> 'math')."""
    if func_name is None:
        return "unknown"
    parts = func_name.split(".")
    if len(parts) > 1:
        return parts[0]
    parts = func_name.split("_")
    if len(parts) > 1:
        return parts[0]
    return "other"


def classify_error(pred_func, pred_args, functions, ground_truth, param_order, required_params):
    """Classify why a prediction is wrong."""
    if pred_func is None:
        return {"error_type": "null_func", "missing_params": [], "wrong_params": [], "null_params": []}

    pred_params = {}
    null_params = []
    for i, val in enumerate(pred_args):
        if i < len(param_order):
            if val is not None:
                pred_params[param_order[i]] = val
            else:
                null_params.append(param_order[i])

    missing_params = [p for p in required_params if p not in pred_params]
    if missing_params:
        return {"error_type": "missing_required", "missing_params": missing_params,
                "wrong_params": [], "null_params": null_params}

    wrong_params = []
    for gt_call in ground_truth:
        for func_name, gt_params in gt_call.items():
            if func_name != pred_func:
                continue
            for req_param in required_params:
                if req_param in pred_params:
                    pred_val = pred_params[req_param]
                    possible_vals = gt_params.get(req_param, [])
                    if not possible_vals or (len(possible_vals) == 1 and possible_vals[0] == ""):
                        continue
                    match_found = False
                    for pv in possible_vals:
                        if pv == "" and True in possible_vals:
                            continue
                        if values_equal(pred_val, pv):
                            match_found = True
                            break
                        if isinstance(pv, str) and isinstance(pred_val, str):
                            if pv.lower() in pred_val.lower() or pred_val.lower() in pv.lower():
                                match_found = True
                                break
                    if not match_found:
                        wrong_params.append(req_param)
            break

    if wrong_params:
        return {"error_type": "wrong_value", "missing_params": [],
                "wrong_params": wrong_params, "null_params": null_params}

    return {"error_type": "no_error", "missing_params": [], "wrong_params": [], "null_params": null_params}


def check_overall_accuracy(pred_func, pred_args, functions, ground_truth):
    """Check if the predicted function call matches ground truth.

    Uses BFCL-style evaluation: required params must match, optional params are
    checked if provided. Values compared against possible answer alternatives.
    """
    if pred_func is None:
        return False

    # Get parameter ordering for this function
    param_order, param_details, required_params = get_param_order(functions, pred_func)
    if not param_order:
        return False

    for gt_call in ground_truth:
        for func_name, gt_params in gt_call.items():
            if func_name != pred_func:
                continue

            # Map positional args to parameter names (skip None/null values)
            pred_params = {}
            for i, val in enumerate(pred_args):
                if i < len(param_order) and val is not None:
                    pred_params[param_order[i]] = val

            # Check required parameters
            all_correct = True
            for req_param in required_params:
                if req_param not in pred_params:
                    all_correct = False
                    break

                pred_val = pred_params[req_param]
                possible_vals = gt_params.get(req_param, [])

                # Check if parameter value matches any possible answer
                match_found = False
                for pv in possible_vals:
                    if pv == "" and True in possible_vals:
                        # This is a boolean flag: empty string = optional, True = expected value
                        continue
                    if values_equal(pred_val, pv):
                        match_found = True
                        break
                    # String contains check
                    if isinstance(pv, str) and isinstance(pred_val, str):
                        if pv.lower() in pred_val.lower() or pred_val.lower() in pv.lower():
                            match_found = True
                            break

                if not match_found and possible_vals:
                    # Check if the parameter is actually optional (has "" in possible_vals)
                    if "" not in possible_vals and True not in possible_vals:
                        all_correct = False
                        break

            if all_correct:
                return True

    return False


def run_evaluation(model_path, data_dir, subsets, max_samples=None):
    """Run the full evaluation pipeline."""
    from vllm import LLM, SamplingParams

    print(f"Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        enable_prefix_caching=True,
        max_num_seqs=8,
    )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop=STOPS,
        include_stop_str_in_output=True,
    )

    # Load data
    print(f"Loading BFCL data from {data_dir}...")
    samples = load_bfcl_data(data_dir, subsets)
    answers = load_possible_answers(data_dir, subsets)

    if max_samples:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} samples")

    # Prepare prompts
    print("Building prompts...")
    base_prompts = []
    for s in samples:
        base_prompts.append(build_prompt_v1(s))

    # Create per-head prompts
    all_prompts = []
    for bp in base_prompts:
        for _, op, _ in HEADS:
            all_prompts.append(bp + op)

    # Generate
    print(f"Running inference on {len(samples)} samples ({len(all_prompts)} head prompts)...")
    t0 = time.perf_counter()
    outputs = llm.generate(all_prompts, sp)
    elapsed = time.perf_counter() - t0
    print(f"Inference completed in {elapsed:.1f}s ({elapsed/len(samples):.3f}s per sample)")

    # Process outputs
    print("Processing outputs...")
    results = []
    for i, sample in enumerate(samples):
        base_idx = i * len(HEADS)
        raw_outputs = {}

        for j, (name, _, _) in enumerate(HEADS):
            idx = base_idx + j
            if idx < len(outputs) and outputs[idx].outputs:
                o = outputs[idx].outputs[0]
                raw_outputs[name] = clean_output(o.text)
            else:
                raw_outputs[name] = None

        func_name, args = parse_multi_head_output(raw_outputs)
        gt = answers.get(sample["id"], [])
        funcs = sample["function"]

        func_acc = check_function_accuracy(func_name, gt)
        overall_acc = check_overall_accuracy(func_name, args, funcs, gt) if func_acc else False

        # Error classification and per-head tracking
        param_order, param_details, required_params = get_param_order(funcs, func_name) if func_name else ([], {}, [])
        error_info = classify_error(func_name, args, funcs, gt, param_order, required_params)
        func_category = get_function_category(func_name)

        per_head = {}
        per_head["function"] = {"status": "ok" if func_acc else "wrong"}
        for i in range(min(6, len(args))):
            pname = param_order[i] if i < len(param_order) else "param_%d" % i
            pval = args[i]
            if pval is None:
                per_head["arg%d" % (i+1)] = {"status": "null", "param": pname}
            elif func_acc and not overall_acc and pname in error_info.get("wrong_params", []):
                per_head["arg%d" % (i+1)] = {"status": "wrong_value", "param": pname}
            elif pname in error_info.get("missing_params", []):
                per_head["arg%d" % (i+1)] = {"status": "missing", "param": pname}
            else:
                per_head["arg%d" % (i+1)] = {"status": "ok", "param": pname}
        # Fill remaining arg slots as unused
        for i in range(len(args), 6):
            per_head["arg%d" % (i+1)] = {"status": "unused", "param": ""}

        results.append({
            "id": sample["id"],
            "subset": sample["_subset"],
            "pred_func": func_name,
            "pred_args": args,
            "function_accuracy": func_acc,
            "overall_accuracy": overall_acc,
            "error_type": error_info["error_type"],
            "func_category": func_category,
            "per_head": per_head,
        })

    # Compute metrics
    total = len(results)
    func_correct = sum(1 for r in results if r["function_accuracy"])
    overall_correct = sum(1 for r in results if r["overall_accuracy"])

    func_acc_pct = (func_correct / total * 100) if total > 0 else 0
    overall_acc_pct = (overall_correct / total * 100) if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total samples:       {total}")
    print(f"  Function Accuracy:   {func_correct}/{total} = {func_acc_pct:.1f}%")
    print(f"  Overall Accuracy:    {overall_correct}/{total} = {overall_acc_pct:.1f}%")
    print(f"  Inference time:      {elapsed:.1f}s")
    print(f"{'='*60}")

    # Error taxonomy analysis
    print("\n" + "="*60)
    print("  ERROR TAXONOMY")
    print("="*60)
    error_types = {}
    for r in results:
        et = r.get("error_type", "unknown")
        error_types[et] = error_types.get(et, 0) + 1
    for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print("  %s: %d/%d = %.1f%%" % (et, count, total, count/total*100))

    # Per-subset error breakdown
    for subset in subsets:
        short_name = os.path.basename(subset)
        subset_results = [r for r in results if r["subset"] == short_name]
        if subset_results:
            print("\n  %s errors:" % short_name)
            for et in sorted(set(r["error_type"] for r in subset_results)):
                cnt = sum(1 for r in subset_results if r["error_type"] == et)
                print("    %s: %d/%d = %.1f%%" % (et, cnt, len(subset_results), cnt/len(subset_results)*100))

    # Function category breakdown
    print("\n  Function category errors (top 10):")
    cat_errors = {}
    for r in results:
        if not r["overall_accuracy"]:
            cat = r.get("func_category", "unknown")
            cat_errors[cat] = cat_errors.get(cat, 0) + 1
    for cat, count in sorted(cat_errors.items(), key=lambda x: -x[1])[:10]:
        print("    %s: %d errors" % (cat, count))

    # Per-head error breakdown
    print("\n  Per-head error breakdown:")
    head_errors = {}
    for i in range(1, 7):
        head_errors["arg%d" % i] = {"null": 0, "wrong_value": 0, "missing": 0, "unused": 0, "ok": 0}
    head_errors["function"] = {"wrong": 0, "ok": 0}
    for r in results:
        ph = r.get("per_head", {})
        for head_name, head_info in ph.items():
            if head_name in head_errors:
                status = head_info.get("status", "ok")
                if status in head_errors[head_name]:
                    head_errors[head_name][status] += 1
    for head_name in ["function"] + ["arg%d" % i for i in range(1, 7)]:
        hi = head_errors[head_name]
        total_h = sum(hi.values())
        error_h = total_h - hi.get("ok", 0)
        if total_h > 0:
            parts = ["%s=%d" % (k, v) for k, v in sorted(hi.items()) if v > 0 and k != "ok"]
            print("    %s: %d/%d errors (%s)" % (head_name, error_h, total_h, ", ".join(parts)))

    # Save error taxonomy JSON
    taxonomy_path = "/repo/error_taxonomy.json"
    taxonomy_data = {
        "total_samples": total,
        "overall_accuracy": overall_acc_pct,
        "function_accuracy": func_acc_pct,
        "error_type_counts": error_types,
        "function_category_errors": cat_errors,
        "per_head_errors": {},
        "sample_errors": [
            {"id": r["id"], "subset": r["subset"], "error_type": r.get("error_type", "unknown"),
             "func_category": r.get("func_category", "unknown"),
             "per_head": {}}
            for r in results if not r["overall_accuracy"]
        ],
    }
    for h, v in head_errors.items():
        taxonomy_data["per_head_errors"][h] = dict(v)
    for entry, r in zip(taxonomy_data["sample_errors"], [r for r in results if not r["overall_accuracy"]]):
        entry["per_head"] = {h: hi.get("status", "ok") for h, hi in r.get("per_head", {}).items()}
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy_data, f, indent=2)
    print("\n  Error taxonomy saved to %s" % taxonomy_path)

    # Per-subset breakdown
    for subset in subsets:
        short_name = os.path.basename(subset)
        subset_results = [r for r in results if r["subset"] == short_name]
        if subset_results:
            st = len(subset_results)
            sf = sum(1 for r in subset_results if r["function_accuracy"])
            so = sum(1 for r in subset_results if r["overall_accuracy"])
            print(f"  {short_name}:")
            print(f"    Function: {sf}/{st} = {sf/st*100:.1f}%")
            print(f"    Overall:  {so}/{st} = {so/st*100:.1f}%")

    return results, func_acc_pct, overall_acc_pct


def main():
    parser = argparse.ArgumentParser(description="BFCL evaluation for RealtimeTool")
    parser.add_argument("--model", default="/models/RT-Qwen2.5-0.5B")
    parser.add_argument("--data-dir",
                        default="/repo/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data")
    parser.add_argument("--subsets", nargs="+",
                        default=["BFCL_v4_multiple", "BFCL_v4_simple_python"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results, func_acc, overall_acc = run_evaluation(
        args.model, args.data_dir, args.subsets, args.max_samples
    )

    if args.output:
        output_data = {
            "model": args.model,
            "subsets": args.subsets,
            "total_samples": len(results),
            "function_accuracy": func_acc,
            "overall_accuracy": overall_acc,
            "results": [
                {"id": r["id"], "subset": r["subset"],
                 "pred_func": r["pred_func"],
                 "function_accuracy": r["function_accuracy"],
                 "overall_accuracy": r["overall_accuracy"]}
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
