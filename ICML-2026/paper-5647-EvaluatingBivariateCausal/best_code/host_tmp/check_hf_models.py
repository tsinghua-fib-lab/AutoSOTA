"""Check available models on Hugging Face and try to load one for coefficient generation."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.insert(0, '/repo')

from huggingface_hub import HfApi, login

# Check token
token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_HUB_TOKEN"))
print("HF_TOKEN available: {}".format(token is not None))

api = HfApi(token=token)

# Search for relevant models
print("\nSearching for models...")
search_terms = ["gemma-3-4b-it", "gemma-2-2b-it", "Qwen/Qwen2.5-1.5B-Instruct"]
for term in search_terms:
    try:
        models = list(api.list_models(search=term, limit=3))
        print("\nSearch: {}".format(term))
        for m in models:
            print("  {} (gated={})".format(m.id, getattr(m, 'gated', 'unknown')))
    except Exception as e:
        print("  Error searching {}: {}".format(term, e))

# Try to load a small model and generate coefficients
print("\n" + "=" * 70)
print("Attempting to load model and generate causal coefficients...")
print("=" * 70)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np
    import re

    from experiments_llm_linear import (
        compute_correlation_matrix, VARIABLES, VARIABLE_DESCRIPTIONS
    )
    from synthetic_experiments_linear import compatibility_score

    # Try Qwen2.5-0.5B-Instruct (tiny, fast)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print("\nTrying to load: {}".format(model_name))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
    )
    print("Tokenizer loaded")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )
    print("Model loaded on device: {}".format(model.device))

    # Get correlation matrix
    corr_df = compute_correlation_matrix()
    corr = corr_df.values
    n = len(VARIABLES)

    # Build prompt matching paper's approach
    var_list = list(VARIABLES)
    corr_str = corr_df.to_string()
    var_desc_block = '\n'.join(
        "- {}: {}".format(v, VARIABLE_DESCRIPTIONS[v]) for v in var_list
    )

    # Step 1: Get causal ordering
    ordering_prompt = (
        "I have observational data on 7 country-level development indicators.\n\n"
        "Correlation matrix:\n{}\n\n"
        "Variable descriptions:\n{}\n\n"
        "Please determine a plausible causal ordering of these 7 variables "
        "(from root causes to downstream effects). Return the ordering as a "
        "comma-separated list of variable names inside <ordering> tags. "
        "Use the exact variable names shown above.\n\n"
        "For example: <ordering>population_density, literacy_rate, daily_income, "
        "sanitation_access, smoking, happiness_score, life_expectancy</ordering>"
    ).format(corr_str, var_desc_block)

    system_msg = (
        "You are a causality expert, tasked to estimate standardized TOTAL "
        "causal effects between country development indicators. "
        "Return your answer concisely in the requested format."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": ordering_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.7,
            do_sample=True,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant response
    if "assistant" in response.lower():
        parts = response.split("assistant")
        if len(parts) > 1:
            response = parts[-1]

    print("\nOrdering response (first 300 chars):")
    print(response[:300])

    # Parse ordering
    match = re.search(r'<ordering>\s*(.+?)\s*</ordering>', response, re.DOTALL)
    if match:
        ordering_raw = match.group(1)
        ordering = [s.strip() for s in ordering_raw.split(',')]
        print("\nParsed ordering: {}".format(ordering))

        # Validate
        valid_ordering = []
        for v in ordering:
            if v in VARIABLES:
                valid_ordering.append(v)

        if len(valid_ordering) == n:
            ordering = valid_ordering
        else:
            print("Incomplete ordering ({}), using default".format(len(valid_ordering)))
            ordering = var_list
    else:
        print("Failed to parse ordering, using default")
        ordering = var_list

    # Step 2: Get coefficient for each pair
    coefficient_prompt = (
        "I will now ask you to estimate the total linear causal coefficient "
        "for several pairs of variables, following the ordering you provided. "
        "For each question, provide your answer in the format: "
        "<answer>CAUSAL_COEFFICIENT: <number></answer>\n\n"
        "The causal coefficient quantifies the expected change of the effect "
        "variable in standard deviations, given an intervention that changes "
        "the cause variable by 1 standard deviation. No other text."
    )
    messages.append({"role": "user", "content": coefficient_prompt})

    # Build coefficient matrix
    A = np.eye(n)
    var_to_idx = {v: i for i, v in enumerate(VARIABLES)}

    pairs = []
    for i_cause, cause in enumerate(ordering):
        if cause not in var_to_idx:
            continue
        for effect in ordering[i_cause+1:]:
            if effect not in var_to_idx:
                continue
            pairs.append((cause, effect))

    print("\nQuerying {} pairs...".format(len(pairs)))

    for cause, effect in pairs:
        q = 'Estimate the total linear causal coefficient for the causal effect of "{}" on "{}".'.format(cause, effect)
        messages.append({"role": "user", "content": q})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.6,
                top_p=0.7,
                do_sample=True,
            )
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "assistant" in resp.lower():
            parts = resp.split("assistant")
            if len(parts) > 1:
                resp = parts[-1]

        match = re.search(r'CAUSAL[_ ]COEFFICIENT:\s*([-+]?\d*\.?\d+)', resp)
        if match:
            coef = float(match.group(1))
            ci = var_to_idx.get(effect)
            cj = var_to_idx.get(cause)
            if ci is not None and cj is not None:
                A[ci, cj] = coef
            print("  {} -> {}: {:.4f}".format(cause, effect, coef))
        else:
            print("  {} -> {}: FAILED to parse".format(cause, effect))

    # Compute compatibility score
    score = compatibility_score(A, corr)
    print("\n" + "=" * 40)
    print("LLM-generated compatibility score: {:.6f}".format(score))
    print("Coefficient matrix A:")
    print(np.array2string(A, precision=3, suppress_small=True))
    print("=" * 40)

except ImportError as e:
    print("Import error: {}".format(e))
except Exception as e:
    print("Error: {}".format(e))
    import traceback
    traceback.print_exc()
