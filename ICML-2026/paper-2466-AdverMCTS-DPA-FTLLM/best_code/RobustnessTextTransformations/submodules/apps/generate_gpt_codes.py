"""
Run a tranined model to generate Python code.
"""

import io
import json
import os
import random

from .classes import ConfigCode
from .reindent import run as run_reindent


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False,
        },
    )

    return ret.getvalue()


def generate_prompt(
    args: ConfigCode,
    folder: str,
    tokenizer,
):

    test_case_path = os.path.join(folder, "input_output.json")
    prompt_path = os.path.join(folder, "question.txt")
    starter_path = (
        os.path.join(folder, "starter_code.py")
        if os.path.exists(os.path.join(folder, "starter_code.py"))
        else None
    )
    solutions_path = os.path.join(folder, "solutions.json")

    _input = "\nQUESTION:\n"
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path is not None:
        with open(starter_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data  # + "\n"
        _input += data
    else:
        # _input += "\n\n"
        pass

    with open(test_case_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"  # \n"
    else:
        _input += "\nUse Call-Based format"  # \n"

    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # Need to do some peeking.

        # Read one example solution
        with open(solutions_path, "r", encoding="utf-8") as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol


def load_generations(args: ConfigCode, tokenizer):
    folders = [
        os.path.join(args.source_folder, f)
        for f in os.listdir(args.source_folder)
        if os.path.isdir(os.path.join(args.source_folder, f))
    ]

    prompt_texts, sample_sols = [], []
    for folder in folders:
        prompt_text, sample_sol = generate_prompt(args, folder, tokenizer)
        prompt_texts.append(prompt_text)
        sample_sols.append(sample_sol)

    return prompt_texts, sample_sols, folders
