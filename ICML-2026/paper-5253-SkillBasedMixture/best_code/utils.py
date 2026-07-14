import json
import re
import os
import torch
import random
import tiktoken
import numpy as np
from backoff import on_exception, expo
from math_verify import parse, verify
    
def write_json(obj, file_name):
    with open(file_name, "w") as f:
        json.dump(obj, f)

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)

def get_number_choice(text):
    if not text:
        return "N/A"
    match = re.findall(r"answer is \((\d)\)", text)
    if match:
        return match[-1]
    else:
        match = re.findall(r"\((\d)\)", text)
        return match[-1] if match else "N/A"
    return "N/A"

def get_alphabet_choice(text, num_choice=4):
    choices = '|'.join([chr(65 + i) for i in range(num_choice)])
    if text:
        # First try to match with parentheses
        match = re.findall(f'([{choices}])\)', text)
        if not match:
            # If no match with parentheses, try without
            match = re.findall(f'([{choices}])', text)
    else:
        return "N/A"
    return match[-1] if match else "N/A"
    
def get_true_false(text):
    if not text:
        return "N/A"
    match = re.findall(r"(true|false)", text, re.IGNORECASE)
    return match[-1].lower() if match else "N/A"

def get_yes_no(text):
    if not text:
        return "N/A"
    match = re.findall(r"(yes|no)", text, re.IGNORECASE)
    return match[-1].lower() if match else "N/A"

def get_keywords(output):
    keywords = output.split("Keywords:")[-1].split(",")
    keywords = [i.strip().lower().replace(".", "") for i in keywords]
    return keywords

def get_token_count(string, encoding_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def is_math_equiv(ref, pred):
    # Test math equivalence of ref and pred, 
    # can also handle answer choices e.g., A vs. (A)
    try:
        if any([verify(parse(f"${ref}$"), parse(f"${pred}$")),
               verify(parse(ref), parse(pred)),
               verify(parse(ref), parse(pred.replace("\\(", "").replace("\\)", "")))]):
            return True
    except:
        return False    
    return False
    
def has_consensus(predictions):
    ref = predictions[0]
    for exp in predictions[1:]:
        if not ref == exp:
            return False
    return True
    
def last_boxed_only_string(string):
    if not string: return "N/A"
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

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

    if right_brace_idx == None:
        retval = string
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return s

def parse_number(text):
    try:
        match = re.findall(r"\$?([0-9]+[\.,]?[0-9]*)", text)
        return float(match[-1].replace(",", "")) if match else "N/A"
    except:
        print(text)

def parse_boxed(s):
    if not s:
        return "N/A"
    s = last_boxed_only_string(s)
    s = remove_boxed(s)
    s = parse_number(s)
    return s