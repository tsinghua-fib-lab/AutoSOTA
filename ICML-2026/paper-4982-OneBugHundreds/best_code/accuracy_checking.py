import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")



def find_all_answer_indices(text: str, key) -> list[int]:
    """
    Returns a list of all start indices where the substring "answer" occurs in `text`.
    If "answer" does not occur, returns an empty list.
    """
    indices = []
    start = 0 
    while True:
        idx = text.find(key, start)
        if idx == -1:
            break
        indices.append(idx)
        # Move past this occurrence to look for the next one
        start = idx + len(key)
    
    return indices

def classification_metrics(fp: int, fn: int, tp: int, tn: int):
    """
    Calculate precision, recall, and accuracy.

    Args:
        fp (int): False positives
        fn (int): False negatives
        tp (int): True positives
        tn (int): True negatives

    Returns:
        tuple of floats: (precision, recall, accuracy)
    """
    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total        if total > 0   else 0.0

    return precision, recall, accuracy

def extract_context_from_string_gpt(text: str, context_len: int = 30):
    """
    Finds all occurrences of the whole word "answer" or "response" in `text`
    and returns a list of (keyword, context) tuples, where `context` is the
    substring containing `context_len` characters before and after the match.
    """
    # \b enforces word-boundaries so we only match “answer” or “response” as standalone words
    patterns = ['answer','response','conclusion','verdict','decision','output','result','assessment','therefore']
    results = []
    text = text.replace("\n","")
    text = text.lower()
    for p in patterns:
        indices = find_all_answer_indices(text,p)
        if indices == []:
            continue
        for idx in indices:
            # Compute slice bounds, clamped to [0, len(text)]
            ctx_start = max(0, idx)
            ctx_end   = min(len(text), idx + context_len)
        
            context = text[ctx_start:ctx_end]
            results.append(context)
    
    return results

def gpt_model_decision(model_outut):
    if "yes" in model_outut[:3].lower():
        return "yes"
    if "no" in model_outut[:3].lower():
        return "no"
    results = extract_context_from_string_gpt(model_outut)
    for model_decision in results:
        if "yes" in model_decision.lower() or "**YES**" in model_outut:
            return "yes"
        elif "no" in model_decision.lower() or "**NO**" in model_outut:
            return "no"     
    return "other"

def llama_model_decision(model_outut):
    model_decision =  model_outut[-10:]
    # simple matching
    if "yes" in model_outut[:3].lower():
        return "yes"
    if "no" in model_outut[:3].lower():
        return "no"
    if "yes" in model_decision.lower() or "**YES**" in model_outut:
        return "yes"
    elif  "no" in model_decision.lower() or "**NO**" in model_outut:
        return "no"
    # keywords finding
    else: 
        results = extract_context_from_string_gpt(model_outut,100)
        for rst in results:
            if "yes" in rst.lower():
                return "yes"
            elif "no" in rst.lower():
                return "no"
        return "other"
    
def claude_model_decision(model_outut):
    model_decision =  model_outut[-10:]
    # simple matching
    if "yes" in model_outut[:3].lower():
        return "yes"
    if "no" in model_outut[:3].lower():
        return "no"
    if "\nyes" in model_outut.lower() or "yes" in model_decision.lower()  or "**YES**" in model_outut:
        return "yes"
    elif  "\nno" in model_outut.lower() or  "no" in model_decision.lower()  or "**NO**" in model_outut:
        return "no"
    # keywords finding
    else: 
        results = extract_context_from_string_gpt(model_outut,100)
        for rst in results:
            if "yes" in rst.lower():
                return "yes"
            elif "no" in rst.lower():
                return "no"
        return "other"

def Model_answer(model_name,model_output):
    model = model_name
    model_decision =""
    if "gpt" in model.lower():
        model_decision = gpt_model_decision(model_output)
    elif "llama" in model.lower():
        model_decision = llama_model_decision(model_output) # for llama3
    else:
        model_decision = claude_model_decision(model_output)
    if "yes" in model_decision.lower():
        return "yes"
    elif "no" in model_decision.lower():
        return "no"
    else:
        return "-"

def OverAllResultsAnalysis(Results_file):

    with open(Results_file, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    
    model_name = Results_file[:-5].split("_")[-1]
    Total_N_cases = len(prompt_data)
    Total_P_cases = len(prompt_data)
    FP = 0.0
    TP = 0.0
    FN = 0.0
    TN = 0.0
    correct_pairs = 0.0
    N_input_tokens = 0.0
    P_input_tokens = 0.0
    N_output_tokens = 0.0
    P_output_tokens = 0.0
    for data in prompt_data:
        N_input_tokens += len(encoding.encode(data["patched_prompt"])) 
        P_input_tokens += len(encoding.encode(data["unpatched_prompt"]))
        N_output_tokens += len(encoding.encode(data["patched_output"]))
        P_output_tokens += len(encoding.encode(data["unpatched_output"]))

        n_answer = Model_answer(model_name,data["patched_output"])
        if n_answer == "no":
            TN += 1
        p_answer = Model_answer(model_name,data["unpatched_output"])
        if p_answer == "yes":
            TP += 1
        if n_answer == "no" and p_answer == "yes":
            correct_pairs += 1

    avg_input = (float(N_input_tokens)+P_input_tokens) / (Total_P_cases+Total_N_cases)
    avg_output = (float(N_output_tokens)+P_output_tokens) / (Total_P_cases+Total_N_cases)
    print("Average input tokens:",avg_input)
    print("Average output tokens:",avg_output)

    FP = Total_N_cases-TN  # avoid no answer cases
    FN = Total_P_cases - TP # avoid no answer cases
    print("Negative cases (Patched):",Total_N_cases,"\tFP:",FP,"\tTN:",TN)
    print("Positive cases (Unpatched):",Total_P_cases,"\tTP:",TP,"\tFN:",FN)
    precision, recall, acc = classification_metrics(FP, FN, TP, TN)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("PAcc: ",correct_pairs/len(prompt_data))

