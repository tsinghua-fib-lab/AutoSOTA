import os
import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import read_json, write_json, get_keywords, get_alphabet_choice, remove_boxed, last_boxed_only_string
from agent import *
import logging
import pandas as pd
from ast import literal_eval
from collections import defaultdict
import re
import glob as glob_module

def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregator", type=str, default="QwenR1")
    parser.add_argument("--task", type=str, default="MMLU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--weighted", action="store_true", default=False,
                        help="Use skill-profile weighted voting instead of majority vote")
    parser.add_argument("--consensus_bypass", action="store_true", default=False,
                        help="Skip aggregator when all experts unanimously agree")
    return parser.parse_args()

def get_model_weights(task="GPQA"):
    """Read profile accuracies and compute normalized model weights."""
    profiles = glob_module.glob("./skills/{task}/*_profile_*.json".format(task=task))
    weights = {}
    for pf in profiles:
        name = pf.split("_profile")[0].split("./skills/{task}/".format(task=task))[-1]
        match = re.search(r"profile_([0-9.]+)\.json", pf)
        if match:
            acc = float(match.group(1))
            weights[name] = max(acc, 1.0)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    return weights

def get_model_name_from_column(col_name):
    """Extract model name from column like QwenR1_answer_1 -> QwenR1"""
    parts = col_name.rsplit("_answer_", 1)
    if len(parts) == 2:
        return parts[0]
    return col_name

def weighted_vote(predictions_with_models, model_weights):
    """Compute weighted vote. predictions_with_models: list of (pred, model_name)"""
    weighted_counts = {}
    for pred, model in predictions_with_models:
        weight = model_weights.get(model, 1.0 / max(len(model_weights), 1))
        weighted_counts[pred] = weighted_counts.get(pred, 0) + weight
    if not weighted_counts:
        return None
    return max(weighted_counts, key=weighted_counts.get)

def get_model_responses(agent_name, prompts, num_gpu):
    model_id = agent_map.get(agent_name)
    print("getting responses from this model: {name}".format(name=agent_name))
    if agent_name in ["Phi", "Mistral"]:
        llm = LLM(model = model_id, enforce_eager=True,
                  download_dir = "/models",
                  max_model_len = 16000,
                  tensor_parallel_size = num_gpu,
                  trust_remote_code = True)
    elif agent_name in ["DeepSeekMath"]:
        llm = LLM(model = model_id, enforce_eager=True,
                  download_dir = "/models",
                  max_model_len = 4096,
                  tensor_parallel_size = num_gpu,
                  trust_remote_code = True)
    else:
        llm = LLM(model = model_id, enforce_eager=True,
                  download_dir = "/models",
                  tensor_parallel_size = num_gpu,
                  trust_remote_code = True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    results = vllm_generate(agent_name, llm, tokenizer, prompts, temperature=0.7)
    return results

def get_valid_responses(row, answer_columns):
    responses = {}
    for col in answer_columns:
        if pd.notna(row[col]) and pd.notnull(row[col]):
            responses[col] = row[col]
    return responses

def get_valid_responses_with_models(row, answer_columns):
    """Same as get_valid_responses but returns (col_name, response) pairs."""
    responses = []
    for col in answer_columns:
        if pd.notna(row[col]) and pd.notnull(row[col]):
            responses.append((col, row[col]))
    return responses

if __name__ == "__main__":

    args = parse_args()
    seed_everything(args.seed)
    start_time = time.time()
    num_choice = 10 if args.task == "MMLU_Pro" else 4
    test_samples = read_json("./test_data/{task}_test.json".format(task=args.task))
    round_zero_df = pd.read_csv("./skills/{task}/round0_seed{seed}.csv".format(task=args.task, seed=args.seed))
    answer_columns = [col for col in round_zero_df.columns if "answer_" in col]

    # Load model weights if weighted voting is enabled
    model_weights = {}
    if args.weighted:
        model_weights = get_model_weights(args.task)
        print("Model weights for weighted voting: {weights}".format(weights=model_weights))

    # --- Initial accuracy computation ---
    num_correct = 0
    consensus_count = 0  # Track unanimous agreement
    for i, row in round_zero_df.iterrows():
        gt = row["gold_answer"]
        if args.weighted:
            # Weighted voting
            valid_with_models = get_valid_responses_with_models(row, answer_columns)
            preds_with_models = []
            for col, r in valid_with_models:
                model = get_model_name_from_column(col)
                if args.task in ["MATH", "AIME24"]:
                    pred = remove_boxed(last_boxed_only_string(r))
                else:
                    pred = get_alphabet_choice(r, num_choice=num_choice)
                preds_with_models.append((pred, model))
            maj = weighted_vote(preds_with_models, model_weights)
        else:
            # Original majority voting
            valid_responses = get_valid_responses(row, answer_columns)
            preds = []
            for r in list(valid_responses.values()):
                if args.task in ["MATH", "AIME24"]:
                    pred = remove_boxed(last_boxed_only_string(r))
                else:
                    pred = get_alphabet_choice(r, num_choice=num_choice)
                preds.append(pred)
            maj = Counter(preds).most_common(1)[0][0]
            # Track consensus
            if len(set(preds)) == 1:
                consensus_count += 1
        if is_math_equiv(maj, gt):
            num_correct += 1
    acc = round(num_correct / round_zero_df.shape[0] * 100, 2)
    vote_method = "weighted" if args.weighted else "majority"
    print("Initial accuracy with {method} voting: {acc}".format(method=vote_method, acc=acc))
    if not args.weighted:
        print("Consensus rate (all experts agree): {c}/{t} = {pct}%".format(
            c=consensus_count, t=len(round_zero_df), pct=round(consensus_count/len(round_zero_df)*100, 1)))

    # --- Build aggregation prompts ---
    # If consensus bypass is enabled, skip QwenR1 for unanimous questions
    bypass_indices = set()
    if args.consensus_bypass:
        for i, row in round_zero_df.iterrows():
            valid_responses = get_valid_responses(row, answer_columns)
            preds = []
            for r in list(valid_responses.values()):
                if args.task in ["MATH", "AIME24"]:
                    pred = remove_boxed(last_boxed_only_string(r))
                else:
                    pred = get_alphabet_choice(r, num_choice=num_choice)
                preds.append(pred)
            if len(set(preds)) == 1:
                bypass_indices.add(i)
        print("Consensus bypass: skipping QwenR1 for {n}/{t} questions ({pct}%)".format(
            n=len(bypass_indices), t=len(round_zero_df),
            pct=round(len(bypass_indices)/len(round_zero_df)*100, 1)))

    agg_prompts = []
    bypass_answers = {}  # index -> answer for bypassed questions
    for i, row in round_zero_df.iterrows():
        q = row["question"]

        if args.consensus_bypass and i in bypass_indices:
            # Skip aggregation, use unanimous answer directly
            valid_responses = get_valid_responses(row, answer_columns)
            first_resp = list(valid_responses.values())[0]
            if args.task in ["MATH", "AIME24"]:
                bypass_answers[i] = remove_boxed(last_boxed_only_string(first_resp))
            else:
                bypass_answers[i] = get_alphabet_choice(first_resp, num_choice=num_choice)
            agg_prompts.append(None)  # placeholder
            continue

        agg_prompt = ("You have been provided with a set of responses from various open-source models to the latest user query. "
                      "Your task is to synthesize these responses into a single, high-quality response. "
                      "It is crucial to critically evaluate the information provided in these responses, "
                      "recognizing that some of it may be biased or incorrect. "
                      "Your response should not simply replicate the given answers but should offer a refined, "
                      "accurate, and comprehensive reply to the instruction. "
                      "Ensure your response is well-structured, coherent, and adheres"
                      "to the highest standards of accuracy and reliability. "
                      "Responses from models:\n\n")

        if args.weighted:
            # Order responses by model weight (highest first)
            valid_with_models = get_valid_responses_with_models(row, answer_columns)
            # Sort by model weight descending
            valid_with_models.sort(key=lambda x: model_weights.get(get_model_name_from_column(x[0]), 0), reverse=True)
            valid_responses_list = [r for _, r in valid_with_models]
        else:
            valid_responses = get_valid_responses(row, answer_columns)
            valid_responses_list = list(valid_responses.values())

        for idx, res in enumerate(valid_responses_list):
            res = res.split("</think>")[-1]
            agg_prompt += "### Model {idx}'s response:\n{res}\n\n".format(idx=idx+1, res=res)

        if args.task in ["MATH", "AIME24"]:
            agg_prompt += ("Question: {q}\n".format(q=q)
                           + "Provide your step-by-step reasoning first, and then print "
                           + "\"The answer is \\boxed{{X}}\", "
                           + "where X is the final answer, at the end of your response.")
        else:
            agg_prompt += ("Question: {q}\n".format(q=q)
                           + "Provide your step-by-step reasoning first, and then print "
                           + "\"The answer is (X)\", "
                           + "where X is the answer choice (one capital letter), at the end of your response.")
        agg_prompts.append(agg_prompt)

    # Filter out None placeholders for vLLM
    valid_agg_prompts = [p for p in agg_prompts if p is not None]
    valid_indices = [i for i, p in enumerate(agg_prompts) if p is not None]

    round_zero_df = round_zero_df.loc[:, ["question", "gold_answer", "keywords", "solvers"]]

    if valid_agg_prompts:
        result = get_model_responses(args.aggregator, valid_agg_prompts, args.gpus)
    else:
        result = []

    # Reconstruct full results with bypass answers
    full_results = []
    result_idx = 0
    num_correct = 0
    for i, ts in enumerate(test_samples):
        gt = ts["gold_answer"]
        if args.consensus_bypass and i in bypass_indices:
            pred = bypass_answers[i]
            full_results.append("[CONSENSUS BYPASS] " + str(pred))
        else:
            r = result[result_idx]
            result_idx += 1
            full_results.append(r)
            if args.task in ["MATH", "AIME24"]:
                pred = remove_boxed(last_boxed_only_string(r))
            else:
                pred = get_alphabet_choice(r, num_choice=num_choice)
        if is_math_equiv(pred, gt):
            num_correct += 1

    acc = round(num_correct / len(test_samples) * 100, 2)
    elapsed = time.time() - start_time
    flags = []
    if args.weighted:
        flags.append("weighted")
    if args.consensus_bypass:
        flags.append("consensus_bypass")
    flag_str = " | " + " + ".join(flags) if flags else ""
    print("acc: {acc} | dataset: {task} | aggregator: {aggr} | seed: {seed}{flags}".format(
        acc=acc, task=args.task, aggr=args.aggregator, seed=args.seed, flags=flag_str))
    print("elapsed: {elapsed:.1f}s".format(elapsed=elapsed))
    write_json(full_results, "./skills/{task}/fixed_{aggr}_round1_seed{seed}_{acc}.json".format(
        task=args.task, aggr=args.aggregator, seed=args.seed, acc=acc))
