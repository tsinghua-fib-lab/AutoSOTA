import torch
import numpy as np
import os
import sys
import json
import traceback
import random
from glob import glob
from collections import defaultdict
from typing import Dict, Union, Any, Optional, Tuple, List
from tqdm.auto import tqdm
from pprint import pprint
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

import spot

from autoregltl.ltl.enforcer import LTLSyntaxEnforcerConfig
from autoregltl import dataset
from autoregltl.ltl.trace_check import pool_iter
from autoregltl.ltl.parser import ParseError, ltl_formula
from autoregltl.utils import describe_statistics, tictoc_histogram, init_plot_font
from autoregltl.eval import get_result_dir_name
from autoregltl.sat import get_assignments, spot_to_pyaiger, is_model

from itertools import permutations

def generate_equivalent_expressions(expression, aps, max_perm=None):
    # Extract unique variables from the expression
    unique_variables = sorted(set([ch for ch in expression.replace("xor", "") if ch.islower()]))
    # List to hold all equivalent expressions
    equivalent_expressions = []
    
    if max_perm is None:
        # Generate all permutations of valid variable names of the length of unique variables
        for perm in permutations(aps, len(unique_variables)):
            # Replace variables with each permutation
            translation_map = {original: new for original, new in zip(unique_variables, perm)}
            new_expression = ''.join([translation_map.get(ch, ch) for ch in expression])
            equivalent_expressions.append(new_expression)
    else:
        used_perms = set()
        while len(used_perms) < max_perm:
            perm = tuple(random.sample(aps, len(unique_variables)))
            if perm not in used_perms:
                used_perms.add(perm)
                # Replace variables with each permutation
                translation_map = {original: new for original, new in zip(unique_variables, perm)}
                new_expression = ''.join([translation_map.get(ch, ch) for ch in expression])
                equivalent_expressions.append(new_expression)
    
    return equivalent_expressions


class ResymbolizeDataset(dataset.SeqDataset):
    def __init__(self, dataset, aps, max_perm=None):
        self.base_dataset = dataset
        self.aps = aps
        new_data = []
        self.group_sizes = []
        for trace, formula in dataset.data:
            resyms = generate_equivalent_expressions(formula, aps, max_perm)
            self.group_sizes.append(len(resyms))
            # Trace is not required for generation
            new_data += [(trace, f) for f in resyms]
        print("Original dataset size:", len(dataset.data))
        print("Permutations dataset size:", len(new_data))
        super().__init__(new_data)
    
    def unresym(self, regrouped_predictions):
        # Each group is a list of (prediction, trace, formula)
        out = []
        for group, (base_trace, base_formula) in zip(regrouped_predictions, self.base_dataset.data):
            unique_variables = sorted(set([ch for ch in base_formula.replace("xor", "") if ch.islower()]))
            predictions = defaultdict(int)
            for item, perm in zip(group, permutations(self.aps, len(unique_variables))):
                # Undo the permutation
                translation_map = {new: original for original, new in zip(unique_variables, perm)}
                prediction = ''.join([translation_map.get(ch, ch) for ch in item[0]])
                predictions[prediction] += 1
            predictions = dict(predictions)
            assert len(group) == sum(predictions.values())
            item = {
                "trace": base_trace,
                "formula": base_formula,
                "predictions": predictions,
                "pred_count": len(predictions),
                "perm_count": len(group),
                "ap_count": len(unique_variables),
                "TRC": 1.0 - ((len(predictions)-1) / (len(group)-1)),
            }
            out.append(item)
        return out


def _eval_item(item):
    """
    `item` is a list element from unresym output
    """
    eval_results = []
    prev_assms = []
    invalids = []
    # formula_automaton = spot.formula(ltl_formula(, 'network-polish').to_str('spot')).translate()
    formula_pyaiger = spot_to_pyaiger(item['formula'])
    for prediction in item['predictions'].keys():
        try:
            assm = get_assignments(prediction)
        except ParseError:
            invalids.append(prediction)
            continue
        for result, prev_assm in zip(eval_results, prev_assms):
            if assm == prev_assm:
                result['list'].append(prediction)
                break
        else:
            assignments_pyaiger = get_assignments(spot_to_pyaiger(prediction))
            # Not equivalent to anything
            try:
                holds = is_model(formula_pyaiger, assignments_pyaiger)
                res = "semantically correct" if holds else "incorrect"
            except KeyError as e:
                res = "incorrect"
            eval_results.append({
                "result": res,
                "list": [prediction],
            })
    if invalids:
        eval_results.append({
            "result": "invalid",
            "list": invalids,
        })
    eval_sum = defaultdict(int)
    for result in eval_results:
        label = result['result']
        for p in result['list']:
            eval_sum[label] += item['predictions'][p]
    eval_sum = dict(eval_sum)
    assert item['perm_count'] == sum(eval_sum.values())
    output = {
        "evaluation": eval_results,
        "eval_sum": eval_sum,
    }
    if (correct := eval_sum.get('semantically correct', None)) is not None:
        output["correct_rate"] = correct / item['perm_count']
    return output


def evaluate_model(model_path, args, load_model, get_gen_args):
    model = load_model(model_path, **vars(args))
    print("Loaded pretrained model:", model_path)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count:_}")

    if args.min_aps is None or args.min_aps < 1:
        print("Forcing min_aps to 1")
        args.min_aps = 1

    dataset_vocab = dataset.get_dataset_vocab(args, model.config)
    # Only input compatibility is required
    if not model.config.vocab.are_inputs_compatible(dataset_vocab):
        sys.exit("Dataset vocabulary is not compatible with the model")
    test_dataset = dataset.get_dataset(args, args.split, dataset.RawLTLDataset, max_samples=args.max_samples)
    resym_dataset = ResymbolizeDataset(test_dataset, dataset_vocab.aps, args.max_perm)

    gen_args = get_gen_args(args)
    if args.syntax_enforcing:
        gen_args['syntax_enforcer'] = LTLSyntaxEnforcerConfig(model.config.vocab)

    result_dir = os.path.join(model_path, args.result_dir_name)
    os.makedirs(result_dir, exist_ok=True)

    if os.path.exists(os.path.join(result_dir, "predictions.json")):
        print("Predictions already exist, skipping generation and loading from file")
        with open(os.path.join(result_dir, "predictions.json"), 'r') as f:
            unresymed = json.load(f)
    else:
        if os.path.exists(os.path.join(result_dir, "raw_predictions.json")):
            print("Raw predictions already exist, skipping generation and loading from file")
            with open(os.path.join(result_dir, "raw_predictions.json"), 'r') as f:
                regrouped = json.load(f)
        else:
            predictions = model.generate_predictions(resym_dataset, args.max_length, gen_args)
            it = iter(predictions)
            regrouped = [[next(it) for _ in range(group_size)] for group_size in resym_dataset.group_sizes]

            with open(os.path.join(result_dir, "raw_predictions.json"), 'w') as f:
                json.dump(regrouped, f, indent=4)
        
        unresymed = resym_dataset.unresym(regrouped)
        with open(os.path.join(result_dir, "predictions.json"), 'w') as f:
            json.dump(unresymed, f, indent=4)

    # Evaluation
    with pool_iter(_eval_item, unresymed, args.eval_threads, args.eval_timeout, tqdm_desc="Evaluate") as iterator:
        for item in unresymed:
            try:
                result = next(iterator)
                item |= result
            except TimeoutError:
                item['evaluation'] = {"result": "timeout", "time": args.eval_timeout}
            except ProcessExpired as e:
                item['evaluation'] = {
                    "result": "runtime error",
                    "error": f"ProcessExpired with exit code {e.exitcode}",
                    "time": 0.0,
                }
            except Exception as e:
                item['evaluation'] = {
                    "result": "runtime error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "time": 0.0,
                }

    # Analysis

    def summarize_label(label):
        all_trcs = [item[label] for item in unresymed if label in item]
        ap_trcs = defaultdict(list)
        for item in unresymed:
            if label in item:
                ap_count = item["ap_count"]
                ap_trcs[ap_count].append(item[label])

        summ = {label: describe_statistics(all_trcs, True)}
        summ |= {f"{k} AP {label}": describe_statistics(v, True) for k, v in ap_trcs.items()}

        # Plot time
        ap_trcs = dict(sorted(ap_trcs.items()))
        ap_trcs["All"] = all_trcs
        resym_plot(f"{label} Box Plot", ap_trcs, os.path.join(result_dir, f"{label}-box.png"), violin=False)
        resym_plot(f"{label} Violin Plot", ap_trcs, os.path.join(result_dir, f"{label}-violin.png"), violin=True)
        return summ
    
    summary = {}
    summary |= summarize_label("correct_rate")
    summary |= summarize_label("TRC")
    print("SUMMARY:")
    pprint(summary)

    with open(os.path.join(result_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    with open(os.path.join(result_dir, "predictions.json"), 'w') as f:
        json.dump(unresymed, f, indent=4)

    print("Done")


def resymbolize_eval(args, load_model, get_gen_args):
    init_plot_font()

    if args.result_dir_name is None:
        args.result_dir_name = "resym-" + get_result_dir_name(args)
        print("Result directory name:", args.result_dir_name)

    model_paths = glob(args.model_path)
    if not model_paths:
        print(f"No models found at {args.model_path}")
        return
    try:
        from natsort import natsorted
        model_paths = natsorted(model_paths)
    except ImportError:
        pass
    for model_path in model_paths:
        try:
            evaluate_model(model_path, args, load_model, get_gen_args)
        except Exception:
            trace = traceback.format_exc()
            print("Error during the evaluation of", model_path)
            print("Trace:")
            print(trace)
        print()


import matplotlib.pyplot as plt

def resym_plot(title, data, save_to, violin):
    figure, ax = plt.subplots(figsize=(7, 5))

    labels = [f'{k}\nn={len(v)}' for k, v in data.items()]
    xvalues = list(data.values())

    if violin:
        ax.violinplot(xvalues, showmeans=False, showmedians=True)
    else:
        ax.boxplot(xvalues)
    ax.set_title(title)

    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(labels))], labels=labels)
    ax.set_xlabel('AP count')
    ax.set_ylabel('Resymbolization Consistency')

    plt.show()
    figure.savefig(save_to, bbox_inches="tight", dpi=192)
