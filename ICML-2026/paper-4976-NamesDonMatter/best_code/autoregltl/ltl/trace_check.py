import sys
import argparse
import math
import time
import traceback
from collections import defaultdict
from functools import reduce, partial
from contextlib import contextmanager
from tqdm.auto import tqdm

import multiprocessing
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import spot

from . parser import LTLTrace, LTLFormula, F_AND, F_IMLIES, F_NEXT, F_GLOBALLY, F_NOT, F_AP
from . parser import ParseError, ltl_formula, ltl_trace

from autoregltl.sat import get_assignments, spot_to_pyaiger, is_model


def per_size_analysis(full_results, **kwargs):
    import matplotlib.pyplot as plt

    colors = {
        'syntactically correct': '#38b547',
        'exact match': '#38b547',
        'equivalent': '#5ed561',
        'only semantically correct': '#85f67c',
        'semantically correct': '#85f67c',
        'incorrect': '#ed974d',
        'invalid': '#fd4a4a',
    }
    results = {k: v for k, v in full_results.items() if k in colors}
    order = {
        'syntactically correct': 0,
        'exact match': 0,
        'equivalent': 1,
        'only semantically correct': 2,
        'semantically correct': 2,
        'incorrect': 3,
        'invalid': 4,
    }
    results = dict(sorted(results.items(), key=lambda pair: order[pair[0]]))

    min_size = min([min(d) if len(d) > 0 else math.inf for d in results.values()])
    max_size = max([max(d) if len(d) > 0 else 0 for d in results.values()])
    x, totals = [], []
    assert not ('total' in results)
    results_complete = {}
    for size in range(min_size, max_size + 1):
        x.append(size)
        totals.append(0)
    bottom_positions = totals.copy()

    for category, dist in results.items():  # dict with sizes to list; not all values may occur in dict
        results_complete[category] = []
        for idx, size in enumerate(range(min_size, max_size + 1)):
            value = dist[size] if size in dist else 0
            results_complete[category].append(value)
            totals[idx] += value
    results_percent = {}
    for category, dist_complete in results_complete.items():
        results_percent[category] = []
        for val, total in zip(dist_complete, totals):
            if total == 0 and val != 0:
                raise RuntimeError()
            results_percent[category].append(val / total * 100 if total > 0 else 0)

    names = {
        'syntactically correct': 'exact match',
        'exact match': 'exact match',
        'equivalent': 'equivalent',
        'only semantically correct': 'correct',
        'semantically correct': 'correct',
        'incorrect': 'incorrect',
        'invalid': 'invalid',
     }
    # Do the plotting
    # thanks to https://chrisalbon.com/python/data_visualization/matplotlib_percentage_stacked_bar_plot/
    # figure, (hist_ax, dist_ax) = plt.subplots(2, figsize=(12,8))
    figure, (dist_ax) = plt.subplots(1, figsize=(12, 5))
    bar_width = 1
    # hist_ax.bar(x, totals, width=bar_width, color='#3071ff', edgecolor='white')
    # hist_ax.set_ylabel('number of items')
    # hist_ax.set_xlabel('formula size')
    for category, dist_percent in results_percent.items():
        dist_ax.bar(x, dist_percent, bottom=bottom_positions, label=names[category], width=bar_width, color=colors[category], edgecolor='white')
        bottom_positions = [acc + q for acc, q in zip(bottom_positions, dist_percent)]  # update positions
    dist_ax.set_ylabel('Percentage')
    dist_ax.set_xlabel('Trace size')
    dist_ax.set_ylim(-10, 110)
    dist_ax.legend()
    if 'save_analysis' in kwargs and kwargs['save_analysis'] is not None:
        figure.savefig(kwargs['save_analysis'] + '.png', bbox_inches="tight", dpi=192)
        figure.savefig(kwargs['save_analysis'] + '.svg', bbox_inches="tight", dpi=192)
    
    plt.close(figure)
    plt.clf()

    # collapse size-wise data for further processing
    results_collapsed = {}
    for category, dist in full_results.items():
        results_collapsed[category] = sum(dist.values())
    return results_collapsed


@contextmanager
def pool_iter(process_item, data, threads=None, timeout=30, tqdm_desc=None, leave_tqdm=True):
    if threads is None:
        threads = multiprocessing.cpu_count()
    with ProcessPool(threads) as pool, tqdm(total=len(data), desc=tqdm_desc, leave=leave_tqdm) as pbar:
        future = pool.map(process_item, data, timeout=timeout)
        callback = lambda _: pbar.update(1)
        for f in future.futures:
            f.add_done_callback(callback)
        iterator = future.result()
        yield iterator


def process_ltl_item(item, formula_format):
    pred_str, label_str, formula_str = item
    start_time = time.time()
    try:
        pred = get_assignments(pred_str)
    except ParseError as e:
        return {"result": "invalid", "error": f"{e}", "time": time.time() - start_time}
    if label_str:
        label = get_assignments(label_str)
        if pred == label:
            return {"result": "exact match", "time": time.time() - start_time}
    # Semantic check
    formula_pyaiger = spot_to_pyaiger(formula_str)
    assignments_pyaiger = get_assignments(spot_to_pyaiger(pred_str))
    try:
        holds = is_model(formula_pyaiger, assignments_pyaiger)
    except KeyError as e:
        return {"result": "incorrect", "error": f"{str(e)} is not in formula", "time": time.time() - start_time}
    except RuntimeError as e:
        return {
            "result": "runtime error",
            "error": repr(e),
            "time": time.time() - start_time,
        }
    result = "semantically correct" if holds else "incorrect"
    return {"result": result, "time": time.time() - start_time}


def evaluate_ltl(data, polish=True, threads=None, timeout=30, leave_tqdm=True):
    """
    Args:
        data: List of tuples (formula, trace, target trace)
    """
    formula_format = 'network-' + ('polish' if polish else 'infix')
    process_item = partial(process_ltl_item, formula_format=formula_format)

    results = []
    with pool_iter(process_item, data, threads, timeout, tqdm_desc="Evaluate", leave_tqdm=leave_tqdm) as iterator:
        for a, b, c in data:
            try:
                result = next(iterator)
            except TimeoutError:
                result = {"result": "timeout", "time": timeout}
            except ProcessExpired as e:
                result = {
                    "result": "runtime error",
                    "error": f"ProcessExpired with exit code {e.exitcode}",
                    "time": 0.0,
                }
            except Exception as e:
                result = {
                    "result": "runtime error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "time": 0.0,
                }
            result.update({"prediction": a, "trace": b, "formula": c})
            results.append(result)

    return results


def analyze_results(results):
    """
    Calculate statistics per size from evaluation results.
    """
    output = defaultdict(lambda: defaultdict(int))
    # Trace format: 1;1;{1;1}
    get_size = lambda x: (x["trace"].count(';') + 1)
    for result in results:
        output[result["result"]][get_size(result)] += 1
    return output