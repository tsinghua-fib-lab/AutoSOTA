from __future__ import generator_stop  # just to be safe with python 3.7
import os
import argparse
import random
import sys
import pickle
from tqdm.auto import tqdm

import spot
from autoregltl.ltl.parser import ParseError, ltl_formula, ltl_trace
from autoregltl.sat import spot_to_pyaiger, pyaiger_to_spot, generate_model

import math
from functools import reduce, partial
from collections import defaultdict
from timeit import default_timer as timer
import time

import multiprocessing as mp
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from pebble.common import SLEEP_UNIT


def tictoc_histogram(results, show=False, save_to=None, figsize=None, logscale=True):
    import matplotlib.pyplot as plt

    num_subplots = len(results)
    if figsize is None:
        figsize = (num_subplots * 5, 5)
    figure, axes = plt.subplots(1, num_subplots, figsize=figsize)
    if num_subplots == 1:
        axes = [axes]
    for idx, (name, vals) in enumerate(results.items()):
        axes[idx].hist(vals)
        axes[idx].set_xlabel('Time (s)')
        axes[idx].title.set_text(name)
        if logscale:
            axes[idx].set_yscale('log', nonpositive='clip')
    if save_to is not None:
        figure.savefig(save_to, bbox_inches="tight", dpi=192)
    if show:
        plt.show()
    else:
        plt.close(figure)


class TicToc():
    def __init__(self):
        self.t = None
        self.results = defaultdict(list)

    def tic(self):
        self.t = timer()

    def toc(self, name):
        if self.t is None:
            raise RuntimeError('Timer not started')
        diff = timer() - self.t
        self.t = None
        self.results[name].append(diff)

    def add(self, name, diff):
        self.results[name].append(diff)

    def histogram(self, show=True, save_to=None, figsize=None):
        tictoc_histogram(dict(self.results), show, save_to, figsize)



def abbrev_count(count):
    log_count = math.floor(math.log10(count))
    k_exponent = math.floor(log_count / 3)
    suffixes = ['', 'k', 'm']
    return '{:g}{}'.format(count / 10**(k_exponent*3), suffixes[k_exponent])


def dataset_name(num_aps, tree_size, num_formulas, polish=True, name_prefix=None, **kwargs):
    folder = name_prefix + '-' if name_prefix is not None else ''

    if isinstance(tree_size, int):
        tree_size = str(tree_size)
    else:
        tree_size = str(tree_size[0]) + '-' + str(tree_size[1])
    folder_substrs = ['na', str(num_aps), 'ts', tree_size, 'nf']
    folder_substrs.append(abbrev_count(num_formulas))
    folder += '-'.join(folder_substrs)
    if polish:
        folder += '-lbt'
    return folder



def generate_samples(num_aps, num_formulas, tree_size, seed, polish, trace_generator, timeout, max_time, **kwargs):
    if num_aps > 26:
        raise ValueError("Cannot generate more than 26 APs")
    aps = list(map(chr, range(97, 97 + num_aps)))

    if isinstance(tree_size, int):
        tree_size = (1, tree_size)
    formula_generator = spot.randltl(aps, seed=seed, tree_size=tree_size,
                                     ltl_priorities='false=1,true=1,not=1,F=0,G=0,X=0,equiv=0.5,implies=0,xor=0.5,R=0,U=0,W=0,M=0,and=1,or=1',
                                     simplify=0)

    tictoc = TicToc()
    max_time = max_time if max_time is not None else float('inf')
    start_time = time.time()

    # generate samples
    print('Generating samples...')
    print("Tree size: {}".format(tree_size))
    discovered = set()
    samples = defaultdict(set)
    unsat_samples = defaultdict(set)
    timeout_samples = defaultdict(set)
    total_samples = 0
    cpus = len(os.sched_getaffinity(0)) if kwargs['cpus'] == 0 else kwargs['cpus']
    print(f'Using {cpus} CPUs')
    if cpus < kwargs['min_cpus']:
        print(f"Not enough CPUs (<{kwargs['min_cpus']}), exiting...")
        sys.exit(1)
    qsize = 0
    maxqsize = 1000
    exhaustive = False
    with (
        tqdm(desc="Generate") as pbar1, 
        tqdm(total=num_formulas, desc="Trace") as pbar, 
        ProcessPool(cpus) as pool,
    ):
        def callback(future, formula_str, formula_size):
            nonlocal pbar, discovered, samples, unsat_samples, timeout_samples, total_samples, qsize, maxqsize
            formula_aps = [i for i in aps if i in formula_str]
            key = (len(formula_aps), formula_size)
            try:
                label_str, elapsed = future.result()  # blocks until results are ready
            except TimeoutError as error:
                timeout_samples[key].add(formula_str)
                return
            except Exception as error:
                print("Function raised %s" % error)
                print(error.traceback)  # traceback of the function
                return
            finally:
                qsize -= 1
            tictoc.add('trace generation', elapsed)
            if label_str is None:
                unsat_samples[key].add(formula_str)
                return
            if total_samples >= num_formulas:
                return
            label_str = ''.join(pyaiger_to_spot(label_str.split(' ')))
            samples[key].add((formula_str, label_str, elapsed))
            total_samples += 1
            pbar.update(1)
        try:
            while total_samples < num_formulas:
                tictoc.tic()
                try:
                    formula_spot = next(formula_generator)
                except StopIteration:
                    print('Generated all formulas')
                    exhaustive = True
                    pool.close()
                    break
                tictoc.toc('formula generation')
                formula_str = formula_spot.to_str()

                # Adjust the AP characters to start from 0
                formula_aps = [i for i in aps if i in formula_str]
                if len(formula_aps) < len(aps):
                    # Make sure the aps are sorted
                    # We don't want a formula like (e & f), it must be (a & b)
                    lookup = {k: aps[i] for i, k in enumerate(formula_aps)}
                    formula_str = "".join([lookup.get(c, c) for c in formula_str])
                
                if formula_str in discovered:
                    continue
                discovered.add(formula_str)

                formula_obj = ltl_formula(formula_str, 'spot')
                formula_size = formula_obj.size()
                formula_str = formula_obj.to_str('network-' + ('polish' if polish else 'infix'))
                polist_pyaiger = spot_to_pyaiger(
                    formula_obj.to_str('network-polish', spacing='all ops').split(' ')
                )
                future = pool.schedule(generate_model, args=(polist_pyaiger, None), timeout=timeout)
                future.add_done_callback(partial(callback, formula_str=formula_str, formula_size=formula_size))
                pbar1.update(1)
                qsize += 1
                while qsize >= maxqsize:
                    time.sleep(SLEEP_UNIT)
                    pbar.set_description('Waiting Queue: {}'.format(qsize))
                    pbar.refresh()
                else:
                    pbar.set_description("Trace")
            else:
                pool.stop()
        except KeyboardInterrupt:
            tqdm.write('KeyboardInterrupt')
            pool.close()

    tictoc.histogram(show=False, save_to='timing_nf{}_ts{:d}-{:d}.png'.format(abbrev_count(num_formulas), tree_size[0], tree_size[1]))         # For timing analysis
    print('Generated {:d} samples, {:d} requested'.format(total_samples, num_formulas))

    return {
        'sat': samples,
        'unsat': unsat_samples,
        'timeout': timeout_samples,
        'timeout_duration': timeout,
        'exhaustive': exhaustive,  # Whether all formulas have been generated
    }


def run():
    parser = argparse.ArgumentParser(
        description='Randomly generates LTL formulas with a corresponding trace.')
    parser.add_argument('--num-aps', '-na', type=int, default=5)
    parser.add_argument('--num-formulas', '-nf', type=int, default=1000)
    parser.add_argument('--tree-size', '-ts', type=str, default='15', metavar='MAX_TREE_SIZE',
                        help="Maximum tree size of generated formulas. Range can be specified as 'MIN-MAX'; default minimum is 1")
    parser.add_argument('--output-dir', '-od', type=str, default="my-eval2d")
    parser.add_argument('--seed', type=int, default=42)
    infix_or_polish = parser.add_mutually_exclusive_group()
    infix_or_polish.add_argument('--polish', dest='polish', action='store_true',
                                 default=True, help='write formulas and traces in polish notation; default')
    infix_or_polish.add_argument('--infix', dest='polish', action='store_false',
                                 default=True, help='write formulas and traces in infix notation')
    parser.add_argument('--trace-generator', type=str, choices=[
                        'spot', 'aalta'], default='spot', help='which tool to get a trace (or unsat) from; default spot')
    parser.add_argument('--timeout', type=float, default=10,
                        help='time in seconds to wait for the trace generator to return, if expired kill and continue with next formula')
    parser.add_argument('--name-prefix', help="Name to prefix the dataset name with")
    parser.add_argument('--max-time', type=float, default=3*60*60)
    parser.add_argument('--cpus', type=int, default=0, help="Number of worker threads")
    parser.add_argument('--min-cpus', type=int, default=0, help="Exit if available CPUs are smaller than this")
    args = parser.parse_args()

    tree_size = args.tree_size.split('-')
    if len(tree_size) == 1:
        tree_size = int(tree_size[0])
    else:
        tree_size = (int(tree_size[0]), int(tree_size[1]))
    args_dict = vars(args)
    args_dict['tree_size'] = tree_size

    path = os.path.join(args.output_dir, dataset_name(**args_dict) + '.pkl')
    os.makedirs(args.output_dir, exist_ok=True)

    res = generate_samples(**args_dict)

    with open(path, 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    run()
