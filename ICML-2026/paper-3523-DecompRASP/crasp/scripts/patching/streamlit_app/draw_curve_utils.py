import click
from pathlib import Path
from streamlit_app.my_paths import output_dir
import glob
import json
import matplotlib.pyplot as plt
from patching_helper_functions import get_full_possible_config_for_pruning
from pruning_model import convert_config_fullpaths_, convert_config_fullpaths_incl_mlp_, convert_full_paths_config_to_prune_inside_kq_config_
import yaml

def get_num_element(d):
    total_len = 0
    for k, v in d.items():
        if type(v) == list:
            total_len += len(v)
        elif type(v) == dict:
            total_len += get_num_element(v)
        else:
            raise RuntimeError("invalid obj", v)
    return total_len

def get_pareto_frontier(points):
    # [(acc, num_edges), ...]
    points = sorted(points, key=lambda x: (x[1], -x[0]))
    best_acc = -1
    frontier = []
    for acc, num_edges, exp_name in points:
        if acc > best_acc:
            best_acc = acc
            frontier.append((acc, num_edges, exp_name))
    return frontier

