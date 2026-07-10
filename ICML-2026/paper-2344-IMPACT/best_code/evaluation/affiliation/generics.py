#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import groupby
from operator import itemgetter
import math
import gzip
import glob
import os

def convert_vector_to_events(vector = [0, 1, 1, 0, 0, 1, 0]):
    """
    Convert a binary vector (indicating 1 for the anomalous instances)
    to a list of events. The events are considered as durations,
    i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).
    
    :param vector: a list of elements belonging to {0, 1}
    :return: a list of couples, each couple representing the start and stop of
    each event
    """
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix : ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))
    
    # Consistent conversion in case of range anomalies (for indexes):
    # A positive index i is considered as the interval [i, i+1),
    # so the last index should be moved by 1
    events = [(x, y+1) for (x,y) in events]
        
    return(events)

def infer_Trange(events_pred, events_gt):
    """
    Given the list of events events_pred and events_gt, get the
    smallest possible Trange corresponding to the start and stop indexes 
    of the whole series.
    Trange will not influence the measure of distances, but will impact the
    measures of probabilities.
    
    :param events_pred: a list of couples corresponding to predicted events
    :param events_gt: a list of couples corresponding to ground truth events
    :return: a couple corresponding to the smallest range containing the events
    """
    if len(events_gt) == 0:
        raise ValueError('The gt events should contain at least one event')
    if len(events_pred) == 0:
        # empty prediction, base Trange only on events_gt (which is non empty)
        return(infer_Trange(events_gt, events_gt))
        
    min_pred = min([x[0] for x in events_pred])
    min_gt = min([x[0] for x in events_gt])
    max_pred = max([x[1] for x in events_pred])
    max_gt = max([x[1] for x in events_gt])
    Trange = (min(min_pred, min_gt), max(max_pred, max_gt))
    return(Trange)

def has_point_anomalies(events):
    """
    Checking whether events contain point anomalies, i.e.
    events starting and stopping at the same time.
    
    :param events: a list of couples corresponding to predicted events
    :return: True is the events have any point anomalies, False otherwise
    """
    if len(events) == 0:
        return(False)
    return(min([x[1] - x[0] for x in events]) == 0)

def _sum_wo_nan(vec):
    """
    Sum of elements, ignoring math.isnan ones
    
    :param vec: vector of floating numbers
    :return: sum of the elements, ignoring math.isnan ones
    """
    vec_wo_nan = [e for e in vec if not math.isnan(e)]
    return(sum(vec_wo_nan))
    
def _len_wo_nan(vec):
    """
    Count of elements, ignoring math.isnan ones
    
    :param vec: vector of floating numbers
    :return: count of the elements, ignoring math.isnan ones
    """
    vec_wo_nan = [e for e in vec if not math.isnan(e)]
    return(len(vec_wo_nan))

def read_gz_data(filename = 'data/machinetemp_groundtruth.gz'):
    """
    Load a file compressed with gz, such that each line of the
    file is either 0 (representing a normal instance) or 1 (representing)
    an anomalous instance.
    :param filename: file path to the gz compressed file
    :return: list of integers with either 0 or 1
    """
    with gzip.open(filename, 'rb') as f:
        content = f.read().splitlines()
    content = [int(x) for x in content]
    return(content)

def read_all_as_events():
    """
    Load the files contained in the folder `data/` and convert
    to events. The length of the series is kept.
    The convention for the file name is: `dataset_algorithm.gz`
    :return: two dictionaries:
        - the first containing the list of events for each dataset and algorithm,
        - the second containing the range of the series for each dataset
    """
    filepaths = glob.glob('data/*.gz')
    datasets = dict()
    Tranges = dict()
    for filepath in filepaths:
        vector = read_gz_data(filepath)
        events = convert_vector_to_events(vector)
        # ad hoc cut for those files
        cut_filepath = (os.path.split(filepath)[1]).split('_')
        data_name = cut_filepath[0]
        algo_name = (cut_filepath[1]).split('.')[0]
        if not data_name in datasets:
            datasets[data_name] = dict()
            Tranges[data_name] = (0, len(vector))
        datasets[data_name][algo_name] = events
    return(datasets, Tranges)

def f1_func(p, r):
    """
    Compute the f1 function
    :param p: precision numeric value
    :param r: recall numeric value
    :return: f1 numeric value
    """
    return(2*p*r/(p+r))
