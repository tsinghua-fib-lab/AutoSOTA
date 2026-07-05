"""
*************
Read BayesNet
from File
*************

Read a BayesNet object from a file. These
functions all currently work, but support
for more formats should be added in the future.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartouth.edu>"""

import json
import numpy as np
import copy

from pyBN.classes.bayesnet import BayesNet
from pyBN.classes.factor import Factor
from pyBN.utils.graph import topsort


def read_bn(path):
    """
    Wrapper function for reading BayesNet objects
    from various file types.

    Arguments
    ---------
    *path* : a string
        The path (relative or absolute) - MUST
        include extension -> that's how we know
        which file reader to call.

    Returns
    -------
    *bn* : a BayesNet object

    Effects
    -------
    None

    Notes
    -----

    """
    if '.bif' in path:
        return read_bif(path)
    elif '.bn' in path:
        return read_json(path)
    elif '.mat' in path:
        return read_mat(path)
    else:
        print("Path Extension not recognized")

def read_bif(path):
    """
    This function reads a .bif file into a
    BayesNet object. It's probably not the 
    fastest or prettiest but it gets the job
    done.

    Arguments
    ---------
    *path* : a string
        The path

    Returns
    -------
    *bn* : a BayesNet object

    Effects
    -------
    None

    Notes
    -----
    *V* : a list of strings
    *E* : a dict, where key = vertex, val = list of its children
    *F* : a dict, where key = rv, val = another dict with
                keys = 'parents', 'values', cpt'

    """
    _parents = {} # key = vertex, value = list of vertices in the scope (includind itself)
    _cpt = {} # key = vertex, value = list (then numpy array)
    _vals = {} # key=vertex, val=list of its possible values

    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if 'variable' in line:
                new_vertex = line.split()[1]

                _parents[new_vertex] = []
                _cpt[new_vertex] = []
                #_vals[new_vertex] = []

                new_line = f.readline()
                new_vals = new_line.replace(',', ' ').split()[6:-1] # list of vals
                _vals[new_vertex] = new_vals
                num_outcomes = len(new_vals)
            elif 'probability' in line:
                line = line.replace(',', ' ')
                child_rv = line.split()[2]
                parent_rvs = line.split()[4:-2]

                if len(parent_rvs) == 0: # prior
                    new_line = f.readline().replace(';', ' ').replace(',',' ').split()
                    prob_values = new_line[1:]
                    _cpt[child_rv].append(map(float,prob_values))
                    #_cpt[child_rv] = map(float,prob_values)
                else: # not a prior
                    _parents[child_rv].extend(list(parent_rvs))
                    while True:
                        new_line = f.readline()
                        if '}' in new_line:
                            break
                        new_line = new_line.replace(',',' ').replace(';',' ').replace('(', ' ').replace(')', ' ').split()
                        prob_values = new_line[-(len(_vals[new_vertex])):]
                        prob_values = map(float,prob_values)
                        _cpt[child_rv].append(prob_values)
            if line == '':
                break

    # CREATE FACTORS
    _F = {}
    _E = {}
    for rv in _vals.keys():
        _E[rv] = [c for c in _vals.keys() if rv in _parents[c]]
        f = {
            'parents' : _parents[rv],
            'values' : _vals[rv],
            'cpt' : [item for sublist in _cpt[rv] for item in sublist]
        }
        _F[rv] = f

    bn = BayesNet()
    bn.F = _F
    bn.E = _E
    bn.V = list(topsort(_E))

    return bn

def read_json(path):
    """
    Read a BayesNet object from the json format. This
    format has the ".bn" extension and is completely
    unique to pyBN.

    Arguments
    ---------
    *path* : a string
        The file path

    Returns
    -------
    None

    Effects
    -------
    - Instantiates and sets a new BayesNet object

    Notes
    -----
    
    This function reads in a libpgm-style format into a bn object

    File Format:
        {
            "V": ["Letter", "Grade", "Intelligence", "SAT", "Difficulty"],
            "E": [["Intelligence", "Grade"],
                ["Difficulty", "Grade"],
                ["Intelligence", "SAT"],
                ["Grade", "Letter"]],
            "Vdata": {
                "Letter": {
                    "ord": 4,
                    "numoutcomes": 2,
                    "vals": ["weak", "strong"],
                    "parents": ["Grade"],
                    "children": None,
                    "cprob": [[.1, .9],[.4, .6],[.99, .01]]
                },
                ...
        }


    """
    def byteify(input):
        if isinstance(input, dict):
            return {byteify(key):byteify(value) for key,value in input.iteritems()}
        elif isinstance(input, list):
            return [byteify(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input

    bn = BayesNet()
    
    f = open(path,'r')
    ftxt = f.read()

    success=False
    try:
        data = byteify(json.loads(ftxt))
        bn.V = data['V']
        bn.E = data['E']
        bn.F = data['F']
        success = True
    except ValueError:
        print("Could not read file - check format")
    bn.V = topsort(bn.E)

    return bn

def read_mat(path, delim=' '):
    """
    Read an adjacency matrix into a BayesNet object.

    NOTE: This is for reading the structure only, and
    therefore no parameters for the BayesNet object will
    be set - they must be learned by calling "mle_estimator"
    or "bayes_estimator" on the object.
    """
    _V = []
    _E = {}
    _F = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.split(delim)
            rv = line[0]
            _E[rv] = []

    bn = BayesNet(_E)

    return bn

















