"""
**************
Write BayesNet
to File
**************

Write a BayesNet object to a file. These
functions all currently work, but support
for more formats should be added in the future.
"""


__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu"""



import json
from collections import OrderedDict

def write_bn(bn, path):
    """
    Wrapper function for writing a BayesNet
    object to file

    Arguments
    ---------
    *bn* : a BayesNet object

    *path* : a string
        The path, absolute or relative. MUST contain
        the extension: '.bn' only support right now

    Returns
    -------
    None

    Effects
    -------
    - Creates a new file on the user's local system

    Notes
    -----
    - Should add support for '.bif' and others

    """
    if '.bn' in path:
        write_json(bn, path)
    else:
        print("File Extension not supported")

def write_json(bn, path):
    """
    Write a BayesNet object to a json format file

    Arguments:
        1. *filename* - the path/name of the file to which the function will write the BKB object.

    Overview
    --------


    Parameters
    ----------


    Returns
    -------


    Notes
    -----

    
    """
    bn_dict = OrderedDict([('V',bn.V),('E',bn.E),('F',bn.F)])
    with open(path, 'w') as outfile:
        json.dump(bn_dict, outfile,indent=2)








