"""
Adapted from https://github.com/pricexu/DisCo/blob/main/moses_benchmark.py
"""

import pickle
import moses
from random import sample


file_name = ""
print(file_name)
with open(file_name, "rb") as fp:
    list_of_generated_smiles = pickle.load(fp)

# metrics = moses.get_all_metrics(sample(list_of_generated_smiles,2000))
metrics = moses.get_all_metrics(list_of_generated_smiles)
print(metrics)
print("Filters: {}".format(metrics['Filters']*100))
print("FCD/Test: {}".format(metrics['FCD/Test']))
print("SNN/Test: {}".format(metrics['SNN/Test']))
print("Scaf/TestSF: {}".format(metrics['Scaf/TestSF']*100))