# dynamic equilibrium
# the data set will be panel data

# We can test whether the status is staying in equilibrium
# by Using Equilibrium index

# Load data from real world or create

import pandas as pd

from AttributionParameter import attribution_parameter
from EquilibriumIndex import equilibrium_index_dynamic
from EquilibriumParameter import equilibrium_parameter
from LoadData import load_data
from ModelChoice import model_choice
from SplitDataSet import split1, split2, split_time

initial_data = pd.read_csv("data.csv")
raw_data = load_data(initial_data)

# split train and test sets
train_dynamic, test_dynamic = split2(raw_data[:, :-2], 0.8)

# choice model
choice_dynamic = model_choice(train_dynamic[:, 2:], train_dynamic)

# split data by time, the data set is panel data. So, we choose last time (t0) data
# for test set, and setting p to get train set from t-p to t-1.
train, test = split_time(raw_data)

# calculate attribution parameter (train set) for train and test datasets in each part
coefficient_train_set = []
for i in train[:, -1]:
    part_num = i
    x_train = train[:, 2:-1]
    y_train = train[:, 1]
    coefficient_train = attribution_parameter(choice_dynamic, x_train, y_train)
    coefficient_train_set[i] = coefficient_train

# calculate attribution parameter (test set) for train and test datasets in each part
x_test = test[:, 2:]
y_test = test[:, 1]
coefficient_test_set = attribution_parameter(choice_dynamic, x_test, y_test)

# calculate the equilibrium parameter for train and test datasets
ep_train = equilibrium_parameter(coefficient_train_set, x_train)
ep_test = equilibrium_parameter(coefficient_test_set, x_test)

# test the equilibrium status
ESI = equilibrium_index_dynamic(ep_test, ep_train, 20)

if ESI >= 0:
    print("Equilibrium status is broken.")
else:
    print("Equilibrium status is keeping.")