# We can test whether the status is staying in equilibrium
# by Using Equilibrium index

# Load data from real world or create
import pandas as pd

from AttributionParameter import attribution_parameter
from EquilibriumIndex import equilibrium_index_dynamic
from EquilibriumParameter import equilibrium_parameter
from LoadData import load_data
from ModelChoice import model_choice
from SplitDataSet import split1, split2

initial_data = pd.read_csv("data.csv")
raw_data = load_data(initial_data)

# split train and test sets
# y is the first column, others are x
#x_train, x_test, y_train, y_test = split1(raw_data, 0.8)

train_static, test_static = split2(raw_data, 0.8)

# choice model
choice_static = model_choice(train_static[:, 2:], train_static)

# calculate attribution parameter for train and test datasets in each part
coefficient_train_set = []
for i in train_static[:, -1]:
    part_num = i
    x_train = train_static[:, 2:-1]
    y_train = train_static[:, 1]
    coefficient_train = attribution_parameter(choice_static, x_train, y_train)
    coefficient_train_set[i] = coefficient_train

x_test = test_static[:, 2:]
y_test = test_static[:, 1]
coefficient_test_set = attribution_parameter(choice_static, x_test, y_test)


# calculate the equilibrium parameter for train and test datasets
ep_train = equilibrium_parameter(coefficient_train_set, x_train)
ep_test = equilibrium_parameter(coefficient_test_set, x_test)

# test the equilibrium status
ESI = equilibrium_index_dynamic(ep_test, ep_train, 20)

if ESI >= 0:
    print("Equilibrium status is broken.")
else:
    print("Equilibrium status is keeping.")
