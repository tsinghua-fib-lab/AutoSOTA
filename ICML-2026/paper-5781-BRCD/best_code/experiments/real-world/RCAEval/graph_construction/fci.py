import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci


def fci_default(data: pd.DataFrame):
    node_names = data.columns.to_list()

    # fill nan by ffill
    data = data.fillna(method="ffill")

    data = data.to_numpy().astype(float)

    output = fci(data, node_names=node_names, verbose=False)
    adj = output[0].graph
    return adj
