"""Utility functions for tests."""


import numpy as np
import pandas as pd


def load_tree_table() -> pd.core.frame.DataFrame:
    """Load an example of table containing a tree structure."""
    tree_dict = {}
    tree_dict["Tree"] = [0]*13
    tree_dict["Node"] = list(range(13))
    tree_dict["ID"] = ["0-0", "0-1", "0-2", "0-3", "0-4", "0-5", "0-6", "0-7",
                       "0-8", "0-9", "0-10", "0-11", "0-12"]
    tree_dict["Feature"] = ["f3", "f2", "f0", "Leaf", "f0", "f1", "f1", "Leaf",
                            "Leaf", "Leaf", "Leaf", "Leaf", "Leaf"]
    tree_dict["Split"] = [-1.74, -1.51, 0.99, np.nan, -0.78, -1.61, 1.69,
                          np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    tree_dict["Yes"] = ["0-1", "0-3", "0-5", np.nan, "0-7", "0-9", "0-11",
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    tree_dict["No"] = ["0-2", "0-4", "0-6", np.nan, "0-8", "0-10", "0-12",
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    tree_dict["Missing"] = tree_dict["No"]
    tree_dict["Gain"] = [360.81, 53.89, 401.29, 1.88, 6.50, 87.59, 237.81,
                         0.96, 0.29, 0.88, 0.08, 0.55, 1.78]
    tree_dict["Cover"] = [500.0, 22.0, 478.0, 12.0, 10.0, 400.0, 78.0, 7.0,
                          3.0, 12.0, 388.0, 60.0, 18.0]
    tree_dict["Category"] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan]
    return pd.DataFrame.from_dict(tree_dict)


def eta_main(x: np.ndarray, rho: float) -> np.ndarray:
    """Compute main effect for the analytical example."""
    return(rho/(1 + rho**2)*(x**2 - 1))


def eta_order2(x: np.ndarray, z: np.ndarray, rho: float) -> np.ndarray:
    """Compute interactions for the analytical example."""
    return(rho*(1 - rho**2)/(1 + rho**2) - rho/(1 + rho**2)*(x**2 + z**2)
           + x*z)
