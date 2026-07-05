from collections import defaultdict

import numpy as np
import pandas as pd


class Logger(defaultdict):
    def __init__(self):
        super().__init__(list)

    def to_dataframe(self):
        df = pd.DataFrame()
        for k, v in self.items():
            if isinstance(v[0], list | np.ndarray):
                history = np.array(v).T
                for i in range(len(history)):
                    df[f"{k}_{i}"] = history[i]
            else:
                df[k] = v

        return df
