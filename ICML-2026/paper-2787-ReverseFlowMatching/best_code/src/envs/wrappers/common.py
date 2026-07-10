# Code adapted from https://github.com/ikostrikov/jaxrl

from typing import Tuple

import numpy as np

TimeStep = Tuple[np.ndarray, float, bool, bool, dict]
