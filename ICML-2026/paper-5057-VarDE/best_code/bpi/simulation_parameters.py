# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

from typing import NamedTuple
from enum import Enum

class EnvType(Enum):
    RIVERSWIM = 'Riverswim'
    FORKED_RIVERSWIM = 'ForkedRiverswim'


class SimulationParameters(NamedTuple):
    env_type: EnvType
    gamma: float
    river_length: int
    horizon: int
    n_sims: int
    frequency_evaluation: int
