# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

from agents.agent import Agent, AgentParameters
from agents.qlearning import QLearning, QLearningParameters
from agents.qucb import QUCB, QUCBParameters
from agents.mfbpi import MFBPIParameters, MFBPI
from agents.bpi import BPIParameters, BPI, BPIType
from agents.obpi import OBPIParameters, OBPI
from agents.psrl import PSRL
from agents.varde import VarDE, VarDEParameters
from agents.mfbpi_varde import VarDEMFBPIParameters, VarDEMFBPI
from enum import Enum

class AgentType(Enum):
    Q_LEARNING = 'Q-Learning'
    VarDE_Q_LEARNING = 'VarDE-Q-Learning'
    Q_UCB = 'Q-UCB'
    BAYES_MFBPI = 'Bayes-MFBPI'
    FORCED_MFBPI = 'Forced-MFBPI'
    MDP_NAS = 'MDP-NaS'
    PSRL = 'PSRL'
    PS_MDP_NAS = 'PS-MDP-NaS'
    O_BPI = 'O-BPI'
    VarDEMFBPI = 'MFBPI-VarDE'
QUCB_PARAMETERS = QUCBParameters(confidence=1e-3)
QLEARNING_PARAMETERS = QLearningParameters()
VarDE_PARAMETERS = VarDEParameters()
BAYES_MFBPI_PARAMETERS = MFBPIParameters(kbar=1, ensemble_size=50)
FORCED_MFBPI_PARAMETERS = MFBPIParameters(kbar=1, ensemble_size=1)
MFBPI_VarDE_PARAMETERS = VarDEMFBPIParameters(kbar=1, ensemble_size=50)
MDP_NAS_PARAMETERS = BPIParameters(
    frequency_computation_greedy_policy=200, frequency_computation_omega=200, kbar=None, enable_posterior_sampling=False, bpi_type=BPIType.ORIGINAL_BOUND)
PS_MDP_NAS_PARAMETERS = BPIParameters(
    frequency_computation_greedy_policy=200, frequency_computation_omega=200, kbar=None, enable_posterior_sampling=True, bpi_type=BPIType.ORIGINAL_BOUND)
OBPI_PARAMETERS = OBPIParameters(frequency_computation=200, kbar=1)

def make_agent(agent_name: AgentType, agent_parameters: AgentParameters) -> Agent:
    match agent_name:
        case AgentType.Q_LEARNING:
            return QLearning(QLEARNING_PARAMETERS, agent_parameters)
        case AgentType.VarDE_Q_LEARNING:
            return VarDE(VarDE_PARAMETERS, agent_parameters)
        case AgentType.Q_UCB:
            return QUCB(QUCB_PARAMETERS, agent_parameters)
        case AgentType.BAYES_MFBPI:
            return MFBPI(BAYES_MFBPI_PARAMETERS, agent_parameters)
        case AgentType.FORCED_MFBPI:
            return MFBPI(FORCED_MFBPI_PARAMETERS, agent_parameters)
        case AgentType.PSRL:
            return PSRL(agent_parameters)
        case AgentType.MDP_NAS:
            return BPI(MDP_NAS_PARAMETERS, agent_parameters)
        case AgentType.PS_MDP_NAS:
            return BPI(PS_MDP_NAS_PARAMETERS, agent_parameters)
        case AgentType.O_BPI:
            return OBPI(OBPI_PARAMETERS, agent_parameters)
        case AgentType.VarDEMFBPI:
            return VarDEMFBPI(MFBPI_VarDE_PARAMETERS, agent_parameters)
        case _:
            raise NotImplementedError(f'Type {agent_name.value} not found.')
