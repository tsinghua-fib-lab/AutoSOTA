# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
import multiprocessing as mp
import pickle
import copy
import lzma
import os
policy_evaluation = None
try:
    import pyximport
    _ = pyximport.install(
        setup_args={"include_dirs": np.get_include()},
        reload_support=True,
    )
    from utils.cutils import policy_evaluation as policy_evaluation
except Exception:
    # Optional Cython acceleration; fall back to pure Python if build/import fails.
    policy_evaluation = None
from envs.riverswim import RiverSwim
from envs.forked_riverswim import ForkedRiverSwim
from tqdm import tqdm
from utils.utils import policy_evaluation as policy_evaluation_2, policy_iteration
if policy_evaluation is None:
    policy_evaluation = policy_evaluation_2
from simulation_parameters import SimulationParameters, EnvType
from agents.agent import AgentParameters, Experience
from utils.utils import Results
from make_agent import AgentType, make_agent
import time
from typing import NamedTuple, Sequence
from numpy.typing import NDArray
from utils.new_mdp_description import NewMDPDescription



class DataResults(NamedTuple):
    simulation_parameters: SimulationParameters
    agent_type: AgentType
    data: Sequence[Sequence[Results]]


class SequencedResults(NamedTuple):
    omega: NDArray[np.float64]
    greedy_policy: NDArray[np.int64]
    total_state_visits: NDArray[np.float64]
    last_visit: NDArray[np.float64]
    exp_visits: NDArray[np.float64]
    eval_greedy: NDArray[np.float64]
    dist_omega: NDArray[np.float64]
    U_omega: NDArray[np.float64]
    dist_value: NDArray[np.float64]
    dist_value_infinity: NDArray[np.float64]
    elapsed_times: NDArray[np.float64]
    simulation_parameters: SimulationParameters

def TV(p,q):
    return np.sum(np.abs(p-q), -1) * 0.5

def get_data(data: DataResults, true_mdp: NewMDPDescription, true_omega: NDArray[np.float64]) -> SequencedResults:
    compute_dist_omega = lambda x,y: TV(x,y)
    compute_dist_value2 = lambda V, mdp: np.linalg.norm(V -mdp.V_greedy[np.newaxis, np.newaxis], axis=-1)
    compute_dist_value_infinity = lambda V, mdp: np.linalg.norm(V -mdp.V_greedy[np.newaxis, np.newaxis], ord= np.inf, axis=-1)
    data_omega = []
    num_visits = []
    policies = []
    last_visit = []
    visits = []
    eval_greedy = []
    elapsed_times = []
    for i in range(data.simulation_parameters.n_sims):
        _, _data_omega, _greedy_policy, _num_visits, _last_visit, _visits, _eval_greedy, _elapsed_times = zip(*data.data[i])
        data_omega.append(_data_omega)
        num_visits.append(_num_visits)
        policies.append(_greedy_policy)
        last_visit.append(_last_visit)
        visits.append(_visits)
        eval_greedy.append(_eval_greedy)
        elapsed_times.append(_elapsed_times)
    
    data_omega = np.array(data_omega)
    num_visits = np.array(num_visits)
    policies = np.array(policies)
    last_visit = np.array(last_visit)
    visits = np.array(visits)
    eval_greedy = np.array(eval_greedy)
    elapsed_times = np.array(elapsed_times)

    
    omega = data_omega.reshape(data_omega.shape[0], data_omega.shape[1], -1)

    dist_omega = compute_dist_omega(omega, true_omega.flatten()[np.newaxis, np.newaxis, ...])
    U_omega = eval_allocations(omega, true_mdp, true_mdp.Q_greedy.shape[1])
    dist_value_2 = compute_dist_value2(eval_greedy, true_mdp)
    dist_value_infty = compute_dist_value_infinity(eval_greedy, true_mdp)
    
    return SequencedResults(data_omega, policies, num_visits, last_visit, visits, eval_greedy,
                            dist_omega, U_omega, dist_value_2,dist_value_infty, elapsed_times, data.simulation_parameters)


def eval_allocations(allocation, mdp, num_actions):
    eval_x = np.zeros((allocation.shape[0], allocation.shape[1]))
    for i in range(allocation.shape[0]):
        for j in range(allocation.shape[1]):
            eval_x[i,j] = mdp.evaluate_allocation(allocation[i,j].reshape(-1, num_actions))
            if np.isinf(eval_x[i,j]):
                eps = 1e-3
                U = np.ones_like(allocation[i,j].reshape(-1, num_actions)) / np.prod(allocation[i,j].shape)
                omega = (1-eps) * allocation[i,j].reshape(-1, num_actions) + eps * U
                eval_x[i,j] = mdp.evaluate_allocation(omega)
    return eval_x

    
def run(agent_type: AgentType, p: SimulationParameters):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    match p.env_type:
        case EnvType.RIVERSWIM:
            env = RiverSwim(num_states=p.river_length)
        case EnvType.FORKED_RIVERSWIM:
            env = ForkedRiverSwim(river_length=p.river_length)
        case _:
            raise NotImplementedError(f'Environment {env_type.value} not found.')

    optimal_V = policy_iteration(p.gamma, env.transitions, env.rewards)[0]
    start_time = time.time()
    s = env.reset()
    discount_factor = p.gamma
    agent_parameters = AgentParameters(dim_state_space=env.ns, dim_action_space=env.na, discount_factor=discount_factor, horizon=p.horizon)
    agent = make_agent(agent_type, agent_parameters)
    
    eval_greedy = policy_evaluation(p.gamma, env.transitions, env.rewards[..., np.newaxis], agent.greedy_policy)
    eval = [Results(0, agent.omega, agent.greedy_policy, agent.total_state_visits, agent.last_visit, agent.exp_visits, eval_greedy, time.time() - start_time)]
    
    
    moving_average_results = []

    for t in range(p.horizon):
        a = agent.forward(s, t)
        next_state, reward = env.step(a)
        exp = Experience(s, a, reward, next_state)
        agent.backward(exp, t)

        s = next_state

        if (t +1) % p.frequency_evaluation == 0:
            eval_greedy = np.asarray(policy_evaluation(p.gamma, env.transitions, env.rewards[..., np.newaxis], agent.greedy_policy))
            moving_average_results.append(np.linalg.norm(eval_greedy - optimal_V))
            eval.append(Results(t, agent.omega, agent.greedy_policy, agent.total_state_visits, agent.last_visit, agent.exp_visits, eval_greedy, time.time() - start_time))
    return eval

def run_agent(seed: int, agent_type: AgentType, parameters: SimulationParameters):
    np.random.seed(seed)
    return run(agent_type, parameters)

if __name__ == '__main__':
    NUM_PROCESSES = 10          # Specify number of processes
    N_SIMS = 10                 # Number of seeds
    FREQ_EVAL_GREEDY = 200      # Frequency evaluation greedy policy
    
    
    types = [
        (5, EnvType.RIVERSWIM, 10000),
        (3, EnvType.FORKED_RIVERSWIM, 20000),
        (10, EnvType.RIVERSWIM, 20000),
        (5, EnvType.FORKED_RIVERSWIM, 20000),
        (20, EnvType.RIVERSWIM, 40000),
        (10, EnvType.FORKED_RIVERSWIM, 100000),
        (30, EnvType.RIVERSWIM, 50000),
        (15, EnvType.FORKED_RIVERSWIM, 200000),
        (50, EnvType.RIVERSWIM, 80000),
        (25, EnvType.FORKED_RIVERSWIM, 300000),
    ]
    
    agents = [
        AgentType.BAYES_MFBPI, 
        AgentType.Q_UCB, 
        AgentType.PSRL,
        AgentType.Q_LEARNING, AgentType.VarDE_Q_LEARNING,
        AgentType.MDP_NAS, AgentType.PS_MDP_NAS, AgentType.O_BPI,
    ]

    for length, env_type, horizon in types:
        print(f'> Computing optimal allocation for {env_type.value}({length})... ')
        if env_type == EnvType.FORKED_RIVERSWIM:
            env = ForkedRiverSwim(length)
        else:
            env = RiverSwim(length)
        mdp = NewMDPDescription(env.transitions, env.rewards, 0.99)
        true_omega, _ = mdp.compute_allocation(navigation_constraints=True)
            
        for agent in agents:
            print(f'> Evaluating {agent.value} on {env_type.value}({length})', end='... ')
    
            path = f'./data/{env_type.value}/'
            if not os.path.exists(path):
                os.makedirs(path)

            data = {}
            
            num_states = length if env_type == EnvType.RIVERSWIM else length * 2

            data['simulation_parameters'] = SimulationParameters(
                env_type=env_type,
                gamma=0.99,
                river_length=length,
                horizon=horizon,
                n_sims = N_SIMS,
                frequency_evaluation=FREQ_EVAL_GREEDY
            )
            data['agent_type'] = agent

            iterations = [(seed, agent, data['simulation_parameters']) for seed in  range(data['simulation_parameters'].n_sims)]
            start_time = time.time()
            if NUM_PROCESSES > 1:
                with mp.Pool(NUM_PROCESSES) as p:
                    data_returned = list(p.starmap(run_agent, iterations))
            else:
                data_returned = list(map(lambda x: run_agent(*x), iterations))

            data['data'] = data_returned
            print(f'done in {np.round(time.time() - start_time, 2)} seconds.')
            
            
            data = DataResults(data['simulation_parameters'], data['agent_type'], data['data'])
            
            data = get_data(data, mdp, true_omega)

            with lzma.open(f'{path}/{agent.value}_{length}.pkl.lzma', 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
