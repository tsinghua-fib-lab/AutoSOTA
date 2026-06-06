from components.operators import *
from components.Population import Population
# from env.optimizer_env import Optimizer
from config import get_options
from env import bbob
from utils.utils import set_seed
from utils.make_dataset import *
from env import DummyVectorEnv,SubprocVectorEnv
from tqdm import tqdm
import torch
import warnings, os
import numpy as np
from env.optimizer_mamba import Optimizer  as Optimizerm

warnings.filterwarnings("ignore")
config = get_options()
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def test(config, aid, batch, agent=None):
    set_seed(config.testseed)
    
    trajectories =[{'state': torch.zeros(config.MaxGen, 9), 'reward': torch.zeros(config.MaxGen), 'action': torch.zeros(config.MaxGen, len(batch[0].get_config_space().keys()), dtype=torch.int64)} for _ in range(len(batch) * config.repeat * 2,)]
    pbar = tqdm(total=config.MaxGen * config.repeat * 2, desc=str(aid))
    
    for o in batch:
        o.strategy_mode = 'RL'
    for rep in range(config.repeat):
        env_list=[lambda e=p: e for p in batch]
        envs = SubprocVectorEnv(env_list)
        envs.seed(config.testseed+rep)
        trng = torch.random.get_rng_state()
        state = torch.FloatTensor(envs.reset()).to(config.device)
        torch.random.set_rng_state(trng)
        for t in range(0, config.MaxGen):
            logits = agent.actor(state, 
                                to_critic=False,
                                detach_state=True,
                                )
                
            trng = torch.random.get_rng_state()
            next_state, rewards, _, info = envs.step(logits.detach().cpu())
            # print(info)
            for j in range(len(batch)):
                trajectories[j*config.repeat + rep]['state'][t] = torch.from_numpy(info[j]['pre_state'])
                ia = 0
                for op in info[j]['action_values']:
                    for a in op:
                        trajectories[j*config.repeat + rep]['action'][t][ia] = a.item()
                        ia += 1
                trajectories[j*config.repeat + rep]['reward'][t] = rewards[j]

            # next_state, rewards, _, info = envs.step(torch.rand(batch_size, config.maxCom, config.maxAct))
            torch.random.set_rng_state(trng)
            state=torch.FloatTensor(next_state).to(config.device)
            pbar.update()
        envs.close()

    for o in batch:
        o.strategy_mode = 'Default'
    for rep in range(config.repeat):
        env_list=[lambda e=p: e for p in batch]
        envs = SubprocVectorEnv(env_list)
        envs.seed(config.testseed+rep)
        trng = torch.random.get_rng_state()
        state = torch.FloatTensor(envs.reset()).to(config.device)
        torch.random.set_rng_state(trng)
        for t in range(config.MaxGen):
            logits = torch.rand(len(batch), config.maxCom, config.maxAct * 2)
            # logits = [None]*len(batch)
            trng = torch.random.get_rng_state()
            next_state, rewards, _, info = envs.step(logits)
            # print(info)
            for j in range(len(batch)):
                trajectories[len(batch)*config.repeat + j*config.repeat + rep]['state'][t] = torch.from_numpy(info[j]['pre_state'])
                ia = 0
                for op in info[j]['action_values']:
                    for a in op:
                        trajectories[len(batch)*config.repeat + j*config.repeat + rep]['action'][t][ia] = a
                        ia += 1
                trajectories[len(batch)*config.repeat + j*config.repeat + rep]['reward'][t] = rewards[j]
            # next_state, rewards, _, info = envs.step(torch.rand(batch_size, config.maxCom, config.maxAct))
            torch.random.set_rng_state(trng)
            state=torch.FloatTensor(next_state).to(config.device)
            pbar.update()
        envs.close()
    pbar.close()
    with open(f'trajectory_set_{aid}.pkl', 'wb') as f:
        pickle.dump(trajectories, f)


def mamba_action_interpret(logits, alg, bins=16):
    space = alg.get_config_space()
    alg_actions = {}
    action = []
    for i, subspace in enumerate(space.keys()):
        rge = space[subspace]
        if isinstance(rge[0], float):
            ac = torch.argmax(logits[i])
            alg_actions[subspace] = (ac / (bins - 1)) * (rge[1] - rge[0]) + rge[0]
            action.append(ac)
        else:
            na = len(rge)
            ac = torch.argmax(logits[i][:na])
            alg_actions[subspace] = ac
            action.append(ac)
    return action, alg_actions


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy('file_system')
    set_seed(config.dataseed)
    algs, tasks = make_algorithms_mamba(config)

