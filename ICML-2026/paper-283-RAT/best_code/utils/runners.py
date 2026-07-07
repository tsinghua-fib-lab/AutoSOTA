import torch
import numpy as np
from abc import ABC, abstractmethod
from utils.utils import model_step, gpt_model_step

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam, adv_type, device, test_mode=False):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.adv_type = adv_type
        self.device = device
        self.test_mode = test_mode

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_values, mb_actions, mb_dones, mb_logits = [],[],[],[],[],[]
        epinfos = []

        if self.model.obs_rms is not None:
            self.model.obs_rms.training = False

        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            # actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            actions, values, logits = model_step(self.model, self.obs, deterministic=self.test_mode)

            mb_obs.append(self.obs.clone())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_logits.append(logits)
            mb_dones.append(self.dones) 

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            if self.model.is_discrete:
                clipped_actions = actions # no clipping for discrete actions
            else:
                # clipped_actions = torch.clamp(actions, -1.0, 1.0) # clip actions to be in the range [-1, 1]
                act_lim = self.env.action_space.high[0]
                clipped_actions = torch.tanh(actions) * act_lim # squash actions to be in the range (-1, 1)

            self.obs, rewards, self.dones, infos = self.env.step(clipped_actions.cpu().numpy()) # done==true: a new episode just started

            for i, d in enumerate(self.dones):
                if d and infos[i].get('TimeLimit.truncated', False):
                    terminal_obs = infos[i]['terminal_observation'].unsqueeze(0)
                    bootstrap_val = model_step(self.model, terminal_obs)[1]
                    rewards[i] += self.gamma * bootstrap_val[0]

            for idx, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                if maybeepinfo: 
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

        if self.test_mode:
            return epinfos

        #batch of steps to batch of rollouts
        mb_obs = torch.stack(mb_obs, dim=0)
        mb_rewards = torch.from_numpy(np.asarray(mb_rewards)).to(self.device)
        mb_actions = torch.stack(mb_actions, dim=0)
        mb_values = torch.stack(mb_values, dim=0)
        mb_logits = torch.stack(mb_logits, dim=0)
        mb_dones = torch.from_numpy(np.asarray(mb_dones).astype(np.float32)).to(self.device)

        last_values = model_step(self.model, self.obs)[1]

        # discount/bootstrap off value fn
        mb_returns = torch.zeros_like(mb_rewards)
        mb_advs = torch.zeros_like(mb_rewards)
        mb_td_res = torch.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - torch.from_numpy(self.dones.astype(np.float32)).to(self.device)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            mb_td_res[t] = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = mb_td_res[t] + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        if self.adv_type == 'td':
            mb_advs = mb_td_res
        else: 
            assert self.adv_type == 'gae'

        # mb_obs = sf01(mb_obs)
        # if self.model.obs_rms is not None:
        #     self.model.obs_rms.training = True
        #     mb_obs = self.model.obs_rms(mb_obs)
        #     self.model.obs_rms.training = False
        # return (mb_obs, *map(sf01, (mb_returns, mb_actions, mb_advs, mb_logits)), epinfos)

        return (*map(sf01, (mb_obs, mb_returns, mb_actions, mb_advs, mb_logits)), epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class TrajRunner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam, device):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.device = device

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logits = [],[],[],[],[],[]
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, logits = model_step(self.model, self.obs)
            mb_obs.append(self.obs.clone())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_logits.append(logits)
            mb_dones.append(self.dones) 

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, rewards, self.dones, infos = self.env.step(actions.cpu().numpy()) # done==true: a new episode just started

            for idx, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                if maybeepinfo: 
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts
        mb_obs = torch.stack(mb_obs, dim=0)
        mb_rewards = torch.from_numpy(np.asarray(mb_rewards)).to(self.device)
        mb_actions = torch.stack(mb_actions, dim=0)
        mb_values = torch.stack(mb_values, dim=0)
        mb_logits = torch.stack(mb_logits, dim=0)
        mb_dones = torch.from_numpy(np.asarray(mb_dones).astype(np.float32)).to(self.device)

        last_values = model_step(self.model, self.obs)[1]
        mb_next_values = torch.zeros_like(mb_values)
        mb_next_values[:-1] = mb_values[1:]
        mb_next_values[-1] = last_values

        mb_advs = torch.zeros_like(mb_rewards)
        lastgaelam = 0

        # shifted dones
        mb_shifted_dones = torch.zeros_like(mb_dones)
        mb_shifted_dones[:-1] = mb_dones[1:]
        mb_shifted_dones[-1:] = torch.from_numpy(self.dones.astype(np.float32)).to(self.device)

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextvalues = last_values
            else:
                nextvalues = mb_values[t+1]
            nextnonterminal = 1.0 - mb_shifted_dones[t]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        return (*map(swap01, (mb_obs, mb_rewards, mb_actions, mb_values, mb_next_values, mb_advs, mb_logits, mb_shifted_dones)), epinfos)

def swap01(arr):
    return arr.transpose(0, 1)

class TrajRunnerAdv(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, device):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.device = device

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_logits = [],[],[],[],[]
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, _, logits = model_step(self.model, self.obs)
            mb_obs.append(self.obs.clone())
            mb_actions.append(actions)
            mb_logits.append(logits)
            mb_dones.append(self.dones) 

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, rewards, self.dones, infos = self.env.step(actions.cpu().numpy()) # done==true: a new episode just started

            for idx, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                if maybeepinfo: 
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts
        mb_obs = torch.stack(mb_obs, dim=0)
        mb_rewards = torch.from_numpy(np.asarray(mb_rewards)).to(self.device)
        mb_actions = torch.stack(mb_actions, dim=0)
        mb_logits = torch.stack(mb_logits, dim=0)
        mb_dones = torch.from_numpy(np.asarray(mb_dones).astype(np.float32)).to(self.device)

        # shifted dones
        mb_shifted_dones = torch.zeros_like(mb_dones)
        mb_shifted_dones[:-1] = mb_dones[1:]
        mb_shifted_dones[-1:] = torch.from_numpy(self.dones.astype(np.float32)).to(self.device)

        return (*map(swap01, (mb_obs, mb_rewards, mb_actions, mb_logits, mb_shifted_dones)), epinfos)


class SeqRunner(AbstractEnvRunner):
    def __init__(self, *, env, model, seq_len, device, is_gpt=False, reset_per_run=False, writer=None, preprocess=None, deterministic=False):
        super().__init__(env=env, model=model, nsteps=seq_len)
        self.is_gpt = is_gpt
        self.device = device
        self.writer = writer
        self.ts_elapsed = torch.zeros(len(self.dones)).long() # timestep starts at 1
        self.ttg_start_idx = torch.zeros(len(self.dones)).long()
        self.reset_per_run = reset_per_run
        self.preprocess = preprocess
        self.deterministic = deterministic

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_timeouts, mb_outputs, mb_tel = [],[],[],[],[],[],[]
        epinfos = []

        if self.is_gpt:
            self.model.eval()
            self.model.clear_cache() # resets the buffer
            if self.model.obs_rms is not None:
                self.model.obs_rms.training = False
        
        if self.reset_per_run:
            self.ts_elapsed = torch.zeros(len(self.dones)).long() # timestep starts at 1
            self.ttg_start_idx = torch.zeros(len(self.dones)).long()
            self.obs[:] = self.env.reset()
            self.dones = [False for _ in range(len(self.dones))]

        self.timeouts = np.zeros_like(self.dones)

        # For n in range number of steps
        for id_step in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            mb_timeouts.append(self.timeouts.copy())

            if self.is_gpt:
                mb_obs.append(self.obs.clone())
                mb_dones.append(self.dones)
                mb_tel.append(self.ts_elapsed.clone())
                actions, _, outputs = gpt_model_step(self.model, mb_obs, mb_dones, mb_tel, self.deterministic)
            else:
                actions, _, outputs = model_step(self.model, self.obs)
                mb_obs.append(self.obs.clone())
                mb_dones.append(self.dones) 
                mb_tel.append(self.ts_elapsed.clone())

            if self.writer is not None:
                self.writer.add_images('obs_image', self.obs, global_step=id_step)

            mb_actions.append(actions)
            mb_outputs.append(outputs)

            for i, d in enumerate(self.dones):
                if d: 
                    self.ts_elapsed[i] = -1

            self.obs, rewards, self.dones, infos = self.env.step(actions.cpu().numpy()) # done==true: a new episode just started
            self.ts_elapsed += 1

            # when done is set by truncation, final observation is saved in infos: terminal_observation
            # self.obs stores the new obs after reset
            self.timeouts = np.zeros_like(self.dones)
            for i, d in enumerate(self.dones):
                self.timeouts[i] = infos[i]['TimeLimit.truncated']
                # if d and (not self.timeouts[i]):
                #     self.obs[i] = torch.zeros_like(self.obs[i]) # reset the obs to 0

            for _, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                if maybeepinfo: 
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

        #batch of step
        mb_obs.append(self.obs.clone())
        mb_obs = torch.stack(mb_obs, dim=0)

        mb_rewards = torch.from_numpy(np.asarray(mb_rewards)).to(self.device)
        mb_actions = torch.stack(mb_actions, dim=0)
        mb_outputs = torch.stack(mb_outputs, dim=0)

        mb_dones.append(self.dones)  # add the last done
        mb_dones = torch.from_numpy(np.asarray(mb_dones).astype(np.float32)).to(self.device)

        mb_timeouts.append(self.timeouts)  # add the last done
        mb_timeouts = torch.from_numpy(np.asarray(mb_timeouts).astype(np.float32)).to(self.device)

        mb_tel.append(self.ts_elapsed.clone())
        mb_tel = torch.stack(mb_tel, dim=0).to(self.device)

        # convert mb_tel to mb_ttg
        # mb_ttg = torch.zeros_like(mb_tel)
        # ttg_start_idx = torch.zeros(len(self.dones)).long()
        # for i in range(self.nsteps+1):
        #     dones = mb_dones[i]
        #     for j, d in enumerate(dones):
        #         if d: 
        #             mb_ttg[ttg_start_idx[j]:i, j] = mb_tel[i-1, j] - mb_tel[ttg_start_idx[j]:i, j]
        #             ttg_start_idx[j] = i
        #         elif i == self.nsteps:
        #             # mb_ttg[ttg_start_idx[j]:, j] = mb_tel[i, j] - mb_tel[ttg_start_idx[j]:, j] # last step
        #             mb_ttg[ttg_start_idx[j]:, j] = torch.tensor(999).to(mb_tel.device) - mb_tel[ttg_start_idx[j]:, j] # last step
        
        mb_rollouts = {}
        mb_rollouts['obs']      = swap01(mb_obs)
        mb_rollouts['rewards']  = swap01(mb_rewards)
        mb_rollouts['actions']  = swap01(mb_actions)
        mb_rollouts['outputs']  = swap01(mb_outputs)
        mb_rollouts['dones']    = swap01(mb_dones)
        mb_rollouts['timeouts'] = swap01(mb_timeouts)
        mb_rollouts['tel']      = swap01(mb_tel)
        mb_rollouts['ttg']      = swap01(mb_tel)

        return (mb_rollouts, epinfos)

class EvalRunner(AbstractEnvRunner):
    def __init__(self, *, env, model, seq_len, device, is_gpt=False, writer=None, preprocess=None, deterministic=False):
        super().__init__(env=env, model=model, nsteps=seq_len)
        self.is_gpt = is_gpt
        self.device = device
        self.writer = writer
        self.idx = torch.zeros(len(self.dones)).long()
        self.preprocess = preprocess
        self.deterministic = deterministic

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_dones, mb_pos = [],[],[],[]

        if self.is_gpt:
            self.model.clear_cache() # resets the buffer
        
        self.idx = torch.zeros(len(self.dones)).long()
        self.obs[:] = self.env.reset()
        self.dones = [False for _ in range(len(self.dones))]

        ## HACK: torch.randn_like changes the randomness state of the GPU
        ## and the next sample will be different
        ## This is a workaround to keep the randomness state
        ## of the GPU the same as before
        ## This is not a problem for CPU, but for GPU it is
        ## very important to keep the randomness state
        ## otherwise the next sample will be different
        rng_state = torch.cuda.get_rng_state() 

        # For n in range number of steps
        idx = torch.zeros(len(self.dones)).long()
        for id_step in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            mb_pos.append(idx)

            if self.is_gpt:
                mb_obs.append(self.obs.clone())
                mb_dones.append(self.dones)
                actions, _, _ = gpt_model_step(self.model, mb_obs, mb_dones, mb_pos, self.deterministic)
            else:
                actions, _, _ = model_step(self.model, self.obs)
                mb_obs.append(self.obs.clone())
                mb_dones.append(self.dones) 

            if self.writer is not None:
                self.writer.add_images('obs_image', self.obs, global_step=id_step)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, rewards, dones, _ = self.env.step(actions.cpu().numpy()) # done==true: a new episode just started
            # when done, final observation is saved in infos: terminal_observation
            # self.obs stores the new obs after reset
            mb_rewards.append(rewards)
            idx += 1

            for i, d in enumerate(dones):
                if d and not self.dones[i]: 
                    self.dones[i] = True
                    self.idx[i] = idx[i]
                    idx[i] = 0

            if np.all(self.dones):
                break

        # in the case no done has been reached, we need to set the idx to the nsteps
        # and set the dones to 1
        for i, d in enumerate(self.dones):
            if not d:
                self.dones[i] = True
                self.idx[i] = self.nsteps

        #batch of step
        mb_rewards = torch.from_numpy(np.asarray(mb_rewards)).to(self.device)
        mb_rewards = swap01(mb_rewards)

        ep_returns = [mb_rewards[i][:self.idx[i]].sum().item() for i in range(len(self.dones))]
        ep_lens = [self.idx[i].item() for i in range(len(self.dones))]
        mean_returns = np.mean(ep_returns)
        mean_lens = np.mean(ep_lens)
        std_returns = np.std(ep_returns)
        std_lens = np.std(ep_lens)

        torch.cuda.set_rng_state(rng_state)

        return mean_returns, mean_lens, std_returns, std_lens