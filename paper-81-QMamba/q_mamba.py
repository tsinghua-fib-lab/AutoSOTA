from network import DAC_block
import torch
import torch.nn.functional as F
import random
import numpy as np
class Q_Mamba():
    def __init__(self,
                 state_dim=9,
                 actions_dim=5,
                 action_bins=16,
                 d_state=32,
                 d_conv=4,
                 expand=2,
                 num_hidden_mlp=32,
                 device='cuda',
                 mamba_num = 1,
                 gamma = 0.99,
                 from_pretrain = None,
                 seed = 999,
                 lr = 5e-3,
                 lr_decay = False):
        self.state_dim = state_dim
        self.actions_dim=actions_dim
        self.action_bins=action_bins
        self.device=device
        self.gamma = gamma
        self.lr_decay =lr_decay
        # initialize the DAC_block network
        if from_pretrain is None:
            # set the random seed
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            self.dac_block = DAC_block(self.state_dim,
                                    self.actions_dim,
                                    self.action_bins,
                                    d_state,
                                    d_conv,
                                    expand,
                                    num_hidden_mlp,
                                    mamba_num).to(self.device)
            self.optimizer = torch.optim.AdamW(self.dac_block.parameters(), lr)
            # optional scheduler
            if lr_decay:
                self.lr_adapter = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[50,75,100,125], 0.8)
        else:
            self.dac_block = self.load_model(from_pretrain).to(self.device)
            self.optimizer = torch.optim.AdamW(self.dac_block.parameters(), lr)
        
    def train(self):
        self.dac_block.train()
    
    def eval(self):
        self.dac_block.eval()
    
    def save_model(self, save_path):
        torch.save(self.dac_block, save_path)

    def load_model(self, load_path):
        return torch.load(load_path)

    def int_to_binary_float_tensor(self, int_tensor):
        device = int_tensor.device
        binary_strings = [bin(i)[2:].zfill(self.actions_dim) for i in int_tensor.view(-1).tolist()]
        float_tensor = torch.tensor([[float(bit) for bit in binary_str] for binary_str in binary_strings])
        return float_tensor.view(*int_tensor.shape, -1).to(device)

    def parse_traj_to_input(self, batch_traj): # batch_traj.shape: bs  *  {[s0, a_0_1, ..., a_0_k, r0], [s1, a_1_1, ....a_1_k, r1],......}
        bs,ls,num_actions = batch_traj['action'].shape
        
        batch_state = batch_traj['state'] # bs * ls * 9
        batch_state_augmented = batch_state.repeat(1,1,num_actions).unsqueeze(-2).reshape(bs,ls*num_actions,-1) # bs * (ls*num_actions) * 9
        batch_ones_actions = torch.ones(bs,ls,1,self.actions_dim).to(self.device)
        batch_action_except_last_one = self.int_to_binary_float_tensor(batch_traj['action'])[:,:,:-1,:] #  bs * ls * (num -1) *action_dim
        batch_action = torch.concat([batch_ones_actions,batch_action_except_last_one], dim=-2).reshape(bs,ls*num_actions,-1)
        return torch.concat([batch_state_augmented,batch_action],dim=-1) # ..shape: bs * (ls * num_actions) * (9 + 5), 
    
    '''
    execute on-policy q-learning: min belman_loss = 0.5 * (q_pred - q_target) **2
    '''
    
    def learn_from_trajectory(self, batch_traj, has_conservative_reg_loss = True, beta = 10.0, alpha = 1.0): # batch_traj.shape: bs  *  {[s0, a_0_1, ..., a_0_k, r0], [s1, a_1_1, ....a_1_k, r1],......}
        bs, ls, num_actions = batch_traj['action'].shape
        batch_inputs = self.parse_traj_to_input(batch_traj).to(self.device)
        batch_qvalues = self.dac_block(batch_inputs) # -> bs * (ls * num_actions) * 16
        batch_action_mask = batch_traj['action'].reshape(bs,-1).unsqueeze(-1) # ..shape: bs * (ls * num_actions) * 1
        # constructing q_pred
        batch_q_pred = torch.gather(batch_qvalues, dim=-1, index=batch_action_mask.to(self.device)).reshape(bs, ls, num_actions)
        # batch_q_pred = batch_qvalues[batch_action_mask].reshape(bs, ls, num_actions) # ..shape: bs * (ls * num_actions) * 1
        # constructing q_target
        batch_reward = batch_traj['reward'] # ..shape: bs * ls 
        batch_q_target = batch_qvalues.detach() # ..shape: bs * (ls * num_actions) * 16
        batch_q_target_max_one = torch.max(batch_q_target, dim=-1).values.reshape(bs, ls, num_actions)
        q_pred_rest_actions, q_pred_last_action      = batch_q_pred[..., :-1], batch_q_pred[..., -1]
        q_target_first_action, q_target_rest_actions = batch_q_target_max_one[..., 0], batch_q_target_max_one[..., 1:]
        # print((q_pred_rest_actions-q_target_rest_actions).min(), (q_pred_rest_actions-q_target_rest_actions).max())
        losses_all_actions_but_last = F.mse_loss(q_pred_rest_actions, q_target_rest_actions)

        # next take care of the very last action, which incorporates the rewards

        q_target_last_action = torch.concat([q_target_first_action[..., 1:], torch.zeros(bs,1).to(self.device)], dim = -1)
        q_target_last_action = batch_reward + (self.gamma * q_target_last_action) # bs * ls
        losses_last_action = F.mse_loss(q_pred_last_action, q_target_last_action)
        # print(losses_all_actions_but_last,losses_last_action)
        # td_loss = losses_all_actions_but_last + losses_last_action
        td_loss = losses_all_actions_but_last + beta * losses_last_action

        if has_conservative_reg_loss:
            q_preds = batch_qvalues.reshape(bs * ls * num_actions , self.action_bins) # shape  bs * (ls * num_actions) * 16 -> (bs * ls * num_actions) * 16
            num_non_dataset_actions = self.action_bins - 1
            actions = batch_traj['action'].reshape(bs * ls * num_actions, 1) # shape bs * (ls * num_actions) -> (bs * ls * num_actions) * 1
            dataset_action_mask = torch.zeros_like(q_preds).scatter_(-1, actions, torch.ones_like(q_preds))
            q_actions_not_taken = q_preds[~dataset_action_mask.bool()]
            conservative_reg_loss = ((q_actions_not_taken) ** 2).mean() 
            total_loss = 0.5*td_loss + 0.5*alpha * conservative_reg_loss
        else:
            total_loss = td_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        

        if has_conservative_reg_loss:
            return total_loss.cpu().item(), td_loss.cpu().item(), conservative_reg_loss.cpu().item()
        return total_loss.cpu().item()
        
    def rollout_trajectory(self, env, maxGens, need_trajectory = False, need_return_cost = False, reset_context_every = 300):
        '''
        return trajectory if need_trajectory is True
        trajectory: {parse_traj_to_input, rewards}
        '''
        with torch.no_grad():
            state = env.reset().unsqueeze(0).unsqueeze(0).float().to(self.device) # 1, 1, 9
            input_embedding = None
            config_space = env.get_config_space()
            total_reward = 0
            if need_trajectory:
                rewards = []
                actions = []
            if need_return_cost:
                best_cost = 0
            
            for gen in range(maxGens):
                # Iter-4: Periodic context reset every 300 gens to prevent long-seq degradation
                if reset_context_every is not None and gen > 0 and gen % reset_context_every == 0:
                    input_embedding = None
                actions_for_this_gen = {}
                action_embedding = torch.ones(1,1,self.actions_dim).to(self.device)
                for action_step, key in enumerate(config_space.keys()):
                    rge = config_space[key]
                    if isinstance(rge[0], float):
                        config_range = self.action_bins
                    else:
                        config_range = len(rge)
                    # if action_step == 0:
                        # input_embedding = torch.concat([state,action_embedding], dim=-1)
                    if input_embedding is None:
                        # print(state.shape, action_embedding.shape)
                        input_embedding = torch.concat([state,action_embedding], dim=-1).to(self.device)
                    else:
                        # print(state.shape, action_embedding.shape, input_embedding.shape, torch.concat([state,action_embedding], dim=-1).shape)
                        input_embedding = torch.concat([input_embedding, torch.concat([state,action_embedding], dim=-1).to(self.device)], dim = 1)
                        # print(input_embedding.shape)
                    q_values = self.dac_block(input_embedding)
                    # print(q_values[:,-1,:config_range])
                    action_id = torch.argmax(q_values[:,-1,:config_range], dim=-1)
                    # print(action_step, action_id)
                    actions_for_this_gen[key] = action_id
                    action_embedding = self.int_to_binary_float_tensor(action_id).unsqueeze(0).to(self.device)
                        # input_embedding = torch.stack([input_embedding, torch.concat([state,action_embedding], dim=-1)], dim = 1)
                    # else:
                        # q_values = self.dac_block(input_embedding)
                        # action_id = torch.argmax(q_values[:,-1,:config_range], dim=-1)
                        # actions_for_this_gen[key] = action_id
                        # action_embedding = self.int_to_binary_float_tensor(action_id)
                        # if action_step < (len(config_space.keys()) -1):
                        #     input_embedding = torch.stack([input_embedding, torch.concat([state,action_embedding], dim=-1)], dim = 1)
                # give the selected action to env
                state, reward, _, cost = env.mamba_step(actions_for_this_gen)
                state = state.unsqueeze(0).unsqueeze(0).float().to(self.device)
                total_reward += reward
                if need_trajectory:
                    actions.append(list(actions_for_this_gen.values()))
                    rewards.append(reward)
                if need_return_cost and gen == maxGens - 1:
                    best_cost = cost
                
            if need_trajectory:
                actions = torch.tensor([actions]).to(self.device).to(torch.int64)
                rewards = torch.tensor([reward]).to(self.device).float()
                trajectory = {"parse_traj_to_input":input_embedding, "rewards":rewards, "actions":actions}
                return trajectory
            if need_return_cost:
                return total_reward, best_cost['gbest_val']
            return total_reward

    def learn_from_online_trajectory(self, env, maxGens, has_conservative_reg_loss = True, alph = 1.0):
        '''
        execute online q-learning
        '''
        bs, ls, num_actions = 1, maxGens, len(env.get_config_space())
        trajectory = self.rollout_trajectory(env, maxGens, need_trajectory=True)
        batch_inputs = trajectory['parse_traj_to_input']
        batch_qvalues = self.dac_block(batch_inputs) # -> bs * (ls * num_actions) * 16
        batch_action_mask = trajectory["actions"].reshape(bs,-1).unsqueeze(-1) # ..shape: bs * (ls * num_actions) * 1
        # constructing q_pred
        batch_q_pred = torch.gather(batch_qvalues, dim=-1, index=batch_action_mask.to(self.device)).reshape(bs, ls, num_actions)
        # batch_q_pred = batch_qvalues[batch_action_mask].reshape(bs, ls, num_actions) # ..shape: bs * (ls * num_actions) * 1
        # constructing q_target
        batch_reward = trajectory['rewards'] # ..shape: bs * ls 
        batch_q_target = batch_qvalues.detach() # ..shape: bs * (ls * num_actions) * 16
        batch_q_target_max_one = torch.max(batch_q_target, dim=-1).values.reshape(bs, ls, num_actions)
        q_pred_rest_actions, q_pred_last_action      = batch_q_pred[..., :-1], batch_q_pred[..., -1]
        q_target_first_action, q_target_rest_actions = batch_q_target_max_one[..., 0], batch_q_target_max_one[..., 1:]
        # print((q_pred_rest_actions-q_target_rest_actions).min(), (q_pred_rest_actions-q_target_rest_actions).max())
        losses_all_actions_but_last = F.mse_loss(q_pred_rest_actions, q_target_rest_actions)

        # next take care of the very last action, which incorporates the rewards

        q_target_last_action = torch.concat([q_target_first_action[..., 1:], torch.zeros(bs,1).to(self.device)], dim = -1)
        q_target_last_action = batch_reward + (self.gamma * q_target_last_action) # bs * ls
        losses_last_action = F.mse_loss(q_pred_last_action, q_target_last_action)
        # print(losses_all_actions_but_last,losses_last_action)
        # td_loss = losses_all_actions_but_last + losses_last_action
        td_loss = losses_all_actions_but_last + 10 * losses_last_action

        if has_conservative_reg_loss:
            q_preds = batch_qvalues.reshape(bs * ls * num_actions , self.action_bins) # shape  bs * (ls * num_actions) * 16 -> (bs * ls * num_actions) * 16
            num_non_dataset_actions = self.action_bins - 1
            actions =  trajectory["actions"].reshape(bs * ls * num_actions, 1) # shape bs * (ls * num_actions) -> (bs * ls * num_actions) * 1
            dataset_action_mask = torch.zeros_like(q_preds).scatter_(-1, actions, torch.ones_like(q_preds))
            q_actions_not_taken = q_preds[~dataset_action_mask.bool()]
            conservative_reg_loss = ((q_actions_not_taken) ** 2).mean() 
            total_loss = 0.5*td_loss + 0.5*alph * conservative_reg_loss
        else:
            total_loss = td_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        

        if has_conservative_reg_loss:
            return total_loss.cpu().item(), td_loss.cpu().item(), conservative_reg_loss.cpu().item()
        return total_loss.cpu().item()
        






