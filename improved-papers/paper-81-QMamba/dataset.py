from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import numpy as np
from einops import pack, unpack, repeat, reduce, rearrange

class My_Dataset(Dataset):
    def __init__(self, dataset_path, device='cuda'):
        super(My_Dataset).__init__()
        self.device = device
        if dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as f:
                self.data = pickle.load(f)   
        else:
            assert False, "Unsupported dataset format"
        
        # self.data 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        actions = data_item["action"]
        rewards = data_item["reward"]
        states = data_item["state"]

        #
        states = states.clone()
        actions = actions.clone().to(dtype=torch.int64)
        rewards = rewards.clone().to(dtype=torch.float32)
        return states.to(self.device), actions.to(self.device), rewards.to(self.device)

def int_to_binary_float_tensor(int_tensor, action_dim):

    device = int_tensor.device
    
    binary_strings = [bin(i)[2:].zfill(action_dim) for i in int_tensor.view(-1).tolist()]
    
    float_tensor = torch.tensor([[float(bit) for bit in binary_str] for binary_str in binary_strings])
    
    return float_tensor.view(*int_tensor.shape, -1).to(device)

def maybe_append_actions(sos_tokens, actions = None, action_dim = 5):
        # print('actions:', actions.shape)
        # print('sos_tokens:', sos_tokens.shape)
        batch = sos_tokens.shape[0]
        if actions is None:
            start_binary = torch.zeros(batch, 1, action_dim, device = sos_tokens.device)
            token = torch.cat((sos_tokens, start_binary), dim=-1)
            # print('not exist actions')
            return token
        
        batch, num_actions = actions.shape
        actions_binary_float = int_to_binary_float_tensor(actions, action_dim)
        start_binary = torch.zeros(batch, 1, action_dim, device = actions.device)
        binary_float = torch.cat((start_binary, actions_binary_float[:,:-1,]), dim=1)
        sos_tokens = repeat(sos_tokens, 'b 1 d -> b n d', n = num_actions)
        # print('sos_tokens:', sos_tokens.shape)
        # print('binary_float:', binary_float.shape)
        token = torch.cat((sos_tokens, binary_float), dim=-1)
        return token
# def test():
#     dataset = Dataset("/home/data3/ZhouJiang2/AAAAAAAAA/Gong/Q-Mamba-Long-term/trajectory_set_16_.pkl")
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#     for states, actions, rewards in dataloader:
#         print(states.shape, actions.shape, rewards.shape)
#         state =rearrange(states, 'b n d -> (b n) d')
#         action = rearrange(actions, 'b n d -> (b n) d')
#         state = reduce(state, 'b d -> b 1 d', 'mean')
#         new = maybe_append_actions(state, action)
#         print(new.shape)
        
        
# if __name__ == "__main__":
#     test()
        
        
        