from tqdm import tqdm
import numpy as np
import torch

def node_id(i, j, N):
        return i * N + j
def get_edge_list(N):
    edge_list=[]
    for i in range(N):
            for j in range(N):
                curr = node_id(i, j, N)
               
                if i + 1 < N:
                    down = node_id(i + 1, j, N)
                    edge_list.append([curr, down])
                    edge_list.append([down, curr])  
              
                if j + 1 < N:
                    right = node_id(i, j + 1, N)
                    edge_list.append([curr, right])
                    edge_list.append([right, curr])  
    return np.transpose(edge_list)


class GraphDataset_paired(torch.utils.data.Dataset):
    def __init__(self, index_list, data_dir, device):
        self.index_list=index_list
        self.l_pos=[]
        self.l_e=[]
        self.l_y=[]   
        self.h_pos=[]
        self.h_e=[]
        self.h_y=[]
        self.device=device
        
       

        
        f= np.load(data_dir+"/cfd32.npy")
      

        l_y=torch.FloatTensor(((f[self.index_list])).reshape(-1,32*32,1))
        self.l_y=(l_y-torch.min(l_y,1, keepdim=True).values)/(torch.max(l_y,1, keepdim=True).values-torch.min(l_y,1, keepdim=True).values)
        self.l_pos=torch.FloatTensor(np.concatenate([np.arange(0,1,1/32).reshape(32,1).repeat(32,1).reshape(-1,1), np.arange(0,1,1/32).reshape(1,32).repeat(32,0).reshape(-1,1)],-1))
        self.l_e=torch.LongTensor(get_edge_list(32))
            
            
        f= np.load(data_dir+"/cfd1024.npy")


      
        h_y=torch.FloatTensor(((f[self.index_list])).reshape(-1,1024*1024,1))
      

            
        self.h_pos=torch.FloatTensor(np.concatenate([np.arange(0,1,1/1024).reshape(1024,1).repeat(1024,1).reshape(-1,1), np.arange(0,1,1/1024).reshape(1,1024).repeat(1024,0).reshape(-1,1)],-1))
        self.h_e=torch.LongTensor(get_edge_list(1024))
        self.h_y=(h_y-torch.min(l_y,1, keepdim=True).values)/(torch.max(l_y,1, keepdim=True).values-torch.min(l_y,1, keepdim=True).values)
        
            
                      
    def __len__(self):
        return len(self.index_list)
    
        
    def __getitem__(self,idx):
        return self.l_pos.to(self.device), self.l_y[idx].to(self.device), self.l_e.to(self.device), self.h_pos.to(self.device), self.h_e.to(self.device), self.h_y[idx].to(self.device)
    



class GraphDataset_unpaired(torch.utils.data.Dataset):
    def __init__(self, index_list, data_dir, device):
        self.index_list=index_list
        self.l_pos=[]
        self.l_e=[]
        self.l_y=[]
        self.h_pos=[]
        self.h_e=[]
        self.device=device
        
           
        
        f= np.load(data_dir+"/cfd32.npy")
        l_y=torch.FloatTensor(((f[self.index_list])).reshape(-1,32*32,1))
        self.l_y=(l_y-torch.min(l_y,1, keepdim=True).values)/(torch.max(l_y,1, keepdim=True).values-torch.min(l_y,1, keepdim=True).values)
        self.l_pos=torch.FloatTensor(np.concatenate([np.arange(0,1,1/32).reshape(32,1).repeat(32,1).reshape(-1,1), np.arange(0,1,1/32).reshape(1,32).repeat(32,0).reshape(-1,1)],-1))
        self.l_e=torch.LongTensor(get_edge_list(32))
            
            
       
            
        self.h_pos=torch.FloatTensor(np.concatenate([np.arange(0,1,1/1024).reshape(1024,1).repeat(1024,1).reshape(-1,1), np.arange(0,1,1/1024).reshape(1,1024).repeat(1024,0).reshape(-1,1)],-1))
        self.h_e=torch.LongTensor(get_edge_list(1024))
        
        
            
                      
    def __len__(self):
        return len(self.index_list)
    
        
    def __getitem__(self,idx):
        return self.l_pos.to(self.device), self.l_y[idx].to(self.device), self.l_e.to(self.device), self.h_pos.to(self.device), self.h_e.to(self.device)
        
        

