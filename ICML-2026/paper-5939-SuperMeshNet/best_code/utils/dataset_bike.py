from tqdm import tqdm
import numpy as np
import torch



class GraphDataset_paired(torch.utils.data.Dataset):
    def __init__(self, index_list, data_dir, device):
        self.index_list=index_list
        self.l_pos=[]
        self.l_e=[]
        self.l_y=[]   
        self.h_pos=[]
        self.h_e=[]
        self.h_y=[]
        
       

        
        for n in tqdm((index_list)):
            l_pos=np.load(data_dir+str(n)+"/L_point.npy")
            l_e=np.load(data_dir+str(n)+"/L_edge.npy")
            l_y=np.load(data_dir+str(n)+"/L_y.npy")
            h_pos=np.load(data_dir+str(n)+"/H_point.npy")
            h_e=np.load(data_dir+str(n)+"/H_edge.npy")
            h_y=np.load(data_dir+str(n)+"/H_y.npy")
            l_e=torch.LongTensor(l_e)
            h_e=torch.LongTensor(h_e)
            self.l_pos.append(torch.FloatTensor((l_pos-np.min(l_pos,0))/(np.max(l_pos,0)-np.min(l_pos,0))).to(device))
            self.l_e.append(l_e.to(device))
            self.l_y.append(torch.FloatTensor(((l_y-np.min(l_y,0, keepdims=True))/(np.max(l_y,0, keepdims=True)-np.min(l_y,0, keepdims=True)))).to(device))
            self.h_pos.append(torch.FloatTensor((h_pos-np.min(h_pos,0))/(np.max(h_pos,0)-np.min(h_pos,0))).to(device))
            self.h_e.append(h_e.to(device))
            self.h_y.append(torch.FloatTensor(((h_y-np.min(l_y,0, keepdims=True))/(np.max(l_y,0, keepdims=True)-np.min(l_y,0, keepdims=True)))).to(device))
           
           
    def __len__(self):
        return len(self.index_list)
    
        
    def __getitem__(self,idx):
        return self.l_pos[idx], self.l_y[idx], self.l_e[idx], self.h_pos[idx], self.h_e[idx], self.h_y[idx]
    



class GraphDataset_unpaired(torch.utils.data.Dataset):
    def __init__(self, index_list, data_dir, device):
        self.index_list=index_list
        self.l_pos=[]
        self.l_e=[]
        self.l_y=[]
        self.h_pos=[]
        self.h_e=[]
        
           
        
        for n in tqdm((index_list)):
            l_pos=np.load(data_dir+str(n)+"/L_point.npy")
            l_e=np.load(data_dir+str(n)+"/L_edge.npy")
            l_y=np.load(data_dir+str(n)+"/L_y.npy")
            h_pos=np.load(data_dir+str(n)+"/H_point.npy")
            h_e=np.load(data_dir+str(n)+"/H_edge.npy")
            l_e=torch.LongTensor(l_e)
            h_e=torch.LongTensor(h_e)
            self.l_pos.append(torch.FloatTensor((l_pos-np.min(l_pos,0))/(np.max(l_pos,0)-np.min(l_pos,0))).to(device))
            self.l_e.append(l_e.to(device))
            self.l_y.append(torch.FloatTensor(((l_y-np.min(l_y,0, keepdims=True))/(np.max(l_y,0, keepdims=True)-np.min(l_y,0, keepdims=True)))).to(device))
            self.h_pos.append(torch.FloatTensor((h_pos-np.min(h_pos,0))/(np.max(h_pos,0)-np.min(h_pos,0))).to(device))
            self.h_e.append(h_e.to(device))
        
           
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self,idx):
        
        return self.l_pos[idx], self.l_y[idx], self.l_e[idx], self.h_pos[idx], self.h_e[idx]
        

