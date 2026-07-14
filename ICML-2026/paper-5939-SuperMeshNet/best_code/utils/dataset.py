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
            l_pos=np.load(data_dir+str(n)+"/L_mesh_geometry.npy")
            l_topology=np.load(data_dir+str(n)+"/L_mesh_topology.npy")
            l_y=np.load(data_dir+str(n)+"/L_y.npy")
            h_pos=np.load(data_dir+str(n)+"/H_mesh_geometry.npy")
            h_topology=np.load(data_dir+str(n)+"/H_mesh_topology.npy")
            h_y=np.load(data_dir+str(n)+"/H_y.npy")
            
      
            l_e1=np.concatenate([l_topology[:,0].reshape(-1,1),l_topology[:,1].reshape(-1,1)], 1)
            l_e2=np.concatenate([l_topology[:,1].reshape(-1,1),l_topology[:,2].reshape(-1,1)], 1)
            l_e3=np.concatenate([l_topology[:,2].reshape(-1,1),l_topology[:,0].reshape(-1,1)], 1)
            l_e4=np.concatenate([l_topology[:,1].reshape(-1,1),l_topology[:,0].reshape(-1,1)], 1)
            l_e5=np.concatenate([l_topology[:,2].reshape(-1,1),l_topology[:,1].reshape(-1,1)], 1)
            l_e6=np.concatenate([l_topology[:,0].reshape(-1,1),l_topology[:,2].reshape(-1,1)], 1)
            l_e=np.concatenate([l_e1,l_e2,l_e3,l_e4,l_e5,l_e6], 0)
                                            
            h_e1=np.concatenate([h_topology[:,0].reshape(-1,1),h_topology[:,1].reshape(-1,1)], 1)
            h_e2=np.concatenate([h_topology[:,1].reshape(-1,1),h_topology[:,2].reshape(-1,1)], 1)
            h_e3=np.concatenate([h_topology[:,2].reshape(-1,1),h_topology[:,0].reshape(-1,1)], 1)
            h_e4=np.concatenate([h_topology[:,1].reshape(-1,1),h_topology[:,0].reshape(-1,1)], 1)
            h_e5=np.concatenate([h_topology[:,2].reshape(-1,1),h_topology[:,1].reshape(-1,1)], 1)
            h_e6=np.concatenate([h_topology[:,0].reshape(-1,1),h_topology[:,2].reshape(-1,1)], 1)
            h_e=np.concatenate([h_e1,h_e2,h_e3,h_e4,h_e5,h_e6], 0)
            
            l_e=torch.transpose(torch.LongTensor(l_e),0,1)
            h_e=torch.transpose(torch.LongTensor(h_e),0,1)

            self.l_pos.append(torch.FloatTensor((l_pos-np.min(l_pos,0))/(np.max(l_pos,0)-np.min(l_pos,0))).to(device))
            self.l_e.append(l_e.to(device))
            self.l_y.append(torch.FloatTensor(((l_y-np.min(l_y))/(np.max(l_y)-np.min(l_y))).reshape(-1,1)).to(device))
            
            self.h_pos.append(torch.FloatTensor((h_pos-np.min(h_pos,0))/(np.max(h_pos,0)-np.min(h_pos,0))).to(device))
            self.h_e.append(h_e.to(device))
            self.h_y.append(torch.FloatTensor(((h_y-np.min(l_y))/(np.max(l_y)-np.min(l_y))).reshape(-1,1)).to(device))
           
           
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
            l_pos=np.load(data_dir+str(n)+"/L_mesh_geometry.npy")
            l_topology=np.load(data_dir+str(n)+"/L_mesh_topology.npy")
            l_y=np.load(data_dir+str(n)+"/L_y.npy")
            h_pos=np.load(data_dir+str(n)+"/H_mesh_geometry.npy")
            h_topology=np.load(data_dir+str(n)+"/H_mesh_topology.npy")
            
      
            l_e1=np.concatenate([l_topology[:,0].reshape(-1,1),l_topology[:,1].reshape(-1,1)], 1)
            l_e2=np.concatenate([l_topology[:,1].reshape(-1,1),l_topology[:,2].reshape(-1,1)], 1)
            l_e3=np.concatenate([l_topology[:,2].reshape(-1,1),l_topology[:,0].reshape(-1,1)], 1)
            l_e4=np.concatenate([l_topology[:,1].reshape(-1,1),l_topology[:,0].reshape(-1,1)], 1)
            l_e5=np.concatenate([l_topology[:,2].reshape(-1,1),l_topology[:,1].reshape(-1,1)], 1)
            l_e6=np.concatenate([l_topology[:,0].reshape(-1,1),l_topology[:,2].reshape(-1,1)], 1)
            l_e=np.concatenate([l_e1,l_e2,l_e3,l_e4,l_e5,l_e6], 0)
           

            h_e1=np.concatenate([h_topology[:,0].reshape(-1,1),h_topology[:,1].reshape(-1,1)], 1)
            h_e2=np.concatenate([h_topology[:,1].reshape(-1,1),h_topology[:,2].reshape(-1,1)], 1)
            h_e3=np.concatenate([h_topology[:,2].reshape(-1,1),h_topology[:,0].reshape(-1,1)], 1)
            h_e4=np.concatenate([h_topology[:,1].reshape(-1,1),h_topology[:,0].reshape(-1,1)], 1)
            h_e5=np.concatenate([h_topology[:,2].reshape(-1,1),h_topology[:,1].reshape(-1,1)], 1)
            h_e6=np.concatenate([h_topology[:,0].reshape(-1,1),h_topology[:,2].reshape(-1,1)], 1)
            h_e=np.concatenate([h_e1,h_e2,h_e3,h_e4,h_e5,h_e6], 0)
            
            l_e=torch.transpose(torch.LongTensor(l_e),0,1)
            h_e=torch.transpose(torch.LongTensor(h_e),0,1)

            
           
            self.l_pos.append(torch.FloatTensor((l_pos-np.min(l_pos,0))/(np.max(l_pos,0)-np.min(l_pos,0))).to(device))
            self.l_e.append(l_e.to(device))
            self.l_y.append(torch.FloatTensor(((l_y-np.min(l_y))/(np.max(l_y)-np.min(l_y))).reshape(-1,1)).to(device))

          
            
            self.h_pos.append(torch.FloatTensor((h_pos-np.min(h_pos,0))/(np.max(h_pos,0)-np.min(h_pos,0))).to(device))
            self.h_e.append(h_e.to(device))
        
           
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self,idx):
        
        return self.l_pos[idx], self.l_y[idx], self.l_e[idx], self.h_pos[idx], self.h_e[idx]
        

