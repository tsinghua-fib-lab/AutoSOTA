import glob
import scipy.io as sio
from torch.utils.data import Dataset


class DataLoaderSL(Dataset):
    def __init__(self , Training_path):
        self.datapath = glob.glob(Training_path + f"kneePD.mat")
        
    def __getitem__(self , index):
      
      ksp      = sio.loadmat(self.datapath[index])[f"kspace"].transpose([0,2,1])    
      coil     = sio.loadmat(self.datapath[index])[f"coils"].transpose([0,2,1])   
      FileName = self.datapath[index].split('/')[-1]

      return ksp , coil, FileName
    
    def __len__(self):
        return len(self.datapath)
        
        
        






 
