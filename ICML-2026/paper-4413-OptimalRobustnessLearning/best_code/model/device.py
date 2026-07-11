import torch
import os 

class DeviceManager:
    def __init__(self):
        self.default_i = None
        self.default_device = None

    def set_device(self, default_device='cpu', enable_multi_device=False):
        if default_device != 'cpu':
            self.default_i = int(default_device.split(":")[1])
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.default_i)
            if torch.cuda.is_available():
                self.default_device = torch.device('cuda:0')
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
            else:
                self.default_i = 'cpu'
                self.default_device = torch.device('cpu')
        else:
            self.default_i = 'cpu'
            self.default_device = torch.device('cpu')
        print(f'DeviceManager: Default Device[{self.default_i} -> {self.default_device}]', flush=True)
    
    def get_default_device(self):
        return self.default_device

device_manager = DeviceManager()