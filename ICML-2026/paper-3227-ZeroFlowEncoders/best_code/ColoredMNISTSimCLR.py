# %%
import os
from ColoredMNIST import ColoredMNISTFlowExperiment
from utils.logistic import acc_classifier, fine_tune
from utils.simclr import SimCLRLoss
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch import nn
from torch.nn import functional as F
# Assumed imports based on your script context
from models.nn import Encoder, VectorField
from utils.mine import MINE
from datasets.MNISTLoader import FastMNISTLoader
from CIFAR10 import CIFARFlowExperiment

class ColoredMNISTSimCLRExperiment(ColoredMNISTFlowExperiment):
    def _init_loader(self):
        self.img_size = self.config.get('img_size', 32)
        from datasets.ColoredMNISTLoader import ColoredMNISTLoader
        return ColoredMNISTLoader(colored=True, 
                                  batch_size=self.config['batch_size'], 
                                  device=self.device)
        
    def _init_models(self):
        self.criterion = SimCLRLoss(temperature=0.1)
        return super()._init_models()   
    
    def _forward_step(self, batch):
        v1, v2, _ = batch
        z1 = self.models['encoder'](v1)
        z2 = self.models['encoder'](v2)

        loss = self.criterion(z1, z2)
        return loss, {'total_loss': loss.item()}
    
    def _visualize_live(self, epoch, metrics):
        print(f"Epoch {epoch}: Loss = {metrics['total_loss']:.4f}")
        
# %% run the experiment
config = {
    'device': 'cuda',
    'batch_size': 256,
    'enc_dim':128,
    'lr': 1e-3,
    'num_test_samples': 10000
}
# %%
if os.path.exists(f'data/ColoredMNISTSimCLRExperiment_checkpoint.pt'):
    os.remove(f'data/ColoredMNISTSimCLRExperiment_checkpoint.pt')
experiment = ColoredMNISTSimCLRExperiment(config, seed = 5)
experiment.train(num_epochs=5000, viz_interval=100)
# %%
experiment.summary()
# %%
experiment.test()

# %%
experiment.bench()
# %%

