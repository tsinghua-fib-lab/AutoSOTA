# %%
from matplotlib import pyplot as plt
import numpy as np
import time

from sklearn.covariance import GraphicalLassoCV
from datasets.ToyChainLoader import ToyChainLoader
from utils.hsic import hsic
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

from models.nntoy import CNNEncoder, Encoder, LSTMEncoder, VectorField 
from datasets.ToyLoader import ToyLoader
from BaseExperiment import BaseExperiment

# %%
class ToyFlowExperiment(BaseExperiment):
    def _init_loader(self):
        self.inputdim = self.config['inputdim']
        self.iter_starttime = None
        
        # # Specific: Uses ToyLoader
        # return ToyLoader(dim=self.inputdim, 
        #                  batch_size=self.config['batch_size'], device=self.device)
        # return ToyChainLoader(batch_size=self.config['batch_size'], device=self.device)
        return ToyLoader(dim=self.inputdim, 
                         batch_size=self.config['batch_size'], device=self.device)

    def _init_models(self):
        # Specific: Uses Generator/VectorField
        
        encodertype = self.config.get('encodertype', 'mlp')
        if encodertype == 'mlp':
            self.encoder = Encoder(inputdim=self.inputdim, hiddn_dim=128).to(self.device)
        elif encodertype == 'lstm':
            self.encoder = LSTMEncoder(inputdim=self.inputdim, hiddn_dim=32).to(self.device)
        elif encodertype == 'cnn':
            self.encoder = CNNEncoder(inputdim=self.inputdim, hidden_dim=32).to(self.device)
        else:  
            self.encoder = Encoder(inputdim=self.inputdim, hiddn_dim=128).to(self.device)  # default to mlp
        self.vf = VectorField(inputdim=self.inputdim, hiddn_dim=256).to(self.device)
        
        opt = torch.optim.AdamW([
            {'params': self.encoder.parameters()},
            {'params': self.vf.parameters()}
        ], lr=self.config['lr'])
        
        # Store models in a dict for easy access
        return {'encoder': self.encoder, 'vectorfield': self.vf}, opt
    
    def _forward_step(self, batch):
        z1, z2, mask = batch  
        x = z1 * mask + torch.zeros_like(z1) * (1 - mask)
        y = z1 * (1 - mask) + torch.zeros_like(z1) * mask
        xprime = z2 * mask + torch.zeros_like(z2) * (1 - mask)
        yprime = z2 * (1 - mask) + torch.zeros_like(z2) * mask
        
        t = torch.rand(x.size(0), 1).to(self.device)
        xt = t * xprime + (1 - t) * x
        enc_yprime = self.models['encoder'](yprime, mask)
        
        pred_dxdt = self.models['vectorfield'](xt, enc_yprime, y, mask, t)
        
        # Rectified Main Loss
        rectloss = torch.mean((xprime*mask - x*mask - pred_dxdt*mask)**2)
        
        # Penalty of the vector field
        enc_y = self.models['encoder'](y, mask)
        # predt = self.models['vectorfield'](xt, enc_y, y, mask, t) * mask
        # pred_1minust = self.models['vectorfield'](xt, enc_y, y, mask, 1 - t) * mask
        # penalty = torch.mean((predt + pred_1minust)**2)
        t = torch.ones_like(t) * 0.5
        # b=0.05
        # t = torch.exp(-torch.abs(t - 0.5) / b)
        predt = self.models['vectorfield'](xt, enc_y, y, mask, t) * mask
        penalty = torch.mean((predt)**2)
        
        # l1 penalty on encoder weights to encourage sparsity
        l1_lambda = self.config.get('l1_lambda', 1e-9)
        l1_penalty = self.models['encoder'].get_gates_sum(mask)
        
        total_loss = rectloss + 1e-1 * penalty + l1_lambda * l1_penalty
        
        # Return loss and any metrics to log
        return total_loss, {'total_loss': total_loss.item(), 'recloss': rectloss.item(), 'penalty': penalty.item(), 'l1_penalty': l1_penalty.item()}

    def _visualize_live(self, epoch, metrics):
        self.models['encoder'].eval() 
        self.models['vectorfield'].eval()
        
        with torch.no_grad():
            if self.iter_starttime is None:
                self.iter_starttime = time.time()
            else:
                elapsed_time = time.time() - self.iter_starttime
                print(f"Time per interval: {elapsed_time:.2f} seconds")
                self.iter_starttime = time.time()
            
            print(f"Epoch {epoch}:")
            print("total loss:", metrics['total_loss'])
            print("recloss:", metrics['recloss'])
            print("penalty:", metrics['penalty'])
            clear_output(wait=True)
            plt.figure(figsize=(4,3))
            idx = torch.zeros((1, self.inputdim), device=self.device)
            idx[0, 1] = 1.0
            pred_gates = self.models['encoder'].get_gates(idx).cpu().numpy()
            plt.plot(pred_gates.flatten(), marker='o', linestyle='None', alpha=0.5)
            plt.title("Encoder Gates Visualization")
            plt.xlabel("Dimension Index")
            plt.ylabel("Gate Value")
            # plt.ylim([-0.1, 1.1])
            plt.show()
        
        self.models['encoder'].train()
        self.models['vectorfield'].train()

    def _visualize_test(self):
        pass

    def _bench(self):
        pass


# %% 

# %% run the experiment
config = {
    'l1_lambda': 1e-11,
    'device': 'cuda',
    'encodertype': 'cnn',  # 'mlp' or 'lstm'
    'inputdim': 225,
    'batch_size': 400,
    'lr': 1e-3,
    'num_test_samples': 1000
}
# %%
experiment = ToyFlowExperiment(config, seed = 999)

def main():
    experiment.train(num_epochs=5_001, viz_interval=100)
    
    experiment.summary()
    with torch.no_grad():
        plt.figure(figsize=(4,3))
        idx = torch.zeros((1, experiment.inputdim), device=experiment.device)
        idx[0, 11] = 1.0
        idx[0, 12] = 1.0
        pred_gates = experiment.models['encoder'].get_gates(idx).cpu().numpy()
        plt.plot(pred_gates.flatten(), marker='o', linestyle='None', alpha=0.5)
        plt.title("Encoder Gates Visualization")
        plt.xlabel("Dimension Index")
        plt.ylabel("Gate Value")
        # plt.ylim([-0.1, 1.1])
        plt.show()
        
    with torch.no_grad():
        plt.figure(figsize=(10,6))
        plt.subplots_adjust(wspace=.1)
        plt.subplot(1, 2, 1)
        Theta = experiment.loader.Sigma.inverse().cpu().numpy(); np.fill_diagonal(Theta, 0)
        plt.imshow(Theta, cmap='bwr')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(1, 2, 2)
        all_gates = []
        for i in range(experiment.inputdim):
            idx = torch.zeros((1, experiment.inputdim), device=experiment.device)
            idx[0, i] = 1.0
            pred_gates = experiment.models['encoder'].get_gates(idx).cpu().numpy()
            all_gates.append(pred_gates.flatten())
        all_gates = np.array(all_gates)
        plt.imshow(all_gates, cmap='viridis')
        # no axis labels
        plt.xticks([]); 
        plt.yticks([])
        plt.show()
        
# %%
    
if __name__ == "__main__":
    main()
# %%