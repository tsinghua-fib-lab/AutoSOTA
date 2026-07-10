# %%
import os
from sklearn.covariance import GraphicalLassoCV
from utils.roc import compute_roc_curve, auc_trapezoid
from datasets.ToyTruncatedLoader import ToyTruncatedLoader
from datasets.ToyChainLoader import ToyChainLoader
from datasets.ToyChainNonpra import ToyNonParanormalLoader
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ToyLattice import ToyFlowExperiment
import numpy as np
import time
from utils.hsic import hsic
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

from models.nntoy import Encoder, LSTMEncoder, VectorField 
from datasets.ToyLoader import ToyLoader
from BaseExperiment import BaseExperiment

# %%

def get_allgates(encoder: Encoder, inputdim: int, device: torch.device):
    with torch.no_grad():
        all_gates = []
        for i in range(inputdim):
            mask = torch.zeros((1, inputdim), device=device)
            mask[0, i] = 1.0
            pred_gates = encoder.get_gates(mask).cpu().numpy()
            all_gates.append(pred_gates.flatten())
        all_gates = np.array(all_gates).T  # shape (inputdim, inputdim)
        return all_gates

class ToyROCFlowExperiment(ToyFlowExperiment):
    def _init_loader(self):
        self.inputdim = self.config['inputdim']
        self.iter_starttime = None
        
        dataset = self.config.get('dataset', 'normal')
        if dataset == 'normal':
            return ToyChainLoader(batch_size=self.config['batch_size'], device=self.device)
        elif dataset == 'nonparanormal':
            return ToyNonParanormalLoader(batch_size=self.config['batch_size'], device=self.device)
        else:
            return ToyTruncatedLoader(batch_size=self.config['batch_size'], device=self.device)

    def _init_models(self):
        # Specific: Uses Generator/VectorField
        
        encodertype = self.config.get('encodertype', 'mlp')
        if encodertype == 'mlp':
            self.encoder = Encoder(inputdim=self.inputdim, hiddn_dim=128).to(self.device)
        elif encodertype == 'lstm':
            self.encoder = LSTMEncoder(inputdim=self.inputdim, hiddn_dim=32).to(self.device)
        else:  
            self.encoder = Encoder(inputdim=self.inputdim, hiddn_dim=128).to(self.device)  # default to mlp

        self.vf = VectorField(inputdim=self.inputdim, hiddn_dim=256).to(self.device)
        
        opt = torch.optim.AdamW([
            {'params': self.encoder.parameters()},
            {'params': self.vf.parameters()}
        ], lr=self.config['lr'])
        
        # Store models in a dict for easy access
        return {'encoder': self.encoder, 'vectorfield': self.vf}, opt


# %% run the experiment
config = {
    'inputdim': 50,
    'batch_size': 400,
    'lr': 1e-3,
    'num_test_samples': 1000
}
# %%
def run_roc_plot_experiment(dataset='truncated'):
    config['dataset'] = dataset  
    
    config['encodertype'] = 'mlp'
    os.remove('data/ToyROCFlowExperiment_checkpoint.pt')
    experiment = ToyROCFlowExperiment(config, seed = 1)
    experiment.train(num_epochs=5000, viz_interval=100)
    experiment.summary()
    all_gates_mlp = get_allgates(experiment.models['encoder'], experiment.inputdim, experiment.device)    

    os.remove('data/ToyROCFlowExperiment_checkpoint.pt')
    config['encodertype'] = 'lstm'
    experiment = ToyROCFlowExperiment(config, seed = 1)
    experiment.train(num_epochs=5000, viz_interval=100)
    experiment.summary()
    all_gates_lstm = get_allgates(experiment.models['encoder'], experiment.inputdim, experiment.device)

    true_prec = experiment.loader.Sigma.inverse().cpu().numpy()
    np.fill_diagonal(true_prec, 0.0)
    true_prec[np.abs(true_prec) < 1e-5] = 0.0

    model = GraphicalLassoCV(cv=5,max_iter=100)
    model.fit(experiment.loader.xdata.cpu().numpy())
    precision = model.precision_
    print(f"Optimal Alpha found: {model.alpha_:.5f}")

    # save data needed for plotting, 
    # make a folder 
    folder = f"data/results_roc_{config['dataset']}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(f'{folder}/true_precision.npy', true_prec)
    np.save(f'{folder}/all_gates_mlp.npy', all_gates_mlp)
    np.save(f'{folder}/all_gates_lstm.npy', all_gates_lstm)
    np.save(f'{folder}/glasso_precision.npy', precision)

    # plot ROC curves
    fontsize = 18
    folder = f"data/results_roc_{config['dataset']}"

    true_prec = np.load(f'{folder}/true_precision.npy')
    all_gates_mlp = np.load(f'{folder}/all_gates_mlp.npy')
    all_gates_lstm = np.load(f'{folder}/all_gates_lstm.npy')
    precision = np.load(f'{folder}/glasso_precision.npy')

    mk_size = 4
    plt.figure(figsize=(6, 5))
    fpr_list, tpr_list = compute_roc_curve(np.abs(true_prec)>1e-5, all_gates_mlp)

    auc_neural = auc_trapezoid(np.array(fpr_list), np.array(tpr_list))
    print(f"AUC (Neural Network Encoder): {auc_neural:.4f}")

    plt.plot(fpr_list, tpr_list, marker='o', label='MLP Encoder', markersize=mk_size)
    plt.plot([0,1], [0,1], linestyle='--', color='gray')

    fpr_list, tpr_list = compute_roc_curve(np.abs(true_prec)>1e-5, all_gates_lstm)
    auc_neural = auc_trapezoid(np.array(fpr_list), np.array(tpr_list))
    print(f"AUC (Neural Network Encoder - LSTM): {auc_neural:.4f}")

    plt.plot(fpr_list, tpr_list, marker='o', label='LSTM Encoder', markersize=mk_size)
    plt.plot([0,1], [0,1], linestyle='--', color='gray')

    fpr_list, tpr_list = compute_roc_curve(np.abs(true_prec)>1e-5, np.abs(precision))
    auc_glasso = auc_trapezoid(np.array(fpr_list), np.array(tpr_list))
    print(f"AUC (Graphical Lasso): {auc_glasso:.4f}")

    plt.plot(fpr_list, tpr_list, marker='o', label='Graphical Lasso', markersize=mk_size)
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.grid(True)
    plt.xlim([-.02, 1.02]); plt.xticks(fontsize=fontsize)
    plt.ylim([-.02, 1.02]); plt.yticks(fontsize=fontsize)
    plt.xlabel("False Positive Rate", fontsize=fontsize)
    plt.ylabel("True Positive Rate", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    # plt.title("ROC Curve for Graphical Lasso Structure Recovery")

# %% # compute AUCs and print mean AUCs and stds
def run_roc_auc_experiments():
    nseed = 10
    res = {}
    for dataset in ['normal', 'nonparanormal', 'truncated']:
        folder = f"data/results_roc_{dataset}"

        for encodertype in ['mlp', 'lstm', 'glasso']:
            config['dataset'] = dataset
            config['encodertype'] = encodertype
            
            for seed in range(nseed):
                print(f"Running Dataset: {dataset}, Encoder: {encodertype}, Seed: {seed}")
                experiment = ToyROCFlowExperiment(config, seed = seed)
                if encodertype == 'glasso':
                    model = GraphicalLassoCV(cv=5, max_iter=100)
                    model.fit(experiment.loader.xdata.cpu().numpy())
                    precision = model.precision_
                else:
                    if os.path.exists('data/ToyROCFlowExperiment_checkpoint.pt'):
                        os.remove('data/ToyROCFlowExperiment_checkpoint.pt')
                    experiment.train(num_epochs=5000, viz_interval=100)
                    experiment.summary()
                    precision = get_allgates(experiment.models['encoder'], experiment.inputdim, experiment.device)    

                true_prec = experiment.loader.Sigma.inverse().cpu().numpy()
                np.fill_diagonal(true_prec, 0.0)
                true_prec[np.abs(true_prec) < 1e-5] = 0.0

                fpr_list, tpr_list = compute_roc_curve(np.abs(true_prec)>1e-5, np.abs(precision))
                auc_score = auc_trapezoid(np.array(fpr_list), np.array(tpr_list))
                res[(dataset, encodertype, seed)] = auc_score
                
    # summarize
    for dataset in ['normal', 'nonparanormal', 'truncated']:
        for encodertype in ['mlp', 'lstm', 'glasso']:
            aucs = [res[(dataset, encodertype, seed)] for seed in range(nseed)]
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            print(f"Dataset: {dataset}, Encoder: {encodertype} => AUC: {mean_auc:.4f} ± {std_auc:.4f}") 
            
# %%
if __name__ == "__main__":
    run_roc_plot_experiment('truncated')
    run_roc_plot_experiment('nonparanormal')
    # run_roc_auc_experiments()
# %%
