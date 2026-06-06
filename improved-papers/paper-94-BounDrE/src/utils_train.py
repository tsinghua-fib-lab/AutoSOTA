from datetime import datetime
class Logger:
    def __init__(self, model_name, basedir='.'):
        self.model_name=model_name
        self.date=str(datetime.now().date()).replace('-','')[2:]
        import os

        self.dir = os.path.join(basedir,'log')
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.logger_file = f'{self.dir}/{self.date}_{self.model_name}'
        
    def __call__(self, text, verbose=True, log=True):
        if log:
            with open(f'{self.logger_file}.log', 'a') as f:
                f.write(f'[{datetime.now().replace(microsecond=0)}] {text}\n')
        if verbose:
            print(f'[{datetime.now().time().replace(microsecond=0)}] {text}')

class EarlyStopper:
    def __init__(self, patience=7, printfunc=print, verbose=True, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): epochs to wait after minimum has been reached.
                            Default: 7
            verbose (bool): whether to print the early stopping message or not.
                            Default: False
            delta (float): minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): path to save the model when validation loss decreases.
                            Default: 'checkpoint.pt'
            printfunc (func): print function to use.
                            Default: python print function
        """
        self.printfunc=printfunc
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, save=True):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.printfunc(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if save:
                self.save_checkpoint(val_loss, model)


    def save_checkpoint(self, val_loss, model):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        ''' Saves model when validation loss decrease. '''
        # if self.verbose:
        #     self.printfunc(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        
import os
import random
import numpy as np
import torch

def set_seed(seed=42,logger=print):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger(f'random seed with {seed}')


def smiles_to_fp(smiles, nBits=1024, radius=2):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import DataStructs
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def evaluate_clf(y_true, y_pred_proba, cut_off = 0.5):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, matthews_corrcoef
    y_pred = y_pred_proba > cut_off

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    return acc, f1, precision, recall, mcc, roc_auc, pr_auc
