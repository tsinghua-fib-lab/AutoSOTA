# MLP
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from src.models import BounDrE, MLP
from src.utils_train import EarlyStopper, set_seed, Logger, evaluate_clf, smiles_to_fp

class Boundary_Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

def train(model, loader, optimizer):
    model.train()
    train_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        dist_loss = model(data)

        dist_loss.backward()
        optimizer.step()
        train_loss += dist_loss.item()
    train_loss /= len(loader)
    return train_loss

def E_step(model, init_trainloader, train_zincloader):
    # E-step: boundary update and identify out-boundary compoudns
    model.get_c(init_trainloader)
    model.get_R(init_trainloader)
    
    ## identify out-boundary compound indices 
    zinc_dists = model.decision_function(train_zincloader).numpy()
    out_zinc_train_idx = np.where(zinc_dists<0)[0] # only outliers

    return model, out_zinc_train_idx

def M_step(model, optimizer, trainset_onlycompound, trainset_onlydrug, out_zinc_train_idx, batch_size):
    # M-step: embedding space update

    ## build train-negative set based on the out-boundary compounds
    trainset_outcompound = data.Subset(trainset_onlycompound, out_zinc_train_idx)
    trainset = data.ConcatDataset([trainset_onlydrug, trainset_outcompound])
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))

    train_loss = train(model, trainloader, optimizer)
    return model, train_loss

def get_ICR(model, init_trainloader, validloader, y_valid):
    model.eval()
    with torch.no_grad():
        model.get_c(init_trainloader)
        model.get_R(init_trainloader)
        
        # in_comps ratio
        # es by in_comps ratio
        in_compounds = model.decision_function(validloader).numpy()[y_valid==0]
        in_compounds = in_compounds>0
        return np.mean(in_compounds)
    
def test_performance(model, testloader, y_test, cut_off=0):
    y_pred_proba = model.decision_function(testloader).numpy()
    acc, f1, precision, recall, mcc, roc_auc, pr_auc = evaluate_clf(y_test, y_pred_proba, cut_off=cut_off)
    y_preds = y_pred_proba > cut_off

    num_drugs = len(y_test[y_test == 1])
    num_comps = len(y_test[y_test == 0])
    in_drugs = y_preds[y_test==1].sum()
    in_comps = y_preds[y_test==0].sum()

    return acc, f1, precision, recall, mcc, roc_auc, pr_auc, in_drugs/num_drugs, in_comps/num_comps, (in_drugs/num_drugs)/(in_comps/num_comps)

def main():
    import os
    import pandas as pd
    import argparse
    from train_multimodal_alignment import Aligner

    parser = argparse.ArgumentParser(description='BounDrE')
    # run parameters
    parser.add_argument('--train_drugs', type=str, default='demo/boundary_data/train_drugs.smi', help='train set of drugs')
    parser.add_argument('--train_compounds', type=str, default='demo/boundary_data/train_compounds.smi', help='train set of compounds')
    parser.add_argument('--valid_drugs', type=str, default='demo/boundary_data/valid_drugs.smi', help='valid set of drugs')
    parser.add_argument('--valid_compounds', type=str, default='demo/boundary_data/valid_compounds.smi', help='valid set of compounds')
    parser.add_argument('--test_drugs', type=str, default='demo/boundary_data/test_drugs.smi', help='test set of drugs')
    parser.add_argument('--test_compounds', type=str, default='demo/boundary_data/test_compounds.smi', help='test set of compounds')
    parser.add_argument('--project_dir', type=str, default='projects/test_run', help='project directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    
    # encoder parameters
    parser.add_argument('--idim', type=int, default=512, help='input dimension')
    parser.add_argument('--odim', type=int, default=2, help='output dimension')
    parser.add_argument('--hdim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lambda_out', type=float, default=1, help='lambda for out_boundary loss')
    parser.add_argument('--nu', type=float, default=0.95, help='percentile for drug boundary')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    args = parser.parse_args()

    # Generate embeddings from Multimoal alignment model
    train_drug_smiles = pd.read_csv(args.train_drugs, header=None)[0].tolist()
    train_comp_smiles = pd.read_csv(args.train_compounds, header=None)[0].tolist()
    valid_drug_smiles = pd.read_csv(args.valid_drugs, header=None)[0].tolist()
    valid_comp_smiles = pd.read_csv(args.valid_compounds, header=None)[0].tolist()
    test_drug_smiles = pd.read_csv(args.test_drugs, header=None)[0].tolist()
    test_comp_smiles = pd.read_csv(args.test_compounds, header=None)[0].tolist()

    train_drug_fps = np.array([smiles_to_fp(smiles) for smiles in train_drug_smiles])
    train_comp_fps = np.array([smiles_to_fp(smiles) for smiles in train_comp_smiles])
    valid_drug_fps = np.array([smiles_to_fp(smiles) for smiles in valid_drug_smiles])
    valid_comp_fps = np.array([smiles_to_fp(smiles) for smiles in valid_comp_smiles])
    test_drug_fps = np.array([smiles_to_fp(smiles) for smiles in test_drug_smiles])
    test_comp_fps = np.array([smiles_to_fp(smiles) for smiles in test_comp_smiles])

    from train_multimodal_alignment import Aligner
    aligner = Aligner(device = args.device)
    os.makedirs(args.project_dir, exist_ok=True)
    try:
        aligner.load_state_dict(torch.load(os.path.join(args.project_dir,'ckpts/multimodal_alignment_model.pt'), map_location=args.device))
    except FileNotFoundError:
        raise FileNotFoundError(f"NO FILE {os.path.join(args.project_dir,'ckpts/multimodal_alignment_model.pt')}. Please train multimodal alignment model first or copy the pretrained model to the ckpts folder under project directory")
    aligner.to(args.device)

    X_train_drug = aligner.encode_fp(torch.Tensor(train_drug_fps).to(args.device)) #numpy
    X_train_comp = aligner.encode_fp(torch.Tensor(train_comp_fps).to(args.device))
    X_valid_drug = aligner.encode_fp(torch.Tensor(valid_drug_fps).to(args.device))
    X_valid_comp = aligner.encode_fp(torch.Tensor(valid_comp_fps).to(args.device))
    X_test_drug = aligner.encode_fp(torch.Tensor(test_drug_fps).to(args.device))
    X_test_comp = aligner.encode_fp(torch.Tensor(test_comp_fps).to(args.device))

    # Prepare dataset
    X_valid = np.concatenate([X_valid_drug, X_valid_comp])
    X_test = np.concatenate([X_test_drug, X_test_comp])
    y_valid = np.array([1]*len(X_valid_drug) + [0]*len(X_valid_comp))
    y_test = np.array([1]*len(X_test_drug) + [0]*len(X_test_comp))

    trainset =  Boundary_Dataset(np.concatenate([X_train_drug, X_train_comp]), np.array([1]*len(X_train_drug) + [0]*len(X_train_comp)))
    trainset_onlydrug = Boundary_Dataset(X_train_drug, np.array([1]*len(X_train_drug)))
    trainset_onlycompound = Boundary_Dataset(X_train_comp, np.array([0]*len(X_train_comp)))
    validset = Boundary_Dataset(X_valid, y_valid)
    testset = Boundary_Dataset(X_test, y_test)

    validloader = data.DataLoader(validset, batch_size=args.batch_size, shuffle=False)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    train_zincloader = data.DataLoader(trainset_onlycompound, batch_size=args.batch_size, shuffle=False)
    init_trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    logger = Logger(basedir=args.project_dir, model_name='boundary_model')
    logger(f'Training boundary_model')
    os.makedirs(f'{args.project_dir}/ckpts', exist_ok=True)
    modelf = f'{args.project_dir}/ckpts/boundary_model.pt'


    # Multiple initialization for EM
    best_valid_loss = float('inf')
    for seed in range(10):
        logger(f'Start initialization {seed}')
        run_seed = args.seed + seed # runseed = seed + 0~9 for 10 different initializations
        set_seed(run_seed, logger)
        encoder = MLP(args.idim, args.odim, args.hdim, args.nlayers, args.dropout).to(args.device)
        model = BounDrE(encoder, nu=args.nu, neg_lambda=args.lambda_out)
        model.encoder.device = args.device
        model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        earlystopper = EarlyStopper(patience=3, verbose=True, path=None, printfunc=logger)

        # Train model
        epoch = 0
        while True:
            epoch += 1
            # E-step
            model, out_zinc_train_idx = E_step(model, init_trainloader, train_zincloader)
            # M-step
            model, train_loss = M_step(model, optimizer, trainset_onlycompound, trainset_onlydrug, out_zinc_train_idx, args.batch_size)
            model.eval()
            with torch.no_grad():
                ICR = get_ICR(model, init_trainloader, validloader, y_valid)
                valid_loss = ICR

            logger(f'Epoch {epoch} | Train Loss: {train_loss} | Valid Loss: {valid_loss}')
            logger(f'Out_boundary_compounds: {len(out_zinc_train_idx)}/{len(trainset_onlycompound)} = {len(out_zinc_train_idx)/len(trainset_onlycompound):.4f}')
            earlystopper(valid_loss, model, save=False)

            if earlystopper.counter == 0:
                # save model with best valid loss
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    logger(f'Save best model with valid loss {valid_loss}')
                    model_dict = {
                        'model': model.state_dict(),
                        'R': model.R.cpu(),
                        'c': model.c.cpu(),
                        'valid_loss': valid_loss,
                    }
                    torch.save(model_dict, modelf)

            if earlystopper.early_stop:
                logger(f'End of training of iteration {seed}')
                break
            elif valid_loss == 0.0:
                logger(f'End of training of iteration {seed}')
                break

    # load best model for testset evaluation
    model_dict = torch.load(modelf, map_location=args.device)
    model.load_state_dict(model_dict['model'])
    model.R = model_dict['R'].to(args.device)
    model.c = model_dict['c'].to(args.device)
    logger(f"Loaded best model from {modelf}, valid ICR: {model_dict['valid_loss']:.4f}")

    acc, f1, precision, recall, mcc, roc_auc, avg_pr, IDR, ICR, DCR = test_performance(model, testloader, y_test, cut_off=0)
    logger('Test set performance')
    logger(f'F1: {f1:.4f} | AUC: {roc_auc:.4f} | Avg PR: {avg_pr:.4f} | IDR: {IDR:.4f} | ICR: {ICR:.4f} | IDR/ICR: {DCR:.4f}')
    logger('='*50)

if __name__ == '__main__':
    main()
