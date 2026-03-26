import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.models import Aligner
from src.utils_train import EarlyStopper, set_seed, Logger, smiles_to_fp
    

def detached(tensor):
    return tensor.detach().cpu().numpy()

class aligndataset(Dataset):
    def __init__ (self, kg_rep, structure_rep):
        self.kg_rep = torch.FloatTensor(kg_rep)
        self.structure_rep = torch.FloatTensor(structure_rep)
        
    def __len__(self):
        return len(self.kg_rep)
    
    def __getitem__(self, idx):
        return self.kg_rep[idx], self.structure_rep[idx], idx

def train(args, model, trainloader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (kg, structure, idx) in enumerate(trainloader):
        kg, structure = kg.to(args.device), structure.to(args.device)
        optimizer.zero_grad()
        loss = model(args, kg, structure, idx)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(trainloader.dataset)
    return train_loss

def eval(args, model, testloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (kg, structure, idx) in enumerate(testloader):
            kg, structure = kg.to(args.device), structure.to(args.device)
            loss = model(args, kg, structure, idx)
            test_loss += loss.item()
    test_loss /= len(testloader.dataset)
    return test_loss

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # run params
    parser.add_argument('--input_smiles', type=str, default='benchmark/compound_sets/approved_drugs/knowledge_alignment/MSI_drug_SMILES.csv')
    parser.add_argument('--input_knowledgegraph', type=str, default='benchmark/compound_sets/approved_drugs/knowledge_alignment/MSI_DREAMwalk_embeddings.npy')
    parser.add_argument('--input_similarity', type=str, default='benchmark/compound_sets/approved_drugs/knowledge_alignment/ATC_similarity_matrix.npy')
    parser.add_argument('--project_dir', type=str, default='projects/test_run')
    parser.add_argument('--seed', type=int, default=42)

    # model params
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hdim', type=int, default=512)
    parser.add_argument('--alpha1', type=float, default=1.0)
    parser.add_argument('--alpha2', type=float, default=1.0)
    parser.add_argument('--alpha3', type=float, default=1.0)
    args = parser.parse_args()

    model_name = f'multimodal_alignment_model'
    logger = Logger(basedir=args.project_dir, model_name='multimodal_alignment_model')
    logger(f'Start training {model_name}')
    set_seed(args.seed, logger)

    smiles = pd.read_csv(args.input_smiles)['SMILES'].tolist()
    structure_rep = np.array([smiles_to_fp(s) for s in smiles])
    kg_rep = np.load(args.input_knowledgegraph)
    sim_matrix = np.load(args.input_similarity)
    
    trainsize = int(0.9*len(kg_rep))
    dataset=aligndataset(kg_rep,structure_rep)
    trainset, validset = data.random_split(dataset, [trainsize, len(kg_rep)-trainsize], torch.Generator().manual_seed(42))
    trainloader = DataLoader(trainset,args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    validoader = DataLoader(validset, args.batch_size, shuffle=False)
    logger(f'Train size: {len(trainset)} Valid size: {len(validset)}')

    model = Aligner(device = args.device, hdim=args.hdim, alpha1=args.alpha1, 
                    alpha2=args.alpha2, alpha3=args.alpha3, ATC_adj=sim_matrix).to(args.device)
    
    os.makedirs(f'{args.project_dir}/ckpts', exist_ok=True)
    modelf = f'{args.project_dir}/ckpts/multimodal_alignment_model.pt'
    early_stopper = EarlyStopper(patience=20, verbose=True, path=modelf, printfunc=logger)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger('Start training multimodal alignment model')
    epoch=0
    while True:
        epoch+=1
        train_loss = train(args, model, trainloader, optimizer)
        valid_loss = eval(args, model, validoader)
        logger('[Epoch {}] Train Loss: {:.4f} Valid Loss: {:.4f}'.format(epoch, train_loss, valid_loss))

        # exception for nan loss for early stopping
        if torch.isnan(torch.tensor(valid_loss)):
            valid_loss = 100
        early_stopper(valid_loss, model)
        if early_stopper.early_stop:
            print('Early stopping')
            break
    
    model.load_state_dict(torch.load(modelf,map_location=args.device))
    model.eval()
    logger(f'Best model "{modelf}", valid loss: {early_stopper.val_loss_min:.4f}')

if __name__ == '__main__':
    main()
