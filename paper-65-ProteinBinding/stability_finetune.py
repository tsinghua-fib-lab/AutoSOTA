import numpy as np
import pandas as pd
import torch
import os
import argparse
import pickle
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import wandb
from stabddg.mpnn_utils import StructureDataset, ProteinMPNN, parse_PDB
from stabddg.model import StaBddG

def validation_step(model, ddG_data, dataset_valid, batch_size=20000, name='val', device='cuda'):
    val_spearman=[]
    val_pearson=[]
    all_pred = []
    all_labels = []
    for sample in tqdm(dataset_valid):
        pdb_name = sample['name']
        ddG = ddG_data[f'{pdb_name}.pdb']['ddG'].to(device)
        mut_seqs = ddG_data[f'{pdb_name}.pdb']['mut_seqs']
        N = mut_seqs.shape[0]
        M = batch_size // mut_seqs.shape[1] # convert number of tokens to number of sequences per batch

        sample_pred = []
        # Batching for mutants
        for batch_idx in range(0, N, M):
            B = min(N - batch_idx, M)
            # ddG prediction
            pred = model.folding_ddG(sample, mut_seqs[batch_idx:batch_idx+B])
            sample_pred.append(pred.detach().cpu())

        pred = torch.cat(sample_pred)
        
        sp, _ = spearmanr(pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())
        val_spearman.append(sp)

        pr, _ = pearsonr(pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())
        val_pearson.append(pr)

        all_pred.append(pred.cpu().detach().numpy())
        all_labels.append(ddG.cpu().detach().numpy())
        
    sp, _ = spearmanr(np.concatenate(all_pred), np.concatenate(all_labels))
    pr, _ = pearsonr(np.concatenate(all_pred), np.concatenate(all_labels))

    return {
        f'{name}_spearman': np.mean(val_spearman),
        f'{name}_pearson': np.mean(val_pearson),
        f'{name}_all_spearman': sp,
        f'{name}_all_pearson': pr,
    }

def finetune(model, dataset_train, dataset_valid, dataset_test, ddG_data, args, batch_size=10000, device='cuda'):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ddG_loss_fn  = torch.nn.MSELoss()

    for e in tqdm(range(args.num_epochs), desc="Epoch"):
        model.train()
        train_sum = []
        spearmans = []

        # Iterate through all training domains.
        for sample in dataset_train:
            pdb_name = sample['name']
            ddG = ddG_data[f'{pdb_name}.pdb']['ddG'].to(device)
            mut_seqs = ddG_data[f'{pdb_name}.pdb']['mut_seqs']
            N = mut_seqs.shape[0]
            M = batch_size // mut_seqs.shape[1] # convert number of tokens to number of sequences per batch

            
            # random shuffling
            permutation = torch.randperm(ddG.shape[0])
            ddG = ddG[permutation]
            mut_seqs = mut_seqs[permutation]

            sample_pred = []

            # Batching for mutants
            for batch_idx in range(0, N, M):
                B = min(N - batch_idx, M)
                optimizer.zero_grad()

                # ddG prediction
                pred = model.folding_ddG(sample, mut_seqs[batch_idx:batch_idx+B])

                ddG_loss = ddG_loss_fn(pred, ddG[batch_idx:batch_idx+B])

                ddG_loss.backward()
                optimizer.step()

                train_sum.append(ddG_loss.item())
                sample_pred.append(pred.detach().cpu())
                if args.single_batch:
                    break

            sample_pred = torch.cat(sample_pred)
            sp, _ = spearmanr(sample_pred.detach().numpy(), ddG.cpu().detach().numpy()[:sample_pred.shape[0]])
            spearmans.append(sp)

        model.eval()
        if (e + 1) % args.model_save_freq == 0:
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            torch.save(model.pmpnn.state_dict(), f"{args.model_save_dir}/{args.run_name}_epoch{e}.pt")
        if args.wandb:
            if e % args.val_freq == 0:
                with torch.no_grad():
                    valid_metrics = validation_step(model, ddG_data, dataset_valid, batch_size=10000, name='valid', device=device)
                    test_metrics = validation_step(model, ddG_data, dataset_test, batch_size=10000, name='test', device=device)

                wandb.log(valid_metrics, step=e+1)
                wandb.log(test_metrics, step=e+1)

            wandb.log({'train_loss': np.mean(train_sum), 'train_spearman': np.mean(spearmans)}, step=e+1)
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=e+1)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    torch.save(model.pmpnn.state_dict(), f"{args.model_save_dir}/{args.run_name}_final.pt")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--run_name", type=str, default="stability_finetune")
    argparser.add_argument("--checkpoint", type=str, default="model_ckpts/proteinmpnn.pt")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--num_epochs", type=int, default=70)
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--model_save_freq", type=int, default=10)
    argparser.add_argument("--model_save_dir", type=str, default="cache/stability_finetuned")
    argparser.add_argument("--wandb", action='store_true')
    # Sample only one batch of mutants per domain during training.
    argparser.add_argument("--single_batch", action='store_true') 
    argparser.add_argument("--val_freq", type=int, default=10) # Train validation frequency
    argparser.add_argument("--noise_level", type=float, default=0.1) # Backbone noise.
    argparser.add_argument("--dropout", type=float, default=0.0) # Dropout during model training. 
    # Do not fix permutation order and backbone noise between mutant and wildtype during decoding.
    argparser.add_argument("--no_antithetic_variates", action='store_true') 
    argparser.add_argument("--lam", type=float, default=0.0) # KL regularization strength.
    argparser.add_argument("--pdb_dir", type=str, default="AlphaFold_model_PDBs")
    argparser.add_argument("--stability_data", type=str, default='Tsuboyama2023_Dataset2_Dataset3_20230416.csv')
    argparser.add_argument("--lr", type=float, default=1e-6)
    argparser.add_argument("--random_init", action='store_true')
    args = argparser.parse_args()

    torch.manual_seed(args.seed)

    # Dataset preprocessing/loading

    # Read split files
    with open('data/rocklin/mega_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    train_names = splits['train']
    val_names = splits['val'].tolist()
    test_names = splits['test'].tolist()

    # Load AF predicted structures
    pdb_dict_train = []
    for name in tqdm(train_names, desc="Loading train set"):
        name = name.split('.pdb', 1)[0] + '.pdb'
        name = name.replace("|", ':')
        path = os.path.join(args.pdb_dir, name)
        pdb_dict_train.append(parse_PDB(path)[0])

    pdb_dict_val = []
    for name in tqdm(val_names, desc="Loading validation set"):
        name = name.split('.pdb', 1)[0] + '.pdb'
        name = name.replace("|", ':')
        path = os.path.join(args.pdb_dir, name)
        pdb_dict_val.append(parse_PDB(path)[0])

    pdb_dict_test = []
    for name in tqdm(test_names, desc="Loading test set"):
        name = name.split('.pdb', 1)[0] + '.pdb'
        name = name.replace("|", ':')
        path = os.path.join(args.pdb_dir, name)
        pdb_dict_test.append(parse_PDB(path)[0])

    # Read ddG data
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    ddG_data = {}

    df_2 = pd.read_csv(args.stability_data, low_memory=False)
    dataset_3 = df_2[df_2['ddG_ML']!='-']
    dataset_3_noindel = dataset_3.loc[~dataset_3.mut_type.str.contains("ins") & ~dataset_3.mut_type.str.contains("del"), :].reset_index(drop=True)

    for name in train_names + val_names + test_names:
        cleaned_name = name.split('.pdb', 1)[0] + '.pdb'
        cleaned_name = cleaned_name.replace("|", ':')
        ddG_data[cleaned_name] = dataset_3_noindel[(dataset_3_noindel['WT_name'] == name) & (dataset_3_noindel['mut_type'] != 'wt')]
    
    for name, mut_df in ddG_data.items():
        ddG_data[name] = {
            'mut_seqs': mut_df['aa_seq'].to_list(),
            'ddG': mut_df['ddG_ML'].to_numpy(dtype=np.float32)
        }

    # Featurize mutations as sequences
    for name, mut_df in ddG_data.items():
        index_matrix = []
        for s in mut_df['mut_seqs']:
            indices = np.asarray([ALPHABET.index(a) for a in s], dtype=np.int64)
            index_matrix.append(indices)
        index_matrix = np.vstack(index_matrix)
        ddG_data[name]['mut_seqs'] = torch.from_numpy(index_matrix)
        ddG_data[name]['ddG'] = torch.tensor(mut_df['ddG'])

    # Mask all input chains
    for dict in pdb_dict_train:
        dict['masked_list'] = ['A']
        dict['visible_list'] = []
    for dict in pdb_dict_val:
        dict['masked_list'] = ['A']
        dict['visible_list'] = []
    for dict in pdb_dict_test:
        dict['masked_list'] = ['A']
        dict['visible_list'] = []

    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=3000) 
    dataset_valid = StructureDataset(pdb_dict_val, truncate=None, max_length=3000)
    dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=3000)
        
    # Load pre-trained ProteinMPNN
    device = torch.device("cuda")

    pmpnn = ProteinMPNN(node_features=128, 
                        edge_features=128, 
                        hidden_dim=128,
                        num_encoder_layers=3, 
                        num_decoder_layers=3, 
                        k_neighbors=48, 
                        dropout=0.0,
                        augment_eps=0.0)
    
    mpnn_checkpoint = torch.load(args.checkpoint)
    if 'model_state_dict' in mpnn_checkpoint.keys():
        pmpnn.load_state_dict(mpnn_checkpoint['model_state_dict'])
    else:
        pmpnn.load_state_dict(mpnn_checkpoint)
    print('Successfully loaded model at', args.checkpoint)

    model = StaBddG(pmpnn=pmpnn, use_antithetic_variates=not args.no_antithetic_variates, 
                    noise_level=args.noise_level, device=device)
    
    model.to(device)
    model.eval()

    # Initialize wandb logging
    if args.wandb:
        print('Initializing weights and biases.')
        wandb.init(
            project="",
            entity='',
            name=args.run_name, 
        )
        print('Weights and biases intialized.')

    finetune(model, dataset_train, dataset_valid, dataset_test, ddG_data, args, batch_size=args.batch_size, device=device)
