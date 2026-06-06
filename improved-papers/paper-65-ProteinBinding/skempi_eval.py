import numpy as np
import pandas as pd
import torch
import os
import argparse
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from stabddg.mpnn_utils import ProteinMPNN
from stabddg.model import StaBddG
from stabddg.ppi_dataset import SKEMPIDataset, YeastDataset

def eval(model, dataset, ensemble=20, batch_size=10000):
    val_spearman=[]
    val_pearson=[]
    val_rmse=[]
    pred_df=[]
    for sample in tqdm(dataset, desc="Interface"):  
        complex, binder1, binder2 = sample['complex'], sample['binder1'], sample['binder2']
        complex_mut_seqs = sample['complex_mut_seqs'].to(device)
        binder1_mut_seqs = sample['binder1_mut_seqs'].to(device)
        binder2_mut_seqs = sample['binder2_mut_seqs'].to(device)
        ddG = sample['ddG']

        binding_ddG_pred_ensemble = []
        for _ in range(ensemble):
            with torch.no_grad(): 
                N = complex_mut_seqs.shape[0]
                M = batch_size // complex_mut_seqs.shape[1] # convert number of tokens to number of sequences per batch

                binding_ddG_pred_ = []
                for batch_idx in range(0, N, M):
                    B = min(N - batch_idx, M)
                    batch_binding_ddG_pred = model(complex, binder1, binder2, complex_mut_seqs[batch_idx:batch_idx+B], 
                                                binder1_mut_seqs[batch_idx:batch_idx+B], binder2_mut_seqs[batch_idx:batch_idx+B])
                    binding_ddG_pred_.append(batch_binding_ddG_pred.cpu().detach())

                binding_ddG_pred_ = torch.cat(binding_ddG_pred_)
                
                binding_ddG_pred_ensemble.append(binding_ddG_pred_.squeeze().cpu())
                
        binding_ddG_pred = torch.stack(binding_ddG_pred_ensemble).mean(dim=0)

        name, mutations = sample['name'], sample['mutation_list']
        data = {
            "#Pdb": [name] * len(mutations),  # Repeat the name for all rows
            "Mutation": mutations,
            "ddG": ddG.cpu().detach().numpy(),
            "Prediction": binding_ddG_pred.cpu().detach().numpy()
        }

        df = pd.DataFrame(data)
        pred_df.append(df)

        sp, _ = spearmanr(binding_ddG_pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())

        pr, _ = pearsonr(binding_ddG_pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())

        rmse = torch.sqrt(torch.mean((binding_ddG_pred.cpu() - ddG.cpu()) ** 2)).item()

        val_spearman.append(sp)
        val_pearson.append(pr)
        val_rmse.append(rmse)

    pred_df = pd.concat(pred_df, ignore_index=True)
    return pred_df#, np.mean(val_spearman), np.mean(val_pearson), np.mean(val_rmse)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--checkpoint", type=str, default="./model_ckpts/stabddg.pt")
    argparser.add_argument("--run_name", type=str, default="eval")
    argparser.add_argument("--skempi_path", type=str, default="data/SKEMPI/filtered_skempi.csv")
    argparser.add_argument("--skempi_pdb_dir", type=str, default="")
    argparser.add_argument("--skempi_pdb_cache_path", type=str, default="cache/skempi_full_mask_pdb_dict.pkl")
    argparser.add_argument("--skempi_split_path", type=str, default="data/SKEMPI/test_pdb.pkl")
    argparser.add_argument("--yeast_path", type=str, default="")
    argparser.add_argument("--yeast_pdb_dir", type=str, default="")
    argparser.add_argument("--ensemble", type=int, default=20)
    argparser.add_argument("--yeast_pdb_cache_path", type=str, default="")
    argparser.add_argument("--output_dir", type=str, default="cache")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--noise_level", type=float, default=0.1, help="amount of backbone noise")
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--device", type=str, default="cuda")
    
    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda" else torch.device("cpu")
    
    dataset = None

    if args.yeast_path:
        dataset = YeastDataset(pdb_dir=args.yeast_pdb_dir, csv_path=args.yeast_path, pdb_dict_cache_path=args.yeast_pdb_cache_path)
    elif args.skempi_path:
        dataset = SKEMPIDataset(split_path=args.skempi_split_path, pdb_dir=args.skempi_pdb_dir, 
                                csv_path=args.skempi_path, pdb_dict_cache_path=args.skempi_pdb_cache_path)
    
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

    model = StaBddG(pmpnn=pmpnn, noise_level=args.noise_level, device=device)
    
    model.to(device)
    model.eval()

    combined_df = None
    with torch.no_grad():
        pred_df = eval(model, dataset, ensemble=args.ensemble, batch_size=args.batch_size)
        combined_df = pred_df[['#Pdb', 'Mutation', 'ddG']]
        combined_df[f'ddG_pred'] = pred_df['Prediction']

    combined_df.to_csv(os.path.join(args.output_dir, f'{args.run_name}.csv'))
