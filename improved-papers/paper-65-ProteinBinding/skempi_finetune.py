import numpy as np
import torch
import argparse
import os
import wandb


from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from stabddg.mpnn_utils import ProteinMPNN
from stabddg.ppi_dataset import SKEMPIDataset, YeastDataset, CombinedDataset
from stabddg.model import StaBddG

def val_epoch(model, val_dataset, loss_fn, trials=3, 
              val_ensemble=3, batch_size=10000, device='cuda'):
    val_spearman=[]
    val_rmse=[]
    all_pred=[]
    all_labels=[]
    overall_sp=[]
    overall_pr=[]
    val_loss=0
    for _ in range(trials):
        for sample in val_dataset:  
            complex, binder1, binder2 = sample['complex'], sample['binder1'], sample['binder2']
            complex_mut_seqs = sample['complex_mut_seqs']
            binder1_mut_seqs = sample['binder1_mut_seqs']
            binder2_mut_seqs = sample['binder2_mut_seqs']
            ddG = sample['ddG'].float()
            N = complex_mut_seqs.shape[0]
            M = batch_size // complex_mut_seqs.shape[1] # convert number of tokens to number of sequences per batch
            ensemble_ddG_pred = 0

            with torch.no_grad():
                for _ in range(val_ensemble):
                    binding_ddG_pred_ = []
                    for batch_idx in range(0, N, M):
                        B = min(N - batch_idx, M)
                        batch_binding_ddG_pred = model(complex, binder1, binder2, complex_mut_seqs[batch_idx:batch_idx+B], 
                                                    binder1_mut_seqs[batch_idx:batch_idx+B], binder2_mut_seqs[batch_idx:batch_idx+B])
                        binding_ddG_pred_.append(batch_binding_ddG_pred.cpu().detach())

                    binding_ddG_pred_ = torch.cat(binding_ddG_pred_)
                    ensemble_ddG_pred += binding_ddG_pred_

            binding_ddG_pred = ensemble_ddG_pred / val_ensemble

            assert binding_ddG_pred.shape == ddG.shape
            # Check that pred1 and ddG do not contain any NaNs
            assert not torch.isnan(binding_ddG_pred).any(), "pred contains NaN values."
            assert not torch.isnan(ddG).any(), "ddG contains NaN values."

            # Optionally, if you want to also check for Infinities:
            assert torch.isfinite(binding_ddG_pred).all(), "pred contains Inf values."
            assert torch.isfinite(ddG).all(), "ddG contains Inf values."

            sp, _ = spearmanr(binding_ddG_pred.numpy(), ddG.numpy())
            rmse = torch.sqrt(torch.mean((binding_ddG_pred - ddG) ** 2)).item()

            val_spearman.append(sp)
            val_rmse.append(rmse)
            val_loss += loss_fn(binding_ddG_pred, ddG).item()

            all_pred.append(binding_ddG_pred.numpy())
            all_labels.append(ddG.numpy())
                
        sp, _ = spearmanr(np.concatenate(all_pred), np.concatenate(all_labels))
        pr, _ = pearsonr(np.concatenate(all_pred), np.concatenate(all_labels))
        overall_sp.append(sp)
        overall_pr.append(pr)

    return (np.mean(val_spearman), np.mean(val_rmse), val_loss/(trials * len(val_dataset)), np.mean(overall_sp), np.mean(overall_pr))

def finetune(model, train_dataset, val_dataset, args, lr=1e-5, batch_size=10000, 
             n_epochs=500, train_val_freq=10, model_save_dir='cache/skempi_finetuned', use_wandb=False, device='cuda'):

    # optimizer = get_std_opt(model.parameters(), 128, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ddG_loss_fn = torch.nn.MSELoss()
    # ddG_loss_fn = torch.nn.L1Loss()

    for epoch in tqdm(range(n_epochs), desc='Epoch'):
        model.train()
        train_sum = 0
        avg_spearman = 0
        avg_rmse = 0
        train_samples = 0
        for sample in train_dataset:
            try:
                complex, binder1, binder2 = sample['complex'], sample['binder1'], sample['binder2']
                complex_mut_seqs = sample['complex_mut_seqs']
                binder1_mut_seqs = sample['binder1_mut_seqs']
                binder2_mut_seqs = sample['binder2_mut_seqs']
                ddG = sample['ddG'].float().to(device)
                N = complex_mut_seqs.shape[0]
                M = batch_size // complex_mut_seqs.shape[1] # convert number of tokens to number of sequences per batch
                binding_ddG_pred = []

                ### shuffling for batching
                permutation = torch.randperm(ddG.shape[0])
                ddG = ddG[permutation]
                complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs = complex_mut_seqs[permutation], binder1_mut_seqs[permutation], binder2_mut_seqs[permutation]
                complex_loss_sum = []

                for batch_idx in range(0, N, M):
                    B = min(N - batch_idx, M)
                    batch_binding_ddG_pred = model(complex, binder1, binder2, complex_mut_seqs[batch_idx:batch_idx+B], 
                                              binder1_mut_seqs[batch_idx:batch_idx+B], binder2_mut_seqs[batch_idx:batch_idx+B])

                    ddG_loss = ddG_loss_fn(batch_binding_ddG_pred, ddG[batch_idx:batch_idx+B])
                    ddG_loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    complex_loss_sum.append(ddG_loss.item())
                    binding_ddG_pred.append(batch_binding_ddG_pred.cpu().detach())
                    train_samples += B

                if len(binding_ddG_pred) > 1:
                    binding_ddG_pred = torch.cat(binding_ddG_pred)
                else:
                    binding_ddG_pred = binding_ddG_pred[0]
                    ddG = ddG[:binding_ddG_pred.shape[0]]

                sp, _ = spearmanr(binding_ddG_pred.numpy(), ddG.cpu().detach().numpy())
                rmse = torch.sqrt(torch.mean((binding_ddG_pred.cpu() - ddG.cpu()) ** 2)).item()

                avg_spearman += sp
                avg_rmse += rmse
                train_sum += np.mean(complex_loss_sum)

            except Exception as e:
                print('Failed on complex', sample['complex']['name'])
                print(e)
        
        if train_val_freq != -1 and (epoch+1) % train_val_freq == 0:
            with torch.no_grad():
                model.eval()
                (val_spearman, 
                 val_rmse, 
                 val_loss,
                 overall_sp, 
                 overall_pr) = val_epoch(model, val_dataset, 
                                ddG_loss_fn,
                                trials=args.val_trials,
                                val_ensemble=args.val_ensembles)
                
                if use_wandb:
                    wandb.log({'val_loss': val_loss, 
                    'val_per_structure_spearman': val_spearman, 
                    'val_per_structure_rmse': val_rmse,
                    'val_overall_spearman': overall_sp,
                    'val_overall_pearson': overall_pr,}, step=epoch+1),
        if use_wandb:
            wandb.log({'train_loss': train_sum/len(train_dataset), 
                    'train_spearman': avg_spearman/len(train_dataset), 
                    'train_rmse': avg_rmse/len(train_dataset)}, step=epoch+1)
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch+1)

        if args.model_save_freq != -1 and (epoch+1) % args.model_save_freq == 0:
            # if model save directory does not exist, create it
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(model.pmpnn.state_dict(), os.path.join(model_save_dir, f"{args.run_name}_{epoch+1}.pt"))
    
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    torch.save(model.pmpnn.state_dict(), os.path.join(model_save_dir, f"{args.run_name}_final.pt"))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--run_name", type=str, default="skempi finetune")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--wandb", action='store_true')
    argparser.add_argument("--checkpoint", type=str, default="cache/megascale_finetuned/early_stopping_noamopt_all_data.pt")
    argparser.add_argument("--model_save_dir", type=str, default="cache/skempi_finetuned")
    argparser.add_argument("--skempi_path", type=str, default="data/SKEMPI/filtered_skempi.csv")
    argparser.add_argument("--skempi_pdb_dir", type=str, default="/home/exx/arthur/data/SKEMPI_v2/PDBs")
    argparser.add_argument("--skempi_pdb_cache_path", type=str, default="cache/skempi_full_mask_pdb_dict.pkl")
    argparser.add_argument("--af_apo_structures", action='store_true') 
    argparser.add_argument("--train_split_path", type=str, default="data/SKEMPI/train_clusters.pkl")
    argparser.add_argument("--second_train_split_path", type=str, default="")
    argparser.add_argument("--yeast_path", type=str, default="")
    argparser.add_argument("--yeast_pdb_dir", type=str, default="")
    argparser.add_argument("--yeast_pdb_cache_path", type=str, default="")
    argparser.add_argument("--yeast_only", action='store_true')
    argparser.add_argument("--val_split_path", type=str, default="")
    argparser.add_argument("--model_save_freq", type=int, default=10)
    argparser.add_argument("--epochs", type=int, default=200)
    argparser.add_argument("--lr", type=float, default=1e-6)
    argparser.add_argument("--train_val_freq", type=int, default=-1)
    argparser.add_argument("--val_trials", type=int, default=5)
    argparser.add_argument("--val_ensembles", type=int, default=5)
    argparser.add_argument("--decode_mut_last", action='store_true')
    argparser.add_argument("--noise_level", type=float, default=0.2, help="amount of backbone noise")
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--device", type=str, default="cuda")
    
    args = argparser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda" else torch.device("cpu")

    if args.yeast_path:
        yeast_dataset = YeastDataset(pdb_dir=args.yeast_pdb_dir, csv_path=args.yeast_path, pdb_dict_cache_path=args.yeast_pdb_cache_path)

    if args.yeast_only:
        train_dataset = yeast_dataset
    else:
        train_dataset = SKEMPIDataset(split_path=args.train_split_path, pdb_dir=args.skempi_pdb_dir, csv_path=args.skempi_path, pdb_dict_cache_path=args.skempi_pdb_cache_path, af_apo_structures=args.af_apo_structures)
        if args.second_train_split_path:
            train_dataset2 = SKEMPIDataset(split_path=args.second_train_split_path, pdb_dir=args.skempi_pdb_dir, csv_path=args.skempi_path, pdb_dict_cache_path=args.skempi_pdb_cache_path, af_apo_structures=args.af_apo_structures)
            train_dataset = CombinedDataset(train_dataset, train_dataset2)
        if args.yeast_path:
            train_dataset = CombinedDataset(train_dataset, yeast_dataset)

    if args.val_split_path == "":
        val_dataset = None
        assert args.train_val_freq == -1, "If no validation set is provided, train_val_freq must be -1."
    else:
        val_dataset = SKEMPIDataset(split_path=args.val_split_path, csv_path=args.skempi_path, pdb_dir=args.skempi_pdb_dir, pdb_dict_cache_path=args.skempi_pdb_cache_path, af_apo_structures=args.af_apo_structures)

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

    model = StaBddG(pmpnn=pmpnn, noise_level=args.noise_level, use_antithetic_variates=True, device=device)
    model.to(device)

    if args.wandb:
        wandb.init(
            project="binding",
            entity='',
            name=args.run_name, 
        )

    finetune(model, train_dataset, val_dataset, args, 
             lr=args.lr, 
             batch_size=args.batch_size, n_epochs=args.epochs, train_val_freq=args.train_val_freq,
             model_save_dir=args.model_save_dir, use_wandb=args.wandb, device=device)
