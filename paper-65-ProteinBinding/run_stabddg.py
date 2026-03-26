import pandas as pd
import torch
import os
import argparse
from tqdm import tqdm
from stabddg.mpnn_utils import ProteinMPNN
from stabddg.model import StaBddG
from stabddg.ppi_dataset import SKEMPIDataset
from stabddg.utils import extract_chains

def run(model, dataset, ensemble=20, batch_size=10000):
    pred_df=[]
    for sample in tqdm(dataset):  
        complex, binder1, binder2 = sample['complex'], sample['binder1'], sample['binder2']
        complex_mut_seqs = sample['complex_mut_seqs'].to(device)
        binder1_mut_seqs = sample['binder1_mut_seqs'].to(device)
        binder2_mut_seqs = sample['binder2_mut_seqs'].to(device)

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
        binding_ddG_pred *= -1 # Convert to convention of negative values for stabilizing mutations

        name, mutations = sample['name'], sample['mutation_list']
        data = {
            "Name": [name] * len(mutations),  # Repeat the name for all rows
            "Mutation": mutations,
            "Prediction": binding_ddG_pred.cpu().detach().numpy()
        }

        df = pd.DataFrame(data)
        pred_df.append(df)

    pred_df = pd.concat(pred_df, ignore_index=True)
    return pred_df

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--checkpoint", type=str, default="model_ckpts/stabddg.pt")
    argparser.add_argument("--run_name", type=str, default="output")
    argparser.add_argument("--csv_path", type=str, default="")
    argparser.add_argument("--pdb_path", type=str, default="examples/one_mutation/1AO7.pdb")
    argparser.add_argument("--mutation", type=str, default="EA63Q,QD30V,KA66A")
    argparser.add_argument("--chains", type=str, default="ABC_DE")
    argparser.add_argument("--pdb_dir", type=str, default="")
    argparser.add_argument("--mc_samples", type=int, default=20)
    argparser.add_argument("--pdb_dict_cache_name", type=str, default="")
    argparser.add_argument("--output_dir", type=str, default="")
    argparser.add_argument("--trials", type=int, default=1)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--noise_level", type=float, default=0.1, help="amount of backbone noise")
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--device", type=str, default="cuda")
    
    args = argparser.parse_args()
    torch.manual_seed(args.seed)

    if not args.pdb_dir:
        if not os.path.exists(args.pdb_path):
            raise FileNotFoundError(f"PDB file {args.pdb_path} does not exist.")
        pdb_file_name = os.path.splitext(os.path.basename(args.pdb_path))[0]
        pdb_dir = os.path.join(os.path.dirname(args.pdb_path), pdb_file_name + "_output")
        if not os.path.exists(pdb_dir):
            os.makedirs(pdb_dir, exist_ok=True)

        binder1_chains, binder2_chains = args.chains.split('_')
            
        ### copy pdb file to pdb_dir
        complex_path = os.path.join(pdb_dir, f"{pdb_file_name}.pdb")
        os.system(f"cp {args.pdb_path} {complex_path}")

        ### create binder pdb files
        extract_chains(complex_path, f"{pdb_dir}/{pdb_file_name}_{binder1_chains}.pdb", binder1_chains,
                       f"{pdb_dir}/{pdb_file_name}_{binder2_chains}.pdb", binder2_chains)
        
        ### create csv file
        # mutations = args.mutation.split(',')
        data = {
            "#Pdb": f"{pdb_file_name}_{binder1_chains}_{binder2_chains}",  # Repeat the name for all rows
            "Mutation(s)_cleaned": args.mutation,
            "ddG": 0.0  # Placeholder for ddG
        }
        df = pd.DataFrame([data])
        csv_path = os.path.join(pdb_dir, f"input.csv")
        df.to_csv(csv_path, index=False)
        pdb_cache_path = os.path.join(pdb_dir, f"{pdb_file_name}_{binder1_chains}_{binder2_chains}_cache.pkl")
        output_dir = pdb_dir
    else:
        output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(args.csv_path), 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        pdb_dir = f"{output_dir}/pdbs"
        if not os.path.exists(pdb_dir):
            os.makedirs(pdb_dir, exist_ok=True)

        ### copy pdbs in pdb_dir to output_dir
        for pdb_file in os.listdir(args.pdb_dir):
            if pdb_file.endswith('.pdb'):
                os.system(f"cp {os.path.join(args.pdb_dir, pdb_file)} {pdb_dir}")
        pdb_cache_path = f'{pdb_dir}/{args.pdb_dict_cache_name}' if args.pdb_dict_cache_name else ''

        ### rename mutation column to be consistent with SKEMPI column name
        mut_csv = pd.read_csv(args.csv_path)
        mut_csv['Mutation(s)_cleaned'] = mut_csv['mutation']
        csv_path = os.path.join(output_dir, "input.csv")
        mut_csv.to_csv(csv_path, index=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda" else torch.device("cpu")
    
    dataset = SKEMPIDataset(pdb_dir=pdb_dir, csv_path=csv_path, pdb_dict_cache_path=pdb_cache_path)
    
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

    model = StaBddG(pmpnn=pmpnn, use_antithetic_variates=True, noise_level=args.noise_level, device=device)
    
    model.to(device)
    model.eval()

    spearmans = []
    pearsons = []
    rmses = []

    combined_df = None
    with torch.no_grad():
        for i in tqdm(range(args.trials)):
            pred_df = run(
                model,
                dataset,
                ensemble=args.mc_samples,
                batch_size=args.batch_size
            )
            if i == 0:
                combined_df = pred_df[['Name', 'Mutation']]
            combined_df[f'pred_{i+1}'] = pred_df['Prediction']

    combined_df.to_csv(os.path.join(output_dir, f'{args.run_name}.csv'))
    print(f'Saved predictions to {os.path.join(output_dir, f"{args.run_name}.csv")}')