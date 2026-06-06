import os
import pickle
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from .mpnn_utils import parse_PDB
from .utils import extract_chains

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        print(f'Combined datasets of sizes {len(dataset1)} and {len(dataset2)} to get a dataset of size {len(dataset1) + len(dataset2)}/')

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, index):
        if index >= len(self.dataset1):
            return self.dataset2[index - len(self.dataset1)]
        return self.dataset1[index]
    
class PPIDataset(Dataset):
    def __init__(self, 
                csv_path,
                split_path='',
                pdb_dir='/home/exx/arthur/data/SKEMPI_v2/PDBs',
                pdb_dict_cache_path='cache/skempi_full_mask_pdb_dict.pkl',
                af_apo_structures=False,
                alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        
        self.alphabet = alphabet
        self.data = []
        self.num_mutants = 0

        ### Process skempi csv, keep only relevant split
        ddG_df = self.preprocess_df(csv_path)

        ### Keep only split clusters
        if split_path:
            with open(split_path, 'rb') as f:
                split_pdbs = pickle.load(f)
            ddG_df = ddG_df[ddG_df['#Pdb'].isin(split_pdbs)]

        ### split the input pdb files into chains
        complex_names = set(ddG_df['#Pdb'].to_list())
        for pdb in complex_names:
            *pdb_base, binder1_chains, binder2_chains = pdb.split('_')
            pdb_base = '_'.join(pdb_base)
            pdb_path = os.path.join(pdb_dir, f"{pdb_base}.pdb")
            
            if not os.path.exists(pdb_path):
                raise FileNotFoundError(f"PDB file {pdb_path} does not exist.")
            
            if not os.path.exists(f"{pdb_dir}/{pdb_base}_{binder1_chains}.pdb") or \
               not os.path.exists(f"{pdb_dir}/{pdb_base}_{binder2_chains}.pdb"):
                extract_chains(pdb_path, f"{pdb_dir}/{pdb_base}_{binder1_chains}.pdb", binder1_chains,
                            f"{pdb_dir}/{pdb_base}_{binder2_chains}.pdb", binder2_chains)
            
        ### Get PDB file names
        self.pdb_names = []
        self.pdb_file_names = []
        for complex_name in complex_names:
            *pdb_id, binder1_chains, binder2_chains = complex_name.split('_')
            pdb_id = '_'.join(pdb_id)

            self.pdb_names.append(pdb_id)
            self.pdb_names.append(pdb_id + '_' + binder1_chains)
            self.pdb_names.append(pdb_id + '_' + binder2_chains)
            
            self.pdb_file_names.append(pdb_id)
            if af_apo_structures and pdb_id != '3Q8D': # no MSA could be found for 3Q8D
                self.pdb_file_names.append(pdb_id + '_' + binder1_chains + '_AF')
                self.pdb_file_names.append(pdb_id + '_' + binder2_chains + '_AF')
            else:
                self.pdb_file_names.append(pdb_id + '_' + binder1_chains)
                self.pdb_file_names.append(pdb_id + '_' + binder2_chains )

        ### Load cached structure dictionary
        structure_dict = {}
        if os.path.exists(pdb_dict_cache_path):
            print('Found cached structure dictionary at', pdb_dict_cache_path)
            with open(pdb_dict_cache_path, 'rb') as f:
                structure_dict = pickle.load(f)
        else:
            print('Did not find a cached structure dictionary, processing structures now.')
            structure_dict = self.preprocess_structures(pdb_dir, pdb_dict_cache_path)

        ### Process mutants
        grouped_ddG_df = ddG_df.groupby('#Pdb')
        if af_apo_structures:
            for x in structure_dict:
                x['name'] = x['name'].replace("_AF", "")

        structure_dict = [x for x in structure_dict if x['name'] in self.pdb_names]
        
        self.name_to_struct = {x['name']: x for x in structure_dict}
        for complex_name, group in grouped_ddG_df:
            *pdb_id, binder1_chains, binder2_chains = complex_name.split('_')
            pdb_id = '_'.join(pdb_id)
            complex = self.name_to_struct[pdb_id]
            binder1 = self.name_to_struct[pdb_id + '_' + binder1_chains]
            binder2 = self.name_to_struct[pdb_id + '_' + binder2_chains]

            mutations_list = group['mutations'].to_list()

            complex_mut_seqs = self.mutations_to_seq(complex, mutations_list)
            binder1_mut_seqs = self.mutations_to_seq(binder1, mutations_list)
            binder2_mut_seqs = self.mutations_to_seq(binder2, mutations_list)

            ddG = group['ddG'].to_numpy()

            self.data.append({
                'name': complex_name,
                'complex': complex,
                'binder1': binder1,
                'binder2': binder2,
                'complex_mut_seqs': torch.from_numpy(complex_mut_seqs),
                'binder1_mut_seqs': torch.from_numpy(binder1_mut_seqs),
                'binder2_mut_seqs': torch.from_numpy(binder2_mut_seqs),
                'ddG': torch.from_numpy(ddG),
                'mutation_list': mutations_list
            })

            self.num_mutants += len(mutations_list)

        print(f'Finished loading dataset with {len(self.data)} complexes and {self.num_mutants} mutants.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def preprocess_structures(
            self,
            pdb_dir: str = '',
            pdb_dict_cache_path: str = 'cache/pdb_dict.pkl'
        ) -> dict:

        file_paths = [os.path.join(pdb_dir, f'{name}.pdb') for name in self.pdb_file_names]

        pdb_dict = []

        for path in tqdm(file_paths, desc="Reading in PDBs"):
            pdb_dict.append(parse_PDB(path)[0])
        
        for dict in pdb_dict:
            all_chains = []
            for key in dict.keys():
                if key.startswith('seq_chain_'):
                    all_chains.append(key.split('_')[-1])
            dict['masked_list'] = all_chains
            dict['visible_list'] = []
        
        if pdb_dict_cache_path != '':
            cache_dir  = os.path.dirname(pdb_dict_cache_path) 
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

            with open(pdb_dict_cache_path, 'wb') as f:
                pickle.dump(pdb_dict, f)
        
        return pdb_dict

    def mutations_to_seq(
            self,
            processed_struct: dict, 
            mutations_list: list, 
        ) -> np.ndarray:
        
        chain_offset = {}
        offset = 0
        for key in processed_struct.keys():
            if key.startswith('seq_chain_'):
                chain_offset[key.split('_')[-1]] = offset
                offset += len(processed_struct[key])

        index_matrix = []
        for mutations in mutations_list:
            mut_seq = list(processed_struct['seq'])
            for mut in set(mutations):
                mut_chain = mut[1]
                wt_aa = mut[0]
                mut_aa = mut[-1]

                if not mut_chain in chain_offset.keys(): continue
                
                mut_chain_offset = chain_offset[mut_chain]
                mut_pos = int(mut[2:-1]) + mut_chain_offset - 1

                assert mut_seq[mut_pos] == wt_aa

                mut_seq[mut_pos] = mut_aa

            mut_seq = [x if x in self.alphabet else 'X' for x in mut_seq]
            indices = np.asarray([self.alphabet.index(a) for a in mut_seq], dtype=np.int64)
            index_matrix.append(indices)

        index_matrix = np.vstack(index_matrix)

        return index_matrix
    
    def preprocess_df(
            self,
            csv_path
        ) -> pd.DataFrame:

        raise NotImplementedError("")

class SKEMPIDataset(PPIDataset):
    def __init__(self, 
            csv_path='data/SKEMPI/filtered_skempi.csv',
            split_path='',
            pdb_dir='/home/exx/arthur/data/SKEMPI_v2/PDBs',
            pdb_dict_cache_path='cache/skempi_full_mask_pdb_dict.pkl', 
            alphabet='ACDEFGHIKLMNPQRSTVWYX',
            af_apo_structures=False): 
              
        super().__init__(
            csv_path=csv_path,
            split_path=split_path,
            pdb_dir=pdb_dir,
            pdb_dict_cache_path=pdb_dict_cache_path, 
            alphabet=alphabet,
            af_apo_structures=af_apo_structures
        )

    def preprocess_df(
            self,
            csv_path='data/SKEMPI/filtered_skempi.csv'
        ) -> pd.DataFrame:

        df = pd.read_csv(csv_path)
        processed_df = []

        for _, row in df.iterrows():
            mutations = row['Mutation(s)_cleaned'].split(',')
            row['mutations'] = mutations
            ### If there is no ddG column, set it to 0
            if 'ddG' not in row or pd.isna(row['ddG']):
                row['ddG'] = 0.0
            processed_df.append(row[['#Pdb', 'mutations', 'ddG']])

        processed_skempi = pd.DataFrame(processed_df)
        return processed_skempi

class YeastDataset(PPIDataset):

    def __init__(self, 
            csv_path='data/SKEMPI/filtered_skempi.csv',
            split_path='',
            pdb_dir='/home/exx/arthur/data/SKEMPI_v2/PDBs',
            pdb_dict_cache_path='cache/yeast_pdb_dict.pkl', 
            alphabet='ACDEFGHIKLMNPQRSTVWYX'): 
        
        super().__init__(
            csv_path=csv_path,
            split_path=split_path,
            pdb_dir=pdb_dir,
            pdb_dict_cache_path=pdb_dict_cache_path, 
            alphabet=alphabet
        )
        
    def preprocess_df(
            self,
            csv_path=''
        ) -> pd.DataFrame:

        df = pd.read_csv(csv_path)
        processed_df = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            row['mutations'] = [row['wt_AA'] + 'A' + str(row['pos']) + row['mut_AA']]
            row['#Pdb'] = row['pdb_path'][:-4] + '_A_B'
            row['ddG'] = -row['ddG']
            processed_df.append(row[['#Pdb', 'mutations', 'ddG']])

        processed_yeast = pd.DataFrame(processed_df)
        return processed_yeast
