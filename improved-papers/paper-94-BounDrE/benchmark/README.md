# Using Benchmark data

Here we provide the 10-Fold CV sets for drug-likeness prediction benchmark, with [drugs](approved_drugs) from `Drugbank` and 100k [non-drugs](ZINC_100k) sampled fom `ZINC20` databases. Specifically, we provide the whole dataset, and pickle files for the indices of train/valid/test sets for each fold. For all CV sets, you can load the train, valid, and test sets using the python code lines below:

```python
import numpy as np
import pickle

drug_smiles = *load ( [drug_SMILES_file] ) *
compound_smiles = *load ( [compound_SMILES_file] ) *

split = 'time' # Options: ['time', 'scaffold']
test_comp_ratio = 10 # The ratio of compound in the testset. options: {1, 5, 10, 50 100}
cv = 0 # in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

with open(f'splits/{split}_splits/{split}_split_indices_1:{ratio}_cv{cv}.pkl', 'rb') as f:
    indices = pickle.load(f)

drug_train_idx, drug_valid_idx, drug_test_idx = indices['drug']
compound_train_idx, compound_valid_idx, compound_test_idx = indices['compound']

train_drug_smiles, valid_drug_smiles, test_drug_smiles =\
    drug_smiles[drug_train_idx], drug_smiles[drug_valid_idx], drug_smiles[drug_test_idx]

train_compound_smiles, valid_compound_smiles, test_compound_smiles =\
    compound_smiles[compound_train_idx], compound_smiles[compound_valid_idx], compound_smiles[compound_test_idx]
```
Please note the order of the drugs and compounds MUST be identical to the order of drugs and compounds in [`drugbank_5.1.12_approved.csv`](compound_sets/approved_drugs/drugbank_5.1.12_approved.csv) and [`ZINC_clean_annot_100k.csv`](compound_sets/ZINC_100k/ZINC_clean_annot_100k.csv).


### Biomedical knowledge graph embeddings of approved drugs
We utilize the biomedical knowledge graph embeddings of approved drugs from the knowledge graph `MSI` network using the node embedding-based algorithm `DREAMwalk`. There are total 1,449 approved drugs in the network. We provide the SMILES strings and the embeddings (`dim`=300) of the 1,449 drug entities under the [`approved_drugs/knowledge_alignment`](compound_sets/approved_drugs/knowledge_alignment/) folder.

### Time-based split
Drug that were annotated with approval time are split into train/valid/test sets of 8:1:1 ratio. ZINC compounds are also split into 10 subsets, matching the number of drugs in a single subset.

### Scaffold-based split
Drugs are grouped using Bemis-Murko Scaffolds through RDkit. Then, they are partitioned equally into 10 subsets. ZINC compounds are also split into 10 subsets, matching the number of drugs in a single subset. Each subset acts as a test set for each CV.

## External compound sets
We further provide the toxic compound sets for measuring the generalizability of the datasets [here](compound_sets/external_validation).

### Toxic compound sets

1. Withdrawn drugs (`DrugBank` ver. 5.1.12) 
2. Hepatotoxic compounds (`TOXRIC` database, accessed 2024.07)
3. Cardiotoxic compounds (`TOXRIC` database, accessed 2024.07)
4. Carcinogenic compounds (`TOXRIC` database, accessed 2024.07)

### Compound sets of diverse drug-discovery stages
1. `Targetdiff`-generated 10k compounds (using **Bcr** protein pocket)
2. `MOOD`-generated 10k compounds
3. Investigational compounds (`ZINC20`)
4. World-approved drugs (`ZINC20`)


### References
```plain
[Drugbank] Craig Knox, Mike Wilson, Christen M Klinger, Mark Franklin, Eponine Oler, Alex Wilson, Allison Pon, Jordan Cox, Na Eun Chin, Seth A Strawbridge, et al. Drugbank 6.0: the drugbank knowledgebase for 2024. Nucleic acids research, 52(D1):D1265–D1275, 2024

[ZINC20] John J Irwin, Khanh G Tang, Jennifer Young, Chinzorig Dandarchuluun, Benjamin R Wong,Munkhzul Khurelbaatar, Yurii S Moroz, John Mayfield, and Roger A Sayle. Zinc20—a free ultralarge-scale chemical database for ligand discovery. Journal of chemical information and modeling, 60(12):6065–6073, 2020.

[MSI] Camilo Ruiz, Marinka Zitnik, and Jure Leskovec. Identification of disease treatment mechanisms through the multiscale interactome. Nature communications, 12(1):1796, 2021.

[DREAMwalk] Dongmin Bang, Sangsoo Lim, Sangseon Lee, and Sun Kim. Biomedical knowledge graph learning for drug repurposing by extending guilt-by-association to multiple layers. Nature Communications, 14(1):3570, 2023.

[TOXRIC] Lianlian Wu, Bowei Yan, Junshan Han, Ruijiang Li, Jian Xiao, Song He, and Xiaochen Bo. Toxric: a comprehensive database of toxicological data and benchmarks. Nucleic Acids Research, 51 (D1):D1432–D1445, 2023.

[Targetdiff] Jiaqi Guan, Wesley Wei Qian, Xingang Peng, Yufeng Su, Jian Peng, and Jianzhu Ma. 3d equivariant diffusion for target-aware molecule generation and affinity prediction. In The Eleventh International Conference on Learning Representations, 2023.

[MOOD] Seul Lee, Jaehyeong Jo, and Sung Ju Hwang. Exploring chemical space with score-based out-of-distribution generation. In International Conference on Machine Learning, pp. 18872–18892. PMLR, 2023.
```