import os
import pathlib
import pickle
from typing import Dict, Literal

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from torchmetrics import MeanSquaredError, MeanAbsoluteError
import time
import warnings

from tqdm import tqdm

### packages for visualization
from src.analysis.rdkit_functions import build_molecule_with_partial_charges, compute_molecular_metrics, mol2smiles
import torch
from torchmetrics import Metric, MetricCollection
from torch import Tensor
import wandb
import torch.nn as nn
import numpy as np
import pandas as pd

import fcd
from fcd_torch import FCD
# Helper functions and cache class for MOSES metrics
from src.metrics.molecular_utils import (
    _scaffold_histogram,
    _cosine_from_counters,
    _extract_unique_scaffolds,
    MosesRefCache,
)

from src.metrics.abstract_metrics import compute_ratios
from src import utils
from src.datasets.abstract_dataset import ProportionManager
from src.datasets.data_configs import ModularAtomInfos


# class TrainMolecularMetrics(nn.Module):
#     def __init__(self, remove_h):
#         super().__init__()
#         self.train_atom_metrics = AtomMetrics(atom_list)
#         self.train_bond_metrics = BondMetrics()

#     def forward(
#         self,
#         masked_pred_epsX,
#         masked_pred_epsE,
#         pred_y,
#         true_epsX,
#         true_epsE,
#         true_y,
#         log: bool,
#     ):
#         self.train_atom_metrics(masked_pred_epsX, true_epsX)
#         self.train_bond_metrics(masked_pred_epsE, true_epsE)
#         if log:
#             to_log = {}
#             for key, val in self.train_atom_metrics.compute().items():
#                 to_log["train/" + key] = val.item()
#             for key, val in self.train_bond_metrics.compute().items():
#                 to_log["train/" + key] = val.item()
#             if wandb.run:
#                 wandb.log(to_log, commit=False)

#     def reset(self):
#         for metric in [self.train_atom_metrics, self.train_bond_metrics]:
#             metric.reset()

#     def log_epoch_metrics(self):
#         epoch_atom_metrics = self.train_atom_metrics.compute()
#         epoch_bond_metrics = self.train_bond_metrics.compute()

#         to_log = {}
#         for key, val in epoch_atom_metrics.items():
#             to_log["train_epoch/epoch" + key] = val.item()
#         for key, val in epoch_bond_metrics.items():
#             to_log["train_epoch/epoch" + key] = val.item()

#         if wandb.run:
#             wandb.log(to_log, commit=False)

#         for key, val in epoch_atom_metrics.items():
#             epoch_atom_metrics[key] = f"{val.item() :.3f}"
#         for key, val in epoch_bond_metrics.items():
#             epoch_bond_metrics[key] = f"{val.item() :.3f}"

#         return epoch_atom_metrics, epoch_bond_metrics



class DistributionMetrics(nn.Module):
    def __init__(self, atom_list,output_dims, proportion_manager: ProportionManager):
        super().__init__()
        self.atom_list = atom_list
        max_nums_of_nodes = proportion_manager.get_max_nums_of_nodes()

        self.metric_names = ["node_nums", "node_types", "edge_types", "valencies"]

        node_type_nums = output_dims["X"]
        edge_type_nums = output_dims["E"]
        self.proportion_metrics = nn.ModuleDict({
            "node_nums": NodeNumsProportionMetric(max_nums_of_nodes),
            "node_types": NodeTypsProportionMetric(node_type_nums),
            "edge_types": EdgesTypesProportionMetric(edge_type_nums),
            "valencies": ValenciesProportionMetric(max_nums_of_nodes)
        })
        self.mae_metrics = nn.ModuleDict()
        for name in self.metric_names:
            target_proportion = proportion_manager.get_proportion(name)
            self.register_buffer(f"target_{name}_proportion", target_proportion)
            self.mae_metrics[name] = HistogramsMAE(target_proportion)

    def forward(self, molecules):
        proportions = {}
        for name, metric in self.proportion_metrics.items():
            metric(molecules)
            proportion = metric.compute()
            proportions[name] = proportion
            self.mae_metrics[name](proportion)

        dist_dict = {}
        dist_log_info = {
            "node_types": {
                "labels": self.atom_list,
                "key_template": "molecular_metrics/{}_dist"
            },
            "edge_types": {
                "labels": ["No bond", "Single", "Double", "Triple", "Aromatic"],
                "key_template": "molecular_metrics/bond_{}_dist"
            },
            "valencies": {
                "labels": range(6),
                "key_template": "molecular_metrics/valency_{}_dist"
            }
        }

        for name, info in dist_log_info.items():
            proportion = proportions[name]
            target_proportion = getattr(self, f"target_{name}_proportion")
            for i, label in enumerate(info["labels"]):
                if i < len(proportion):
                    generated_prop = proportion[i]
                    target_prop = target_proportion[i]
                    key = info["key_template"].format(label)
                    dist_dict[key] = (generated_prop - target_prop).item()

        mae_dict = {}
        for name, metric in self.mae_metrics.items():
            key = f"basic_metrics/{name}_mae"
            mae_dict[key] = metric.compute()

        return dist_dict, mae_dict

    def reset(self):
        for metric in self.proportion_metrics.values():
            metric.reset()
        for metric in self.mae_metrics.values():
            metric.reset()


class SmilesProvider:
    def __init__(self, datamodule, cfg, atom_list):
        self.datamodule = datamodule
        self.cfg = cfg
        self.atom_list = atom_list
        self.reference_smiles = self._init_smiles()
    
    def _init_smiles(self):
        splits = ["train","val","test"]
        reference_smiles = {}
        for split in splits:
            smiles_path = self._get_smiles_path(self.cfg, split)
            if os.path.exists(smiles_path):
                print(f"Loading smiles from {smiles_path}")
                with open(smiles_path, "rb") as f:
                    smiles = np.load(f)
            else:
                print(f"Computing smiles for {split}")
                smiles = self._compute_smiles(split)
                with open(smiles_path, "wb") as f:
                    print(f"Saving smiles to {smiles_path}")
                    np.save(f, smiles)
            reference_smiles[split] = smiles
        return reference_smiles

    def _get_smiles_path(self, cfg, split:Literal["train", "val", "test"]):
        datadir = cfg.dataset.datadir
        remove_h = cfg.dataset.remove_h
        name = cfg.dataset.name
        root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
        if name == "qm9":
            smiles_file_name = (
                    f"{split}_smiles_no_h.npy" if remove_h else f"{split}_smiles_h.npy"
                )
        else:
            smiles_file_name = f"{split}_smiles.npy"
        smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
        return smiles_path
    
    # def get_smiles(self, split: Literal["train", "val", "test"]):
    #     return self.reference_smiles[split]

    def get_smiles(self):
        return self.reference_smiles

    def _compute_smiles(self,split:Literal["train", "val", "test"], only_connected=True):
        """
        Guacamol and MOSES use only connected molecules
        qm9 includes disconnected molecules
        Should only include molecules with a single connected component
        """
        mols_smiles = []
        invalid = 0
        disconnected = 0
        _data_loader_map ={
            "train": self.datamodule.train_dataloader(),
            "val": self.datamodule.val_dataloader(),
            "test": self.datamodule.test_dataloader(),
        }
        dataloader = _data_loader_map[split]
        for data in tqdm(dataloader, desc="Processing data"):
            dense_data, node_mask = utils.pyg_data_to_place_holder(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            dense_data = dense_data.mask(node_mask, one_hot_to_index=True)
            X, E = dense_data.X, dense_data.E

            n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

            molecule_list = []
            for k in range(X.size(0)):
                n = n_nodes[k]
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                molecule_list.append([atom_types, edge_types])

            for l, molecule in enumerate(molecule_list):
                mol = build_molecule_with_partial_charges(
                    molecule[0], molecule[1], self.atom_list
                )
                smile = mol2smiles(mol)
                if smile is not None:
                    # # if remove_not_connected:
                    # mols_smiles.append(smile)
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True
                    )
                    if only_connected:
                        if len(mol_frags) == 1:
                            mols_smiles.append(smile)
                    else:
                        mols_smiles.append(smile)
                    if len(mol_frags) > 1:
                        disconnected += 1
                else:
                    invalid += 1
        print("Number of invalid molecules", invalid)
        print("Number of disconnected molecules", disconnected)
        return mols_smiles

class SamplingMolecularMetrics(nn.Module):
    def __init__(self, datamodule, cfg, output_dims, proportion_manager: ProportionManager):
        super().__init__()
        modular_atom_infos = ModularAtomInfos(cfg)
        self.atom_list = modular_atom_infos.get_atom_list()
        self.cfg = cfg
        self.distribution_metrics = DistributionMetrics(self.atom_list, output_dims, proportion_manager)
        self.smiles_provider = SmilesProvider(datamodule, cfg, self.atom_list)
        self.reference_smiles = self._init_reference_smiles()

        # Configuration switches: backward compatible with old fields, support new fine-grained fields
        self.compute_fcd = getattr(cfg.dataset, "compute_fcd", False)
        self.compute_moses = getattr(cfg.dataset, "compute_moses", False)
        self.compute_guacamol  = getattr(cfg.dataset, "compute_guacamol", False)
 

        # Initialize FCD metric using fcd_torch for acceleration
        self._fcd_metric = None
        if self.compute_fcd:
            # Determine device (prefer CUDA if available)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self._fcd_metric = FCD(device=device, n_jobs=8, batch_size=1024)

        # Initialize unified cache (for MOSES metrics + FCD)
        self._cache = None
        # if self.compute_moses or  self.compute_fcd:
        #     self._cache = MosesRefCache(
        #         self.reference_smiles, 
        #         cfg, 
        #         None, 
        #         fcd_metric=self._fcd_metric,
        #         cache_name_prefix="metrics_ref"
        #     )
        self.save_smiles_dir = getattr(cfg.dataset, "save_smiles_dir", None)
    def _init_reference_smiles(self):
        return self.smiles_provider.get_smiles()

    def forward(
        self,
        molecules: list,
        split: Literal["train", "val", "test"],
        labels=None,
    ):
        self.reset()
        result_dict = self._compute_metrics(molecules, labels,split)
        # ratios = self._compute_ratios(result_dict, split)
        # result_dict.update(ratios)
        return result_dict

    def _compute_metrics(self, molecules, labels, split: Literal["test", "val"]) -> Dict[str, float]:
        stability, rdkit_metrics, all_smiles, relaxed_valid_smiles,result_dict = compute_molecular_metrics(
            molecules, self.reference_smiles['train'], self.atom_list, labels, self.cfg,
        )

        # FCD (keep existing interface)
        if self.compute_fcd and not self.compute_moses and not self.compute_guacamol:
            result_dict["fcd"] = self._compute_fcd(all_smiles, split)

        if self.compute_moses or self.compute_guacamol:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"all_smiles_{split}_{timestamp}.pkl"
            if self.save_smiles_dir is not None:
                file_name = os.path.join(self.save_smiles_dir, file_name)
            dir_path = os.path.dirname(file_name)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            print("saveing smiles to: ", os.path.abspath(file_name))
            with open(file_name, "wb") as fp:
                if self.compute_guacamol:
                    smiles_to_save = relaxed_valid_smiles #From the paper of DeFoG: "Note that Guacamol includes molecules with charges; therefore, the generated graphs are converted to charged molecules based on the relaxed validity criterion"
                else:
                    smiles_to_save = all_smiles
                pickle.dump(smiles_to_save, fp)
            result_dict['file_name'] = file_name            
        # Other distribution/MAE metrics
        dist_dict, mae_dict = self.distribution_metrics(molecules)
        result_dict.update(dist_dict)
        result_dict.update(mae_dict) 
        return result_dict
    
    def _init_ref_metrics(self, datamodule):
        def save_pickle(array, path):
            with open(path, "wb") as f:
                pickle.dump(array, f)

        def load_pickle(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        def get_ref_metrics_path(datamodule):
            ref_metrics_path = os.path.join(
                datamodule.train_dataloader().dataset.root, "ref_metrics_new.pkl"
            )
            if hasattr(datamodule, "remove_h"):
                if datamodule.remove_h:
                    ref_metrics_path = ref_metrics_path.replace(".pkl", "_no_h.pkl")
                else:
                    ref_metrics_path = ref_metrics_path.replace(".pkl", "_h.pkl")
            return ref_metrics_path

        ref_metrics_path = get_ref_metrics_path(datamodule)
        if os.path.exists(ref_metrics_path):
            ref_metrics = load_pickle(ref_metrics_path)
            return ref_metrics

        ref_metrics = {}
        ref_metrics["train"] = self._compute_fcd(self.reference_smiles['train'],
                                                     "train")
        ref_metrics["val"] = self._compute_fcd(self.reference_smiles['val'],
                                                   "train")
        ref_metrics["test"] = self._compute_fcd(self.reference_smiles['test'],
                                                    "train")
        save_pickle(ref_metrics, ref_metrics_path)
        return ref_metrics
    # ---------- New: Filters / SNN independent functions ----------
    def _compute_filters(self, smiles) -> float:
        """Compute fraction of valid molecules that pass MOSES filters."""
        valid = mm.remove_invalid([s for s in (smiles or []) if s])
        return float(mm.fraction_passes_filters(valid)) if len(valid) > 0 else -1.0

    

    # ---------- Modified: Scaf uses MOSES official implementation (unique scaffold overlap percentage) ----------
    def _compute_scaf(self, generated_smiles) -> float:
        """
        MOSES-Scaf: percentage of unique scaffolds in generated set that appear in training set.
        
        Computes: 100 × |unique_scaffolds(gen) ∩ unique_scaffolds(train)| / |unique_scaffolds(gen)|
        
        This measures how many of the generated molecules have scaffolds that were 
        seen in the training data (i.e., scaffold memorization/coverage).
        
        Args:
            generated_smiles: List of generated SMILES strings
            
        Returns:
            Percentage (0-100) of generated scaffolds found in training set,
            or -1.0 if no valid scaffolds found
            
        Note:
            - Uses generic scaffolds (MakeScaffoldGeneric): heteroatoms → C, bonds → single
            - Reference set: training set (MOSES convention)
            - Higher values (→100) indicate generated scaffolds are mostly from training set
            - Lower values (→0) indicate novel scaffolds not in training set
            
        Example values (from MOSES paper):
            - Training set: ~99% (almost all scaffolds appear in training)
            - Good generative models: 14-23% (mostly novel scaffolds)
        """
        if not self._cache:
            raise ValueError("Scaf cache not initialized")
        gen = [s for s in (generated_smiles or []) if s]
        gen_valid = mm.remove_invalid(gen)
        if len(gen_valid) == 0:
            return -1.0
        
        # Get reference unique scaffolds (from training set, cached)
        ref_scaffolds = self._cache.ensure_ref_unique_scaffolds()
        # Extract unique scaffolds from generated set
        gen_scaffolds = _extract_unique_scaffolds(gen_valid, generic=True)
        
        if len(gen_scaffolds) == 0:
            return -1.0
        
        # Compute overlap percentage
        overlap = gen_scaffolds & ref_scaffolds
        percentage = 100.0 * len(overlap) / len(gen_scaffolds)
        
        return float(percentage)

    # ---------- FCD: use fcd_torch for acceleration ----------
    def _compute_fcd(self, generated_smiles, split: Literal["train", "test", "val"]) -> float:
        """Compute FCD using fcd_torch with cached precomputed reference embeddings."""
        if self._fcd_metric is None or not self._cache:
            # Fallback to old implementation if fcd_torch not initialized
            return compute_fcd(val_smiles=self.reference_smiles[split], generated_smiles=generated_smiles)
        
        # Filter out None values
        gen_smiles_clean = [s for s in (generated_smiles or []) if s is not None]
        
        if len(gen_smiles_clean) == 0:
            return -1.0
        
        try:
            return compute_fcd(val_smiles=self.reference_smiles[split], generated_smiles=generated_smiles)
            fcd_score = self._fcd_metric(gen=gen_smiles_clean, pref=pref)
            return float(fcd_score)
        except Exception as e:
            warnings.warn(f"FCD computation failed: {e}, set FCD to -1.0")
            return -1.0


    def _compute_guacamol_kl(self, generated_smiles):
        """Compute KL divergence per Guacamol via centralized wrapper."""
        ref_train = self.reference_smiles.get("train")
        return benchmarks_compute_guacamol_kl(generated_smiles, ref_train)
    
    def _compute_ratios(self, result_dict, split: Literal["test", "val"])->Dict[str, float]:
        ratios = compute_ratios(
            gen_metrics=result_dict,
            ref_metrics=self.ref_metrics[split],
            metrics_keys=["fcd"],
        )
        return ratios
    
    def reset(self):
        self.distribution_metrics.reset()

def compute_fcd(val_smiles, generated_smiles):
    """smiles have must be a list of str"""

    print("Starting FCD computation")
    start = time.time()

    # not using fcd.canonical_smiles because both smiles are already in canonical form (result from the Chem.MolToSmiles)
    # filter out None values (not sanitizable molecules)
    generated_smiles = [smile for smile in generated_smiles if smile is not None]

    # supress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            fcd_score = fcd.get_fcd(generated_smiles, val_smiles)
        except Exception as e:
            print(f"Error in FCD computation. Setting FCD to -1.")
            fcd_score = -1

    end = time.time()
    print("FCD computation time:", end - start, "FCD score is", fcd_score)

    return fcd_score


def smiles_to_generic_scaffold(smiles: str):
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return None
        scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def compute_scaffold_similarity(ref_smiles, generated_smiles):
    """Compute scaffold similarity between reference and generated sets.

    Returns dict with keys: precision, recall, avg_max_tanimoto.
    - precision: |Scaf(gen) ∩ Scaf(ref)| / |Scaf(gen)|
    - recall:    |Scaf(gen) ∩ Scaf(ref)| / |Scaf(ref)|
    - avg_max_tanimoto: For each generated scaffold, max Tanimoto to any ref scaffold (RDKFingerprint), averaged.
    """
    print("Computing scaffold similarity")
    start_time = time.time()
    # Build scaffold lists (do not deduplicate for similarity averaging)
    gen_scaff_smiles = [sm for sm in (smiles_to_generic_scaffold(s) for s in generated_smiles) if sm]
    ref_scaff_smiles = [sm for sm in (smiles_to_generic_scaffold(s) for s in ref_smiles) if sm]

    gen_set = set(gen_scaff_smiles)
    ref_set = set(ref_scaff_smiles)

    if len(gen_set) == 0 or len(ref_set) == 0:
        return {"precision": -1.0, "recall": -1.0, "avg_max_tanimoto": -1.0}

    inter = gen_set & ref_set
    precision = len(inter) / len(gen_set) if len(gen_set) > 0 else -1.0
    recall = len(inter) / len(ref_set) if len(ref_set) > 0 else -1.0

    # Fingerprints for Tanimoto (ECFP4 on Bemis–Murcko scaffolds)
    ref_mols_all = [Chem.MolFromSmiles(s) for s in ref_scaff_smiles]
    gen_mols_all = [Chem.MolFromSmiles(s) for s in gen_scaff_smiles]
    ref_mols_all = [m for m in ref_mols_all if m is not None]
    gen_mols_all = [m for m in gen_mols_all if m is not None]
    if len(ref_mols_all) == 0 or len(gen_mols_all) == 0:
        avg_max_tanimoto = -1.0
    else:
        ref_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in ref_mols_all]
        gen_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in gen_mols_all]
        max_sims = []
        for gfp in gen_fps:
            if len(ref_fps) == 0:
                max_sims.append(0.0)
                continue
            sims = DataStructs.BulkTanimotoSimilarity(gfp, ref_fps)
            max_sims.append(max(sims) if len(sims) > 0 else 0.0)
        avg_max_tanimoto = float(np.mean(max_sims)) if len(max_sims) > 0 else -1.0
    end_time = time.time()
    print("Scaffold similarity computation time:", end_time - start_time)
    return {
        "precision": precision,
        "recall": recall,
        "avg_max_tanimoto": avg_max_tanimoto,
    }


class NodeNumsProportionMetric(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class NodeTypsProportionMetric(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                    int(atom_type) != -1
                ), "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class EdgesTypesProportionMetric(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class MeanNumberEdge(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total_edge", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples


class ValenciesProportionMetric(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            edge_types[edge_types == 5] = 0.0  # zero out virtual states
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)


class MSEPerClass(MeanSquaredError):
    full_state_update = False

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds = preds[..., self.class_id]
        target = target[..., self.class_id]
        super().update(preds, target)


class HydroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


# Bonds MSE


class NoBondMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetrics(MetricCollection):
    def __init__(self, atom_list):
        self.atom_list = atom_list

        types = {
            "H": 0,
            "C": 1,
            "N": 2,
            "O": 3,
            "F": 4,
            "B": 5,
            "Br": 6,
            "Cl": 7,
            "I": 8,
            "P": 9,
            "S": 10,
            "Se": 11,
            "Si": 12,
        }

        class_dict = {
            "H": HydroMSE,
            "C": CarbonMSE,
            "N": NitroMSE,
            "O": OxyMSE,
            "F": FluorMSE,
            "B": BoronMSE,
            "Br": BrMSE,
            "Cl": ClMSE,
            "I": IodineMSE,
            "P": PhosphorusMSE,
            "S": SulfurMSE,
            "Se": SeMSE,
            "Si": SiMSE,
        }

        metrics_list = []
        for i, atom_type in enumerate(self.atom_list):
            metrics_list.append(class_dict[atom_type](i))

        super().__init__(metrics_list)


class BondMetrics(MetricCollection):
    def __init__(self):
        mse_no_bond = NoBondMSE(0)
        mse_SI = SingleMSE(1)
        mse_DO = DoubleMSE(2)
        mse_TR = TripleMSE(3)
        mse_AR = AromaticMSE(4)
        super().__init__([mse_no_bond, mse_SI, mse_DO, mse_TR, mse_AR])


if __name__ == "__main__":
    pass