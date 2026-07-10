import argparse
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from itertools import combinations
from os.path import join

import numpy as np
import torch

# RDKit import first, like the repo.
from rdkit import Chem
from rdkit.Chem import AllChem

import utils
from configs.datasets_config import get_dataset_info
from qm9 import dataset as qm9_dataset
from qm9.models import get_model, get_autoencoder, get_latent_diffusion
from qm9.sampling import sample as sample_qm9
from qm9.utils import compute_mean_mad
from tqdm import tqdm


BOND_TYPE_TO_STR = {
    Chem.rdchem.BondType.SINGLE: "single",
    Chem.rdchem.BondType.DOUBLE: "double",
    Chem.rdchem.BondType.TRIPLE: "triple",
    Chem.rdchem.BondType.AROMATIC: "aromatic",
}


def masked_tensor_to_mols(one_hot, x, node_mask):
    mols = []
    node_mask = node_mask.to(x.device)
    one_hot = one_hot.to(x.device)
    # QM9 loader can return bool one-hot tensors.
    if one_hot.dtype == torch.bool:
        atom_types = one_hot.to(torch.long).argmax(dim=-1)
    else:
        atom_types = one_hot.argmax(dim=-1)

    for i in range(one_hot.size(0)):
        mask = node_mask[i].squeeze(-1) > 0.5
        pos_i = x[i][mask].detach().cpu().to(torch.float64)
        atom_i = atom_types[i][mask].detach().cpu().to(torch.long)
        mols.append((pos_i, atom_i))
    return mols


def canonical_bond_label(sym1, sym2, bond_type=None):
    a, b = sorted([sym1, sym2])
    if bond_type is None:
        return f"{a}-{b}"
    return f"{a}-{b}:{bond_type}"


def canonical_angle_label(sym1, sym2, sym3):
    a, c = sorted([sym1, sym3])
    return f"{a}-{sym2}-{c}"


def js_divergence_from_values(x, y, bins):
    if len(x) == 0 or len(y) == 0:
        return None
    hx, _ = np.histogram(x, bins=bins, density=False)
    hy, _ = np.histogram(y, bins=bins, density=False)
    px = hx.astype(np.float64)
    py = hy.astype(np.float64)
    px = px / max(px.sum(), 1.0)
    py = py / max(py.sum(), 1.0)
    m = 0.5 * (px + py)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    return 0.5 * kl(px, m) + 0.5 * kl(py, m)


class GeometryEvaluator:
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info
        self.atom_decoder = dataset_info["atom_decoder"]
        # Reuse the repo's deterministic bond inference to stay consistent
        # with the existing validity/stability pipeline.
        from qm9.rdkit_functions import build_xae_molecule

        self.build_xae_molecule = build_xae_molecule

    def build_mol_with_coords(self, positions, atom_types):
        X, A, E = self.build_xae_molecule(positions, atom_types, self.dataset_info)
        rw = Chem.RWMol()
        for atom_idx in X.tolist():
            rw.AddAtom(Chem.Atom(self.atom_decoder[int(atom_idx)]))

        added = set()
        for i, j in torch.nonzero(A, as_tuple=False).tolist():
            key = tuple(sorted((i, j)))
            if key in added:
                continue
            bond_order = int(E[i, j].item())
            if bond_order <= 0:
                continue
            bond_type = {
                1: Chem.rdchem.BondType.SINGLE,
                2: Chem.rdchem.BondType.DOUBLE,
                3: Chem.rdchem.BondType.TRIPLE,
                4: Chem.rdchem.BondType.AROMATIC,
            }[bond_order]
            rw.AddBond(int(i), int(j), bond_type)
            added.add(key)

        mol = rw.GetMol()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            p = positions[i].tolist()
            conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
        mol.AddConformer(conf, assignId=True)
        return mol

    def sanitize_mol(self, mol):
        mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def extract_geometry(self, positions, atom_types):
        mol = self.build_mol_with_coords(positions, atom_types)
        mol = self.sanitize_mol(mol)
        if mol is None:
            return None

        conf = mol.GetConformer()
        bond_lengths = []
        bond_lengths_by_type = defaultdict(list)
        bond_lengths_by_pair = defaultdict(list)
        adjacency = defaultdict(list)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pi = conf.GetAtomPosition(i)
            pj = conf.GetAtomPosition(j)
            d = math.dist((pi.x, pi.y, pi.z), (pj.x, pj.y, pj.z))
            s1 = mol.GetAtomWithIdx(i).GetSymbol()
            s2 = mol.GetAtomWithIdx(j).GetSymbol()
            btype = BOND_TYPE_TO_STR.get(bond.GetBondType(), str(bond.GetBondType()))
            bond_lengths.append(d)
            bond_lengths_by_pair[canonical_bond_label(s1, s2)].append(d)
            bond_lengths_by_type[canonical_bond_label(s1, s2, btype)].append(d)
            adjacency[i].append(j)
            adjacency[j].append(i)

        angles = []
        angles_by_type = defaultdict(list)
        for center in range(mol.GetNumAtoms()):
            nbrs = adjacency.get(center, [])
            if len(nbrs) < 2:
                continue
            pc = conf.GetAtomPosition(center)
            vc = np.array([pc.x, pc.y, pc.z], dtype=np.float64)
            csym = mol.GetAtomWithIdx(center).GetSymbol()
            for i, k in combinations(nbrs, 2):
                pi = conf.GetAtomPosition(i)
                pk = conf.GetAtomPosition(k)
                vi = np.array([pi.x, pi.y, pi.z], dtype=np.float64) - vc
                vk = np.array([pk.x, pk.y, pk.z], dtype=np.float64) - vc
                denom = np.linalg.norm(vi) * np.linalg.norm(vk)
                if denom < 1e-12:
                    continue
                cosang = np.clip(np.dot(vi, vk) / denom, -1.0, 1.0)
                ang = math.degrees(math.acos(cosang))
                isym = mol.GetAtomWithIdx(i).GetSymbol()
                ksym = mol.GetAtomWithIdx(k).GetSymbol()
                angles.append(ang)
                angles_by_type[canonical_angle_label(isym, csym, ksym)].append(ang)

        ff_stats = self.compute_forcefield_stats(mol)
        return {
            "mol": mol,
            "n_atoms": mol.GetNumAtoms(),
            "bond_lengths": bond_lengths,
            "bond_lengths_by_pair": dict(bond_lengths_by_pair),
            "bond_lengths_by_type": dict(bond_lengths_by_type),
            "angles": angles,
            "angles_by_type": dict(angles_by_type),
            "ff": ff_stats,
        }

    def compute_forcefield_stats(self, mol):
        n_atoms = max(mol.GetNumAtoms(), 1)

        # Prefer MMFF94, fall back to UFF.
        ff_kind = None
        raw_energy = None
        min_energy = None

        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                ff_kind = "MMFF94"
                props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                raw_energy = float(ff.CalcEnergy())

                mol_opt = Chem.Mol(mol)
                props_opt = AllChem.MMFFGetMoleculeProperties(mol_opt, mmffVariant="MMFF94")
                ff_opt = AllChem.MMFFGetMoleculeForceField(mol_opt, props_opt)
                ff_opt.Minimize(maxIts=200)
                min_energy = float(ff_opt.CalcEnergy())
            else:
                ff_kind = "UFF"
                ff = AllChem.UFFGetMoleculeForceField(mol)
                raw_energy = float(ff.CalcEnergy())

                mol_opt = Chem.Mol(mol)
                ff_opt = AllChem.UFFGetMoleculeForceField(mol_opt)
                ff_opt.Minimize(maxIts=200)
                min_energy = float(ff_opt.CalcEnergy())
        except Exception:
            ff_kind = None

        if ff_kind is None or raw_energy is None or min_energy is None:
            return {
                "ff_type": None,
                "raw_energy": None,
                "raw_energy_per_atom": None,
                "min_energy": None,
                "min_energy_per_atom": None,
                "delta_energy": None,
                "delta_energy_per_atom": None,
            }

        delta = raw_energy - min_energy
        return {
            "ff_type": ff_kind,
            "raw_energy": raw_energy,
            "raw_energy_per_atom": raw_energy / n_atoms,
            "min_energy": min_energy,
            "min_energy_per_atom": min_energy / n_atoms,
            "delta_energy": delta,
            "delta_energy_per_atom": delta / n_atoms,
        }


def aggregate_geometry(geom_list):
    out = {
        "n_total": len(geom_list),
        "n_valid": 0,
        "n_invalid": 0,
        "bond_lengths": [],
        "bond_lengths_by_pair": defaultdict(list),
        "bond_lengths_by_type": defaultdict(list),
        "angles": [],
        "angles_by_type": defaultdict(list),
        "sizes": [],
        "ff_type_counter": Counter(),
        "raw_energy_per_atom": [],
        "min_energy_per_atom": [],
        "delta_energy_per_atom": [],
    }

    for geom in geom_list:
        if geom is None:
            out["n_invalid"] += 1
            continue
        out["n_valid"] += 1
        out["sizes"].append(geom["n_atoms"])
        out["bond_lengths"].extend(geom["bond_lengths"])
        out["angles"].extend(geom["angles"])
        for k, v in geom["bond_lengths_by_pair"].items():
            out["bond_lengths_by_pair"][k].extend(v)
        for k, v in geom["bond_lengths_by_type"].items():
            out["bond_lengths_by_type"][k].extend(v)
        for k, v in geom["angles_by_type"].items():
            out["angles_by_type"][k].extend(v)

        ff = geom["ff"]
        if ff["ff_type"] is not None:
            out["ff_type_counter"][ff["ff_type"]] += 1
            out["raw_energy_per_atom"].append(ff["raw_energy_per_atom"])
            out["min_energy_per_atom"].append(ff["min_energy_per_atom"])
            out["delta_energy_per_atom"].append(ff["delta_energy_per_atom"])

    out["bond_lengths_by_pair"] = dict(out["bond_lengths_by_pair"])
    out["bond_lengths_by_type"] = dict(out["bond_lengths_by_type"])
    out["angles_by_type"] = dict(out["angles_by_type"])
    out["ff_type_counter"] = dict(out["ff_type_counter"])
    return out


def summarize_distribution_shift(ref_vals, gen_vals, bins, min_count=100):
    if len(ref_vals) < min_count or len(gen_vals) < min_count:
        return None
    return {
        "count_ref": len(ref_vals),
        "count_generated": len(gen_vals),
        "mean_ref": float(np.mean(ref_vals)),
        "std_ref": float(np.std(ref_vals)),
        "mean_generated": float(np.mean(gen_vals)),
        "std_generated": float(np.std(gen_vals)),
        "jsd": float(js_divergence_from_values(ref_vals, gen_vals, bins)),
    }


def compare_aggregates(ref_agg, gen_agg, top_k=12):
    bond_bins = np.linspace(0.6, 2.4, 181)
    angle_bins = np.linspace(0.0, 180.0, 181)

    summary = {
        "reference": basic_stats(ref_agg),
        "generated": basic_stats(gen_agg),
        "bond_length_all": summarize_distribution_shift(
            ref_agg["bond_lengths"], gen_agg["bond_lengths"], bond_bins, min_count=100
        ),
        "angle_all": summarize_distribution_shift(
            ref_agg["angles"], gen_agg["angles"], angle_bins, min_count=100
        ),
        "bond_length_by_pair": {},
        "bond_length_by_type": {},
        "angle_by_type": {},
    }

    pair_keys = sorted(
        set(ref_agg["bond_lengths_by_pair"].keys()) & set(gen_agg["bond_lengths_by_pair"].keys()),
        key=lambda k: len(ref_agg["bond_lengths_by_pair"][k]) + len(gen_agg["bond_lengths_by_pair"][k]),
        reverse=True,
    )
    for k in pair_keys[:top_k]:
        s = summarize_distribution_shift(
            ref_agg["bond_lengths_by_pair"][k],
            gen_agg["bond_lengths_by_pair"][k],
            bond_bins,
            min_count=50,
        )
        if s is not None:
            summary["bond_length_by_pair"][k] = s

    type_keys = sorted(
        set(ref_agg["bond_lengths_by_type"].keys()) & set(gen_agg["bond_lengths_by_type"].keys()),
        key=lambda k: len(ref_agg["bond_lengths_by_type"][k]) + len(gen_agg["bond_lengths_by_type"][k]),
        reverse=True,
    )
    for k in type_keys[:top_k]:
        s = summarize_distribution_shift(
            ref_agg["bond_lengths_by_type"][k],
            gen_agg["bond_lengths_by_type"][k],
            bond_bins,
            min_count=50,
        )
        if s is not None:
            summary["bond_length_by_type"][k] = s

    angle_keys = sorted(
        set(ref_agg["angles_by_type"].keys()) & set(gen_agg["angles_by_type"].keys()),
        key=lambda k: len(ref_agg["angles_by_type"][k]) + len(gen_agg["angles_by_type"][k]),
        reverse=True,
    )
    for k in angle_keys[:top_k]:
        s = summarize_distribution_shift(
            ref_agg["angles_by_type"][k],
            gen_agg["angles_by_type"][k],
            angle_bins,
            min_count=50,
        )
        if s is not None:
            summary["angle_by_type"][k] = s

    return summary


def basic_stats(agg):
    def maybe_stats(vals):
        if len(vals) == 0:
            return None
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        }

    return {
        "n_total": agg["n_total"],
        "n_valid": agg["n_valid"],
        "n_invalid": agg["n_invalid"],
        "valid_fraction": agg["n_valid"] / max(agg["n_total"], 1),
        "n_bonds": len(agg["bond_lengths"]),
        "n_angles": len(agg["angles"]),
        "size": maybe_stats(agg["sizes"]),
        "raw_energy_per_atom": maybe_stats(agg["raw_energy_per_atom"]),
        "min_energy_per_atom": maybe_stats(agg["min_energy_per_atom"]),
        "delta_energy_per_atom": maybe_stats(agg["delta_energy_per_atom"]),
        "ff_type_counter": agg["ff_type_counter"],
    }


def collect_training_molecules(args, eval_args):
    dataloaders, _ = qm9_dataset.retrieve_dataloaders(args)
    collected = []
    for batch in tqdm(dataloaders["train"], desc="Training molecules"):
        mols = masked_tensor_to_mols(
            batch["one_hot"],
            batch["positions"],
            batch["atom_mask"].unsqueeze(-1),
        )
        collected.extend(mols)
        if len(collected) >= eval_args.n_reference:
            return collected[: eval_args.n_reference]
    return collected


def load_model_for_sampling(model_path):        
    with open(join(model_path, "args.pickle"), "rb") as f:
        args = pickle.load(f)
    if not hasattr(args, "normalization_factor"):
        args.normalization_factor = 1
    if not hasattr(args, "aggregation_method"):
        args.aggregation_method = "sum"
    args.StP = getattr(args, "StP", False)
    args.data_norm = getattr(args, "data_norm", False)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    args.datadir = "../processed_dataset/qm9"
    utils.create_folders(args)
    dataloaders, _ = qm9_dataset.retrieve_dataloaders(args)
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)
    fn = "generative_model_ema_40.npy" if args.ema_decay > 0 else "generative_model.npy"
    state = torch.load(join(model_path, fn), map_location=device)
    generative_model.load_state_dict(state)
    generative_model.eval()
    return args, device, dataset_info, generative_model, nodes_dist, prop_dist


def collect_generated_molecules(args):
    model_args, device, dataset_info, generative_model, nodes_dist, prop_dist = load_model_for_sampling(
        args.model_path
    )
    collected = []
    batch_size = min(args.batch_size_gen, args.n_generated)
    assert args.n_generated % batch_size == 0, "n_generated must be divisible by batch_size_gen"

    with torch.no_grad():
        for _ in tqdm(range(args.n_generated // batch_size), desc="Generating molecules"):
            nodesxsample = nodes_dist.sample(batch_size)
            one_hot, charges, x, node_mask = sample_qm9(
                model_args,
                device,
                generative_model,
                dataset_info,
                prop_dist=prop_dist,
                nodesxsample=nodesxsample,
            )
            mols = masked_tensor_to_mols(one_hot, x, node_mask)
            collected.extend(mols)
    return collected[: args.n_generated], dataset_info


def maybe_cache(path, obj=None, load_only=False):
    if path is None or path == "":
        return None if load_only else obj

    # Load if requested and file exists
    if load_only:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    # Save
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path like outputs/edm_qm9")
    parser.add_argument("--n_reference", type=int, default=10000)
    parser.add_argument("--n_generated", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_gen", type=int, default=100)
    parser.add_argument("--reference_cache", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--top_k", type=int, default=12)
    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
        
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'
    args.StP = getattr(args, "StP", False)
    args.data_norm = getattr(args, "data_norm", False)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    utils.create_folders(args)
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    evaluator = GeometryEvaluator(dataset_info)

    ref_agg = maybe_cache(eval_args.reference_cache, load_only=True)

    if ref_agg is not None:
        print(f"Loaded cached reference stats from {eval_args.reference_cache}")
    else:
        print("Collecting training molecules...")
        ref_mols = collect_training_molecules(args, eval_args)
        print("Extracting reference geometry...")
        ref_geom = [
            evaluator.extract_geometry(pos, atom_types)
            for pos, atom_types in tqdm(ref_mols, desc="Reference geometry")
        ]
        ref_agg = aggregate_geometry(ref_geom)
        maybe_cache(eval_args.reference_cache, ref_agg)
        if eval_args.reference_cache:
            print(f"Saved reference stats to {eval_args.reference_cache}")

    print("Sampling generated molecules...")
    gen_mols, dataset_info_from_model = collect_generated_molecules(eval_args)
    if dataset_info_from_model["atom_decoder"] != dataset_info["atom_decoder"]:
        raise ValueError("Model dataset_info does not match requested QM9 config.")

    print("Extracting generated geometry...")
    gen_geom = [
        evaluator.extract_geometry(pos, atom_types)
        for pos, atom_types in tqdm(gen_mols, desc="Generated geometry")
    ]
    gen_agg = aggregate_geometry(gen_geom)

    summary = compare_aggregates(ref_agg, gen_agg, top_k=eval_args.top_k)
    summary["config"] = {
        "model_path": eval_args.model_path,
        "n_reference": eval_args.n_reference,
        "n_generated": eval_args.n_generated,
        "remove_h": args.remove_h,
    }

    print(json.dumps(summary, indent=2))

    out_json = eval_args.output_json or join(eval_args.model_path, "geometry_eval_qm9.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {out_json}")


if __name__ == "__main__":
    main()