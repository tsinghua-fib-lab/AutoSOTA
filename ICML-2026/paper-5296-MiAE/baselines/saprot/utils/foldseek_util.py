import os
import time
import json
import numpy as np
import re
import sys

from Bio.PDB import PDBParser, MMCIFParser


sys.path.append(".")


# Get structural seqs from pdb file
def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = "auto",
                  plddt_threshold: float = 70.,
                  foldseek_verbose: bool = False) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.

        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 \'{path}\' \'{tmp_save_path}\'"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 \'{path}\' \'{tmp_save_path}\'"
    os.system(cmd)
    
    # Check whether the structure is predicted by AlphaFold2
    if plddt_mask == "auto":
        with open(path, "r") as r:
            plddt_mask = True if "alphafold" in r.read().lower() else False
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            # Mask low plddt
            if plddt_mask:
                try:
                    plddts = extract_plddt(path, chain)
                    assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                    
                    # Mask regions with plddt < threshold
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)
                
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Failed to mask plddt for {name}")
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    #os.remove(tmp_save_path)
    #os.remove(tmp_save_path + ".dbtype")
    try:
        os.remove(tmp_save_path)
    except FileNotFoundError:
        pass
    dbtype_file = tmp_save_path + ".dbtype"
    if os.path.exists(dbtype_file):
        try:
            os.remove(dbtype_file)
        except FileNotFoundError:
            pass
    index_file = tmp_save_path + ".index.0"
    try:
        os.remove(index_file)
    except FileNotFoundError:
        pass
    return seq_dict


def extract_plddt(pdb_path: str, chain: str) -> np.ndarray:
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """
    if pdb_path.endswith(".cif"):
        parser = MMCIFParser()
    elif pdb_path.endswith(".pdb"):
        parser = PDBParser()
    else:
        raise ValueError("Invalid file format for plddt extraction. Must be '.cif' or '.pdb'.")
    
    structure = parser.get_structure('protein', pdb_path)
    if structure is None or len(structure) == 0:
        raise ValueError(f"Failed to parse structure from {pdb_path}")
    model = structure[0]
    chain = model[chain]

    # Extract plddt scores
    plddts = []
    for residue in chain:
        residue_plddts = []
        for atom in residue:
            plddt = atom.get_bfactor()
            residue_plddts.append(plddt)
        
        plddts.append(np.mean(residue_plddts))

    plddts = np.array(plddts)
    return plddts


def transform_pdb_dir(foldseek: str, pdb_dir: str, seq_type: str, save_path: str):
    """
    Transform a directory of pdb files into a fasta file.
    Args:
        foldseek: Binary executable file of foldseek.
        
        pdb_dir: Directory of pdb files.
        
        seq_type: Type of sequence to be extracted. Must be "aa" or "foldseek"
        
        save_path: Path to save the fasta file.
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert seq_type in ["aa", "foldseek"], f"seq_type must be 'aa' or 'foldseek'!"
    
    tmp_save_path = f"get_struc_seq_{time.time()}.tsv"
    cmd = f"{foldseek} structureto3didescriptor --chain-name-mode 1 {pdb_dir} {tmp_save_path}"
    os.system(cmd)
    
    with open(tmp_save_path, "r") as r, open(save_path, "w") as w:
        for line in r:
            protein_id, aa_seq, foldseek_seq = line.strip().split("\t")[:3]
            
            if seq_type == "aa":
                w.write(f">{protein_id}\n{aa_seq}\n")
            else:
                w.write(f">{protein_id}\n{foldseek_seq.lower()}\n")
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    

if __name__ == '__main__':
    foldseek = "/sujin/bin/foldseek"
    # test_path = "/sujin/Datasets/PDB/all/6xtd.cif"
    test_path = "/sujin/Datasets/FLIP/meltome/af2_structures/A0A061ACX4.pdb"
    plddt_path = "/sujin/Datasets/FLIP/meltome/af2_plddts/A0A061ACX4.json"
    res = get_struc_seq(foldseek, test_path, plddt_path=plddt_path, plddt_threshold=70.)
    print(res["A"][1].lower())
