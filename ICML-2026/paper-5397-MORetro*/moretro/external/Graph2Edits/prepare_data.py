import argparse
import copy
import os
import sys
from typing import Any, Tuple

import joblib
import torch
from rdkit import Chem

from Graph2Edits.utils.collate_fn import get_batch_graphs, prepare_edit_labels
from Graph2Edits.utils.reaction_actions import (AddGroupAction, AtomEditAction,
                                                BondEditAction, Termination)
from Graph2Edits.utils.rxn_graphs import MolGraph, RxnGraph, Vocab


def apply_edit_to_mol(mol: Chem.Mol, edit: Tuple, edit_atom: Any) -> Chem.Mol:
    """ Apply edits to molecular graph """

    if edit[0] == 'Change Atom':
        edit_exe = AtomEditAction(
            edit_atom, *edit[1], action_vocab='Change Atom')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Delete Bond':
        edit_exe = BondEditAction(
            *edit_atom, *edit[1], action_vocab='Delete Bond')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Change Bond':
        edit_exe = BondEditAction(
            *edit_atom, *edit[1], action_vocab='Change Bond')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Add Bond':
        edit_exe = BondEditAction(
            *edit_atom, *edit[1], action_vocab='Add Bond')
        new_mol = edit_exe.apply(mol)

    if edit[0] == 'Attaching LG':
        edit_exe = AddGroupAction(
            edit_atom, edit[1], action_vocab='Attaching LG')
        new_mol = edit_exe.apply(mol)

    return new_mol


def process_batch(batch_graphs, args):
    lengths = torch.tensor([len(graph_seq)
                           for graph_seq in batch_graphs], dtype=torch.long)
    max_length = max([len(graph_seq) for graph_seq in batch_graphs])

    bond_vocab_file = f'data/{args.dataset}/{args.mode}/bond_vocab.txt'
    atom_vocab_file = f'data/{args.dataset}/{args.mode}/atom_lg_vocab.txt'
    bond_vocab = Vocab(joblib.load(bond_vocab_file))
    atom_vocab = Vocab(joblib.load(atom_vocab_file))

    graph_seq_tensors = []
    edit_seq_labels = []
    seq_mask = []

    for idx in range(max_length):
        graphs_idx = [copy.deepcopy(batch_graphs[i][min(idx, length-1)]).get_components(attrs=['prod_graph', 'edit_to_apply', 'edit_atom'])
                      for i, length in enumerate(lengths)]
        mask = (idx < lengths).long()
        prod_graphs, edits, edit_atoms = list(zip(*graphs_idx))
        assert all([isinstance(graph, MolGraph) for graph in prod_graphs])

        edit_labels = prepare_edit_labels(
            prod_graphs, edits, edit_atoms, bond_vocab, atom_vocab)
        current_graph_tensors = get_batch_graphs(
            prod_graphs, use_rxn_class=args.use_rxn_class)

        graph_seq_tensors.append(current_graph_tensors)
        edit_seq_labels.append(edit_labels)
        seq_mask.append(mask)

    seq_mask = torch.stack(seq_mask).long()
    assert seq_mask.shape[0] == max_length
    assert seq_mask.shape[1] == len(batch_graphs)

    return graph_seq_tensors, edit_seq_labels, seq_mask


def prepare_data(args: Any) -> None:
    """ 
    prepare data batches for edits prediction
    """
    datafile = f'data/{args.dataset}/{args.mode}/{args.mode}.file.kekulized'
    rxns_data = joblib.load(datafile)

    batch_graphs = []
    batch_num = 0

    if args.use_rxn_class:
        savedir = f'data/{args.dataset}/{args.mode}/with_rxn_class/'
    else:
        savedir = f'data/{args.dataset}/{args.mode}/without_rxn_class/'
    os.makedirs(savedir, exist_ok=True)

    for idx, rxn_data in enumerate(rxns_data):
        graph_seq = []
        rxn_smi = rxn_data.rxn_smi
        r, p = rxn_smi.split('>>')
        r_mol = Chem.MolFromSmiles(r)
        p_mol = Chem.MolFromSmiles(p)
        Chem.Kekulize(p_mol)

        if len(rxn_data.edits) > args.max_steps:
            print(f'Edits step exceed max_steps. Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        int_mol = p_mol
        for i, edit in enumerate(rxn_data.edits):
            if int_mol is None:
                print("Interim mol is None")
                break
            if edit == 'Terminate':
                graph = RxnGraph(prod_mol=Chem.Mol(
                    int_mol), edit_to_apply=edit, reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=args.use_rxn_class)
                graph_seq.append(graph)
                edit_exe = Termination(action_vocab='Terminate')
                try:
                    pred_mol = edit_exe.apply(Chem.Mol(int_mol))
                    final_smi = Chem.MolToSmiles(pred_mol)
                except Exception as e:
                    final_smi = None
            else:
                graph = RxnGraph(prod_mol=Chem.Mol(int_mol), edit_to_apply=edit,
                                 edit_atom=rxn_data.edits_atom[i], reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=args.use_rxn_class)
                graph_seq.append(graph)
                int_mol = apply_edit_to_mol(
                    Chem.Mol(int_mol), edit, rxn_data.edits_atom[i])

        if len(graph_seq) == 0 or final_smi is None:
            print(f"No valid states found. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        batch_graphs.append(graph_seq)
        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(rxns_data)} {args.mode} reactions processed.")
            sys.stdout.flush()

        if (len(batch_graphs) % args.batch_size == 0) and len(batch_graphs):
            batch_tensors = process_batch(batch_graphs, args)
            torch.save(batch_tensors, os.path.join(
                savedir, f'batch-{batch_num}.pt'))

            batch_num += 1
            batch_graphs = []

    print(f"All {args.mode} reactions complete.")
    sys.stdout.flush()

    batch_tensors = process_batch(batch_graphs, args)
    print("Saving..")
    torch.save(batch_tensors, os.path.join(savedir, f'batch-{batch_num}.pt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_50k',
                        help='dataset: USPTO_50k or uspto_full')
    parser.add_argument('--mode', type=str, default='train',
                        help='Type of dataset being prepared: train or valid or test')
    parser.add_argument("--use_rxn_class", default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument("--batch_size", default=32,
                        type=int, help="Number of shards")
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')
    parser.add_argument('--print_every', type=int,
                        default=1000, help='Print during preprocessing')
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    prepare_data(args=args)


if __name__ == "__main__":
    main()
