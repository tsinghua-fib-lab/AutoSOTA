from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from Graph2Edits.prepare_data import apply_edit_to_mol
from rdkit import Chem
from Graph2Edits.utils.collate_fn import get_batch_graphs
from Graph2Edits.utils.rxn_graphs import MolGraph, Vocab

from Graph2Edits.models.encoder import Global_Attention, MPNEncoder
from Graph2Edits.models.model_utils import (creat_edits_feats, index_select_ND,
                                            unbatch_feats)


class Graph2Edits(nn.Module):
    def __init__(self,
                 config: Dict,
                 atom_vocab: Vocab,
                 bond_vocab: Vocab,
                 device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        config: Dict, Model arguments
        atom_vocab: atom and LG edit labels
        bond_vocab: bond edit labels
        device: str, Device to run the model on.
        """
        super(Graph2Edits, self).__init__()

        self.config = config
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.atom_outdim = len(atom_vocab)
        self.bond_outdim = len(bond_vocab)
        self.device = device

        self._build_layers()

    def _build_layers(self) -> None:
        """Builds the different layers associated with the model."""
        config = self.config
        self.encoder = MPNEncoder(atom_fdim=config['n_atom_feat'],
                                  bond_fdim=config['n_bond_feat'],
                                  hidden_size=config['mpn_size'],
                                  depth=config['depth'],
                                  dropout=config['dropout_mpn'],
                                  atom_message=config['atom_message'])

        self.W_vv = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)
        nn.init.eye_(self.W_vv.weight)
        self.W_vc = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)

        if config['use_attn']:
            self.attn = Global_Attention(
                d_model=config['mpn_size'], heads=config['n_heads'])

        self.atom_linear = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.atom_outdim))
        self.bond_linear = nn.Sequential(
            nn.Linear(config['mpn_size'] * 2, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.bond_outdim))

        self.graph_linear = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], 1))

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            tensors = [tensor.to(self.device, non_blocking=True)
                       for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")

    def compute_edit_scores(self, prod_tensors: Tuple[torch.Tensor],
                            prod_scopes: Tuple[List], prev_atom_hiddens: torch.Tensor = None,
                            prev_atom_scope: Tuple[List] = None) -> Tuple[torch.Tensor]:
        """Computes the edit scores given product tensors and scopes.

        Parameters
        ----------
        prod_tensors: Tuple[torch.Tensor]:
            Product tensors
        prod_scopes: Tuple[List]
            Product scopes. Scopes is composed of atom and bond scopes, which
            keep track of atom and bond indices for each molecule in the 2D
            feature list
        prev_atom_hiddens: torch.Tensor, default None,
            Previous hidden state of atoms.
        """
        prod_tensors = self.to_device(prod_tensors)
        atom_scope, bond_scope = prod_scopes
        if prev_atom_hiddens is None:
            n_atoms = prod_tensors[0].size(0)
            prev_atom_hiddens = torch.zeros(
                n_atoms, self.config['mpn_size'], device=self.device)

        a_feats = self.encoder(prod_tensors, mask=None)
        if self.config['use_attn']:
            feats, mask = creat_edits_feats(a_feats, atom_scope)
            attention_score, feats = self.attn(feats, mask)
            a_feats = unbatch_feats(feats, atom_scope)

        if a_feats.shape[0] != prev_atom_hiddens.shape[0]:
            n_atoms = a_feats.shape[0]
            new_ha = torch.zeros(
                n_atoms, self.config['mpn_size'], device=self.device)
            for idx, ((st_n, le_n), (st_p, le_p)) in enumerate(zip(*(atom_scope, prev_atom_scope))):
                new_ha[st_n: st_n + le_p] = prev_atom_hiddens[st_p: st_p + le_p]
            prev_atom_hiddens = new_ha

        assert a_feats.shape == prev_atom_hiddens.shape
        atom_feats = F.relu(self.W_vv(prev_atom_hiddens) + self.W_vc(a_feats))
        prev_atom_hiddens = atom_feats.clone()
        prev_atom_scope = atom_scope

        node_feats = atom_feats.clone()
        
        # The branch is modified by Mianchu
        if prod_tensors[-1].size(1) == 0:
            bond_feats = torch.zeros(atom_feats.shape[0], atom_feats.shape[1] * 2).to(device=self.device)
        else:
            bond_starts = index_select_ND(atom_feats, index=prod_tensors[-1][:, 0])
            bond_ends = index_select_ND(atom_feats, index=prod_tensors[-1][:, 1])
            bond_feats = torch.cat([bond_starts, bond_ends], dim=1)

        graph_vecs = torch.stack(
            [atom_feats[st: st + le].sum(dim=0) for st, le in atom_scope])

        atom_outs = self.atom_linear(node_feats)
        bond_outs = self.bond_linear(bond_feats)
        graph_outs = self.graph_linear(graph_vecs)

        edit_scores = [torch.cat([bond_outs[st_b: st_b + le_b].flatten(),
                                  atom_outs[st_a: st_a + le_a].flatten(), graph_outs[idx]], dim=-1)
                       for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope)))]

        return edit_scores, prev_atom_hiddens, prev_atom_scope

    def forward(self, prod_seq_inputs: List[Tuple[torch.Tensor, List]]) -> Tuple[torch.Tensor]:
        """
        Forward propagation step.

        Parameters
        ----------
        prod_seq_inputs: List[Tuple[torch.Tensor, List]]
            List of prod_tensors for edit sequence
        """
        max_seq_len = len(prod_seq_inputs)
        assert len(prod_seq_inputs[0]) == 2

        prev_atom_hiddens = None
        prev_atom_scope = None
        seq_edit_scores = []

        for idx in range(max_seq_len):
            prod_tensors, prod_scopes = prod_seq_inputs[idx]
            edit_scores, prev_atom_hiddens, prev_atom_scope = self.compute_edit_scores(
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope)
            seq_edit_scores.append(edit_scores)

        return seq_edit_scores

    def predict(self, prod_smi: str, rxn_class: int = None, max_steps: int = 9):
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        rxn_class: int, default None
            Associated reaction class for the product
        max_steps: int, default 8
            Max number of edit steps allowed
        """
        use_rxn_class = False
        if rxn_class is not None:
            use_rxn_class = True

        done = False
        steps = 0
        edits = []
        edits_atom = []
        prev_atom_hiddens = None
        prev_atom_scope = None

        products = Chem.MolFromSmiles(prod_smi)
        Chem.Kekulize(products)
        prod_graph = MolGraph(mol=Chem.Mol(products),
                              rxn_class=rxn_class, use_rxn_class=use_rxn_class)
        prod_tensors, prod_scopes = get_batch_graphs(
            [prod_graph], use_rxn_class=use_rxn_class)

        while not done and steps <= max_steps:
            if prod_tensors[-1].size() == (1, 0):
                edit = 'Terminate'
                edits.append(edit)
                done = True
                break

            edit_logits, prev_atom_hiddens, prev_atom_scope = self.compute_edit_scores(
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope)
            idx = torch.argmax(edit_logits[0])
            val = edit_logits[0][idx]

            max_bond_idx = products.GetNumBonds() * self.bond_outdim

            if idx.item() == len(edit_logits[0]) - 1:
                edit = 'Terminate'
                edits.append(edit)
                done = True
                break

            elif idx.item() < max_bond_idx:
                bond_logits = edit_logits[0][:products.GetNumBonds(
                ) * self.bond_outdim]
                bond_logits = bond_logits.reshape(
                    products.GetNumBonds(), self.bond_outdim)
                idx_tensor = torch.where(bond_logits == val)

                idx_tensor = [indices[-1] for indices in idx_tensor]

                bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
                a1 = products.GetBondWithIdx(
                    bond_idx).GetBeginAtom().GetAtomMapNum()
                a2 = products.GetBondWithIdx(
                    bond_idx).GetEndAtom().GetAtomMapNum()

                a1, a2 = sorted([a1, a2])
                edit_atom = [a1, a2]
                edit = self.bond_vocab.get_elem(edit_idx)

            else:
                atom_logits = edit_logits[0][max_bond_idx:-1]

                assert len(atom_logits) == products.GetNumAtoms() * \
                    self.atom_outdim
                atom_logits = atom_logits.reshape(
                    products.GetNumAtoms(), self.atom_outdim)
                idx_tensor = torch.where(atom_logits == val)

                idx_tensor = [indices[-1] for indices in idx_tensor]
                atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()

                a1 = products.GetAtomWithIdx(atom_idx).GetAtomMapNum()
                edit_atom = a1
                edit = self.atom_vocab.get_elem(edit_idx)

            try:
                products = apply_edit_to_mol(mol=Chem.Mol(
                    products), edit=edit, edit_atom=edit_atom)
                prod_graph = MolGraph(mol=Chem.Mol(
                    products),  rxn_class=rxn_class, use_rxn_class=use_rxn_class)
                prod_tensors, prod_scopes = get_batch_graphs(
                    [prod_graph], use_rxn_class=use_rxn_class)

                edits.append(edit)
                edits_atom.append(edit_atom)
                steps += 1

            except:
                steps += 1
                continue

        return edits, edits_atom

    def get_saveables(self) -> Dict:
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}
        saveables['config'] = self.config
        saveables['atom_vocab'] = self.atom_vocab
        saveables['bond_vocab'] = self.bond_vocab

        return saveables
