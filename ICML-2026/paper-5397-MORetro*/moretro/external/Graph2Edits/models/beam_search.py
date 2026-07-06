import numpy as np
from typing import List, Union
from collections import defaultdict
import torch
import torch.nn.functional as F
from rdkit import Chem

from Graph2Edits.utils.rxn_graphs import MolGraph
from Graph2Edits.utils.collate_fn import get_batch_graphs
from Graph2Edits.prepare_data import apply_edit_to_mol
from Graph2Edits.utils.reaction_actions import (AddGroupAction, AtomEditAction,
                                    BondEditAction, Termination)


class BeamSearch:
    def __init__(self, model, step_beam_size, beam_size, use_rxn_class):
        self.model = model
        self.step_beam_size = step_beam_size
        self.beam_size = beam_size
        self.use_rxn_class = use_rxn_class

    def process_paths(self, paths: List[dict], rxn_class):
        if not paths:
            return []

        # Prepare batch of graphs
        # We prefer using cached MolGraph if available, else recreate
        mol_graphs = []
        for p in paths:
            if 'prod_graph' in p:
                mol_graphs.append(p['prod_graph'])
            else:
                mol_graphs.append(MolGraph(mol=p['prod_mol'], rxn_class=rxn_class, use_rxn_class=self.use_rxn_class))

        # Create batched tensors
        prod_tensors, prod_scopes = get_batch_graphs(mol_graphs, use_rxn_class=self.use_rxn_class)
        prod_tensors = self.model.to_device(prod_tensors)
        
        # Prepare batched previous states
        # If any path has state None (step 0), all should be None or handled
        if paths[0]['state'] is None:
            prev_atom_hiddens = None
            prev_atom_scope = None
        else:
            # Concatenate states from all paths
            # path['state'] is (num_atoms, hidden)
            prev_atom_hiddens = torch.cat([p['state'] for p in paths], dim=0)
            
            # Construct scope list for the batched state
            prev_atom_scope = []
            curr_offset = 0
            for p in paths:
                # p['state_scope'] is [(0, num_atoms)] relative to that path's state tensor
                count = p['state_scope'][0][1]
                prev_atom_scope.append((curr_offset, count))
                curr_offset += count

        # Run Model
        # output: list of score tensors, next hidden states (batched), next scopes
        edit_logits_batch, next_state_batch, _ = self.model.compute_edit_scores(
            prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope)
        
        new_paths = []
        atom_scopes = prod_scopes[0] # List of (start, len) for atoms in the current batch

        for i, path in enumerate(paths):
            prod_mol = path['prod_mol']
            steps = path['steps'] + 1
            
            # Get logits for this graph
            logits = edit_logits_batch[i]
            logits = F.softmax(logits, dim=-1)
            
            # Get next state corresponding to this graph
            # The next_state_batch is aligned with prod_tensors atoms
            start_idx, num_atoms = atom_scopes[i]
            path_next_state = next_state_batch[start_idx : start_idx + num_atoms]
            path_next_state_scope = [(0, num_atoms)] # Normalized for single storage

            k = self.step_beam_size
            top_k_vals, top_k_idxs = torch.topk(logits, k=k)

            for topk_idx, val in zip(top_k_idxs, top_k_vals):
                val_item = round(val.item(), 4)
                new_prob = path['prob'] * val_item
                
                edit, edit_atom = self.get_edit_from_logits(
                    mol=prod_mol, edit_logits=logits, idx=topk_idx, val=val)

                if edit == 'Terminate':
                    edits_prob = list(path['edits_prob'])
                    edits_prob.append(val_item)
                    edits = list(path['edits'])
                    edits.append(edit)
                    
                    final_path = {
                        'prod_mol': prod_mol,
                        'prod_graph': mol_graphs[i], # keep graph
                        'steps': steps,
                        'prob': new_prob,
                        'edits_prob': edits_prob,
                        'tensors': path['tensors'], # Keep original tensors? Or update? usually not needed for terminated
                        'scopes': path['scopes'],
                        'state': path_next_state,
                        'state_scope': path_next_state_scope,
                        'edits': edits,
                        'edits_atom': path['edits_atom'],
                        'finished': True,
                        'root_id': path.get('root_id', 0)
                    }
                    new_paths.append(final_path)

                else:
                    try:
                        int_mol = apply_edit_to_mol(mol=Chem.Mol(prod_mol), edit=edit, edit_atom=edit_atom)
                        
                        # Pre-compute next step graph/tensors for this specific child
                        # This is needed because next iter will expect 'prod_graph' and 'tensors'
                        # Although 'tensors' is only needed if we process individually.
                        # Since we batch in process_paths, we mainly need 'prod_graph'.
                        # But let's compute them to maintain full info and compatibility if mixed matching
                        
                        child_graph = MolGraph(mol=Chem.Mol(int_mol), rxn_class=rxn_class, use_rxn_class=self.use_rxn_class)
                        child_tensors, child_scopes = get_batch_graphs([child_graph], use_rxn_class=self.use_rxn_class)
                        
                        edits_prob = list(path['edits_prob'])
                        edits_prob.append(val_item)
                        edits = list(path['edits'])
                        edits.append(edit)
                        edits_atom = list(path['edits_atom'])
                        edits_atom.append(edit_atom)
                        
                        new_path = {
                            'prod_mol': int_mol,
                            'prod_graph': child_graph,
                            'steps': steps,
                            'prob': new_prob,
                            'edits_prob': edits_prob,
                            'tensors': child_tensors,
                            'scopes': child_scopes,
                            'state': path_next_state,
                            'state_scope': path_next_state_scope,
                            'edits': edits,
                            'edits_atom': edits_atom,
                            'finished': False,
                            'root_id': path.get('root_id', 0)
                        }
                        new_paths.append(new_path)
                    except Exception:
                        continue

        return new_paths

    def get_top_k_paths(self, paths):
        # paths is a mixed list from different root_ids
        # Group by root_id
        grouped = defaultdict(list)
        for p in paths:
            grouped[p.get('root_id', 0)].append(p)
            
        filtered_paths = []
        for root_id, group_paths in grouped.items():
            k = min(len(group_paths), self.beam_size)
            path_argsort = np.argsort([-path['prob'] for path in group_paths])
            filtered_paths.extend([group_paths[i] for i in path_argsort[:k]])
            
        return filtered_paths

    def get_edit_from_logits(self, mol, edit_logits, idx, val):
        max_bond_idx = mol.GetNumBonds() * self.model.bond_outdim

        if idx.item() == len(edit_logits) - 1:
            edit = 'Terminate'
            edit_atom = []

        elif idx.item() < max_bond_idx:
            bond_logits = edit_logits[:mol.GetNumBonds(
            ) * self.model.bond_outdim]
            bond_logits = bond_logits.reshape(
                mol.GetNumBonds(), self.model.bond_outdim)
            idx_tensor = torch.where(bond_logits == val)

            idx_tensor = [indices[-1] for indices in idx_tensor]

            bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()
            a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()
            a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()

            a1, a2 = sorted([a1, a2])
            edit_atom = [a1, a2]
            edit = self.model.bond_vocab.get_elem(edit_idx)

        else:
            atom_logits = edit_logits[max_bond_idx:-1]

            assert len(atom_logits) == mol.GetNumAtoms() *                 self.model.atom_outdim
            atom_logits = atom_logits.reshape(
                mol.GetNumAtoms(), self.model.atom_outdim)
            idx_tensor = torch.where(atom_logits == val)

            idx_tensor = [indices[-1] for indices in idx_tensor]
            atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()

            a1 = mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()
            edit_atom = a1
            edit = self.model.atom_vocab.get_elem(edit_idx)

        return edit, edit_atom

    def run_search(self, prod_smi: Union[str, List[str]], max_steps: int = 8, rxn_class: int = None) -> Union[List[dict], List[List[dict]]]:
        is_single = isinstance(prod_smi, str)
        prod_smis = [prod_smi] if is_single else prod_smi
        
        paths = []
        for i, smi in enumerate(prod_smis):
            product = Chem.MolFromSmiles(smi)
            Chem.Kekulize(product)
            prod_graph = MolGraph(mol=Chem.Mol(product), rxn_class=rxn_class, use_rxn_class=self.use_rxn_class)
            prod_tensors, prod_scopes = get_batch_graphs([prod_graph], use_rxn_class=self.use_rxn_class)

            start_path = {
                'prod_mol': product,
                'prod_graph': prod_graph,
                'steps': 0,
                'prob': 1.0,
                'edits_prob': [],
                'tensors': prod_tensors,
                'scopes': prod_scopes,
                'state': None,
                'state_scope': None,
                'edits': [],
                'edits_atom': [],
                'finished': False,
                'root_id': i
            }
            paths.append(start_path)

        for step_i in range(max_steps):
            followed_path = [path for path in paths if not path['finished']]
            if len(followed_path) == 0:
                break

            finished_paths_list = [path for path in paths if path['finished']]

            # Batched processing of all active paths
            new_paths = self.process_paths(followed_path, rxn_class)
            
            paths = finished_paths_list + new_paths
            paths = self.get_top_k_paths(paths)

            if all(path['finished'] for path in paths):
                break

        # Process results
        final_results = []
        for path in paths:
            if path['finished']:
                try:
                    root_id = path.get('root_id', 0)
                    product_smi = prod_smis[root_id]
                    product = Chem.MolFromSmiles(product_smi)
                    Chem.Kekulize(product)
                    int_mol = product
                    
                    path['rxn_actions'] = []
                    path_failed = False
                    
                    for i, edit in enumerate(path['edits']):
                        if int_mol is None: # Should not happen if edits are valid
                            path_failed = True
                            break
                        
                        if edit == 'Terminate':
                            edit_exe = Termination(action_vocab='Terminate')
                            path['rxn_actions'].append(edit_exe)
                            pred_mol = edit_exe.apply(int_mol)
                            try:
                                pred_mol = Chem.MolFromSmiles(Chem.MolToSmiles(pred_mol))
                                final_smi = Chem.MolToSmiles(pred_mol)
                                path['final_smi'] = final_smi
                            except:
                                path['final_smi'] = 'final_smi_unmapped'

                        elif edit[0] == 'Change Atom':
                            edit_exe = AtomEditAction(
                                path['edits_atom'][i], *edit[1], action_vocab='Change Atom')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)
                            
                        # ... other edit types need to be copied ...
                        elif edit[0] == 'Delete Bond':
                            edit_exe = BondEditAction(
                                *path['edits_atom'][i], *edit[1], action_vocab='Delete Bond')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                        elif edit[0] == 'Change Bond':
                            edit_exe = BondEditAction(
                                *path['edits_atom'][i], *edit[1], action_vocab='Change Bond')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)

                        elif edit[0] == 'Attaching LG':
                            edit_exe = AddGroupAction(
                                path['edits_atom'][i], edit[1], action_vocab='Attaching LG')
                            path['rxn_actions'].append(edit_exe)
                            int_mol = edit_exe.apply(int_mol)
                    
                    if not path_failed:
                        final_results.append(path)
                        
                except Exception as e:
                    # print(f'Exception while final mol to Smiles: {str(e)}')
                    path['final_smi'] = 'final_smi_unmapped'
                    final_results.append(path)

        # Reshape results
        if is_single:
            return final_results
        else:
            grouped = defaultdict(list)
            for p in final_results:
                grouped[p.get('root_id', 0)].append(p)
            return [grouped[i] for i in range(len(prod_smis))]
