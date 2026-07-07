from collections import defaultdict
from knowledge.distiller import distill_transformation
import itertools
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
import random
from rdkit.Chem import rdFingerprintGenerator

# Knowledge Base
class KnowledgeManager:
    def __init__(self):
        # {scaffold: {action_key: {alpha, beta, ...}}}
        self.repo = defaultdict(dict)
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def update_action_after_eval(self, source_scaffold, action_key, delta_score, weight=1.0):

        if action_key not in self.repo[source_scaffold]:
            self.repo[source_scaffold][action_key] = {
                "alpha": 1.0,
                "beta": 1.0,
                "rewards": []
            }

        entry = self.repo[source_scaffold][action_key]

        if delta_score > 0:
            entry["alpha"] += weight
        else:
            entry["beta"] += weight

        entry["rewards"].append(delta_score)

    def extract_and_update_knowledge(self, m_seed, m_new):
        """
        R-group / Scaffold Hopping
        """
        scaf_a = m_seed.scaffold_smiles
        scaf_b = m_new.scaffold_smiles
        delta = m_seed.score - m_new.score

        if scaf_a == scaf_b:
            trans_list_f = distill_transformation(m_seed, m_new)
            if not trans_list_f: return 0

            num_actions = len(trans_list_f)
            split_weight = 1.0 / num_actions

            for trans in trans_list_f:
                self.update(scaf_a, trans, weight=split_weight)

                trans_b = trans.copy()
                trans_b.update({
                    "from_smiles": trans["to_smiles"],
                    "to_smiles": trans["from_smiles"],
                    "delta_score": -trans["delta_score"]
                })
                self.update(scaf_b, trans_b, weight=split_weight)
            return num_actions * 2
        else:

            hopping_action_f = {
                "from_smiles": f"[SCAFFOLD]{scaf_a}",
                "to_smiles": f"[SCAFFOLD]{scaf_b}",
                "delta_score": delta
            }
            self.update(scaf_a, hopping_action_f, weight=1.0)

            hopping_action_b = {
                "from_smiles": f"[SCAFFOLD]{scaf_b}",
                "to_smiles": f"[SCAFFOLD]{scaf_a}",
                "delta_score": -delta
            }
            self.update(scaf_b, hopping_action_b, weight=1.0)

            print(f">> Scaffold Hopping detected: {scaf_a} -> {scaf_b}")
            return 2

    def update(self, scaffold_smiles, transformation, weight=1.0):
        f_smiles = transformation["from_smiles"]
        t_smiles = transformation["to_smiles"]
        action_key = f"{f_smiles}>>{t_smiles}"
        delta = transformation["delta_score"]

        if action_key not in self.repo[scaffold_smiles]:
            self.repo[scaffold_smiles][action_key] = {
                "alpha": 1.0,
                "beta": 1.0,
                "rewards": []
            }

        entry = self.repo[scaffold_smiles][action_key]

        if delta > 0:
            entry["alpha"] += weight
            entry["rewards"].append(delta)
        else:
            entry["beta"] += weight
            entry["rewards"].append(delta)

    def warm_start(self, clusters, min_delta=0.5):
        total_actions = 0
        for scaf_smiles, mol_list in clusters.items():
            for m_from, m_to in itertools.permutations(mol_list, 2):

                delta = m_from.score - m_to.score
                if abs(delta) < min_delta:
                    continue

                trans_list = distill_transformation(m_from, m_to)

                if trans_list:
                    split_weight = 1 / len(trans_list)
                    split_delta = delta / len(trans_list)

                    for trans in trans_list:
                        trans["delta_score"] = split_delta
                        self.update(scaf_smiles, trans, weight=split_weight)
                        total_actions += 1
        print(f"[KnowledgeManager] warm start completed，Action number:{total_actions}")

    def _get_most_similar_scaffold(self, target_scaffold_smiles):
        if not self.repo:
            return None, 0.0

        target_mol = Chem.MolFromSmiles(target_scaffold_smiles)
        if not target_mol: return None, 0.0
        target_fp = self.fp_gen.GetFingerprint(target_mol)

        best_sim = 0.0
        best_scaf = None

        for scaf_smiles in self.repo.keys():
            ref_mol = Chem.MolFromSmiles(scaf_smiles)
            if not ref_mol: continue
            ref_fp = self.fp_gen.GetFingerprint(ref_mol)
            sim = DataStructs.TanimotoSimilarity(target_fp, ref_fp)

            if sim > best_sim:
                best_sim = sim
                best_scaf = scaf_smiles

        return best_scaf, best_sim

    def select_action(self, current_molecule, sim_threshold=0.3, use_thompson=True, use_knowledge=True):
        # lower-level TS
        if not use_knowledge:
            return None
        if not use_thompson:
            all_possible_actions = []
            for scaf, actions in self.repo.items():
                for a_key, a_stats in actions.items():
                    all_possible_actions.append((scaf, a_key, a_stats))

            if not all_possible_actions:
                return None
            rand_scaf, rand_action_key, rand_stats = random.choice(all_possible_actions)
            return {
                "action_key": rand_action_key,
                "stats": rand_stats,
                "expected_utility": 0.0,
                "is_exact_match": False,
                "source_scaffold": rand_scaf,
                "similarity": 0.0
            }

        scaf_smiles = current_molecule.scaffold_smiles
        is_exact_match = True
        similarity = 1.0

        if scaf_smiles not in self.repo:
            is_exact_match = False
            scaf_smiles, similarity = self._get_most_similar_scaffold(scaf_smiles)

        # free explore
        if not scaf_smiles or similarity < sim_threshold:
            return None

        actions = self.repo.get(scaf_smiles, {})
        best_utility = -float('inf')
        selected_action = None

        for action_key, stats in actions.items():
            theta = np.random.beta(stats["alpha"], stats["beta"])

            rewards = stats["rewards"]
            pos_rewards = [r for r in rewards if r > 0]
            neg_rewards = [r for r in rewards if r < 0]

            e_pos = np.mean(pos_rewards) if pos_rewards else 0.0
            e_neg = max(np.mean(neg_rewards), -1.0) if neg_rewards else 0.0

            expected_utility = (theta * e_pos) + ((1 - theta) * e_neg)

            if expected_utility > best_utility:
                best_utility = expected_utility
                selected_action = {
                    "action_key": action_key,
                    "stats": stats,
                    "expected_utility": expected_utility,
                    "is_exact_match": is_exact_match,
                    "source_scaffold": scaf_smiles,
                    "similarity": similarity
                }

        return selected_action