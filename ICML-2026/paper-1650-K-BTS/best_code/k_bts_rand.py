from core.molecule import Molecule
from core.scaffold import cluster_molecules
from core.selector import SeedSelector
from knowledge.manager import KnowledgeManager
from utils.prompt_tools import get_system_instruction, generate_user_task_prompt
from utils.llm_tools import LLMHandler
from utils.evaluator import Evaluator
from utils.molecule_forest import MolecularForest
import os
import pandas as pd
import numpy as np
import random
import torch
import time
import pickle
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_initial_data(exp_path):
    file_path = os.path.join(exp_path, 'init_score.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File Not Found: {file_path}")

    df = pd.read_csv(file_path)
    init_docking_score = df["docking_scores"].tolist()
    init_smiles = df["smile"].tolist()

    mols = []
    seen_smiles = {}

    for i in range(len(init_docking_score)):
        score = init_docking_score[i]
        if score == 1000:
            score = 0

        m = Molecule(init_smiles[i], score=score)

        if m.mol:
            can_smiles = m.smiles

            if can_smiles not in seen_smiles:
                seen_smiles[can_smiles] = m
            else:
                if score < seen_smiles[can_smiles].score:
                    seen_smiles[can_smiles] = m

    mols = list(seen_smiles.values())
    print(f"load unique init mols: {len(mols)}")
    return mols


def run_optimization_step(selector, kb_manager):
    # upper
    m_seed = selector.select_seed(tau=2.0)

    # lower
    action_info = kb_manager.select_action(m_seed, sim_threshold=0.3)

    action_key = action_info["action_key"] if action_info else None

    status = f"Exact Match" if action_info and action_info['is_exact_match'] else "Fuzzy/Discovery"
    print(f">> Selected Seed: {m_seed.smiles} (Score: {m_seed.score:.2f})")
    print(f">> Suggested Action: {action_key if action_key else 'Free Exploration'} [{status}]")

    return m_seed, action_key, action_info

def main(random_seed, exp_path=None, name_protein=None):

    print(f"current protein id: {name_protein}")
    time_record = {"TS_time":0, "LLM_time":0, "Evaluation_time":0}
    start_time = time.time()
    CSV_PATH = os.path.join(exp_path, f"{random_seed}_result.csv")
    POSE_PATH = os.path.join(exp_path, f"pose.pkl")
    set_seed(random_seed)
    mols = get_initial_data(exp_path)
    forest = MolecularForest()
    for m in mols:
        mol = Chem.MolFromSmiles(m.smiles)
        if mol:
            qed_val = QED.qed(mol)
            sa_norm = max(0.0, min(1.0, (10.0 - sascorer.calculateScore(mol)) / 9.0))
            scaffold = m.scaffold_smiles
            forest.add_root(
                smiles=m.smiles,
                score=m.score,
                qed=qed_val,
                sa=sa_norm,
                scaffold=scaffold
            )
    history_set = {m.smiles for m in mols}

    selector = SeedSelector(mols)
    evaluator = Evaluator(output_file=CSV_PATH, pose_pkl=POSE_PATH,clear_old=True)
    clusters = cluster_molecules(mols)

    kb_manager = KnowledgeManager()
    kb_manager.warm_start(clusters, min_delta=0.5)
    system_msg = get_system_instruction()
    llm = LLMHandler(model_type="deepseek", api_key="your api key")
    total_new_mols = 0
    TARGET_COUNT = 100
    MAX_ITERS = 500
    time_record["TS_time"] = time.time() - start_time
    iter_time = time.time()
    print(f"\n=== Starting Optimization Loop (Target: {TARGET_COUNT} molecules) ===")

    for i in range(MAX_ITERS):
        print(f">> Iteration {i+1}/{MAX_ITERS}, new mols num: {total_new_mols}")
        if total_new_mols >= TARGET_COUNT:
            print(f"\n[Terminated] Target reached: {total_new_mols} new molecules.")
            break

        m_seed, action_key, action_info = run_optimization_step(selector, kb_manager)

        user_msg = generate_user_task_prompt(m_seed, action_info, history_set)

        time_record["TS_time"] += time.time() - iter_time
        iter_time = time.time()
        try:
            raw_response = llm.ask(prompt=user_msg, system_prompt=system_msg, temperature=0.7)
            rationale, new_smiles = llm.extract_smiles_and_rationale(raw_response)
        except Exception as e:
            print(f"LLM Error: {e}")
            continue

        if not new_smiles:
            continue
        time_record["LLM_time"] += time.time() - iter_time
        iter_time = time.time()
        eval_res = evaluator.run(
            new_smiles=new_smiles,
            m_seed=m_seed,
            rationale=rationale,
            user_prompt=user_msg,
            protein_name=name_protein,
            history_set=history_set
        )
        time_record["Evaluation_time"] += time.time() - iter_time
        iter_time = time.time()
        if eval_res:
            new_score, final_smiles, qed, sa = eval_res
            total_new_mols += 1

            m_new = Molecule(final_smiles, new_score)
            scaffold = m_new.scaffold_smiles
            forest.add_molecule(
                smiles=final_smiles,
                score=new_score,
                qed=qed,
                sa=sa,
                scaffold=scaffold,
                parent_smiles=m_seed.smiles,
                action=action_key
            )

            delta = m_seed.score - m_new.score
            if action_info:
                kb_manager.update_action_after_eval(
                    source_scaffold=action_info["source_scaffold"],
                    action_key=action_info["action_key"],
                    delta_score=delta
                )

            kb_manager.extract_and_update_knowledge(m_seed, m_new)

            if new_score < 500:
                selector.add_new_molecule(m_new, parent_smiles=m_seed.smiles)
                selector.update_node(m_seed.smiles, new_score)
                print(f"   [{total_new_mols}] Success: {new_score:.2f}")
            else:
                print(f"   [{total_new_mols}] Docking Failed (500), not added to seed pool.")
        time_record["TS_time"] += time.time() - iter_time
        iter_time = time.time()
    print(f"\n=== Run Finished. Total: {total_new_mols} molecules in {CSV_PATH} ===")
    checkpoint_dir = os.path.join(exp_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    selector_path = os.path.join(checkpoint_dir, f"selector_seed_{random_seed}.pkl")
    kb_path = os.path.join(checkpoint_dir, f"kb_manager_seed_{random_seed}.pkl")
    history_path = os.path.join(checkpoint_dir, f"history_set_{random_seed}.pkl")
    time_record_path = os.path.join(checkpoint_dir, f"time_record_{random_seed}.pkl")
    forest_save_path = os.path.join(checkpoint_dir, f"forest_seed_{random_seed}.pkl")
    forest.save(forest_save_path)
    forest.generate_svg(os.path.join(exp_path, "full_forest"))

    with open(selector_path, 'wb') as f:
        pickle.dump(selector, f)

    with open(kb_path, 'wb') as f:
        pickle.dump(kb_manager.repo, f)

    with open(history_path, 'wb') as f:
        pickle.dump(history_set, f)

    with open(time_record_path, 'wb') as f:
        pickle.dump(time_record, f)

    print(f">> All core objects saved to {checkpoint_dir}")
def crossdocked_main():
    seed = 1
    exp_path = 'results/your_path'
    # exp_path = 'results/rand'
    for i in range(100):
        name_protein = str(i)
        main(seed, os.path.join(exp_path, name_protein), name_protein)

if __name__ == "__main__":
    crossdocked_main()