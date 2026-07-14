import pyAgrum as gum
import random
import numpy as np
import pandas as pd
import os
import sys
import argparse
from tqdm import tqdm
import json
import networkx as nx

# Add the current directory to the Python path
sys.path.insert(0, os.getcwd())


def generate_bn_with_rc_child_confounder(n, names, domain_size, initial_seed,
                                        preferred_rc_name=None, max_attempts=100):
    """
    Generates a random BN and ensures a specific confounding structure exists or can be added.

    1. Generates a random BN.
    2. Tries to find a node (RC) with at least one child (Child). It prioritizes
       `preferred_rc_name` if provided and valid, otherwise searches other nodes.
    3. Tries to find another node (C) that is not RC, not Child, and not a descendant of RC.
    4. If successful, adds arcs C -> RC and C -> Child (removing conflicting arcs first).
    5. If any step fails (no node with children, no suitable C), regenerates the BN
       with an incremented seed and tries again, up to max_attempts.

    Args:
        n (int): Number of nodes.
        names (list): List of node names.
        domain_size (int): Domain size for variables.
        initial_seed (int): The starting seed for randomization.
        preferred_rc_name (str, optional): The name of the node to prioritize as RC. Defaults to None.
        max_attempts (int): Maximum number of BN generation attempts.

    Returns:
        tuple: (modified_bn, rc_name, child_name, confounder_name) if successful.

    Raises:
        RuntimeError: If failed to find a suitable structure after max_attempts.
    """
    current_seed = initial_seed

    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts} with seed {current_seed}...")
        gum.initRandom(current_seed)
        # Ensure names are consistent if BN generator reorders them (shouldn't happen with provided names)
        bn = gum.randomBN(n=n, names=names, domain_size=domain_size)
        # Use names from the generated BN to be safe
        current_names = bn.names()
        name_to_id = {name: bn.idFromName(name) for name in current_names}
        id_to_name = {v: k for k, v in name_to_id.items()}

        selected_rc_id = -1
        selected_rc_name = None
        children_ids = set()

        # --- 1. Find suitable Root Cause (RC) ---
        # Check preferred RC first
        if preferred_rc_name and preferred_rc_name in name_to_id:
            rc_id_candidate = name_to_id[preferred_rc_name]
            candidate_children = bn.children(rc_id_candidate)
            if candidate_children:
                selected_rc_id = rc_id_candidate
                selected_rc_name = preferred_rc_name
                children_ids = candidate_children
                print(f"  -> Preferred RC '{selected_rc_name}' has children.")

        # If preferred RC not suitable or not specified, search others
        if selected_rc_id == -1:
            node_ids_shuffled = list(bn.nodes())
            random.shuffle(node_ids_shuffled) # Randomize search order
            for rc_id_candidate in node_ids_shuffled:
                # Skip if it's the preferred one we already checked
                if id_to_name[rc_id_candidate] == preferred_rc_name:
                    continue
                candidate_children = bn.children(rc_id_candidate)
                if candidate_children:
                    selected_rc_id = rc_id_candidate
                    selected_rc_name = id_to_name[selected_rc_id]
                    children_ids = candidate_children
                    print(f"  -> Found suitable RC: '{selected_rc_name}'")
                    break # Found one

        # If still no RC found, this graph is unsuitable
        if selected_rc_id == -1:
            print("  -> No node found with children in this graph. Regenerating...")
            current_seed += 1
            continue # Try next attempt with a new graph

        # --- 2. Select a Child ---
        # Pick one child randomly from the identified children
        selected_child_id = random.choice(list(children_ids))
        selected_child_name = id_to_name[selected_child_id]
        print(f"  -> Selected Child: '{selected_child_name}' (child of '{selected_rc_name}')")

        # --- 3. Find a suitable Confounder (C) ---
        # C cannot be RC, Child, or any descendant of RC
        rc_descendants_ids = bn.descendants(selected_rc_id)
        # The descendants set includes the node itself IF it's part of a cycle,
        # but typically means nodes reachable *from* RC. Let's be explicit.
        forbidden_ids = {selected_rc_id, selected_child_id} | rc_descendants_ids

        potential_c_ids = [nid for nid in bn.nodes() if nid not in forbidden_ids]

        if not potential_c_ids:
            print(f"  -> Found RC ({selected_rc_name}) and Child ({selected_child_name}), but no suitable Confounder node exists. Regenerating...")
            current_seed += 1
            continue # Try next attempt

        # --- 4. Select C and Add Arcs ---
        selected_c_id = random.choice(potential_c_ids)
        selected_c_name = id_to_name[selected_c_id]
        print(f"  -> Selected Confounder (C): '{selected_c_name}'")

        # Modify Arcs (remove potential conflicts first)
        # Remove RC -> C edge if it exists
        if bn.existsArc(selected_rc_id, selected_c_id):
            bn.eraseArc(selected_rc_id, selected_c_id)
            print(f"     Removed conflicting arc: {selected_rc_name} -> {selected_c_name}")
        # Remove Child -> C edge if it exists
        if bn.existsArc(selected_child_id, selected_c_id):
            bn.eraseArc(selected_child_id, selected_c_id)
            print(f"     Removed conflicting arc: {selected_child_name} -> {selected_c_name}")

        # Add confounding arcs C -> RC and C -> Child
        try:
            # Check if adding would create a cycle (basic check, might not catch all complex cases before addArc)
            # gum doesn't have a simple 'would adding arc create cycle?' check readily available AFAIK
            # We rely on the descendant check and addArc's internal checks
            bn.addArc(selected_c_id, selected_rc_id)
            bn.addArc(selected_c_id, selected_child_id)
        except gum.GumException as e:
             # This might happen if addArc detects a cycle despite our descendant check
             print(f"  -> ERROR adding arcs for C={selected_c_name} -> RC={selected_rc_name}/Child={selected_child_name}: {e}. Regenerating...")
             current_seed += 1
             continue # Try next attempt

        print(f"Success on attempt {attempt + 1}!")
        print(f"  Final Configuration: {selected_c_name} -> {selected_rc_name}, {selected_c_name} -> {selected_child_name}")

        # Return the modified BN and the names
        return bn, selected_rc_name, selected_child_name, selected_c_name

    # If loop finishes without success
    raise RuntimeError(f"Failed to generate a suitable BN structure and add confounder after {max_attempts} attempts.")




def generate_data(n, obs_sample_size, int_sample_size, seed, intervened_node=None, add_backdoor=False, multiple_rootcauses=0,  hard_intervention=False):
    """
    Generate observational and interventional data for a random Bayesian network.
    
    Args:
        n (int): Number of nodes in the Bayesian network
        obs_sample_size (int): Number of observational samples to generate
        int_sample_size (int): Number of interventional samples to generate
        seed (int): Random seed for reproducibility
        intervened_node (int, optional): Specific node to intervene on. If None, randomly selected.
    
    Returns:
        tuple: (df_obs, df_int, true_root_cause_str, obs_bn, int_bn, nodenames_str)
    """
    # Generate node names with 'V' prefix to avoid BIF format issues
    nodenames_str = [f'V{i}' for i in range(n)]

    # Select node for intervention if not specified
    if intervened_node is None:
        nodes = list(range(n))
        random.seed(seed) # seed
        intervened_node = random.choice(nodes)
    
    true_root_cause_str = f'V{intervened_node}'

    if multiple_rootcauses:
        shuffled_nodes = random.sample(nodes, multiple_rootcauses)
        true_root_cause_str = [f'V{i}' for i in shuffled_nodes]


    if add_backdoor:
        try:
            final_bn, rc_name, child_name, confounder_name = generate_bn_with_rc_child_confounder(
                n=n,
                names=nodenames_str,
                domain_size=4,
                initial_seed=seed, # Seed for the first BN generation attempt
                preferred_rc_name=true_root_cause_str, # Pass the preference
                max_attempts=50 # Limit the number of retries
            )
            
            print("\nFinal Modified BN DAG:")
            print(final_bn)
            print(f"\nStructure Added: {confounder_name} -> {rc_name} (RC), {confounder_name} -> {child_name} (Child)")
            print(f"Final node count: {final_bn.size()}")
            assert final_bn.size() == n

            true_root_cause_str = rc_name
            obs_bn = final_bn


        except RuntimeError as e:
            print(f"\n{e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
    else:
        # Create random Bayesian network
        gum.initRandom(seed) # seed
        obs_bn = gum.randomBN(n=n, names=nodenames_str, domain_size=4)
    

    # obs_bn = gum.fastBN("V0->V5; V0->V8; V2->V0; V3->V6; V5->V4;V5->V7;V6->V0;V7->V4;V7->V1;V9->V8;V9->V5;V9->V6")
    
    # Generate observational data
    g = gum.BNDatabaseGenerator(obs_bn)
    g.drawSamples(obs_sample_size)
    df_obs = g.to_pandas()
    

    # true_root_cause_str  = str(intervened_node)
    #true_root_cause_str = 'V1'
    
    # Generate interventional data
    int_bn = gum.BayesNet(obs_bn)
    gum.initRandom(seed) # seed
    if multiple_rootcauses:
        for root_cause in true_root_cause_str:
            if hard_intervention:
                copy_bn = gum.BayesNet(int_bn)
                copy_cpt = copy_bn.generateCPT(root_cause)
                print(copy_cpt)
                copy_cpt.fillwith(0)
                copy_cpt[root_cause] = 1.0
                int_bn.cpt(root_cause)[:] = copy_cpt
                print(int_bn.cpt(root_cause))
            else:
                int_bn.generateCPT(root_cause)
    else:
        if hard_intervention:
            copy_bn = gum.BayesNet(int_bn)
            copy_cpt = copy_bn.generateCPT(true_root_cause_str)
            print(copy_cpt)
            copy_cpt.fillwith(0)
            copy_cpt[root_cause] = 1.0
            int_bn.cpt(root_cause)[:] = copy_cpt
            print(int_bn.cpt(root_cause))
        else:
            int_bn.generateCPT(true_root_cause_str)
    g = gum.BNDatabaseGenerator(int_bn)
    g.drawSamples(int_sample_size)
    df_int = g.to_pandas()
    
    return df_obs, df_int, true_root_cause_str, obs_bn, int_bn, nodenames_str

def save_experiment_data(df_obs, df_int, true_root_cause_str, obs_bn, int_bn, nodenames_str, 
                        output_dir, int_sample_size, graph_size, obs_sample_size, experiment_idx, args):
    """
    Save experiment data with index to specified directory.
    """
    # Create experiment-specific directory
    exp_dir = os.path.join(output_dir, f'experiment_{experiment_idx}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # save the bn files
    obs_bn.saveBIF(os.path.join(exp_dir, f'observational_bn.bif'))
    int_bn.saveBIF(os.path.join(exp_dir, f'interventional_bn.bif'))
    
    # Save data to files
    df_obs.to_csv(os.path.join(exp_dir, f'observational_data.csv'), index=False)
    df_int.to_csv(os.path.join(exp_dir, f'interventional_data.csv'), index=False)
    
    # Save metadata
    metadata = {
        'n': graph_size,
        'obs_sample_size': obs_sample_size,
        'int_sample_size': int_sample_size,
        'seed': args.seed + experiment_idx,  # Use different seed for each experiment
        'true_root_cause': true_root_cause_str,
        'node_names': nodenames_str,
        'experiment_index': experiment_idx
    }
    
    with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

def main():
    # you can run this script on terminal as follows
    # python data_generation.py --n 10 --obs_sample_size 100000 --int_sample_sizes 10 50 100 250 500 --num_repetitions 50 --output_dir data

    parser = argparse.ArgumentParser(description='Generate observational and interventional data for Bayesian network analysis')
    parser.add_argument('--n', type=int, nargs='+', default=[1000], help='Number of nodes in the Bayesian network')
    parser.add_argument('--obs_sample_sizes', type=int, nargs='+', default=[10000], help='Number of observational samples to generate')
    parser.add_argument('--int_sample_sizes', type=int, nargs='+', default=[10000], help='A list of interventional sample sizes to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--add_backdoor', type=bool, default=False, help='Whether to adding backdoor on root causes')
    parser.add_argument('--intervened_node', type=int, help='Specific node to intervene on (optional)')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save the generated data')
    parser.add_argument('--num_repetitions', type=int, default=100, help='Number of times to repeat the experiment for each graph size')
    parser.add_argument('--all_sizes_under_n', type=bool, default=True, help='Should we generate each sample size under each graph size')
    parser.add_argument('--multiple_rootcauses', type=int, default=0, help='how many root causes for each experiment, 0 means single root cause')
    parser.add_argument('--hard_intervention', type=int, default=0, help='how many root causes for each experiment, 0 means single root cause')

    args = parser.parse_args()
    
    # Create main output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # generate every combination of graph size, obs sample size, and int sample size
    if args.all_sizes_under_n:
        for graph_size in args.n:
            for obs_sample_size in args.obs_sample_sizes:
                for int_sample_size in args.int_sample_sizes:
                    # Create directory for this sample size
                    size_dir = os.path.join(args.output_dir, f'n_{graph_size}_obsSampleSize_{obs_sample_size}_intSampleSize_{int_sample_size}')
                    os.makedirs(size_dir, exist_ok=True)
                    
                    # Generate data for each repetition
                    for experiment_idx in tqdm(range(args.num_repetitions), desc=f"Generating experiments for graph size {graph_size}, obs size {obs_sample_size}, int size {int_sample_size}", leave=False):
                        # Generate data with different seed for each experiment
                        df_obs, df_int, true_root_cause_str, obs_bn, int_bn, nodenames_str = generate_data(
                            graph_size, obs_sample_size, int_sample_size, 
                            args.seed + experiment_idx, args.intervened_node, args.add_backdoor, args.multiple_rootcauses, args.hard_intervention
                        )
                        
                        # Save experiment data
                        save_experiment_data(df_obs, df_int, true_root_cause_str, obs_bn, int_bn, nodenames_str,
                                        size_dir, int_sample_size, graph_size, obs_sample_size,  experiment_idx, args)

    else:
        # generate every combination for each pair of graph size and int sample size
        for graph_size, int_sample_size in zip(args.n, args.int_sample_sizes):
            # Generate data for each sample size and repetition
            for obs_sample_size in args.obs_sample_sizes: 
            
                # Create directory for this sample size
                size_dir = os.path.join(args.output_dir, f'n_{graph_size}_obsSampleSize_{obs_sample_size}_intSampleSize_{int_sample_size}')
                os.makedirs(size_dir, exist_ok=True)
                
                # Generate data for each repetition
                for experiment_idx in tqdm(range(args.num_repetitions), desc=f"Generating experiments for graph size {graph_size}, obs size {obs_sample_size}, int size {int_sample_size}", leave=False):
                    # Generate data with different seed for each experiment
                    df_obs, df_int, true_root_cause_str, obs_bn, int_bn, nodenames_str = generate_data(
                        graph_size, obs_sample_size, int_sample_size, 
                        args.seed + experiment_idx, args.intervened_node, args.add_backdoor, args.multiple_rootcauses, args.hard_intervention
                    )
                    
                    # Save experiment data
                    save_experiment_data(df_obs, df_int, true_root_cause_str, obs_bn, int_bn, nodenames_str,
                                    size_dir, int_sample_size, graph_size, obs_sample_size,  experiment_idx, args)
    
    print(f"Generated {args.num_repetitions} experiments for each of {len(args.int_sample_sizes)} sample sizes in {args.output_dir}/")
    print(f"Sample sizes: {args.int_sample_sizes}")
    print(f"Each experiment is saved in its own subdirectory (experiment_0 to experiment_{args.num_repetitions-1})")

if __name__ == "__main__":
    main()