import gzip
import pickle
from pathlib import Path
from utils.utils import get_scip_params, create_generator
import ecole
import numpy as np


DATA_MAX_SAMPLES = 1000
DATASET = 'facilities' # setcover or cauctions or indset or facilities
SAVE_DIR = f"examples/brancher/data/{DATASET}"

generator = create_generator(DATASET, 'easy')

class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()
        self.is_root_node = True  # Track if we're at the root node
        self.node_counter = 0

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)
        self.is_root_node = True  # Reset to root node when environment resets
        self.node_counter = 0

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        Always use strong branching for root node, otherwise use probability.
        """
        self.node_counter += 1
        if self.is_root_node:
            # Always use strong branching for the root node
            self.is_root_node = False  # No longer at root node after this
            return (self.strong_branching_function.extract(model, done), True, self.node_counter)
        else:
            # For non-root nodes, use probability-based selection
            probabilities = [1 - self.expert_probability, self.expert_probability]
            expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
            if expert_chosen:
                return (self.strong_branching_function.extract(model, done), True, self.node_counter)
            else:
                return (self.pseudocosts_function.extract(model, done), False, self.node_counter)

# SCIP parameters for branching
scip_parameters = {
    **get_scip_params(),
    'limits/nodes': 3000
} 

# Create environment with observation functions
env = ecole.environment.Branching(
    observation_function= {
        "Branch": ExploreThenStrongBranch(expert_probability=0.6),
        "Node": ecole.observation.NodeBipartite(),
        "Addition": ecole.observation.Khalil2016()
    },
    scip_params=scip_parameters,
)

# Seed for reproducibility
env.seed(0)

def generate_setcover_samples():
    # Create output directory
    Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)
    
    episode_counter = 0
    sample_counter = 0
    rng = np.random.RandomState(0)
    
    while sample_counter < DATA_MAX_SAMPLES:
        print(f"Episode {episode_counter}, collecting samples...")
        instance = next(generator)
        episode_seed = rng.randint(2**32)
        
        # Reset environment with new instance
        observation, action_set, _, done, _ = env.reset(instance)
        
        # Track if we've saved root data for this episode
        root_saved = False
        root_observation = None
        root_action = None
        root_action_set = None
        root_scores = None
        node_samples_saved = 0
        
        while not done:
            (scores, scores_are_expert, node_id) = observation["Branch"]
            
            # Extract node bipartite observation and additional features
            node_observation = {
                "Node": observation["Node"],
                "Addition": observation["Addition"]
            }
            
            # Choose the action with the highest score
            action = action_set[scores[action_set].argmax()]
            
            # If expert scores are used, save the sample
            if scores_are_expert and sample_counter < DATA_MAX_SAMPLES:
                # If this is the first node (root), save root data
                if not root_saved:
                    root_saved = True
                    root_observation = node_observation
                    root_action = action
                    root_action_set = action_set
                    root_scores = scores
                    
                    # Save root data
                    root_filename = f"{SAVE_DIR}/sample_root_0_{episode_counter}.pkl"
                    
                    with gzip.open(root_filename, 'wb') as f:
                        pickle.dump({
                            'type': 'root',
                            'episode': episode_counter,
                            'seed': episode_seed,
                            'stats': {'nnodes': 0, 'time': 0, 'gap': 0, 'nobs': 0},
                            'root_state': [root_observation, root_action_set, action, scores],
                            'obss': [root_observation, action, action_set, scores]
                        }, f)
                    
                    sample_counter += 1
                    print(f"Root sample saved: {sample_counter}/{DATA_MAX_SAMPLES}")
                
                # For non-root nodes, save node data with reference to root
                else:
                    node_filename = f"{SAVE_DIR}/sample_node_{node_id}_{episode_counter}.pkl"
                    
                    with gzip.open(node_filename, 'wb') as f:
                        pickle.dump({
                            'type': 'node',
                            'episode': episode_counter,
                            'seed': episode_seed,
                            'stats': {'nnodes': 0, 'time': 0, 'gap': 0, 'nobs': 0},
                            'root_state': f"{SAVE_DIR}/sample_root_0_{episode_counter}.pkl",
                            'obss': [node_observation, action, action_set, scores]
                        }, f)
                    
                    sample_counter += 1
                    node_samples_saved += 1
                    
                    if sample_counter >= DATA_MAX_SAMPLES:
                        break
            
            # Take a step in the environment
            observation, action_set, _, done, _ = env.step(action)
        
        # Print episode summary
        if root_saved:
            print(f"Episode {episode_counter}: Saved root + {node_samples_saved} node samples, total: {sample_counter}/{DATA_MAX_SAMPLES}")
        else:
            print(f"Episode {episode_counter}: No samples saved")
        
        episode_counter += 1

if __name__ == "__main__":
    generate_setcover_samples()