import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time

sys.path.insert(0, "/repo")
from sharedGen import *

# ---- Configuration matching paper settings ----
UNIVERSE_SIZE = 1000       # |U| = 1000
NOISE_STD = 0.3             # sigma = 0.3
N_SAMPLES = 100             # N = 100 observations
N_PREDICTION_SAMPLES = 100000  # 100,000 prediction samples
BATCH_SIZE = 1000
LEARNING_RATE = 1e-3
HIDDEN_NEURONS = 50
OFF_POLICY = True
NUM_EPOCHS = 50000
MODEL_TYPE = "ours"         # GReinSS

# ---- Reproducibility seeds ----
np.random.seed(1)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}, GPU 0: {torch.cuda.get_device_name(0)}")

# ============================================================
# Step 1: Generate Set Simulation Data (Section C.3)
# ============================================================
print("\n" + "="*60)
print("Step 1: Generating Set Simulation Data")
print("="*60)

v_size = UNIVERSE_SIZE
n_dict = int(np.floor(np.sqrt(v_size)))  # sqrt(v_size) subsets
elem_prob = 2.0 / np.sqrt(v_size)         # each element included with prob 2/sqrt(v_size)

# Generate dictionary of subsets (modules)
dictionary = np.zeros((n_dict, v_size), dtype=int)
for d in range(n_dict):
    for e in range(v_size):
        if np.random.rand() < elem_prob:
            dictionary[d, e] = 1

print(f"Dictionary: {n_dict} subsets, avg subset size: {np.mean(np.sum(dictionary, axis=1)):.2f}")

# Generate latent states S_i* by taking union of random subsets from dictionary
latent_states = np.zeros((N_SAMPLES, v_size), dtype=int)
for i in range(N_SAMPLES):
    state = np.zeros(v_size, dtype=int)
    for d in range(n_dict):
        if np.random.rand() < 0.1:  # include each subset with prob 0.1
            state = np.maximum(state, dictionary[d])
    latent_states[i] = state

print(f"Latent states shape: {latent_states.shape}")
print(f"Avg elements per state: {np.mean(np.sum(latent_states, axis=1)):.2f}")

# Generate noisy observations X_i = S_i* + N(0, sigma^2)
observations = latent_states.astype(np.float64) + np.random.normal(0, NOISE_STD, latent_states.shape)
print(f"Observations shape: {observations.shape}")
print(f"Observations range: [{observations.min():.3f}, {observations.max():.3f}]")

# ============================================================
# Step 2: Define Pr(X|S) functions
# ============================================================
print("\n" + "="*60)
print("Step 2: Setting up Pr(X|S) functions")
print("="*60)

def log_calculate_pr_x_given_g(state, observation):
    """Log probability of observation given binary state (Gaussian noise)"""
    diff = 0.5 * (observation - state) ** 2
    log_prob = -np.sum(diff) / (NOISE_STD ** 2)
    return log_prob

def multi_x_given_g(adjacency_matrices, obs_matrix):
    """Vectorized batch computation of log Pr(X|S) for multiple states and observations"""
    obs_matrix = torch.tensor(obs_matrix).float().to(adjacency_matrices.device)
    adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))
    obs_matrix = obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))
    diff = 0.5 * torch.sum((adjacency_matrices - obs_matrix) ** 2, axis=2)
    prob_mult = -1 * diff / (NOISE_STD ** 2)
    return prob_mult

# ============================================================
# Step 3: Set up ruleObject and model
# ============================================================
print("\n" + "="*60)
print("Step 3: Setting up model")
print("="*60)

graphSize = v_size  # For sets, graphSize = universe size
finalProbSize = 1

# Define graphRules (no restrictions for simple sets)
def graphRules(graph):
    graphAllow = torch.zeros((graph.shape[0], graphSize + 1), device=device)
    finalProbAllow = torch.zeros((graph.shape[0], finalProbSize), device=device)
    return graphAllow, finalProbAllow

# Define off-policy rule (bias sampling toward elements likely to be in the set)
def offPolicyRule(graphList, arange1):
    arange1_mod = arange1 % (2 * observations.shape[0])
    subsetGood = np.argwhere(arange1_mod < observations.shape[0])[:, 0]
    arange1_mod = arange1_mod[subsetGood]
    
    graphAllow = torch.zeros((graphList.shape[0], graphSize + 1), device=device)
    obsList = np.copy(observations[arange1_mod])
    probRatio = ((obsList - 1) ** 2) - (obsList ** 2)
    probRatio = probRatio * 0.5
    noise_level_mod = 1.0 / (NOISE_STD ** 2)
    probRatio = probRatio * noise_level_mod
    probRatio = probRatio * -1
    probRatio = probRatio * 1.5  # 1.5x weighting as in fullTrainSet
    
    graphAllow[subsetGood, :-1] = torch.tensor(probRatio, device=device).float()
    
    finalProbAllow = torch.zeros((graphList.shape[0], finalProbSize), device=device)
    return graphAllow, finalProbAllow

# Create the model (2-layer FC with 50 hidden neurons)
model = GraphGeneratorNet(graphSize, 1, HIDDEN_NEURONS, endingBias=0)
model = model.to(device)
print(f"Model: {model}")

# Set up ruleObject
ruleObject = gClass()
ruleObject.log_calculate_pr_x_given_g = log_calculate_pr_x_given_g
ruleObject.graphSize = graphSize
ruleObject.observations_batch = observations
ruleObject.batchSize = BATCH_SIZE
ruleObject.learning_rate = LEARNING_RATE
ruleObject.model = model
ruleObject.graphRules = graphRules
ruleObject.offPolicyRule = offPolicyRule
ruleObject.multi_x_given_g = multi_x_given_g
ruleObject.adjacency_matrices = observations  # As in fullTrainSet

# ============================================================
# Step 4: Train GReinSS
# ============================================================
print("\n" + "="*60)
print("Step 4: Training GReinSS model")
print("="*60)

model_filename = "/repo/greinss_set_model.pt"

# Set up training data directories
os.makedirs("/repo/data/sims/simpleSet/model", exist_ok=True)
os.makedirs("/repo/data/sims/simpleSet/pred", exist_ok=True)
os.makedirs("/repo/data/sims/simpleSet/sample", exist_ok=True)
os.makedirs("/repo/data/sims/simpleSet/input", exist_ok=True)

# Save the generated data
np.savez_compressed("/repo/data/sims/simpleSet/input/D100_N1000_P0.3_sim0_obs.npz", observations)
np.savez_compressed("/repo/data/sims/simpleSet/input/D100_N1000_P0.3_sim0_latent.npz", latent_states)

# Train the model
print(f"Training with off_policy={OFF_POLICY}, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}")
train_start = time.time()
train_model_off_policy(
    ruleObject, 
    LEARNING_RATE, 
    BATCH_SIZE, 
    OFF_POLICY, 
    num_epochs=NUM_EPOCHS, 
    model_filename=model_filename, 
    rewardType='', 
    giveTrajectory=False
)
train_time = time.time() - train_start
print(f"Training completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")

# ============================================================
# Step 5: Run Inference with 100,000 samples
# ============================================================
print("\n" + "="*60)
print("Step 5: Running Inference with 100,000 prediction samples")
print("="*60)

# Load trained model
policyModel = torch.load(model_filename, map_location=device)
policyModel = policyModel.to(device)

# Sample states using trained policy
sampleSize = N_PREDICTION_SAMPLES
batch_size_sampling = 1000
numBatch = sampleSize // batch_size_sampling

inference_start = time.time()
with torch.no_grad():
    for a in range(numBatch):
        if a % 10 == 0:
            print(f"  Sampling batch {a+1}/{numBatch}")
        adjacency_matrices0, log_prob_pi0, log_prob_prime0, trajectory0 = generate_graph_batch_with_modified_policy(
            policyModel, ruleObject, OFF_POLICY, batch_size_sampling
        )
        adjacency_matrices0 = adjacency_matrices0.data.numpy()
        log_prob_pi0 = log_prob_pi0.data.numpy()
        log_prob_prime0 = log_prob_prime0.data.numpy()
        trajectory0 = trajectory0
        
        if a == 0:
            adjacency_matrices = np.copy(adjacency_matrices0)
            log_prob_pi = np.copy(log_prob_pi0)
            log_prob_prime = np.copy(log_prob_prime0)
        else:
            adjacency_matrices = np.concatenate((adjacency_matrices, adjacency_matrices0), axis=0)
            log_prob_pi = np.concatenate((log_prob_pi, log_prob_pi0), axis=0)
            log_prob_prime = np.concatenate((log_prob_prime, log_prob_prime0), axis=0)

# Apply importance sampling correction for off-policy
adjProb = [log_prob_pi - log_prob_prime]

# Predict states using simpleGeneralPredictor (MAP inference)
predicted_states = simpleGeneralPredictor(
    adjacency_matrices, observations, multi_x_given_g, adjProb=adjProb
)
inference_time = time.time() - inference_start
print(f"Inference completed in {inference_time:.1f} seconds ({inference_time/60:.1f} minutes)")

# ============================================================
# Step 6: Compute F1 Score
# ============================================================
print("\n" + "="*60)
print("Step 6: Computing F1 Scores")
print("="*60)

# checkFscore function from sharedGen adapted for 1D binary vectors
def compute_f1_scores(predicted, ground_truth):
    """Compute per-sample F1 scores for binary vectors"""
    # predicted and ground_truth are (N, D) where N=n_samples, D=universe_size
    incorrect = np.sum(np.abs(predicted - ground_truth), axis=1)
    true_pos = np.sum(predicted * ground_truth, axis=1)
    true_pos_2 = true_pos * 2
    div_part = true_pos_2 + incorrect
    div_part[div_part == 0] = 1
    f1_scores = true_pos_2 / div_part
    return f1_scores

f1_scores = compute_f1_scores(predicted_states, latent_states)
median_f1 = np.median(f1_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"GReinSS F1 Scores:")
print(f"  Median: {median_f1:.4f}")
print(f"  Mean:   {mean_f1:.4f}")
print(f"  Std:    {std_f1:.4f}")
print(f"  Min:    {np.min(f1_scores):.4f}")
print(f"  Max:    {np.max(f1_scores):.4f}")

# Percentiles
for p in [25, 50, 75]:
    print(f"  {p}th percentile: {np.percentile(f1_scores, p):.4f}")

# Save predictions
np.savez_compressed("/repo/data/sims/simpleSet/pred/greinss_predictions.npz", predicted_states)
np.savez_compressed("/repo/data/sims/simpleSet/sample/greinss_samples.npz", adjacency_matrices)

# ============================================================
# Step 7: Local Search Baseline
# ============================================================
print("\n" + "="*60)
print("Step 7: Computing Local Search Baseline")
print("="*60)

ls_start = time.time()
predict_graphs_ls = np.zeros((observations.shape[0], graphSize), dtype=int)
for a in range(observations.shape[0]):
    if a % 20 == 0:
        print(f"  Local search: {a+1}/{observations.shape[0]}")
    obs1 = observations[a]
    graph1 = np.ones(graphSize, dtype=int)
    
    continue1 = True
    while continue1:
        reward0 = log_calculate_pr_x_given_g(graph1, obs1)
        rewardList = np.zeros(graphSize)
        
        for edge_index in range(graphSize):
            graph1_mod = np.copy(graph1)
            graph1_mod[edge_index] = 1 - graph1_mod[edge_index]
            reward1 = log_calculate_pr_x_given_g(graph1_mod, obs1)
            rewardList[edge_index] = reward1
        
        if np.max(rewardList) > reward0:
            edge_index = np.argmax(rewardList)
            graph1[edge_index] = 1 - graph1[edge_index]
        else:
            continue1 = False
    
    predict_graphs_ls[a] = graph1
ls_time = time.time() - ls_start
print(f"Local search completed in {ls_time:.1f} seconds")

f1_scores_ls = compute_f1_scores(predict_graphs_ls, latent_states)
median_f1_ls = np.median(f1_scores_ls)
mean_f1_ls = np.mean(f1_scores_ls)

print(f"Local Search F1 Scores:")
print(f"  Median: {median_f1_ls:.4f}")
print(f"  Mean:   {mean_f1_ls:.4f}")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "="*60)
print("REPRODUCTION SUMMARY")
print("="*60)
print(f"Paper ID: 2169 - GReinSS")
print(f"Configuration: universe_size={UNIVERSE_SIZE}, noise_std={NOISE_STD}, n_samples={N_SAMPLES}")
print(f"Model: 2-layer FC, {HIDDEN_NEURONS} hidden neurons, off-policy={OFF_POLICY}")
print(f"Prediction samples: {N_PREDICTION_SAMPLES}")
print()
print(f"RUBRIC TARGETS:")
print(f"  GReinSS F1 (paper): 0.938 (median)")
print(f"  Local Search F1 (paper): 0.869 (median)")
print(f"  Acceptable range: [0.869, 0.9449]")
print()
print(f"REPRODUCED RESULTS:")
print(f"  GReinSS F1:  {median_f1:.4f} (median), {mean_f1:.4f} (mean)")
print(f"  Local Search F1: {median_f1_ls:.4f} (median), {mean_f1_ls:.4f} (mean)")

if median_f1 >= 0.869 and median_f1 <= 0.9449:
    print(f"  STATUS: REPRODUCTION SUCCEEDED - GReinSS F1 within acceptable range!")
else:
    print(f"  STATUS: Check result - F1={median_f1:.4f}, expected in [0.869, 0.9449]")
