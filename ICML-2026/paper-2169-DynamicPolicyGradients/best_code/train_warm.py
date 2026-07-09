"""Warm-start training: load baseline model, continue training with improvements."""
import numpy as np
import torch
import sys, os, time
sys.path.insert(0, "/repo")
from sharedGen import *

UNIVERSE_SIZE = 1000
NOISE_STD = 0.3
N_SAMPLES = 100
BATCH_SIZE = 1000
LEARNING_RATE = 1e-3
HIDDEN_NEURONS = 50
OFF_POLICY = True
NUM_EPOCHS = 50000

np.random.seed(1)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Generate data
v_size = UNIVERSE_SIZE
n_dict = int(np.floor(np.sqrt(v_size)))
elem_prob = 2.0 / np.sqrt(v_size)
dictionary = np.zeros((n_dict, v_size), dtype=int)
for d in range(n_dict):
    for e in range(v_size):
        if np.random.rand() < elem_prob:
            dictionary[d, e] = 1
latent_states = np.zeros((N_SAMPLES, v_size), dtype=int)
for i in range(N_SAMPLES):
    state = np.zeros(v_size, dtype=int)
    for d in range(n_dict):
        if np.random.rand() < 0.1:
            state = np.maximum(state, dictionary[d])
    latent_states[i] = state
observations = latent_states.astype(np.float64) + np.random.normal(0, NOISE_STD, latent_states.shape)
print(f"Data generated: {observations.shape}", flush=True)

def log_calculate_pr_x_given_g(state, observation):
    diff = 0.5 * (observation - state) ** 2
    return -np.sum(diff) / (NOISE_STD ** 2)

def multi_x_given_g(adjacency_matrices, obs_matrix):
    obs_matrix = torch.tensor(obs_matrix).float().to(adjacency_matrices.device)
    adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))
    obs_matrix = obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))
    diff = 0.5 * torch.sum((adjacency_matrices - obs_matrix) ** 2, axis=2)
    return -1 * diff / (NOISE_STD ** 2)

graphSize = v_size
finalProbSize = 1

def graphRules(graph):
    graphAllow = torch.zeros((graph.shape[0], graphSize + 1), device=device)
    finalProbAllow = torch.zeros((graph.shape[0], finalProbSize), device=device)
    return graphAllow, finalProbAllow

def offPolicyRule(graphList, arange1):
    arange1_mod = arange1 % (2 * observations.shape[0])
    subsetGood = np.argwhere(arange1_mod < observations.shape[0])[:, 0]
    arange1_mod = arange1_mod[subsetGood]
    graphAllow = torch.zeros((graphList.shape[0], graphSize + 1), device=device)
    obsList = np.copy(observations[arange1_mod])
    probRatio = ((obsList - 1) ** 2) - (obsList ** 2)
    probRatio = probRatio * 0.5 / (NOISE_STD ** 2) * (-1) * 1.5
    graphAllow[subsetGood, :-1] = torch.tensor(probRatio, device=device).float()
    finalProbAllow = torch.zeros((graphList.shape[0], finalProbSize), device=device)
    return graphAllow, finalProbAllow

# Load pre-trained model
model_filename = "/repo/greinss_u1000_model.pt"
model = torch.load(model_filename, map_location=device)
model = model.to(device)
model.train()
print(f"Model loaded from {model_filename}, params: {sum(p.numel() for p in model.parameters())}", flush=True)

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
ruleObject.adjacency_matrices = observations

print(f"Warm-start training with off_policy={OFF_POLICY}, lr={LEARNING_RATE}", flush=True)
train_start = time.time()
train_model_off_policy(ruleObject, LEARNING_RATE, BATCH_SIZE, OFF_POLICY,
    num_epochs=NUM_EPOCHS, model_filename=model_filename, rewardType="", giveTrajectory=False)
train_time = time.time() - train_start
print(f"Training completed in {train_time:.1f}s ({train_time/60:.1f}m)", flush=True)
