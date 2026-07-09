import sys, os, time, numpy as np, torch
sys.path.insert(0, '/repo')
from sharedGen import *

UNIVERSE_SIZE = 1000
NOISE_STD = 0.3
N_SAMPLES = 100
N_PREDICTION_SAMPLES = 100000
BATCH_SIZE = 1000
HIDDEN_NEURONS = 50
OFF_POLICY = True

np.random.seed(1)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# ---- Regenerate data with same seeds ----
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
print(f'Data: {observations.shape}, avg elements/state: {np.mean(np.sum(latent_states, axis=1)):.1f}', flush=True)

# ---- Pr(X|S) functions ----
def log_calculate_pr_x_given_g(state, observation):
    diff = 0.5 * (observation - state) ** 2
    return -np.sum(diff) / (NOISE_STD ** 2)

def multi_x_given_g(adjacency_matrices, obs_matrix):
    obs_matrix = torch.tensor(obs_matrix).float().to(adjacency_matrices.device)
    adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))
    obs_matrix = obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))
    diff = 0.5 * torch.sum((adjacency_matrices - obs_matrix) ** 2, axis=2)
    return -1 * diff / (NOISE_STD ** 2)

# ---- Rules ----
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

# ---- Load model ----
model = GraphGeneratorNet(graphSize, 1, HIDDEN_NEURONS, endingBias=0)
model_filename = '/repo/greinss_u1000_model.pt'
try:
    model = torch.load(model_filename, map_location=device)
    print(f'Model loaded from {model_filename}', flush=True)
except:
    print(f'ERROR: Model file {model_filename} not found or invalid', flush=True)
    sys.exit(1)

model = model.to(device)
model.eval()

# ---- Setup ruleObject ----
ruleObject = gClass()
ruleObject.log_calculate_pr_x_given_g = log_calculate_pr_x_given_g
ruleObject.graphSize = graphSize
ruleObject.observations_batch = observations
ruleObject.batchSize = BATCH_SIZE
ruleObject.learning_rate = 1e-3
ruleObject.model = model
ruleObject.graphRules = graphRules
ruleObject.offPolicyRule = offPolicyRule
ruleObject.multi_x_given_g = multi_x_given_g
ruleObject.adjacency_matrices = observations

# ---- Inference ----
print(f'Starting inference with {N_PREDICTION_SAMPLES} samples...', flush=True)
sampleSize = N_PREDICTION_SAMPLES
batch_size_sampling = 1000
numBatch = sampleSize // batch_size_sampling

inf_start = time.time()
with torch.no_grad():
    for a in range(numBatch):
        if a % 20 == 0:
            print(f'  Sampling {a+1}/{numBatch} ({(time.time()-inf_start)/60:.1f}m elapsed)', flush=True)
        adj0, lpi0, lpp0, _ = generate_graph_batch_with_modified_policy(model, ruleObject, OFF_POLICY, batch_size_sampling)
        adj0, lpi0, lpp0 = adj0.cpu().numpy(), lpi0.cpu().numpy(), lpp0.cpu().numpy()
        if a == 0:
            adj_matrices, log_pi, log_pp = adj0.copy(), lpi0.copy(), lpp0.copy()
        else:
            adj_matrices = np.concatenate((adj_matrices, adj0), axis=0)
            log_pi = np.concatenate((log_pi, lpi0), axis=0)
            log_pp = np.concatenate((log_pp, lpp0), axis=0)

adjProb = [log_pi - log_pp]
predicted = simpleGeneralPredictor(adj_matrices, observations, multi_x_given_g, adjProb=adjProb)
inf_time = time.time() - inf_start
print(f'Inference done in {inf_time:.1f}s ({inf_time/60:.1f}m)', flush=True)

# ---- F1 ----
def compute_f1(pred, gt):
    incorrect = np.sum(np.abs(pred - gt), axis=1)
    tp = np.sum(pred * gt, axis=1)
    tp2 = tp * 2
    div = tp2 + incorrect
    div[div == 0] = 1
    return tp2 / div

f1s = compute_f1(predicted, latent_states)
print(f'\n===== GReinSS F1 (|U|={UNIVERSE_SIZE}, sigma={NOISE_STD}) =====', flush=True)
print(f'Median: {np.median(f1s):.4f}', flush=True)
print(f'Mean:   {np.mean(f1s):.4f}', flush=True)
print(f'Std:    {np.std(f1s):.4f}', flush=True)
for p in [25, 50, 75]:
    print(f'  {p}th percentile: {np.percentile(f1s, p):.4f}', flush=True)
print(f'Min: {np.min(f1s):.4f}, Max: {np.max(f1s):.4f}', flush=True)

# ---- Local Search Baseline ----
print(f'\nRunning local search baseline...', flush=True)
ls_start = time.time()
predict_ls = np.zeros((observations.shape[0], graphSize), dtype=int)
for a in range(observations.shape[0]):
    if a % 20 == 0:
        print(f'  LS {a+1}/{observations.shape[0]} ({(time.time()-ls_start)/60:.1f}m)', flush=True)
    obs1 = observations[a]
    graph1 = np.ones(graphSize, dtype=int)
    while True:
        r0 = log_calculate_pr_x_given_g(graph1, obs1)
        rl = np.zeros(graphSize)
        for e in range(graphSize):
            gm = np.copy(graph1)
            gm[e] = 1 - gm[e]
            rl[e] = log_calculate_pr_x_given_g(gm, obs1)
        if np.max(rl) > r0:
            graph1[np.argmax(rl)] = 1 - graph1[np.argmax(rl)]
        else:
            break
    predict_ls[a] = graph1

ls_time = time.time() - ls_start
f1_ls = compute_f1(predict_ls, latent_states)
print(f'Local Search: Median={np.median(f1_ls):.4f}, Mean={np.mean(f1_ls):.4f}', flush=True)

# ---- Save ----
np.savez_compressed('/repo/greinss_u1000_pred.npz', predicted)
np.savez_compressed('/repo/greinss_u1000_ls.npz', predict_ls)

# ---- Summary ----
print(f'\n===== SUMMARY =====', flush=True)
print(f'Paper targets:', flush=True)
print(f'  GReinSS F1: 0.938 (median)', flush=True)
print(f'  Local Search F1: 0.869 (median)', flush=True)
print(f'  Acceptable range: [0.869, 0.9449]', flush=True)
print(f'', flush=True)
print(f'Reproduced:', flush=True)
print(f'  GReinSS F1: {np.median(f1s):.4f} (median)', flush=True)
print(f'  Local Search F1: {np.median(f1_ls):.4f} (median)', flush=True)

if np.median(f1s) >= 0.869:
    print(f'STATUS: GReinSS F1 above lower bound - REPRODUCTION SUCCEEDED!', flush=True)
elif np.median(f1s) >= 0.8:
    print(f'STATUS: GReinSS F1 {np.median(f1s):.4f} - partial reproduction, training may need more epochs', flush=True)
else:
    print(f'STATUS: F1 {np.median(f1s):.4f} below expected range', flush=True)

print(f'\nTotal eval time: {(time.time()-inf_start+ls_time)/60:.1f}m', flush=True)
