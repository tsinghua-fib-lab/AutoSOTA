import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
from scipy.special import logsumexp
import time

import math

# Parameters
#num_simulations = 20
#num_data_points = 100
#num_nodes = 10  # Reduced to 10 nodes for faster runtimes
#num_nodes = 3
#num_paths_per_graph = 10



import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

np.random.seed(1)
torch.manual_seed(0)




def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data



def uniqueValMaker(X):

    _, vals1 = np.unique(X[:, 0], return_inverse=True)

    for a in range(1, X.shape[1]):

        #vals2 = np.copy(X[:, a])
        #vals2_unique, vals2 = np.unique(vals2, return_inverse=True)
        vals2_unique, vals2 = np.unique(X[:, a], return_inverse=True)

        vals1 = (vals1 * vals2_unique.shape[0]) + vals2
        _, vals1 = np.unique(vals1, return_inverse=True)

    return vals1


# Neural network for graph generation
class GraphGeneratorNet(nn.Module):
    def __init__(self, graphSize, finalProbSize, Nhidden, endingBias=-7):
        super(GraphGeneratorNet, self).__init__()
        self.input_size = graphSize
        self.hidden_size = Nhidden
        self.finalProbSize = finalProbSize
        self.output_size = graphSize + 1  # All edges + stop action

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        
        #self.fc_finalProb1 = nn.Linear(self.input_size, 30)
        #self.fc_finalProb2 = nn.Linear(30, self.finalProbSize)

        self.fc_finalProb1 = nn.Linear(self.input_size, Nhidden)
        self.fc_finalProb2 = nn.Linear(Nhidden, self.finalProbSize)

        self.endingBias = endingBias

        #self.fc_finalProb = nn.Linear(self.hidden_size, self.finalProbSize)

        #print ("A")
        #quit()


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        logits = self.fc2(x)
        
        #logits[:, -1] = logits[:, -1] - 7
        logits[:, -1] = logits[:, -1] + self.endingBias
        return logits
    
    def finalProb(self, x):
        x = F.leaky_relu(self.fc_finalProb1(x))
        #print (x.shape)
        logits = self.fc_finalProb2(x)

        #scale = logits0[:, :1]
        #logits_other = nn.LogSoftmax(dim=1)(logits0[:, 1:])

        #logits = logits_other + scale
        #x = F.leaky_relu(self.fc1(x))
        #x = F.leaky_relu(self.fc_finalProb1(x))
        #logits =  self.fc_finalProb2(x)
        #logits = self.fc_finalProb(x)
        return logits




class GeneratorNet(nn.Module):
    def __init__(self, graphSize, Nhidden):
        super(GeneratorNet, self).__init__()
        self.input_size = graphSize
        self.hidden_size = Nhidden
        self.output_size = graphSize + 1 
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc_finalProb1 = nn.Linear(self.input_size, Nhidden)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        logits = self.fc2(x)
        logits[:, -1] = logits[:, -1]
        return logits
    


def checkFscore(predicted_graphs, adjacency_matrices):

    incorrect = np.sum(np.abs(predicted_graphs - adjacency_matrices), axis=(1, 2))

    truePos = np.sum(predicted_graphs * adjacency_matrices, axis=(1, 2))

    truePos_2 = truePos * 2

    divPart = truePos_2 + incorrect 

    divPart[divPart == 0] = 1

    Fscore = truePos_2/ divPart

    return Fscore



def log_start_end_prob(adjacency_matrix):

    num_nodes = adjacency_matrix.shape[1]
    M_exit = np.zeros((num_nodes, num_nodes))
    M_next = np.zeros((num_nodes, num_nodes))

    for node in range(num_nodes):
        out_edges = np.where(adjacency_matrix[node] == 1)[0]
        total_options = len(out_edges) + 1
        M_exit[node, node] = 1 / total_options
        for succ in out_edges:
            M_next[node, succ] = 1 / total_options

    I = np.eye(num_nodes)
    M_se = np.linalg.inv(I - M_next) @ M_exit

    return M_se


# Function to calculate log Pr(X | G)
def sim1_log_calculate_pr_x_given_g(adjacency_matrix, observations):


    if len(adjacency_matrix.shape) == 1:
        #print (adjacency_matrix.shape)
        num_node = int(np.floor( adjacency_matrix.shape[0] ** 0.5 )) + 1
        #print (num_node)
        #quit()

        assert adjacency_matrix.shape[0] == (num_node - 1) * num_node

        eye1 = np.eye(num_node)
        arg1 = np.argwhere(eye1 == 0)
        adjacency_matrix0 = np.zeros((num_node, num_node), dtype=int)

        adjacency_matrix0[arg1[:, 0], arg1[:, 1]] = adjacency_matrix
        adjacency_matrix = adjacency_matrix0

    

    
    M_se = log_start_end_prob(adjacency_matrix)

    M_se_bad = np.zeros(M_se.shape)
    M_se_bad[M_se == 0] = 1
    bad1 = np.sum(observations * M_se_bad)

    if bad1 > 0:
        log_prob = - np.inf
    else:


        M_se[M_se == 0] = 1e-10
        M_se_log = np.log(M_se)


        log_prob = np.sum(M_se_log * observations)

    return log_prob



def sim1_multi_x_given_g(adjacency_matrices, observations_batch):

    

    pr_x_given_g_matrix = torch.tensor([
            [sim1_log_calculate_pr_x_given_g(adjacency_matrices[j].detach().numpy(), observations_batch[i])
             for i in range(observations_batch.shape[0])] for j in range(adjacency_matrices.shape[0])
        ], dtype=torch.float32)

    return pr_x_given_g_matrix


def sim1_fast_multi(adjacency_matrices, observations_batch):

    pr_x_given_g_matrix = np.zeros((adjacency_matrices.shape[0], observations_batch.shape[0]  ))

    M_se_list = np.zeros(observations_batch.shape)

    for a in range(adjacency_matrices.shape[0]):
        adjacency_matrix = adjacency_matrices[a]

        if len(adjacency_matrix.shape) == 1:
            num_node = int(np.floor( adjacency_matrix.shape[0] ** 0.5 )) + 1

            assert adjacency_matrix.shape[0] == (num_node - 1) * num_node

            eye1 = np.eye(num_node)
            arg1 = np.argwhere(eye1 == 0)
            adjacency_matrix0 = np.zeros((num_node, num_node), dtype=int)

            adjacency_matrix0[arg1[:, 0], arg1[:, 1]] = adjacency_matrix
            adjacency_matrix = adjacency_matrix0



        M_se0 = log_start_end_prob(adjacency_matrix)
        #M_se_list[a] = M_se

        M_se = M_se0.reshape((1, M_se0.shape[0], M_se0.shape[1]))


        M_se_bad = np.zeros(M_se.shape)
        M_se_bad[M_se == 0] = 1
        bad1 = np.sum(observations_batch * M_se_bad, axis=(1, 2))

        #print (bad1[0])
        #quit()

        pr_x_given_g_matrix[a, bad1 > 0] = -np.inf 
        
        

        M_se[M_se == 0] = 1e-10
        M_se_log = np.log(M_se)

        log_prob_good = np.sum(M_se_log * observations_batch[bad1 <= 0], axis=(1, 2))
        

        pr_x_given_g_matrix[a, bad1 <= 0] = log_prob_good



        if False:
            for b in range(observations_batch.shape[0]):

                M_se = np.copy(M_se0)

                observations = observations_batch[b]

                M_se_bad = np.zeros(M_se.shape)
                M_se_bad[M_se == 0] = 1
                bad1 = np.sum(observations * M_se_bad)

                if bad1 > 0:
                    log_prob = - np.inf
                else:


                    M_se[M_se == 0] = 1e-10
                    M_se_log = np.log(M_se)


                    log_prob = np.sum(M_se_log * observations)

                    pr_x_given_g_matrix[a, b] = log_prob
    
    pr_x_given_g_matrix = torch.tensor(pr_x_given_g_matrix).float()
    
    return pr_x_given_g_matrix






def causal_x_given_g(adjacency_matrix, observations):

    #observations = torch.tensor(observations_np).float()

    #print (adjacency_matrix.shape)
    #print (observations.shape)

    if len(adjacency_matrix.shape) == 1:
        num_node = int(adjacency_matrix.shape[0] ** 0.5)

        adjacency_matrix = adjacency_matrix.reshape((num_node, num_node))



    X_data = observations[0]
    Y_data = observations[1]

    #print (X_data.shape, adjacency_matrix.shape)

    Y_pred = np.matmul( X_data, adjacency_matrix )

    error = np.sum((Y_pred - Y_data) ** 2) * -1

    return error


def causal_multi_x_given_g(adjacency_matrices, observations_batch):

    pr_x_given_g_matrix = torch.tensor([
            [causal_x_given_g(adjacency_matrices[j].detach().numpy(), observations_batch[i])
             for i in range(observations_batch.shape[0])] for j in range(adjacency_matrices.shape[0])
        ], dtype=torch.float32)

    return pr_x_given_g_matrix



def causal_fast_multi(adjacency_matrices, observations_batch):


    doVerify = False

    if doVerify:
        old_values = causal_multi_x_given_g(adjacency_matrices, observations_batch)


    if len(adjacency_matrices.shape) == 2:
        num_node = int(adjacency_matrices.shape[1] ** 0.5)

        adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], num_node, num_node))
    

    X_data = observations_batch[:, 0]
    X_data = X_data.reshape(( 1, X_data.shape[0], X_data.shape[1], X_data.shape[2] ))
    Y_data = observations_batch[:, 1]
    Y_data = Y_data.reshape(( 1, Y_data.shape[0], Y_data.shape[1], Y_data.shape[2] ))

    adjacency_matrices = adjacency_matrices.reshape(( adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1], adjacency_matrices.shape[2] ))

    #print (adjacency_matrices.shape)
    #print (X_data.shape)
    #quit()

    Y_pred = np.matmul( X_data, adjacency_matrices )

    error = torch.sum(  (Y_pred - Y_data) ** 2 ,  axis=(2, 3)) * -1

    if doVerify:
        assert torch.mean(torch.abs(error - old_values)) < 0.001

    return error
    


def raw_generate_graph_batch(model, graphSize, batch_size):
    
    adjacency_matrices = torch.zeros((batch_size, graphSize), requires_grad=False)
    trajectories = np.zeros((batch_size, graphSize), dtype=int) - 1
    
    
    log_prob_pi_trajectories = torch.zeros(batch_size)

    #edge_indices = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    #stop_index = len(edge_indices)
    stop_index = graphSize

    #finished = torch.zeros(batch_size, dtype=torch.bool)
    finished_new = np.zeros(batch_size, dtype=int)

    iter1 = -1
    while np.mean(finished_new) < 1:
        iter1 += 1 

        argNotDone = np.argwhere(finished_new == 0)[:, 0]
        adjacency_tensors = adjacency_matrices.reshape(batch_size, -1)[argNotDone]
        logits = model(adjacency_tensors)

        # Exclude already existing edges by setting their logits to -infinity
        for idx in range(argNotDone.shape[0]):
            idx2 = argNotDone[idx]
            #for edge_idx, (i, j) in enumerate(edge_indices):
            for edge_idx in range(graphSize):
                if adjacency_matrices[idx2, edge_idx] == 1:
                    logits[idx, edge_idx] = -float('inf')  # Set logit to -inf for existing edge

        

        #probs_pi = F.softmax(logits, dim=1)  # Original policy probabilities
        log_probs_pi = nn.LogSoftmax(dim=1)(logits)
        probs_pi = torch.exp(log_probs_pi)

        m_pi = torch.distributions.Categorical(probs_pi)
        actions_pi = m_pi.sample()

        
        log_probs_pi = log_probs_pi.gather(1, actions_pi.unsqueeze(1)).squeeze(1)
        log_prob_pi_trajectories[argNotDone] += log_probs_pi
        

        for idx in range(argNotDone.shape[0]):
            idx2 = argNotDone[idx]
            if actions_pi[idx].item() == stop_index:
                finished_new[idx2] = 1
            else:
                #i, j = edge_indices[actions_pi[idx].item()]
                action_now = actions_pi[idx].item()
                adjacency_matrices = adjacency_matrices.clone()

                adjacency_matrices[idx2, action_now] = 1
                trajectories[idx2, action_now] = iter1
                #adjacency_tensors = adjacency_matrices.reshape(batch_size, -1)

    return adjacency_matrices, log_prob_pi_trajectories, trajectories




# Function to generate graphs using off-policy sampling with pi'
def generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batch_size):

    observations_batch = ruleObject.observations_batch
    graphSize = ruleObject.graphSize
    graphRules = ruleObject.graphRules
    offPolicyRule = ruleObject.offPolicyRule
    #finalProbSize = ruleObject.finalProbSize

    #num_nodes = observations_batch.shape[1]
    #batch_size = observations_batch.shape[0] * dupGen
    #batch_size = 

    device1 = next(model.parameters()).device

    
    adjacency_matrices = torch.zeros((batch_size, graphSize), requires_grad=False)
    trajectories = np.zeros((batch_size, graphSize), dtype=int) - 1

    adjacency_matrices = adjacency_matrices.to(device1)
    
    
    log_prob_pi_trajectories = torch.zeros(batch_size).to(device1)
    log_prob_pi_prime_trajectories = torch.zeros(batch_size).to(device1)

    #edge_indices = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    #stop_index = len(edge_indices)
    stop_index = graphSize

    #finished = torch.zeros(batch_size, dtype=torch.bool)
    finished_new = np.zeros(batch_size, dtype=int)

    

    #print ("BANANA")
    #print (offPolicy)
    
    timeList = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    iter1 = -1
    while np.mean(finished_new) < 1:
        iter1 += 1 

        #print (iter1)

        time1 = time.time()

        time2 = time.time()

        argNotDone = np.argwhere(finished_new == 0)[:, 0]
        adjacency_tensors = adjacency_matrices.reshape(batch_size, -1)[argNotDone]

        #print ('argNotDone', argNotDone)

        #print ('adjacency_tensors[14]', adjacency_tensors[14])

        timeList[1] += (time.time() - time2)

        time3 = time.time()

        with torch.no_grad():
            graphAllow, _ = graphRules(adjacency_tensors)

        timeList[2] += (time.time() - time3)

        time4 = time.time()


        if offPolicy:
            #print (offPolicy)
            #quit()
            #graphAllow_offPolicy, _ = offPolicyRule(adjacency_tensors, argNotDone % observations_batch.shape[0] )
            graphAllow_offPolicy, _ = offPolicyRule(adjacency_tensors, argNotDone )

        
        #print ("A")
        #print (adjacency_tensors[0])

        

        #print ("Bananan")

        #print (adjacency_tensors)

        logits = model(adjacency_tensors)#.to('cpu')

        


        
        #print (logits)

        #time4 = time.time()


        logits[:, :-1] = logits[:, :-1].masked_fill(adjacency_tensors == 1, -float('inf'))



        logits = logits + graphAllow#.to(device1)
        
        logits_clone = logits.clone()  # Clone logits to prevent in-place modification

        


        

        if False:
            # Exclude already existing edges by setting their logits to -infinity
            for idx in range(argNotDone.shape[0]):
                idx2 = argNotDone[idx]
                for edge_idx in range(graphSize):#, (i, j) in enumerate(edge_indices):
                    if adjacency_matrices[idx2, edge_idx] == 1:
                        assert logits[idx, edge_idx] == -float('inf') 
                        logits[idx, edge_idx] = -float('inf')  # Set logit to -inf for existing edge
                        logits_clone[idx, edge_idx] = -float('inf')  # Apply the same for modified policy
                        

                current_adj_matrix = adjacency_matrices[idx2].detach().numpy()
                log_prob1 = log_calculate_pr_x_given_g(current_adj_matrix, observations_batch[idx2]) 

                #log_prob2 = log_calculate_pr_x_given_g(adjacency_tensors[idx], observations_batch[idx2]) 
                #print (idx)
                #print (log_prob1, log_prob2)

                if log_prob1 == -np.inf:
                    logits_clone[idx, stop_index] = -float('inf')  # Prevent stop action


        
        #print (offPolicy)
        #quit()
        if offPolicy:
            logits_clone = logits_clone + graphAllow_offPolicy#.to(device1)

        #print ('graphAllow[14]', graphAllow[14])
        #print ('graphAllow_offPolicy[14]', graphAllow_offPolicy[14])
        
        #print ("Q")
        #print (logits_clone[0])

        #print (logits_clone)
        #print ('observations_batch', np.argwhere(  observations_batch[0] == 1)[:, 0]  )
        #print (logits_clone[0, :-1][  observations_batch[0] == 0 ]  )
        #assert torch.max(logits_clone[0, :-1][  observations_batch[0] == 0 ] ) < -1000
        #quit()

        #probs_pi = F.softmax(logits, dim=1)  # Original policy probabilities
        log_probs_pi = nn.LogSoftmax(dim=1)(logits)
        
        probs_pi_prime = F.softmax(logits_clone, dim=1)  # Modified policy probabilities


        timeList[3] += time.time() - time4

        time5 = time.time()


        #time5 = time.time()
            
        if True:

            #print (probs_pi_prime)
            #m_pi_prime = torch.distributions.Categorical(probs_pi_prime)

            
            timeList[4] += time.time() - time5
            time6 = time.time()

            if False:
                actions_pi_prime = m_pi_prime.sample()
            else:
                u = torch.rand_like(logits_clone)                  # uniform(0,1)
                gumbel = -torch.log(-torch.log(u))                 # Gumbel(0,1)
                actions_pi_prime = (logits_clone + gumbel).argmax(dim=1)

            
            timeList[5] += time.time() - time6
            time7 = time.time()

            #time3 = time.time()
            #log_probs_pi_prime = m_pi_prime.log_prob(actions_pi_prime)
            #timeList[2] += time.time() - time3

            log_probs_pi_prime = torch.log(probs_pi_prime).gather(1, actions_pi_prime.unsqueeze(1)).squeeze(1)


            timeList[6] += time.time() - time7
          

        
        
        


        #print (log_prob_pi_prime_trajectories)
        #quit()

        #print ('log_probs_pi', torch.exp(log_probs_pi))

        log_prob_pi_prime_trajectories[argNotDone] += log_probs_pi_prime

        

        # Importance sampling correction
        #log_probs_pi = torch.log(probs_pi.gather(1, actions_pi_prime.unsqueeze(1)).squeeze(1))
        log_probs_pi = log_probs_pi.gather(1, actions_pi_prime.unsqueeze(1)).squeeze(1)
        log_prob_pi_trajectories[argNotDone] += log_probs_pi

        


        

        if False:

            for idx in range(argNotDone.shape[0]):
                idx2 = argNotDone[idx]
                if actions_pi_prime[idx].item() == stop_index:
                    finished_new[idx2] = 1
                else:
                    #i, j = edge_indices[actions_pi_prime[idx].item()]
                    actionIndex = actions_pi_prime[idx].item()

                    #time3 = time.time()
                    #adjacency_matrices = adjacency_matrices.clone()
                    #timeList[2] += time.time() - time3

                    adjacency_matrices[idx2, actionIndex] = 1
                    trajectories[idx2, actionIndex] = iter1
                    #adjacency_tensors = adjacency_matrices.reshape(batch_size, -1)

        else:

            

            stop_mask = (actions_pi_prime == stop_index)                  # [M]
            # global batch indices that just finished
            argDoneNow = argNotDone[stop_mask.cpu().numpy()]              # numpy indices
            finished_new[argDoneNow] = 1                                  # numpy array

            # 2) Entries that did NOT choose "stop"
            cont_mask = ~stop_mask                                        # [M]
            # local indices within argNotDone
            cont_local_idx = torch.nonzero(cont_mask, as_tuple=False).squeeze(1)  # [K]

            

            if cont_local_idx.numel() > 0:
                # Global batch indices (convert argNotDone to tensor on same device)
                argNotDone_tensor = torch.from_numpy(argNotDone).to(actions_pi_prime.device)
                batch_idx = argNotDone_tensor[cont_local_idx]             # [K]

                # Action indices for those batches
                action_idx = actions_pi_prime[cont_local_idx]             # [K], dtype long

                # 3) Update adjacency_matrices (torch)
                with torch.no_grad():
                    adjacency_matrices[batch_idx, action_idx] = 1

                # 4) Update trajectories (numpy) – need CPU indices
                batch_idx_np  = batch_idx.cpu().numpy()
                action_idx_np = action_idx.cpu().numpy()
                trajectories[batch_idx_np, action_idx_np] = iter1
            

            
            

            

        #timeList[2] += time.time() - time3

        #timeList[1] += time.time() - time2
        #if 0 in argNotDone:
        #    if iter1 <= np.sum(observations_batch[0] ):
        #        print ("A")
        #        print (actions_pi_prime[0])
        #        print (trajectories[0])

        #print (timeList[:6])

        timeList[0] += time.time() - time1 


    #print ('timeList', timeList)
    #quit()

    #print (timeList[:2])
    #quit()



    #print (np.argwhere(  observations_batch[0] == 1)[:, 0] )
    #print (np.argsort(trajectories[0])[:10]  )
    #quit()
    
    

    return adjacency_matrices, log_prob_pi_trajectories, log_prob_pi_prime_trajectories, trajectories






def mini_spread_graph_batch(model, ruleObject, offPolicy, minProb):

    observations_batch = ruleObject.observations_batch
    graphSize = ruleObject.graphSize
    graphRules = ruleObject.graphRules
    offPolicyRule = ruleObject.offPolicyRule
    finalProbSize = ruleObject.finalProbSize

    #num_nodes = observations_batch.shape[1]
    #batch_size = observations_batch.shape[0] * dupGen
    #batch_size = 

    
    adjacency_matrices = torch.zeros((1, graphSize), requires_grad=False)
    trajectories = np.zeros((1, graphSize), dtype=int) - 1
    log_prob_pi_trajectories = torch.zeros(1)
    log_prob_pi_prime_trajectories = torch.zeros(1)


    maxSize = int(1.0 / minProb) + 1
    indicesDone = 0
    adjacency_matrices_done = torch.zeros((maxSize, graphSize), requires_grad=False)
    trajectories_done = np.zeros((maxSize, graphSize), dtype=int) - 1
    log_prob_pi_trajectories_done = torch.zeros(maxSize)
    log_prob_pi_prime_trajectories_done = torch.zeros(maxSize)

    
    stop_index = graphSize
    #finished_new = np.zeros(1, dtype=int)

    

    iter1 = -1
    while adjacency_matrices.shape[0] >= 1:
        iter1 += 1 

        #argNotDone = np.argwhere(finished_new == 0)[:, 0]
        #print ('adjacency_matrices', adjacency_matrices.shape)
        adjacency_tensors = adjacency_matrices.reshape(adjacency_matrices.shape[0], -1)#[argNotDone]

        #print ('adjacency_tensors[14]', adjacency_tensors[14])

        graphAllow, _ = graphRules(adjacency_tensors)

        #print ('graphAllow end', torch.max(graphAllow[:, -1]))
        

        logits = model(adjacency_tensors)

        logits[:, :-1] = logits[:, :-1].masked_fill(adjacency_tensors == 1, -float('inf'))

        logits = logits + graphAllow
        logits_clone = logits.clone()  # Clone logits to prevent in-place modification

        

        #if offPolicy:
        #    logits_clone = logits_clone + graphAllow_offPolicy

        log_probs_pi = nn.LogSoftmax(dim=1)(logits)
        probs_pi_prime = F.softmax(logits_clone, dim=1)  # Modified policy probabilities

        argIssue = np.argwhere( np.isnan(probs_pi_prime.data.numpy() ))
        if argIssue.shape[0] > 0:
            print ('argIssue', argIssue)
            

        m_pi_prime = torch.distributions.Categorical(probs_pi_prime)

        prob_pi_prime_trajectories_temp = torch.exp(log_prob_pi_prime_trajectories).reshape((-1, 1)) * probs_pi_prime
        #print ('sizes', prob_pi_prime_trajectories_temp.shape, stop_index)
        #print (  'max end', np.max( prob_pi_prime_trajectories_temp[:, -1].data.numpy() ))
        #print (  'max end2', np.max( probs_pi_prime[:, -1].data.numpy() ))

        

        highProbChoices = np.argwhere(prob_pi_prime_trajectories_temp.data.numpy() > minProb)

        #119
        #print (highProbChoices)
        #quit()

        #print (np.max(prob_pi_prime_trajectories_temp.data.numpy()))
        #print (minProb)

        log_probs_pi_prime = torch.log(probs_pi_prime)[highProbChoices[:, 0], highProbChoices[:, 1]]

        log_prob_pi_prime_trajectories = log_prob_pi_prime_trajectories[highProbChoices[:, 0]] + log_probs_pi_prime

        #if log_prob_pi_prime_trajectories.shape[0] > 0:
        #    print ('miniMax',  np.max(np.exp( log_prob_pi_prime_trajectories.data.numpy()  )))

        # Importance sampling correction
        log_probs_pi = log_probs_pi[highProbChoices[:, 0], highProbChoices[:, 1]]
        log_prob_pi_trajectories = log_prob_pi_trajectories[highProbChoices[:, 0]] +  log_probs_pi


        


        adjacency_matrices = adjacency_matrices[highProbChoices[:, 0]]
        trajectories = trajectories[highProbChoices[:, 0]]

        

        argDone = np.argwhere(highProbChoices[:, 1] == stop_index)
        #print ('argDone', argDone.shape)
        if argDone.shape[0] >= 1:
            argDone = argDone[:, 0]
            sizeDone = argDone.shape[0]
            #print ('sizeDone')
            #indicesDone
            #argDone
            adjacency_matrices_done[indicesDone:indicesDone+sizeDone] = adjacency_matrices[argDone]
            trajectories_done[indicesDone:indicesDone+sizeDone] = trajectories[argDone]
            log_prob_pi_trajectories_done[indicesDone:indicesDone+sizeDone] = log_prob_pi_trajectories[argDone]
            log_prob_pi_prime_trajectories_done[indicesDone:indicesDone+sizeDone] = log_prob_pi_prime_trajectories[argDone]
            indicesDone = indicesDone + sizeDone



        argContinue = np.argwhere(highProbChoices[:, 1] != stop_index)[:, 0]
        #print ('argContinue', argContinue.shape)
        

        adjacency_matrices[argContinue, highProbChoices[argContinue, 1]] = 1
        trajectories[argContinue, highProbChoices[argContinue, 1]] = iter1

        


        
        adjacency_matrices = adjacency_matrices[argContinue]
        trajectories = trajectories[argContinue]
        log_prob_pi_trajectories = log_prob_pi_trajectories[argContinue]
        log_prob_pi_prime_trajectories = log_prob_pi_prime_trajectories[argContinue]

        

        #if 1 in adjacency_matrices[:, 119]:
        #    print ("HI")
        #    quit()



    adjacency_matrices_done = adjacency_matrices_done[:indicesDone]
    trajectories_done = trajectories_done[:indicesDone]
    log_prob_pi_trajectories_done = log_prob_pi_trajectories_done[:indicesDone]
    log_prob_pi_prime_trajectories_done = log_prob_pi_prime_trajectories_done[:indicesDone]
    
    
    

    return adjacency_matrices_done, log_prob_pi_trajectories_done, log_prob_pi_prime_trajectories_done, trajectories_done




def spread_graph_batch(model, ruleObject, offPolicy, batch_size):

    

    #minProb = 0.2 / float(batch_size)
    minProb = 0.05 / float(batch_size)
    #minProb = 0.2 / float(batch_size)
    adjacency_matrices_spread, log_prob_pi_trajectories_spread, log_prob_pi_prime_trajectories_spread, trajectories_spread = mini_spread_graph_batch(model, ruleObject, offPolicy, minProb)
    if adjacency_matrices_spread.shape[0] > batch_size / 2:
        print ("Excess")
        minProb = 1.0 / float(batch_size)
        adjacency_matrices_spread, log_prob_pi_trajectories_spread, log_prob_pi_prime_trajectories_spread, trajectories_spread = mini_spread_graph_batch(model, ruleObject, offPolicy, minProb)
    
    batchRemain = batch_size - adjacency_matrices_spread.shape[0]
    adjacency_matrices_sample, log_prob_pi_trajectories_sample, log_prob_pi_prime_trajectories_sample, trajectories_sample = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batchRemain)

    if adjacency_matrices_spread.shape[0] == 0:
        adjacency_matrices, log_prob_pi_trajectories, log_prob_pi_prime_trajectories, trajectories = adjacency_matrices_sample, log_prob_pi_trajectories_sample, log_prob_pi_prime_trajectories_sample, trajectories_sample
    else:

        #plt.imshow(adjacency_matrices_spread.data.numpy())
        #plt.show()

        #plt.imshow(adjacency_matrices_sample[:100].data.numpy())
        #plt.show()

        #print (log_prob_pi_trajectories_spread)
        spreadProbs = np.exp(log_prob_pi_trajectories_spread.data.numpy())
        print ('total spreaded', np.sum(spreadProbs), spreadProbs.shape)
        #quit()

        inverse1 = np.concatenate((  adjacency_matrices_spread.data.numpy(), adjacency_matrices_sample.data.numpy() ), axis=0 )
        inverse1 = uniqueValMaker(inverse1)
        inverse_spread, inverse_sample = inverse1[:adjacency_matrices_spread.shape[0]],  inverse1[adjacency_matrices_spread.shape[0]:]
        #print ('intersect', np.intersect1d(inverse_spread, inverse_sample))
        argGood = np.argwhere( np.isin( inverse_sample, inverse_spread ) == False )[:, 0]

        if False: #TODO REMOVE!!!!
            argGood = np.zeros(0, dtype=int)

        #print (adjacency_matrices_sample.shape, inverse_sample.shape, argGood.shape)
        #quit()
        originalSize = adjacency_matrices_sample.shape[0]
        adjacency_matrices_sample, log_prob_pi_trajectories_sample, log_prob_pi_prime_trajectories_sample, trajectories_sample = adjacency_matrices_sample[argGood], log_prob_pi_trajectories_sample[argGood], log_prob_pi_prime_trajectories_sample[argGood], trajectories_sample[argGood]

        print ('A', originalSize, argGood.shape )

        #TODO, the generating part shouldn't generate the "common" ones.
        #print ("todo implement")
        #samplingRatio = adjacency_matrices_spread.shape[0] / batchRemain
        samplingRatio = -1 * np.log( (adjacency_matrices_spread.shape[0] + batchRemain) )
        log_prob_pi_prime_trajectories_spread[:] = samplingRatio

        adjacency_matrices = torch.cat((adjacency_matrices_spread, adjacency_matrices_sample), axis=0)
        log_prob_pi_trajectories = torch.cat((log_prob_pi_trajectories_spread, log_prob_pi_trajectories_sample), axis=0)
        log_prob_pi_prime_trajectories = torch.cat((log_prob_pi_prime_trajectories_spread, log_prob_pi_prime_trajectories_sample), axis=0)
        #finalProb = torch.cat((finalProb_spread, finalProb_sample), axis=0)
        trajectories = np.concatenate((trajectories_spread, trajectories_sample), axis=0)

    
    

    return adjacency_matrices, log_prob_pi_trajectories, log_prob_pi_prime_trajectories, trajectories



def giveTrajectoryProbs(model, num_nodes, observations_batch, trajectories):

    batch_size = trajectories.shape[0]

    numSteps = np.max(trajectories, axis=(1, 2))
    maxStep = int(np.max(numSteps))


    adjacency_matrices = torch.zeros((batch_size, num_nodes, num_nodes), requires_grad=False)
    
    log_prob_pi_trajectories = torch.zeros(batch_size)
    log_prob_pi_prime_trajectories = torch.zeros(batch_size)

    edge_indices = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_indices_array =  np.array([  [i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j] )
    stop_index = len(edge_indices)

    #finished = torch.zeros(batch_size, dtype=torch.bool)
    finished_new = np.zeros(batch_size, dtype=int)

    for iter1 in range(maxStep + 1):
        
        #print (numSteps)

        argNotDone = np.argwhere(numSteps >= iter1 - 1)[:, 0]
        adjacency_tensors = adjacency_matrices.reshape(batch_size, -1)[argNotDone]
        logits = model(adjacency_tensors)
        logits_clone = logits.clone()  # Clone logits to prevent in-place modification
        
        # Exclude already existing edges by setting their logits to -infinity
        for idx in range(argNotDone.shape[0]):
            idx2 = argNotDone[idx]
            for edge_idx, (i, j) in enumerate(edge_indices):
                if adjacency_matrices[idx2, i, j] == 1:
                    logits[idx, edge_idx] = -float('inf')  # Set logit to -inf for existing edge
                    logits_clone[idx, edge_idx] = -float('inf')  # Apply the same for modified policy

            current_adj_matrix = adjacency_matrices[idx2].detach().numpy()

            if type(observations_batch) != str:
                log_prob1 = log_calculate_pr_x_given_g(current_adj_matrix, observations_batch[idx2]) 

                if log_prob1 == -np.inf:
                    logits_clone[idx, stop_index] = -float('inf')  # Prevent stop action

        probs_pi = F.softmax(logits, dim=1)  # Original policy probabilities
        probs_pi_prime = F.softmax(logits_clone, dim=1)  # Modified policy probabilities

        probs_true = trajectories[argNotDone][:, edge_indices_array[:, 0], edge_indices_array[:, 1]]
        actions_pi_prime_part = np.argwhere(probs_true == iter1)#[:, 0]
        actions_pi_prime = np.zeros(probs_true.shape[0], dtype=int)
        actions_pi_prime[:] = stop_index
        actions_pi_prime[actions_pi_prime_part[:, 0]] = actions_pi_prime_part[:, 1]
        actions_pi_prime = torch.tensor(actions_pi_prime)
        




        m_pi_prime = torch.distributions.Categorical(probs_pi_prime)
        
        
        
        #actions_pi_prime = m_pi_prime.sample()
        #print (logits_clone.shape, actions_pi_prime.shape)
        log_probs_pi_prime = m_pi_prime.log_prob(actions_pi_prime)

        log_prob_pi_prime_trajectories[argNotDone] += log_probs_pi_prime


        


        #if iter1 <= 3:
        #    print (log_probs_pi_prime[:5])

        # Importance sampling correction
        log_probs_pi = torch.log(probs_pi.gather(1, actions_pi_prime.unsqueeze(1)).squeeze(1))
        log_prob_pi_trajectories[argNotDone] += log_probs_pi

        for idx in range(argNotDone.shape[0]):
            idx2 = argNotDone[idx]
            if actions_pi_prime[idx].item() == stop_index:
                finished_new[idx2] = 1
            else:
                i, j = edge_indices[actions_pi_prime[idx].item()]
                adjacency_matrices = adjacency_matrices.clone()

                adjacency_matrices[idx2, i, j] = 1
                assert trajectories[idx2, i, j] == iter1
                #adjacency_tensors = adjacency_matrices.reshape(batch_size, -1)

    return log_prob_pi_trajectories, log_prob_pi_prime_trajectories, adjacency_matrices






# Function to train the model and save it
def train_model_off_policy(ruleObject, learning_rate, batchSize, offPolicy, num_epochs=10000, model_filename='', rewardType='', giveTrajectory=False, saveTrainingLoss=''):


    #observations_copy = np.copy(observations_all)

    if False:#batchSize > observations_all.shape[0]:
        #batchSize = observations_all.shape[0]

        assert batchSize % observations_all.shape[0] == 0
        
        #observations_all = observations_all[np.arange(batchSize) % observations_all.shape[0]]
        #observations_all = observations_all[np.arange(observations_all.shape[0]).repeat(  batchSize // observations_all.shape[0]  )  ]
        observations_all = observations_all[np.arange(observations_all.shape[0]).repeat(dupGen)  ]




    graphSize = ruleObject.graphSize
    #finalProbSize = ruleObject.finalProbSize
    model = ruleObject.model
    observations_all = ruleObject.observations_batch


    #model = GraphGeneratorNet(graphSize, finalProbSize, Nhidden)

    #learning_rate = 1e-2


    if rewardType == 'easy':
        maxReward = -np.inf

    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate , betas=(0.9, 0.99)) #good

    if rewardType == '':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate , alpha=0.99) #good
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000, eta_min=1e-5)
    else:
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate , betas=(0.9, 0.99)) #good
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate , alpha=0.99)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate , alpha=0.9)

    bestTrajectories = np.zeros((1, graphSize), dtype=int)


    #batchNum = obs_size // batchSize
    batchNum = 1

    


    metricMax = -np.inf
    #nearMetric = np.zeros(100 // min(batchNum, 5)) 
    #nearMetric = np.zeros(100)
    nearMetric = np.zeros(2000)
    #
    if rewardType == 'easy':
        nearMetric = np.zeros(500)
    if rewardType == 'GFlow':
        nearMetric = np.zeros(500)
    #nearMetric = np.zeros(1000)
    nearMetric[:] = -np.inf

    #print (nearMetric.shape)


    continue1 = True
    

    if False:
        mps_device = torch.device("mps")

        model.to(mps_device)

        #imageData_C = torch.tensor(imageData_C).float()
        #imageData_C = imageData_C.to(mps_device)

    

    #randomPerm1 = np.random.permutation(observations_all.shape[0])
    #observations_all = observations_all[randomPerm1]

    rewardLiss = []
    
    epoch = -1
    while continue1: 
        epoch += 1

        #for a in range(20):
        #    print ('')

        

        metricNow = 0.0
        
        #randomPerm1 = np.random.permutation(observations_all.shape[0])


        #for batch_index in range(batchNum):
        if True:

            observations_batch = observations_all
            
            if epoch == num_epochs - 1:
                continue1 = False
        
            model.train()
            optimizer.zero_grad()

            time1 = time.time()

            #offPolicy = True
            #if rewardType == 'GFlow':
            #    offPolicy = False 


            
            time1 = time.time()


            #if True:
            #    simPart = 'D' + str(1000) +  '_N' + str(100) + '_P' + str(0.5) + '_sim' + str(1)
            #    model = torch.load('./data/sims/simpleSet/model/graph_' + simPart + '_' + 'ours' + '_onPolicy2.pt')

            
            
            
            #adjacency_matrices, log_prob_pi, trajectories = raw_generate_graph_batch( model, graphSize, obs_size)
            adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batchSize)
            #adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories = spread_graph_batch(model, ruleObject, offPolicy, batchSize)

            #print (adjacency_matrices.shape)
            #quit()

            time1 = time.time() - time1

            time2 = time.time()

            
            
            
            
            if rewardType == '':
                
                
                if False:#with torch.no_grad():
                    #print ("A")
                    #print (adjacency_matrices.shape)
                    #adjacency_matrices_inverse = uniqueValMaker(adjacency_matrices.data.numpy())
                    #_, adjacency_matrices_index = np.unique(adjacency_matrices_inverse, return_index=True)

                    X_probs, rewards, info = ruleObject.rewardFunction(adjacency_matrices, observations_batch, log_prob_pi, log_prob_pi_prime, finalProb, giveInfo=True)
                    #print ("B")
                    #print (X_probs.shape)
                    #print (rewards.shape)
                    #quit()

            
            if giveTrajectory:
                #print (adjacency_matrices[0])
                #print (trajectories[0])
                #quit()
                adjacency_matrices = torch.tensor(trajectories)
                
            

            if rewardType == 'easy':
                
                pr_x_given_g_matrix = ruleObject.multi_x_given_g(adjacency_matrices, observations_batch)
                rewards_pre = torch.logsumexp(pr_x_given_g_matrix, dim=1)
                #print (pr_x_given_g_matrix)
                #print (rewards_pre)
                max1 = np.max(rewards_pre.data.numpy())
                maxReward = max(max1, maxReward)

                rewards = torch.exp(rewards_pre - maxReward)
                #print (rewards)

                

                #print (rewards)


                #rewards_pre_mean = torch.mean(rewards_pre * importance_weights) / torch.mean(importance_weights)
                rewards_pre_mean = torch.mean(rewards_pre) 

            elif rewardType == 'GFlow':

                pr_x_given_g_matrix = ruleObject.multi_x_given_g(adjacency_matrices, observations_batch)

                #print (pr_x_given_g_matrix[0])

                rewards_log = torch.logsumexp(pr_x_given_g_matrix, dim=1)
                rewards_log = rewards_log.detach()


                topReward = np.max(rewards_log.data.numpy())
                #print ('topReward', topReward)


                min1 =  torch.min(rewards_log[rewards_log != -np.inf])
                rewards_log[rewards_log == -np.inf] = min1


            
            if rewardType != '':
                importance_weights_log = (log_prob_pi - log_prob_pi_prime).detach()
                pr_x_given_g_matrix = pr_x_given_g_matrix + importance_weights_log.reshape((-1, 1))
                X_probs = torch.logsumexp(pr_x_given_g_matrix, dim=0)
                info = [0, X_probs]
            
            #print ("HI2")

            # Importance sampling corrected loss
            if rewardType == 'GFlow':

                ZmultFactor = 100

                #numEdgesInGraph = torch.sum(adjacency_matrices, axis=1)
                #maxEdge = adjacency_matrices.shape[1] * (adjacency_matrices.shape[1] - 1)
                #logFactorial = -1 * torch.lgamma(  (maxEdge - numEdgesInGraph) +1)

                #log_prob_pi = log_prob_pi + logFactorial

                
                
                diff1 =  log_prob_pi - rewards_log

                #print (log_prob_pi[:10] , rewards_log[:10])

                diff1 = diff1 - torch.mean(diff1.detach())

                loss = torch.mean(diff1 ** 2)

                #print (loss)


                metricMini = (loss.item() * -1)

            elif rewardType == 'easy':

                importance_weights = torch.exp(log_prob_pi - log_prob_pi_prime)

                loss = -((rewards * importance_weights * log_prob_pi).mean())
                #loss = -((rewards * (log_prob_pi - log_prob_pi_prime) ).mean())
            else:

                #print ("HI3")

                #time2 = time.time()
                #print ('adjacency_matrices.shape', adjacency_matrices.shape)
                loss, info = ruleObject.lossFunction(adjacency_matrices, observations_batch, log_prob_pi, log_prob_pi_prime, giveInfo=True)

                #time2 = time.time() - time2 


            #print (time1, time2)

            #print ("HI4")
            #quit()
                


            X_probs = info[1]

            #print (X_probs.shape)
            #quit()
            #print ('X_probs', X_probs)
            avg_reward = X_probs.mean().item() 



            #print ('avg_reward', avg_reward)
            #quit()

            if False:
                avg_reward = avg_reward / observations_batch.mean().item()
            med_reward = X_probs.median().item()
            #print ('med_reward', med_reward)

            #rewardLiss.append(avg_reward)
            rewardLiss.append(med_reward)


            if rewardType == 'easy':
                metricMini = rewards_pre_mean
            elif rewardType == '':
                metricMini = avg_reward
            #else:
            #    metricMini = avg_reward

            #print ('metricMini', metricMini)
            #print (avg_reward)


            time2 = time.time() - time2
            time3 = time.time()


            #print ('loss', loss)
            #quit()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            metricNow += metricMini

            time3 = time.time() - time3

            


            #print ('times', time1, time2, time3)
        
            
        metricNow = metricNow / batchNum


        if epoch % 10 == 0:
            #print (model.giveBias())
            if rewardType == 'GFlow':
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Reward: {med_reward}", 'loss', loss, 'topReward',  topReward)
            elif rewardType == 'easy':
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Reward: {med_reward}", 'loss', rewards_pre_mean)
            
            else:
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Reward: {metricNow}, Median: {med_reward}")
                #print ('X_probs', np.round(X_probs.data.numpy()*1000)/1000   )

                
                #quit()

        #print (model_filename)

        #print (metricNow)
        #print (np.max(nearMetric), metricMax)

        if not np.isnan(loss.detach().cpu().numpy()):
            #print ("Save")
            torch.save(model, model_filename) 

        extraIterStable = 0
        if rewardType == 'GFlow':
            extraIterStable = 200

        
        if epoch >= extraIterStable:
            metricMax = max(metricMax, nearMetric[epoch % nearMetric.shape[0]])
        nearMetric[epoch % nearMetric.shape[0]] = metricNow
            
        
        if saveTrainingLoss != '':
            np.savez_compressed(saveTrainingLoss, np.array(rewardLiss) )

        #print (np.max(nearMetric), metricMax)

        if epoch > nearMetric.shape[0] + extraIterStable:

            if epoch % 10 == 0:
                print (np.max(nearMetric), metricMax)
                
            #if not (np.max(nearMetric) > (metricMax + 1e-3)):
            if not (np.max(nearMetric) > (metricMax + 1e-7)):
                continue1 = False



def simpleTrainModel(ruleObject, model_filename='', offPolicy=False):
    train_model_off_policy(ruleObject, ruleObject.learning_rate, ruleObject.batchSize, offPolicy, num_epochs=50000, model_filename=model_filename, rewardType='', giveTrajectory=False)


def simpleInference(ruleObject, model_filename, offPolicy=False):

    #In the code, states are sometimes called graphs since an early version of this method was specifically for graphs. 

    sampleSize = 100000
    #offPolicy = False
    policyModel = torch.load(model_filename)
    adjacency_matrices, log_prob_pi, log_prob_prime, trajectory = batchSamplerOurs(policyModel, ruleObject, offPolicy, sampleSize)
    adjProb = []
    observations_batch = ruleObject.observations_batch
    if offPolicy:
        adjProb = [ log_prob_pi - log_prob_prime ]
    predicted_graphs = simpleGeneralPredictor(adjacency_matrices, observations_batch, ruleObject.multi_x_given_g, adjProb=adjProb)
    return predicted_graphs



def combineSampledGraphs(adjacency_matrices, log_prob_pi):

    print (adjacency_matrices.shape, log_prob_pi.shape)

    adjacency_matrices_np = adjacency_matrices.data.numpy()
    log_prob_pi = log_prob_pi.data.numpy()

    inverse1 = uniqueValMaker(adjacency_matrices_np)
    _, indices = np.unique(inverse1, return_index=True)
    max1 = indices.shape[0] #  int(np.max(inverse1+1))

    adjacency_matrices_new = adjacency_matrices[indices]

    pasted_prob = np.zeros(max1) - (np.min(log_prob_pi) * 10)

    for a in range(max1):
        arg1 = np.argwhere(inverse1==a)[:, 0]
        log_prob_pi_now = log_prob_pi[arg1]
        log_prob = scipy.special.logsumexp(log_prob_pi_now)
        pasted_prob[a] = log_prob
        #print (log_prob_pi_now)
        #print (log_prob)
    #quit()

    #print (pasted_prob)
    #print (adjacency_matrices_new.shape, pasted_prob.shape)
    
    pasted_prob = torch.tensor(pasted_prob).float()

    return adjacency_matrices_new, pasted_prob




def batchSamplerOurs(model, ruleObject, offPolicy, sampleSize):

    batch_size = 1000
    with torch.no_grad():

        
        numBatch = sampleSize // batch_size
        for a in range(numBatch):

            print (a)
            adjacency_matrices0, log_prob_pi0, log_prob_prime0, trajectory0  = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batch_size)
            adjacency_matrices0, log_prob_pi0, log_prob_prime0, trajectory0 = adjacency_matrices0.data.numpy(), log_prob_pi0.data.numpy(), log_prob_prime0.data.numpy(),  trajectory0

            if a == 0:
                adjacency_matrices, log_prob_pi, log_prob_prime, trajectory = np.copy(adjacency_matrices0), np.copy(log_prob_pi0), np.copy(log_prob_prime0), np.copy(trajectory0)
            else:
                adjacency_matrices = np.concatenate(( adjacency_matrices,  adjacency_matrices0 ), axis=0)
                log_prob_pi = np.concatenate(( log_prob_pi,  log_prob_pi0 ), axis=0)
                log_prob_prime = np.concatenate(( log_prob_prime,  log_prob_prime0 ), axis=0)
                trajectory = np.concatenate(( trajectory,  trajectory0 ), axis=0)

    return adjacency_matrices, log_prob_pi, log_prob_prime, trajectory



def simpleGeneralPredictor(adjacency_matrices, observations_batch, multi_x_given_g, adjProb=[], return_xProbs=False):

    with torch.no_grad():
        
        inverse2 = uniqueValMaker(adjacency_matrices)
        unique2, index2 = np.unique(inverse2, return_index=True)

        if len(adjProb) == 0:
            log_prob0 = np.zeros(adjacency_matrices.shape[0])
        else:
            log_prob0 = adjProb[0]

        score_adjuster = torch.logsumexp(torch.tensor(log_prob0), axis=0)
        score_adjuster = score_adjuster - np.log(log_prob0.shape[0])

        log_prob_pi = np.zeros(unique2.shape[0])
        adjacency_matrices = adjacency_matrices[index2]

        for a in range(unique2.shape[0]):
            arg1 = np.argwhere(inverse2 == unique2[a])[:, 0]
            #sum1 = np.log(arg1.shape[0])
            sum1 = logsumexp(log_prob0[arg1])
            log_prob_pi[a] = sum1        

        adjacency_matrices = torch.tensor(adjacency_matrices)
        log_prob_pi = torch.tensor(log_prob_pi)

        batch_size = 1000
        batch_indices = np.arange(adjacency_matrices.shape[0]) // batch_size
        batch_unique = np.unique(batch_indices)


        #pr_x_given_g_matrix_pre = multi_x_given_g(adjacency_matrices, observations_batch)

        if True:
            pr_x_given_g_matrix = torch.zeros( (adjacency_matrices.shape[0], observations_batch.shape[0] ) )
            for batch_index in range(batch_unique.shape[0]):
                argBatch = np.argwhere(batch_indices == batch_index)[:, 0]
                #pr_x_given_g_matrix0 = multi_x_given_g(adjacency_matrices[argBatch], observations_batch)
                #pr_x_given_g_matrix0 = pr_x_given_g_matrix0.data.numpy()
                #pr_x_given_g_matrix[argBatch] = pr_x_given_g_matrix0.clone()
                pr_x_given_g_matrix[argBatch] = multi_x_given_g(adjacency_matrices[argBatch], observations_batch)

                #print (torch.mean( torch.abs(pr_x_given_g_matrix_pre[argBatch] - pr_x_given_g_matrix[argBatch] ) ))
                #quit()

        #print (torch.mean( torch.abs(pr_x_given_g_matrix_pre - pr_x_given_g_matrix ) ))
        #quit()


        log_posterior_matrix = log_prob_pi.unsqueeze(1) + pr_x_given_g_matrix

        #print (log_posterior_matrix.shape)

        #score = torch.mean(torch.logsumexp(log_posterior_matrix, axis=0))

        X_probs = torch.logsumexp(log_posterior_matrix, axis=0) - score_adjuster
        score = torch.mean(X_probs)

        print (score)
        #quit()


        best_graph_indices = torch.argmax(log_posterior_matrix, dim=0)
        predicted_graphs = [adjacency_matrices[j].detach().numpy() for j in best_graph_indices]
        predicted_graphs = np.array(predicted_graphs)

    if return_xProbs:
        return predicted_graphs, X_probs
    else:
        return predicted_graphs



def graphProbSampler(model, ruleObject, offPolicy, sampleSize):

    
    with torch.no_grad():

        adjacency_matrices, log_prob_pi, trajectory = batchSamplerOurs(model, ruleObject, offPolicy, sampleSize)

        inverse2 = uniqueValMaker(adjacency_matrices)
        unique2, index2 = np.unique(inverse2, return_index=True)

        log_prob_pi_sum = np.zeros(unique2.shape[0])
        adjacency_matrices_unique = adjacency_matrices[index2]

        for a in range(unique2.shape[0]):
            arg1 = np.argwhere(inverse2 == unique2[a])[:, 0]
            #sum1 = logsumexp(log_prob_pi[arg1])
            sum1 = np.log(arg1.shape[0])

            #sum1 = np.log( float(arg1.shape[0]) - 0.99999) #TODO UNDO

            log_prob_pi_sum[a] = sum1

        #print (adjacency_matrices_unique.shape, log_prob_pi_sum.shape)
        #quit()


        return adjacency_matrices_unique, log_prob_pi_sum







# Function to predict graphs for given observations
def predict_graphs(model, ruleObject, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g, graphRules, offPolicyRule, offPolicy, giveTrajectory=False):
    

    with torch.no_grad():

        print ("BNNN")
        
        batch_size = 10000

        sampleSize = 1000000
        #adjacency_matrices, log_prob_pi = graphProbSampler(model, ruleObject, offPolicy, sampleSize)
        #quit()

        #adjacency_matrices, log_prob_pi = torch.tensor(adjacency_matrices), torch.tensor(log_prob_pi)
        
        # Generate a graph for each observation using the pi' policy
        #adjacency_matrices, log_prob_pi, _, _ = generate_graph_batch_with_modified_policy(model, observations_batch, log_calculate_pr_x_given_g, graphSize)
        adjacency_matrices, log_prob_pi, _, trajectory  = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batch_size)

        #if giveTrajectory:
        #    adjacency_matrices = torch.tensor(trajectory).float()

        
        pr_x_given_g_matrix = multi_x_given_g(adjacency_matrices, observations_batch)

        size1 = pr_x_given_g_matrix.shape[0]

        log_posterior_matrix = log_prob_pi.unsqueeze(1) + pr_x_given_g_matrix
        #log_posterior_matrix = (log_prob_pi.unsqueeze(1) * 0.1) + pr_x_given_g_matrix #TODO SET BACK TO HOW IT WAS !!!!

        log_posterior_matrix_np = log_posterior_matrix.data.numpy()
        

        best_graph_indices = torch.argmax(log_posterior_matrix, dim=0)


        #print (log_posterior_matrix.shape)

        #print (torch.max(log_posterior_matrix))

        #quit()
        
        # Retrieve the best adjacency matrices
        predicted_graphs = [adjacency_matrices[j].detach().numpy() for j in best_graph_indices]
        predicted_graphs = np.array(predicted_graphs)
    
    return predicted_graphs



# Function to predict graphs for given observations
def batch_predict_graphs(model, ruleObject, observations_all, multi_x_given_g, log_calculate_pr_x_given_g, batchSize, offPolicy, giveTrajectory=False):

    with torch.no_grad():

        graphRules, offPolicyRule, graphSize = ruleObject.graphRules, ruleObject.offPolicyRule, ruleObject.graphSize

        size_original = observations_all.shape[0]

        #if batchSize > size_original:
        #    observations_all = observations_all[np.arange(batchSize) % size_original]

        
        numBatches = observations_all.shape[0] // batchSize

        if observations_all.shape[0] <= batchSize:
            predicted_graphs = predict_graphs(model, ruleObject, observations_all, multi_x_given_g, log_calculate_pr_x_given_g, graphRules, offPolicyRule, offPolicy, giveTrajectory=giveTrajectory)

        else:

            print ('numBatches', numBatches)
            #print (numBatches, numBatches)
            for batch_index in range(numBatches):
                print ('batch_index', batch_index)
                observations_batch = observations_all[ batch_index * batchSize: (batch_index+1) * batchSize  ]

                predicted_batch = predict_graphs(model, ruleObject, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g, graphRules, offPolicyRule, offPolicy, giveTrajectory=giveTrajectory)

                

                #[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.
                #0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1.
                #1. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.
                #1. 0. 1. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]


                #[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.
                #0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1.
                #1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
                #1. 1. 1. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

                #quit()

                if batch_index == 0:
                    predicted_graphs = np.zeros(( observations_all.shape[0], predicted_batch.shape[1] ), dtype=int)

                predicted_graphs[ batch_index * batchSize: (batch_index+1) * batchSize  ] = predicted_batch

        if batchSize > size_original:
            predicted_graphs = predicted_graphs[:size_original]
            observations_all = observations_all[:size_original]
    
    return predicted_graphs






class gClass:
    def __init__(self):
        True


    
    def log_calculate_pr_x_given_g(adjacency_matrix, obs):
        print ("TODO implement")
        return 0.0

    
    def multi_x_given_g(self, adjacency_matrices, obs_matrix):
        #For quick code, make sure to replace this with a vectorized version for one's application. 

        time3 = time.time()

        prob_mult = torch.tensor([
            [self.log_calculate_pr_x_given_g(adjacency_matrices[j].detach().numpy(), obs_matrix[i])
             for i in range(obs_matrix.shape[0])] for j in range(adjacency_matrices.shape[0])
        ], dtype=torch.float32)


        time3 = time.time() - time3

        #print ('time3', time3)

        return prob_mult

    
    #def multi_x_given_g(self, adjacency_matrices, obs_matrix):
    #    print ("TODO implement")
    #    prob_mult = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]))
    #    return prob_mult
    

    
    def graphRules_G(self, adjacency_matrices):
        graphNew = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]+1))
        return graphNew

    def graphRules_F(self, adjacency_matrices):
        finalProbAllow = torch.zeros((adjacency_matrices.shape[0], 1))# self.finalProbSize))
        return finalProbAllow
    
    def offPolicyRule_G(self, adjacency_matrices):
        graphNew = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]+1))
        return graphNew

    def offPolicyRule_F(self, adjacency_matrices):
        finalProbAllow = torch.zeros((adjacency_matrices.shape[0], self.finalProbSize))
        return finalProbAllow


    def graphRules(self, adjacency_matrices):

        graphNew = self.graphRules_G(adjacency_matrices)
        finalProbAllow = self.graphRules_F(adjacency_matrices)

        return graphNew, finalProbAllow

    def offPolicyRule(self, adjacency_matrices, arange1):

        graphNew = self.offPolicyRule_G(adjacency_matrices)
        finalProbAllow = self.offPolicyRule_F(adjacency_matrices)
        
        return graphNew, finalProbAllow
    

    def rewardFunction(self, adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=False):


        #print (adjacency_matrices[0])
        #print (obs_matrix[0])
        #print (noise_level)
        #quit()

        pr_x_given_g_matrix = self.multi_x_given_g(adjacency_matrices, obs_matrix) 

        importance_weights_log = (log_prob_pi - log_prob_pi_prime).detach() #* 0

        if True:
            normalizer = torch.logsumexp(importance_weights_log, axis=0) - np.log(importance_weights_log.shape[0])
            #Valid normalization "Self-normalized importance sampling (SNIS)"
            #importance_weights_log = torch.nn.LogSoftmax(dim=0)(importance_weights_log)
            importance_weights_log = importance_weights_log - normalizer
        
        #importance_weights = torch.exp(importance_weights_log)  # Detach to prevent gradient flow    

        #imp_exp =np.round(np.exp(importance_weights_log.data.numpy()) , decimals=2 ) 
        #print ( imp_exp )

        #plt.hist(importance_weights_log.data.numpy(), bins=100)
        #plt.show()
        #quit()

        if True:
            pr_x_given_g_matrix_copy = pr_x_given_g_matrix.detach().cpu().numpy()
        

        
        pr_x_given_g_matrix = pr_x_given_g_matrix + importance_weights_log.reshape((-1, 1))

        X_probs = torch.logsumexp(pr_x_given_g_matrix, dim=0)

        #plt.plot(X_probs.data.numpy())
        #plt.show()
        #quit()

        if True: 
            minValue = torch.min(X_probs[X_probs != -np.inf]) - 1000 #Essentially negative infinity but stable. 
            X_probs[X_probs == -np.inf] = minValue


        pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix - X_probs.reshape((1, -1))

        #pr_x_given_g_matrix_adjusted = X_probs.reshape((1, -1)) - pr_x_given_g_matrix
        #pr_x_given_g_matrix_adjusted = torch.log(torch.exp(pr_x_given_g_matrix_adjusted) - 0.99)
        #pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix_adjusted * -1

        
        #if False:
        #    batchCorrection = np.log(batchSize - 1) - np.log(observations_all.shape[0] - 1)
        #    pr_x_given_g_matrix_adjusted[np.arange(batchSize*dupGen), np.arange(batchSize).repeat(dupGen)] -= batchCorrection

        #print ('pr_x_given_g_matrix_adjusted')
        #print (pr_x_given_g_matrix_adjusted)

        #print ('rrr')

        #print (np.round(np.exp(pr_x_given_g_matrix_adjusted.data.numpy()), decimals=2))

        #print ('max1',  np.max( np.exp(pr_x_given_g_matrix_adjusted.data.numpy()) ) )

        #if False:
        #    rewards = pr_x_given_g_matrix_adjusted[np.arange(pr_x_given_g_matrix.shape[0]), np.arange(pr_x_given_g_matrix.shape[0]) % pr_x_given_g_matrix.shape[1] ]

        #else:
        
        rewards = torch.logsumexp(pr_x_given_g_matrix_adjusted, dim=1)

        #argZero = np.argwhere(np.abs(rewards.data.numpy()) < 0.001)[:, 0]

        #argMax1 = np.argwhere(pr_x_given_g_matrix_adjusted.data.numpy() == np.max(pr_x_given_g_matrix_adjusted.data.numpy()) )[0]
        #print (argMax1)

        #plt.plot(pr_x_given_g_matrix_copy[:, argMax1[1]]  )
        #plt.plot(importance_weights_log.data.numpy())
        #plt.show()
        #quit()
        
        #plt.plot(pr_x_given_g_matrix_copy[argMax1])
        #plt.plot()


        #print (pr_x_given_g_matrix_adjusted[argZero[0]])

        #print (rewards)

        #plt.plot(rewards.data.numpy())
        #plt.plot(importance_weights_log.data.numpy())
        #plt.show()
        #quit()

        



        #print ('mean1', torch.mean(pr_x_given_g_matrix), 'mean2', torch.mean(importance_weights_log) )
        #print ('R', torch.exp(rewards[argMax1])   )

        #print ('state', torch.mean(adjacency_matrices[argMax1]),  adjacency_matrices[argMax1]  )
        #print ('argmax', argMax1   )
        #print ('i', importance_weights_log[argMax1])
        #print ('log prob', log_prob_pi[argMax1] , log_prob_pi_prime[argMax1])
        #print ('imp', np.min(importance_weights_log.data.numpy()), np.max(importance_weights_log.data.numpy()))

        #if np.max(importance_weights_log.data.numpy()) > 0:
        #    for printIter in range(100): 
        #        print ('')
        #    print ("MAX", np.max(importance_weights_log.data.numpy()) )
        #    for printIter in range(100): 
        #        print ('')

            #quit()




        rewards = torch.exp(rewards)


        

        if False:
            #print (torch.sum(rewards))
            #rewards = rewards - (torch.exp(importance_weights_log) * pr_x_given_g_matrix_adjusted.shape[1])
            rewards = rewards - torch.exp(importance_weights_log)
            #print (torch.sum(rewards))

            #rewards_raw = (rewards / torch.exp(importance_weights_log)).data.numpy()
        #quit()

        #print (np.round(rewards.data.numpy(), decimals=2))

        rewards = rewards * (pr_x_given_g_matrix_adjusted.shape[0] / pr_x_given_g_matrix_adjusted.shape[1])

        #print (rewards)
        #quit()

        if giveInfo:
            info = [pr_x_given_g_matrix]
            #return X_probs - normalizer, rewards, info
            return X_probs, rewards, info
        else:
            #return X_probs - normalizer, rewards
            return X_probs, rewards


    def lossFunction(self, adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=False):
        
        with torch.no_grad():
            if giveInfo:
                X_probs, rewards, info = self.rewardFunction(adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=giveInfo)
                info.append(X_probs)
                info.append(rewards)
            else:
                X_probs, rewards = self.rewardFunction(adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=giveInfo)
                

        loss = -((rewards * log_prob_pi).mean())

        if giveInfo:
            return loss, info
        else:
            return loss








class sample_gClass(gClass):


    def __init__(self, finalProbSize):
        super().__init__()  # call parent init to set self.value
        self.finalProbSize = finalProbSize


    def rewardFunction(self, adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=False):

        pr_x_given_g_matrix = self.multi_x_given_g(adjacency_matrices, obs_matrix) 


        if self.finalProbSize >= 2:
            importance_weights_log = (log_prob_pi - log_prob_pi_prime.reshape(((-1, 1)))  ).detach()
        else:
            importance_weights_log = (log_prob_pi - log_prob_pi_prime).detach()

        
        
        importance_weights = torch.exp(importance_weights_log)  # Detach to prevent gradient flow    

        if self.finalProbSize >= 2:
            pr_x_given_g_matrix = pr_x_given_g_matrix + importance_weights_log.reshape((importance_weights_log.shape[0], 1, importance_weights_log.shape[1]))
        else:
            pr_x_given_g_matrix = pr_x_given_g_matrix + importance_weights_log.reshape((-1, 1))
        
        X_probs = torch.logsumexp(pr_x_given_g_matrix, dim=0)


        if self.finalProbSize >= 2:
            pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix - X_probs.reshape((1, X_probs.shape[0], X_probs.shape[1]))

        else:
            pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix - X_probs.reshape((1, -1))
        

        rewards = torch.logsumexp(pr_x_given_g_matrix_adjusted, dim=1)

        rewards = torch.exp(rewards)
        rewards = rewards * (pr_x_given_g_matrix_adjusted.shape[0] / pr_x_given_g_matrix_adjusted.shape[1])

        #print (rewards)
        #quit()

        if giveInfo:
            info = [pr_x_given_g_matrix]
            return X_probs, rewards, info
        else:
            return X_probs, rewards



    def lossFunction(self, adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=False):


        if (self.finalProbSize >= 2) and (adjacency_matrices.shape[0] >= 1):
            _, finalProbAllow = self.graphRules(adjacency_matrices)
            finalProb = self.model.finalProb(adjacency_matrices)
            finalProb = finalProb + finalProbAllow
            finalProb = nn.LogSoftmax(dim=1)(finalProb)
        else:
            finalProb = torch.zeros((adjacency_matrices.shape[0], 1))
    


        log_prob_pi = log_prob_pi.reshape((-1, 1)) + finalProb
        
        with torch.no_grad():
            if giveInfo:
                X_probs, rewards, info = self.rewardFunction(adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, finalProb, giveInfo=giveInfo)
                info.append(X_probs)
                info.append(rewards)
            else:
                X_probs, rewards = self.rewardFunction(adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, finalProb, giveInfo=giveInfo)
                

        loss = -((rewards * log_prob_pi).mean())

        if giveInfo:
            return loss, info
        else:
            return loss








def OLD_fullTrainModel():


    doTrain = True
    #doTrain = False


    #modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'diffusion'
    #modelType = 'localSolver'
    #modelType = 'metropolas'

    
    #graphSize = 100
    #graphSize = 90

    simList = [] #data points, nodes, paths
    simList.append([100, 10, 10])  #default 

    simList.append([1, 10, 10])  #modified number of data points 
    simList.append([5, 10, 10]) 
    simList.append([10, 10, 10]) 
    simList.append([25, 10, 10]) 
    simList.append([250, 10, 10])
    simList.append([1000, 10, 10])


    #simList.append([100, 5, 10])  #modified number of nodes
    #simList.append([100, 15, 10])  #modified number of nodes
    #simList.append([100, 20, 10])  #modified number of nodes


    #simList.append([100, 10, 1])  #modified number of paths
    #simList.append([100, 10, 10])  #modified number of paths
    #simList.append([100, 10, 100])  #modified number of paths
    #simList.append([100, 10, 1000])  #modified number of paths
    #simList.append([100, 10, 10000])  #modified number of paths


    for simParamIndex in range(0, len(simList)):
        print ('simParamIndex', simParamIndex)
        for simIndex in range(1):

            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            num_paths_per_graph =  simList[simParamIndex][2]

            for a in range(5):
                print ('')
            print ('Sim Index ' + str(simIndex))
            for a in range(5):
                print ('')

            #simPart = f'causal_N10_D100_M5_{simIndex}'
            #simPart = f'N10_P100_{simIndex}'
            #simPart = f'N10_P10_{simIndex}'
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

            #simPart = 'D' + str(1000) +  '_N' + str(10) + '_P' + str(1) + '_' + str(simIndex)
            #simPart = 'D' + str(10) +  '_N' + str(10) + '_P' + str(100) + '_' + str(simIndex)

            print (simPart, modelType)

            
            observations_batch = loadnpz('./data/sims/initial/' + simPart + '_obs.npz')
            adjacency_matrices = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')


            #if 'causal' in simPart:
            #    graphSize = 100
            #    predFile =   './data/sims/causal/pred/graph_' + simPart + '_' + modelType + '.npz'
            #    modelFile = './data/sims/causal/model/graph_' + simPart + '_' + modelType + '.pt'
            #else:
            graphSize = 90
            predFile =   './data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '_mod.npz'
            modelFile = './data/sims/startEnd/model/graph_' + simPart + '_' + modelType + '_mod.pt' #1


            #if 'causal' in simPart:
            #    multi_x_given_g, log_calculate_pr_x_given_g = causal_fast_multi, causal_x_given_g  
            #else:
            multi_x_given_g, log_calculate_pr_x_given_g = sim1_fast_multi, sim1_log_calculate_pr_x_given_g  



            #print (adjacency_matrices.shape)
            #quit()
            
                
            #duplicationNum = 1

            #batchSize = 100
            #batchSize = 25
            #batchSize = 50
            #batchSize = 300
            #batchSize = 600

            #batchSize = adjacency_matrices.shape[0]
            #dupGen = 1

            #print (observations_batch.shape)

            #observations_batch = observations_batch[np.arange(250) // 25]

            #observations_batch = observations_batch[np.arange(300) // 30]

            #observations_batch = observations_batch[np.arange(1000) // 100]

            finalProbSize = 1

            ruleObject = gClass()
            def graphRules(graph):
                graphAllow = torch.zeros((graph.shape[0], graphSize+1))
                finalProbAllow = torch.zeros((graph.shape[0], finalProbSize))
                return graphAllow, finalProbAllow

            def offPolicyRule(graphList, arange1):

                #log_calculate_pr_x_given_g, observations_batch
                probList = np.zeros(graphList.shape[0])

                #print ('observations_batch2', observations_batch.shape)
                #print (observations_batch[2])

                for a in range(probList.shape[0]):
                    graphNow = graphList[a]
                    prob1 = log_calculate_pr_x_given_g(graphNow, observations_batch[arange1[a]]  )
                    probList[a] = prob1

                #print (probList)

                argNotDone = np.argwhere(probList ==  -float('inf'))[:, 0]

                #print (argNotDone)
                

                graphAllow = torch.zeros((graphList.shape[0], graphSize+1))
                graphAllow[argNotDone, -1] =  -float('inf')

                finalProbAllow = torch.zeros((graphList.shape[0], finalProbSize))
                return graphAllow, finalProbAllow


            ruleObject.graphRules = graphRules
            ruleObject.offPolicyRule = offPolicyRule
            ruleObject.multi_x_given_g = multi_x_given_g
            ruleObject.log_calculate_pr_x_given_g = log_calculate_pr_x_given_g
            ruleObject.graphSize = graphSize
            ruleObject.observations_batch = observations_batch
            ruleObject.adjacency_matrices = adjacency_matrices
            ruleObject.batchSize = 1000
            batchSize = ruleObject.batchSize
            


            offPolicy = True

            

            #generic_TrainModel(multi_x_given_g, log_calculate_pr_x_given_g, observations_batch, adjacency_matrices, graphSize, finalProbSize, graphRules, offPolicyRule,
            #                   simPart, modelType, predFile, modelFile, doTrain, batchSize, dupGen, offPolicy)
            generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, doTrain, batchSize, offPolicy)



            alsoPred = True
            if doTrain and alsoPred:
                generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, False, batchSize, offPolicy)

