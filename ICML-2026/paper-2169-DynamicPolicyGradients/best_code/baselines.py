import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
from scipy.special import softmax
import time

import math
import os

from sharedGen import *
import pandas as pd


# Parameters
#num_simulations = 20
#num_data_points = 100
#num_nodes = 10  # Reduced to 10 nodes for faster runtimes
#num_nodes = 3
#num_paths_per_graph = 10


#Epoch 541/5000, Average Reward: -2.018215434968489


print ("Testing modification of local history")

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


np.random.seed(1)
torch.manual_seed(0)






# Neural network for graph generation
#class GraphGeneratorNet(nn.Module):
#    def __init__(self, graphSize, finalProbSize, Nhidden):
#        super(GraphGeneratorNet, self).__init__()
#        self.input_size = graphSize
#        self.hidden_size = 100# Nhidden
#        self.finalProbSize = finalProbSize
#        self.output_size = graphSize + 1  # All edges + stop action

#        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
#        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        
        #self.fc_finalProb1 = nn.Linear(self.input_size, 30)
        #self.fc_finalProb2 = nn.Linear(30, self.finalProbSize)

#        self.fc_finalProb1 = nn.Linear(self.input_size, 200)
#        self.fc_finalProb2 = nn.Linear(200, self.finalProbSize)

        #self.fc_finalProb = nn.Linear(self.hidden_size, self.finalProbSize)


#    def forward(self, x):
#        x = F.leaky_relu(self.fc1(x))
#        logits = self.fc2(x)
#        return logits
    
#    def finalProb(self, x):
#        x = F.leaky_relu(self.fc_finalProb1(x))
        #print (x.shape)
#        logits = self.fc_finalProb2(x)

        #scale = logits0[:, :1]
        #logits_other = nn.LogSoftmax(dim=1)(logits0[:, 1:])

        #logits = logits_other + scale
        #x = F.leaky_relu(self.fc1(x))
        #x = F.leaky_relu(self.fc_finalProb1(x))
        #logits =  self.fc_finalProb2(x)
        #logits = self.fc_finalProb(x)
#        return logits


# Define the Graph VAE
class GraphVAE(nn.Module):
    def __init__(self, graphSize, Nhidden, latent_dim=16):
        super(GraphVAE, self).__init__()
        #self.node_dim = node_dim
        self.graphSize = graphSize
        self.latent_dim = latent_dim

        # Encoder: Map adjacency matrix to latent space
        self.encoder_fc1 = nn.Linear(graphSize, Nhidden)
        self.encoder_fc2_mean = nn.Linear(Nhidden, latent_dim)  # Mean of latent distribution
        self.encoder_fc2_logvar = nn.Linear(Nhidden, latent_dim)  # Log variance of latent distribution

        # Decoder: Map latent space back to adjacency matrix
        self.decoder_fc1 = nn.Linear(latent_dim, Nhidden)
        self.decoder_fc2 = nn.Linear(Nhidden, graphSize)  # Output a flattened adjacency matrix

    def encode(self, adj_matrix):
        #x = adj_matrix.view(-1, self.node_dim * self.node_dim)  # Flatten adjacency matrix
        x = adj_matrix
        x = F.relu(self.encoder_fc1(x))
        mu = self.encoder_fc2_mean(x)
        logvar = self.encoder_fc2_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Sample z from N(mu, sigma^2)

    def decode(self, z):
        x = F.relu(self.decoder_fc1(z))
        x = self.decoder_fc2(x)
        #adj_reconstructed = torch.sigmoid(x).view(-1, self.node_dim, self.node_dim)  # Reshape back to adjacency matrix
        adj_reconstructed = torch.sigmoid(x)

        adj_reconstructed = (adj_reconstructed * (  1.0 - 2e-8 )) + 1e-8

        return adj_reconstructed

    def forward(self, adj_matrix):
        mu, logvar = self.encode(adj_matrix)
        z = self.reparameterize(mu, logvar)
        adj_reconstructed = self.decode(z)
        return adj_reconstructed, mu, logvar

# Loss function: ELBO = reconstruction loss + KL divergence
def loss_function(adj_reconstructed, adj_original, mu, logvar):

    recon_loss = F.binary_cross_entropy(adj_reconstructed, adj_original, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence term
    return recon_loss + kl_div


def ELBO_bound(adj_reconstructed, adj_original, mu, logvar):


    #print (adj_reconstructed)
    #quit()
    recon_loss_sum = F.binary_cross_entropy(adj_reconstructed, adj_original, reduction='sum')
    recon_loss = (torch.log(adj_reconstructed) * adj_original) + (torch.log(1 - adj_reconstructed) * (1-adj_original))
    recon_loss = recon_loss * -1
    #print (recon_loss.shape)
    #recon_loss = torch.sum(recon_loss, axis=(1, 2))
    recon_loss = torch.sum(recon_loss, axis=1)
    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence term
    kl_div = torch.sum(kl_div, axis=1)

    #print (recon_loss)
    #print (kl_div)
    #quit()

    prob_est = recon_loss + kl_div

    #print (prob_est)
    #quit()

    return prob_est
    




def sample_graphs(model, num_samples=5):
    """
    Samples new graphs from the trained GraphVAE model.
    
    Parameters:
        model (GraphVAE): Trained GraphVAE model.
        num_samples (int): Number of graphs to generate.
        node_dim (int): Number of nodes in the graph (adjacency matrix size).
    
    Returns:
        List of sampled adjacency matrices.
    """
    device = next(model.parameters()).device  # Ensure sampling is done on the correct device

    # Sample latent variables from a standard normal distribution
    z = torch.randn(num_samples, model.latent_dim).to(device)

    # Decode latent variables to generate adjacency matrices
    with torch.no_grad():
        sampled_adj_matrices = model.decode(z)

    # Convert to binary adjacency matrices (thresholding at 0.5)
    #sampled_adj_matrices = (sampled_adj_matrices > 0.5).float()

    sampled_adj_matrices = torch.bernoulli(sampled_adj_matrices).float()

    

    return sampled_adj_matrices#.cpu().numpy()



def VAE_inferGivenSample(graphs_all, observations_batch, model, multi_x_given_g, doSimple=True, doRandom=True):

    #print (graphs_all.shape)

    if not doSimple:
        adj_batch, mu, logvar = model(graphs_all)
        prob_est = ELBO_bound(adj_batch, adj_batch, mu, logvar)

    #print (prob_est)
    #quit()


    pr_x_given_g_matrix = multi_x_given_g(graphs_all, observations_batch)



    X_probs = torch.logsumexp(pr_x_given_g_matrix, dim=0)

    

    pr_x_given_g_matrix = pr_x_given_g_matrix.data.numpy()

    if not doSimple:
        prob_est = prob_est.data.numpy()
        pr_x_given_g_matrix_adj = pr_x_given_g_matrix + prob_est.reshape((-1, 1))
    else:
        pr_x_given_g_matrix_adj = pr_x_given_g_matrix


    if doRandom:
        pr_x_given_g_matrix_adj[np.isnan(pr_x_given_g_matrix_adj)] = -np.inf
        min1 = -100000
        if pr_x_given_g_matrix_adj[pr_x_given_g_matrix_adj != -np.inf].shape[0] >= 1:
            min1 = np.min(pr_x_given_g_matrix_adj[pr_x_given_g_matrix_adj != -np.inf]) - 100000

        pr_x_given_g_matrix_adj[pr_x_given_g_matrix_adj == -np.inf] = min1

        #print (pr_x_given_g_matrix_adj[:, 0])
        #print (pr_x_given_g_matrix_adj[:, 0])

        #probs = torch.softmax(torch.tensor(pr_x_given_g_matrix_adj).float(), dim=0)
        probs = torch.tensor(softmax(pr_x_given_g_matrix_adj, axis=0)).float()

        #print (probs)
        #print (probs[:, 0])
        #print (probs[:, 0])

        #print (torch.min(probs[0]))
        #print (torch.max(probs[0]))
        repeatNum = probs.shape[0] // probs.shape[1]
        if repeatNum == 0:
            randomSubset = np.random.permutation(probs.shape[1])[:probs.shape[0]]

            argBest = torch.multinomial(probs.T[randomSubset], num_samples=1)


        else:
            argBest = torch.multinomial(probs.T, num_samples=repeatNum)
        #print (argBest.shape)
        #quit()
        argBest = argBest.reshape((-1,))
        argBest = argBest.data.numpy()
    else:
        argBest = np.argmax( pr_x_given_g_matrix_adj , axis=0 )

    #print (pr_x_given_g_matrix_adj[  argBest,  np.arange(argBest.shape[0]) ])

    return argBest, X_probs






# Training loop
def train_graph_vae(ruleObject, Nhidden, graphSize, observations_batch, multi_x_given_g, num_epochs=5000, batch_size=32, learning_rate=0.001, model_filename=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #num_nodes = observations_batch.shape[1]
    #batchSize = observations_batch.shape[0]
    batchSize = ruleObject.batchSize
    
    model = GraphVAE(graphSize, Nhidden).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #graphs = torch.zeros(observations_batch.shape)
    #graphs = torch.zeros((batch_size, graphSize))

    

    metricMax = -np.inf
    nearMetric = np.zeros(500) 
    #nearMetric = np.zeros(1000) 
    nearMetric[:] = -np.inf


    continue1 = True
    
    epoch = -1
    while continue1 and (epoch < num_epochs): 
        epoch += 1

        with torch.no_grad():
            #sample1 = sample_graphs(model, num_samples=graphs.shape[0])
            #adj_reconstructed, mu, logvar = model(graphs)
            #adj_reconstructed = (adj_reconstructed > 0.5).float()
            #adj_reconstructed = adj_reconstructed

            #graphs_all = torch.cat(( graphs, sample1 ), axis=0)
            #argBest, X_probs = VAE_inferGivenSample(graphs_all, observations_batch, model, multi_x_given_g,  doSimple=True)
            #graphs = graphs_all[argBest]

            #graphs_all = sample_graphs(model, num_samples=batchSize)
            #argBest, X_probs = VAE_inferGivenSample(graphs_all, observations_batch, model, multi_x_given_g, doSimple=True, doRandom=True)
            #graphs = graphs_all[argBest]

            ####graphs_all = sampleAutoreg(model, batch_size)
            graphs_all = sample_graphs(model, num_samples=batchSize)
            graphs_all = graphs_all.data.numpy()
            graphs, X_probs = simpleGeneralPredictor(graphs_all, observations_batch, multi_x_given_g, return_xProbs=True)
            graphs = torch.tensor(graphs).float()

            #print (torch.sum(graphs, axis= (1, 2) ))

        

        total_loss = 0
        
        #print (batch[0].shape)
        #quit()
        adj_batch = graphs
        optimizer.zero_grad()

        adj_reconstructed, mu, logvar = model(adj_batch)

        #print (adj_batch.shape)
        #print (adj_reconstructed.shape)
        
        


        loss = loss_function(adj_reconstructed, adj_batch, mu, logvar)

        loss = loss / batchSize

        #print (torch.sum(prob_est))
        #print (loss)
        #quit()
        loss.backward()
        optimizer.step()

        metricMax = max(metricMax, nearMetric[epoch % nearMetric.shape[0]] )
        nearMetric[epoch % nearMetric.shape[0]] = -1 * loss.item()

        total_loss += loss.item()

        if model_filename != '':
            torch.save(model, model_filename)


        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss }", 'Rewards', torch.mean(X_probs), 'Rewards median ', torch.median(X_probs))

        if epoch > nearMetric.shape[0]:
            print (np.max(nearMetric), metricMax)
            if not (np.max(nearMetric) > (metricMax + 1e-3)):
                continue1 = False

    return model, graphs



def VAE_predict(observations_batch, model, multi_x_given_g):

    sample1 = sample_graphs(model, num_samples=10000)
    argBest, _ = VAE_inferGivenSample(sample1, observations_batch, model, multi_x_given_g, doSimple=False, doRandom=False)
    sample1 = sample1[argBest]
    sample1 = sample1.data.numpy()

    return sample1


def sampleVAE(model, sampleSize):

    with torch.no_grad():
        batchSize = 1000
        numBatch = sampleSize // batchSize
        for batchIndex in range(numBatch):
            graphs_all0 = sample_graphs(model, num_samples=batchSize)
            graphs_all0 = graphs_all0.data.numpy()
            if batchIndex == 0:
                graphs_all = np.copy(graphs_all0)
            else:
                graphs_all = np.concatenate(( graphs_all, graphs_all0 ), axis=0)
    return graphs_all


def flowMatching(trajectories, adjacency_matrices, rewards_log, model):

    numSteps = np.max(trajectories, axis=(1, 2))
    maxSteps = np.max(numSteps)

    stateList = []

    num_nodes = trajectories.shape[1]
    edge_indices = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_indices = np.array(edge_indices)
    edge_indices_inverse = np.zeros((num_nodes, num_nodes), dtype=int)
    edge_indices_inverse[edge_indices[:, 0], edge_indices[:, 1]] = np.arange(edge_indices.shape[0])
    

    for index1 in range(trajectories.shape[0]):
        for step1 in range(1, numSteps[index1]+1):
            graph1 = np.copy(trajectories[index1])
            graph1[graph1 > step1] = 0
            graph1[graph1 > 0] = 1
            graph1[graph1 == -1] = 0

            stateList.append(np.copy(graph1))

    stateList = np.array(stateList)

    flowOut = model(torch.tensor(stateList).float().reshape((-1, num_nodes*num_nodes))  )
    flowOut = torch.logsumexp(flowOut, axis=1)

    flowIn = torch.zeros(flowOut.shape[0])


    for index1 in range(stateList.shape[0]):

        stateNow = stateList[index1:index1+1]
        args1 = np.argwhere(stateList[index1] == 1)
        stateNowParents = np.copy(stateNow)[np.zeros(args1.shape[0], dtype=int )]
        stateNowParents[np.arange(args1.shape[0]), args1[:, 0], args1[:, 1]] = 0
        stateNowParents = torch.tensor(stateNowParents).float()

        choice_indices = edge_indices_inverse[args1[:, 0], args1[:, 1]]
        flowIn_mini = model(stateNowParents.reshape((-1, num_nodes*num_nodes)) )
        flowIn_mini = flowIn_mini[np.arange(choice_indices.shape[0]), choice_indices]
        flowIn_mini = torch.logsumexp(flowIn_mini, axis=0)

        #print (flowIn_mini.data.numpy())


        flowIn[index1] = flowIn_mini


    #print (flowIn[:5])

    error1 = torch.sum( ( flowOut - flowIn) ** 2)

    #print ('error1', error1)

    #print ('max', torch.max(torch.abs(flowOut)), torch.max(torch.abs(flowIn)))

    
    flowInReward = model(torch.tensor(adjacency_matrices.reshape((-1, num_nodes*num_nodes))   ).float())
    flowInReward = flowInReward[:, -1]

    error2 =  torch.sum( ( rewards_log - flowInReward) ** 2)

    error_all = (error1 + error2) / (flowIn.shape[0] + flowInReward.shape[0])

    #print ('error1, 2', error1, error2)

    return error_all, flowInReward




def fast_flowMatching(trajectories, adjacency_matrices, rewards_log, model):

    numSteps = np.max(trajectories, axis=(1, 2))
    maxSteps = np.max(numSteps)

    stateList = []

    num_nodes = trajectories.shape[1]
    edge_indices = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_indices = np.array(edge_indices)
    edge_indices_inverse = np.zeros((num_nodes, num_nodes), dtype=int)
    edge_indices_inverse[edge_indices[:, 0], edge_indices[:, 1]] = np.arange(edge_indices.shape[0])
    

    for index1 in range(trajectories.shape[0]):
        for step1 in range(1, numSteps[index1]+1):
            graph1 = np.copy(trajectories[index1])
            graph1[graph1 > step1] = 0
            graph1[graph1 > 0] = 1
            graph1[graph1 == -1] = 0

            stateList.append(np.copy(graph1))

    stateList = np.array(stateList)

    inverse1 = uniqueValMaker(stateList.reshape((stateList.shape[0], stateList.shape[1]*stateList.shape[2] )))
    _, index1 = np.unique(inverse1, return_index=True)



    flowOut = model(torch.tensor(stateList[index1]).float().reshape((-1, num_nodes*num_nodes))  )
    flowOut = torch.logsumexp(flowOut, axis=1)
    flowOut = flowOut[inverse1]

    #flowIn = torch.zeros(flowOut.shape[0])

    stateList_parent = np.zeros(( trajectories.shape[0] * maxSteps * maxSteps, trajectories.shape[1], trajectories.shape[2] ), dtype=int)
    choice_indices_all = np.zeros(stateList_parent.shape[0], dtype=int)
    count1 = 0
    for index1 in range(stateList.shape[0]):

        stateNow = stateList[index1:index1+1]
        args1 = np.argwhere(stateList[index1] == 1)
        stateNowParents = np.copy(stateNow)[np.zeros(args1.shape[0], dtype=int )]
        stateNowParents[np.arange(args1.shape[0]), args1[:, 0], args1[:, 1]] = 0

        choice_indices = edge_indices_inverse[args1[:, 0], args1[:, 1]]

        size1 = stateNowParents.shape[0]
        stateList_parent[count1:count1+size1] = stateNowParents
        choice_indices_all[count1:count1+size1] = choice_indices
        count1 += size1 
    
    stateList_parent = stateList_parent[:count1]
    choice_indices_all = choice_indices_all[:count1]

    inverse2 = uniqueValMaker(stateList_parent.reshape((stateList_parent.shape[0], stateList.shape[1]*stateList.shape[2] )))
    _, index2 = np.unique(inverse2, return_index=True)

    

    #stateNowParents = torch.tensor(stateNowParents).float()

    #choice_indices = edge_indices_inverse[args1[:, 0], args1[:, 1]]
    flowIn = model(  torch.tensor(stateList_parent[index2].reshape((-1, num_nodes*num_nodes)) ).float()    )
    flowIn = flowIn[inverse2]
    flowIn = flowIn[np.arange(choice_indices_all.shape[0]), choice_indices_all]
    #flowIn = torch.logsumexp(flowIn, axis=0)

    count1 = 0
    flowInSum = torch.zeros(flowOut.shape[0])
    for index1 in range(stateList.shape[0]):
        stateNow = stateList[index1:index1+1]
        args1 = np.argwhere(stateList[index1] == 1)
        size1 = args1.shape[0]

        flowIn_now = flowIn[count1:count1+size1]
        flowIn_now = torch.logsumexp(flowIn_now, axis=0)
        flowInSum[index1] = flowIn_now



        count1 += size1 




    #print (flowInSum[:5])
    #quit()

    error1 = torch.sum( ( flowOut - flowInSum) ** 2)

    #print ('error1', error1)


    
    flowInReward = model(torch.tensor(adjacency_matrices.reshape((-1, num_nodes*num_nodes))   ).float())
    flowInReward = flowInReward[:, -1]

    error2 =  torch.sum( ( rewards_log - flowInReward) ** 2)

    error_all = (error1 + error2) / (flowInSum.shape[0] + flowInReward.shape[0])

    #print ('error1, 2', error1, error2)

    return error_all


# Function to train the model and save it
def train_GFlowNet(learning_rate, num_nodes, batch_size, observations_batch, num_epochs, model_filename, rewardType=''):

    model = GraphGeneratorNet(num_nodes)
    #learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate , betas=(0.9, 0.99)) #good
    
    

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Generate graphs using pi'
        adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories = generate_graph_batch_with_modified_policy(
            model, num_nodes, batch_size, observations_batch, graphSize)
        
        pr_x_given_g_matrix = multi_x_given_g(adjacency_matrices, observations_batch)

        
        importance_weights_log = (log_prob_pi - log_prob_pi_prime).detach()
        importance_weights = torch.exp(importance_weights_log)  # Detach to prevent gradient flow       
        pr_x_given_g_matrix = pr_x_given_g_matrix + importance_weights_log.reshape((-1, 1))


        X_probs = torch.logsumexp(pr_x_given_g_matrix, dim=0)
        
            
        rewards_log = torch.logsumexp(pr_x_given_g_matrix, dim=1)
        rewards_log = rewards_log.detach()

        min1 =  torch.min(rewards_log[rewards_log != -np.inf])
        rewards_log[rewards_log == -np.inf] = min1


        #time1 = time.time()
        loss, flowInReward = flowMatching(trajectories, adjacency_matrices, rewards_log, model)


        print ("A")
        print (flowInReward[:10])
        print (rewards_log[:10])

        #time2 = time.time()

        #loss_fast = fast_flowMatching(trajectories, adjacency_matrices, rewards_log, model)

        #time3 = time.time()

        #print ('time', time2 - time1,  time3 - time2)

        #print ('loss', loss, loss_fast)
        #quit()



        # Backpropagation
        loss.backward()
        optimizer.step()

        avg_reward = X_probs.mean().item()


        print(f"Epoch {epoch + 1}/{num_epochs}, Average Reward: {avg_reward}", 'loss', loss)
        

        # Save the trained model
        torch.save(model, model_filename)
        #print(f"Model saved to {model_filename}")



def localSolver(graphSize, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g):

    batch_size = observations_batch.shape[0]


    #predict_graphs = np.zeros(observations_batch.shape, dtype=int)
    predict_graphs = np.zeros( (batch_size, graphSize) , dtype=int)

    #edge_indices = np.array([ [i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j])

    for a in range(observations_batch.shape[0]):
        print (a)
        obs1 = observations_batch[a]

        graph1 = np.ones(graphSize, dtype=int)
        

        continue1 = True 
        while continue1:
            #edge_indices_rand = edge_indices[np.random.permutation(edge_indices.shape[0])]

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
                reward0 = np.max(rewardList)

            else:
                continue1 = False

            #print (np.sum(graph1))
            #print (reward0)
        predict_graphs[a] = graph1
        #quit()
        #quit()
    
    return predict_graphs
                    


def metropolas(graphSize, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g):


    def local_totalReward(graph, observations_batch, multi_x_given_g):

        with torch.no_grad():
            graph = torch.tensor(graph).float()
            #print (graph.shape)
            #quit()
            rewardList = multi_x_given_g(graph.reshape((1, graph.shape[0])), observations_batch)
            rewardList = torch.logsumexp(rewardList[0], axis=0)
            rewardList = rewardList.data.numpy()
        
        return rewardList


    #num_nodes = observations_batch.shape[1]
    batch_size = observations_batch.shape[0]

    #edge_indices = np.array([ [i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j])

    graph = np.ones(graphSize, dtype=int)
    #graph = np.zeros(graphSize, dtype=int)


    epochs = 10000
    #epochs = 1000
    #samples = np.zeros((1000, observations_batch.shape[1], observations_batch.shape[2]), dtype=int)
    samples = np.zeros((1000, graphSize), dtype=int)
    M_sample = epochs // samples.shape[0]


    reward_old = local_totalReward(graph, observations_batch, multi_x_given_g)

    for iter in range(-1000, epochs):

        if iter >= 0:
            if iter % M_sample == 0:
                print (iter)
                print (np.sum(graph))
                samples[iter // M_sample] = np.copy(graph)



        #randEdge = edge_indices[  np.random.randint(edge_indices.shape[0]) ]  
        randEdge = np.random.randint(graphSize)

        graph_mod = np.copy(graph)
        graph_mod[randEdge] = 1 - graph_mod[randEdge]

        reward_now = local_totalReward(graph_mod, observations_batch, multi_x_given_g)

        #print (reward_now)

        alpha1 = np.exp(reward_now - reward_old)

        #print (reward_old, reward_now, alpha1)

        randU = np.random.random()

        if randU < alpha1:
            graph = np.copy(graph_mod)
            reward_old = reward_now


    #samples_flat = samples.reshape((samples.shape[0], samples.shape[1]*samples.shape[2]))
    #inverse1 = uniqueValMaker(samples_flat)
    #_, index1 = np.unique(inverse1, return_index=True)
    #_, count1 = np.unique(inverse1, return_counts=True)

    #samples = samples[index1]
    #logProb = np.log(count1)


    #print (inverse1.shape)
    #print (np.unique(inverse1).shape)
    #quit()

    

    rewardList = multi_x_given_g( torch.tensor(samples).float() , observations_batch)
    rewardList = rewardList.data.numpy()


    #rewardList = rewardList + logProb.reshape((-1, 1))

    #print (rewardList.shape)
    #print (logProb.shape)
    #quit()

    argBest = np.argmax(rewardList, axis=0)

    #print ('mean reward', np.mean(rewardList[argBest, np.arange(argBest.shape[0])]))
    #print ('max reward', np.max(rewardList[argBest, np.arange(argBest.shape[0])] ))

    predicted_graphs = samples[argBest]

    #print (np.max(argBest))

    #print (argBest.shape)
    #print (samples.shape)

    #print ('r1', rewardList[argBest, np.arange(argBest.shape[0])][0])


    #reward_new = multi_x_given_g( torch.tensor(predicted_graphs).float() , observations_batch)
    #reward_new = reward_new.data.numpy()
    #reward_new = reward_new[np.arange(reward_new.shape[0]), np.arange(reward_new.shape[0])]

    #print ('r2', reward_new[0])

    #print ('mean 2', np.mean(reward_new))
    
    return predicted_graphs


# ------------------------------------------------------
# A custom linear layer that applies an elementwise mask
# ------------------------------------------------------
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask):
        """
        Args:
            in_features: size of each input sample.
            out_features: size of each output sample.
            mask: a binary tensor of shape (out_features, in_features)
                  that will be multiplied elementwise with the weight matrix.
        """
        super().__init__(in_features, out_features)
        # Register the mask as a buffer (it’s not a parameter)
        self.register_buffer("mask", mask)

    def forward(self, input):
        # Multiply weight elementwise by mask before applying linear transformation.

        #print ("SIZES: ",  self.mask.shape,  self.weight.shape )

        return F.linear(input, self.mask * self.weight, self.bias)

# ------------------------------------------------------
# Autoregressive Model using masked fully-connected layers
# ------------------------------------------------------
class AutoregressiveMatrixModel(nn.Module):
    def __init__(self, graphSize, Nhidden):
        """
        Args:
            matrix_size: the height/width of the (square) matrix.
            hidden_dim: dimension of the hidden layer. For simplicity,
                        if not provided we use the same dimension as the sequence length.
        """
        super().__init__()
        self.graphSize = graphSize
        # Total tokens: flattened matrix entries plus one start token.
        self.seq_length = graphSize + 1
        
        # Our vocabulary: {0, 1} for the matrix entries and 2 for the start token.
        # We embed tokens into a scalar (1-dimensional space) so that our network input
        # is a vector of fixed length (seq_length).
        self.embedding = nn.Embedding(3, 1)
        
        # For simplicity, if no hidden dimension is provided, use the sequence length.
        #if hidden_dim is None:
        #    hidden_dim = self.seq_length

        # --------------------------------------------
        # Create masks that enforce autoregressive order.
        # For a fully connected layer with input/output dimension = N,
        # a strictly lower-triangular mask (zeros on and above the diagonal)
        # forces the i-th output to depend only on inputs with index < i.
        # --------------------------------------------
        # Mask for the first layer: shape (seq_length, seq_length)
        mask_in_hidden = torch.tril(torch.ones(self.seq_length, self.seq_length), diagonal=-1)
        
        # For the output layer, we need to produce two logits per token.
        # We think of the output as having shape (seq_length, 2); that is,
        # for each token position i the network produces 2 logits.
        # We first build a base mask (for seq_length outputs) and then repeat each row twice.
        mask_hidden_out_base = torch.tril(torch.ones(self.seq_length, self.seq_length), diagonal=-1)
        mask_hidden_out = mask_hidden_out_base.repeat_interleave(2, dim=0)  # shape: (seq_length*2, seq_length)

        #print ('mask_hidden_out', mask_in_hidden.shape)

        # Two masked layers:
        # 1. From the flattened input (of dimension seq_length) to a hidden vector.
        self.fc1 = MaskedLinear(self.seq_length, Nhidden, mask=mask_in_hidden)
        # 2. From hidden layer to output logits (of dimension seq_length*2)
        self.fc2 = MaskedLinear(Nhidden, self.seq_length * 2, mask=mask_hidden_out)
    
    def forward(self, x):
        """
        Args:
            x: LongTensor of shape (batch, seq_length) representing token indices.
        Returns:
            logits: FloatTensor of shape (batch, seq_length, 2) giving
                    the binary logits for each token position.
        """

        if x.shape[1] == self.graphSize:
            startPart = torch.ones((x.shape[0], 1)).int() * 2
            x = torch.cat((startPart, x), axis=1).int()

        batch_size = x.size(0)
        # Embed each token; result is (batch, seq_length, 1).
        x_emb = self.embedding(x)
        # Squeeze the last dimension to obtain a (batch, seq_length) vector.
        x_flat = x_emb.squeeze(-1)
        # Apply the first masked layer and ReLU nonlinearity.

        #print ("A")
        #print (x_flat.shape)
        #print (self.seq_length)
        h = self.fc1(x_flat)
        h = F.relu(h)
        # Apply the output masked layer.
        out = self.fc2(h)  # shape: (batch, seq_length*2)
        # Reshape to (batch, seq_length, 2); each token gets two logits.
        logits = out.view(batch_size, self.seq_length, 2)
        return logits

    @torch.no_grad()
    def generate(self, batch_size=1):
        """
        Autoregressively generate a batch of new binary matrices.
        
        Args:
            device: The device on which to run generation.
            batch_size: Number of matrices to generate.
        
        Returns:
            generated: Tensor of shape (batch_size, matrix_size, matrix_size)
                    containing the generated binary matrices.
        """
        self.eval()
        # Start with a batch of start tokens (token index 2).
        generated = torch.full((batch_size, 1), 2, dtype=torch.long)#, device=device)
        
        # Generate until we have a complete sequence of tokens (seq_length tokens).
        for i in range(1, self.seq_length):
            # For positions not yet generated, pad with zeros.
            # (These padding tokens won't affect the output because of the mask.)

            
            if generated.size(1) < self.seq_length:
                #print ("HI1")
                pad = torch.zeros((batch_size, self.seq_length - generated.size(1)),
                                dtype=torch.long)#, device=device)
                cur = torch.cat([generated, pad], dim=1)
            else:
                #print ("HI2")
                cur = generated
            #print (cur.shape)
            logits = self.forward(cur)  # shape: (batch_size, seq_length, 2)
            # Use the logits at position i for each sequence in the batch.
            logits_i = logits[:, i, :]   # shape: (batch_size, 2)
            probs = F.softmax(logits_i, dim=-1)
            # Sample the next token for each sequence in the batch.
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (batch_size, 1)
            generated = torch.cat([generated, next_token], dim=1)


        #print ('generated')
        #print (generated.shape)
        # Remove the start token and reshape the tokens to matrices.
        # The generated tensor is of shape (batch_size, seq_length) with seq_length = matrix_size*matrix_size + 1.
        # We remove the first token and then reshape each sequence to (matrix_size, matrix_size).
        generated = generated[:, 1:]  # shape: (batch_size, matrix_size*matrix_size)
        #generated = generated.view(batch_size, self.matrix_size, self.matrix_size)
        return generated




def autoregressive_inferGivenSample(graphs_all, observations_batch, model, multi_x_given_g, doSimple=True, doRandom=False):

    time1 = time.time()

    if not doSimple:
        input_seq = graphs_all.long()
        logits = model(input_seq)
        logits = logits[:, 1:, :]
        logits = nn.LogSoftmax(dim=2)(logits)

        #print (logits[:, :, 0].shape, input_seq.shape)
        
        #prob_est = torch.mean(logits[:, :, 0] * (1-input_seq) , axis=1) + torch.mean(logits[:, :, 1] * input_seq, axis=1)
        prob_est = torch.sum(logits[:, :, 0] * (1-input_seq) , axis=1) + torch.sum(logits[:, :, 1] * input_seq, axis=1)

        #quit()

    #print (time.time() - time1)
    #time2 = time.time()
    
    #print (graphs_all.shape, observations_batch.shape)
    #quit()

    pr_x_given_g_matrix = multi_x_given_g(graphs_all, observations_batch)

    #print (time.time() - time2)
    #time3 = time.time()


    X_probs = torch.logsumexp(pr_x_given_g_matrix, dim=0)

    pr_x_given_g_matrix = pr_x_given_g_matrix.data.numpy()

    if not doRandom:##doSimple:
        #prob_est = prob_est.data.numpy()

        inverse1 = uniqueValMaker(graphs_all.data.numpy())
        _, count1 = np.unique(inverse1, return_counts=True)
        prob_est = np.log(count1[inverse1])

        pr_x_given_g_matrix_adj = pr_x_given_g_matrix + prob_est.reshape((-1, 1))

        argBest = np.argmax( pr_x_given_g_matrix_adj , axis=0 )

    else:
        
        
        
        pr_x_given_g_matrix[np.isnan(pr_x_given_g_matrix)] = -np.inf
        pr_x_given_g_matrix[pr_x_given_g_matrix == -np.inf] = np.min(pr_x_given_g_matrix[pr_x_given_g_matrix != -np.inf]) - 100000

        
        probs = torch.tensor(softmax(pr_x_given_g_matrix, axis=0)).float()

        #repeatNum = probs.shape[0] // probs.shape[1]

        randomSubset = np.random.choice(probs.shape[1], size=probs.shape[0])
        argBest = torch.multinomial(probs.T[randomSubset], num_samples=1)


        #if repeatNum == 0:
        #    randomSubset = np.random.permutation(probs.shape[1])[:probs.shape[0]]
        #    argBest = torch.multinomial(probs.T[randomSubset], num_samples=1)
        #else:
        #    argBest = torch.multinomial(probs.T, num_samples=repeatNum)
        
        argBest = argBest.reshape((-1,))
        argBest = argBest.data.numpy()

    #print (time.time() - time3)

    #print (argBest.shape)
    #print (repeatNum)
    #quit()
    
    return argBest, X_probs




# ------------------------------------------------------
# Training loop
# ------------------------------------------------------
def autoregressive_train_model(Nhidden, ruleObject, observations_batch, learning_rate, model_filename, multi_x_given_g, num_epochs=10):
    

    #batch_size = observations_batch.shape[0]

    graphSize = ruleObject.graphSize
    batch_size = ruleObject.batchSize

    #print (batch_size)
    #quit()

    #Initialize the masked fully-connected autoregressive model.
    model = AutoregressiveMatrixModel(graphSize, Nhidden=graphSize+1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    #graphs = torch.zeros(observations_batch.shape)
    graphs = torch.zeros( (batch_size, graphSize ) )

    metricMax = -np.inf
    #nearMetric = np.zeros(100) 
    nearMetric = np.zeros(500) 
    nearMetric[:] = -np.inf
    continue1 = True 

    model.train()
    epoch = -1
    while continue1 and (epoch < num_epochs):
        epoch += 1

        if True:#epoch % 10 == 0:
            with torch.no_grad():
                #time1 = time.time()
                #graphs_all = model.generate(batch_size=batch_size)
                #time2 = time.time()
                #graphs_all = torch.cat(( graphs, sample1 ), axis=0)
                #argBest, X_probs = autoregressive_inferGivenSample(graphs_all, observations_batch, model, multi_x_given_g, doSimple=False, doRandom=False)
                #graphs = graphs_all[argBest]

                graphs_all = sampleAutoreg(model, batch_size)
                graphs, X_probs = simpleGeneralPredictor(graphs_all, observations_batch, multi_x_given_g, return_xProbs=True)
                graphs = torch.tensor(graphs)

                #print (time.time() - time2)
                #quit()

        
        input_seq = graphs.long()
        #input_seq = input_seq.to(device).long()    # shape: (batch, seq_length)
        optimizer.zero_grad()
        logits = model(input_seq)           # shape: (batch, seq_length, 2)
        # We only compute loss for positions corresponding to actual matrix entries;
        # ignore the first position (the start token).
        logits = logits[:, 1:, :]            # shape: (batch, seq_length-1, 2)
        loss = F.cross_entropy(logits.reshape(-1, 2), input_seq.reshape(-1))
        loss.backward()
        optimizer.step()

        metricMax = max(metricMax, nearMetric[epoch % nearMetric.shape[0]] )
        nearMetric[epoch % nearMetric.shape[0]] = -1 * loss.item()

        #if epoch % 100 == 0:
        torch.save(model, model_filename)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss:" , loss.item(), 'median reward', torch.median(X_probs), 'mean reward', torch.mean(X_probs))

        if epoch > nearMetric.shape[0]:
            print (np.max(nearMetric), metricMax)
            if not (np.max(nearMetric) > (metricMax + 1e-3)):
                continue1 = False

        


# ------------------------------------------------------
# Main: create dataset, train, and sample from the model.
# ------------------------------------------------------


def autoregressive_pred(model, observations_batch, multi_x_given_g):


    #batch_size = observations_batch.shape[0]
    graphs_all = model.generate(batch_size=10000)
    argBest, X_probs = autoregressive_inferGivenSample(graphs_all, observations_batch, model, multi_x_given_g, doSimple=False, doRandom=False)

    predict_graphs = graphs_all[argBest]
    predict_graphs = predict_graphs.data.numpy()

    return predict_graphs

def sampleAutoreg(model, sampleSize):

    with torch.no_grad():
        batchSize = 1000
        numBatch = sampleSize // batchSize
        for batchIndex in range(numBatch):
            graphs_all0 = model.generate(batch_size=batchSize)
            graphs_all0 = graphs_all0.data.numpy()
            if batchIndex == 0:
                graphs_all = np.copy(graphs_all0)
            else:
                graphs_all = np.concatenate(( graphs_all, graphs_all0 ), axis=0)
    return graphs_all



class MLPDiffusionModel(nn.Module):
    def __init__(self, input_dim, TIMESTEPS, Nhidden, time_embed_dim=16):
        super(MLPDiffusionModel, self).__init__()

        self.time_embed_dim = time_embed_dim
        self.TIMESTEPS = TIMESTEPS
        # Time Embedding Layer (maps scalar time index to a vector)
        #self.time_embedding = nn.Embedding(TIMESTEPS, time_embed_dim)

        # Fully Connected Network (MLP)
        # The input dimension is increased by time_embed_dim
        self.fc1 = nn.Linear(input_dim + time_embed_dim, Nhidden)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(Nhidden, input_dim)  # Predict noise (raw logits)

    def forward(self, A_t, t):
        """
        Predicts flip probabilities (as raw logits) for each edge in each graph in the batch.
        A_t: Tensor of shape (batch_size, INPUT_DIM)
        t: Tensor of shape (batch_size,)
        """
        # Get time embeddings: shape (batch_size, time_embed_dim)
        #t_embed = self.time_embedding(t)  
        time_embed_dim = self.time_embed_dim
        t_embed = torch.stack([ torch.cos(   t.float()/self.TIMESTEPS  *  np.pi * freq ) for freq in range(1, time_embed_dim + 1) ]).T
        # Concatenate along last dimension: (batch_size, INPUT_DIM + time_embed_dim)

        #print (A_t)
        #print (t_embed.shape)
        x = torch.cat([A_t, t_embed], dim=-1)  
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        noise_pred = self.fc3(x)  # Output raw logits; shape: (batch_size, INPUT_DIM)
        return noise_pred



def forward_diffusion(A, beta, t):
    """
    Adds noise (random edge flips) to a batch of adjacency matrices.
    A: Tensor of shape (batch_size, INPUT_DIM)
    beta: Tensor of shape (TIMESTEPS,) defining noise schedule.
    t: Tensor of shape (batch_size,) containing a time index for each graph.
    Returns:
      A_noisy: Noisy adjacency matrices of shape (batch_size, INPUT_DIM)
      noise_A: Indicator (0 or 1) of which entries were flipped (same shape)
    """
    # beta[t] gives a tensor of shape (batch_size,). Unsqueeze to (batch_size, 1)
    b = beta[t].unsqueeze(1)
    mask = torch.rand_like(A.float()) < b  # mask shape: (batch_size, INPUT_DIM)
    A_noisy = A.clone()
    A_noisy[mask] = 1 - A_noisy[mask]  # Flip edges
    noise_A = (A_noisy != A).long()
    return A_noisy, noise_A

def predict_noise(model, A_t, t):
    """
    Predicts flip probabilities (via raw logits) for every edge in the batch.
    A_t: Noisy adjacency matrices, shape (batch_size, INPUT_DIM)
    t: Time indices, shape (batch_size,)
    Returns:
      noise_logits: Raw logits from the model (for BCEWithLogitsLoss), shape (batch_size, INPUT_DIM)
      predicted_noise: Binary decisions sampled from sigmoid probabilities, shape (batch_size, INPUT_DIM)
    """
    noise_logits = model(A_t, t)  # Raw logits, shape: (batch_size, INPUT_DIM)
    flip_probs = torch.sigmoid(noise_logits)  # Convert logits to probabilities
    predicted_noise = torch.bernoulli(flip_probs)  # Sample binary flip decisions
    return noise_logits, predicted_noise




def diffusion_sample_graphs(model, timesteps, batch_size, INPUT_DIM):
    """
    Samples graphs from the trained diffusion model.
    
    The process starts with a random binary graph and then iteratively
    reverses the diffusion steps. At each reverse step, the model predicts
    which edges should be flipped, and we flip them accordingly.
    
    Returns:
      A_sample: Tensor of shape (batch_size, num_nodes * num_nodes) containing
                the sampled graphs (flattened).
    """

    # Start with a completely random graph for each sample
    A_t = torch.randint(0, 2, (batch_size, INPUT_DIM)).float()#.to(device)
    
    # Iterate backward over timesteps
    for t_val in reversed(range(timesteps)):
        # Create a tensor filled with the current timestep (for each graph in the batch)
        t_tensor = torch.full((batch_size,), t_val, dtype=torch.long)#, device=device)
        # Predict noise for the current noisy graph
        noise_logits, predicted_noise = predict_noise(model, A_t, t_tensor)
        # Reverse the diffusion step:
        # For edges where predicted_noise == 1, flip the edge; else leave it unchanged.
        A_t = torch.where(predicted_noise == 1, 1 - A_t, A_t)
        
    return A_t





def diffusion_train(Nhidden, ruleObject, multi_x_given_g, log_calculate_pr_x_given_g, observations_batch, model_filename, TIMESTEPS):

    

    #TIMESTEPS = 10             # Number of diffusion steps
    graphSize = ruleObject.graphSize
    batchSize = ruleObject.batchSize
    batch_size = batchSize
    #batch_size = observations_batch.shape[0]
    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPDiffusionModel(graphSize, TIMESTEPS, Nhidden).to(device)


    # Optimizer and loss function
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) #Used
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    loss_fn = nn.BCEWithLogitsLoss()  # Binary classification loss

    # Define noise schedule (beta values) as a tensor of shape (TIMESTEPS,)
    beta = torch.linspace(0.01, 0.1, TIMESTEPS).to(device)  # Increasing corruption over time

    #graphs = torch.zeros( (batch_size, graphSize ) )
    # Training loop
    Niter = 10000

    metricMax = -np.inf
    #nearMetric = np.zeros(1000) 
    nearMetric = np.zeros(5000) 
    nearMetric[:] = -np.inf
    continue1 = True

    #for epoch in range(Niter):  
    epoch = -1
    while continue1:
        epoch += 1

        if True:#epoch % 10 == 0:
            with torch.no_grad():
                graphs_all = diffusion_sample_graphs(model, TIMESTEPS, batchSize, graphSize)
                #graphs_all = sample1#graphs_all = torch.cat(( graphs, sample1 ), axis=0)
                #argBest, X_probs = VAE_inferGivenSample(graphs_all, observations_batch, model, multi_x_given_g, doSimple=True, doRandom=True)
                #argBest, X_probs = VAE_inferGivenSample(graphs_all, observations_batch, model, multi_x_given_g, doSimple=True, doRandom=False)
                #graphs = graphs_all[argBest]

                graphs, X_probs = simpleGeneralPredictor(graphs_all, observations_batch, multi_x_given_g, return_xProbs=True)
                graphs = torch.tensor(graphs).float()

                #print (graphs.shape)
                #quit()


        
        optimizer.zero_grad()
        
        # Pick a random timestep for each graph in the batch: shape (BATCH_SIZE,)
        t = torch.randint(0, TIMESTEPS, (graphs.shape[0],)).to(device)
        
        # Apply forward diffusion to obtain noisy graphs and the corresponding noise indicator
        A_noisy, noise_A = forward_diffusion(graphs, beta, t)
        
        # Predict noise (flip probabilities) using the model
        noise_logits, predicted_noise_A = predict_noise(model, A_noisy, t)
        
        # BCEWithLogitsLoss expects target values as floats
        noise_A = noise_A.float()
        
        # Compute loss between the model's raw logits and the true noise indicator
        loss = loss_fn(noise_logits, noise_A)

        metricMax = max(metricMax, nearMetric[epoch % nearMetric.shape[0]]) 

        nearMetric[epoch % nearMetric.shape[0]] = loss.item() * -1
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}", 'Rewards', torch.mean(X_probs))

        torch.save(model, model_filename)

        if False:#epoch == 10000:
            continue1 = False

        if True:#epoch > nearMetric.shape[0]:

            if epoch % 100 == 0:
                print (np.max(nearMetric), metricMax)

            if not (np.max(nearMetric) > (metricMax + 1e-3)):
                continue1 = False
                print (np.max(nearMetric), metricMax)



def diffusion_pred(model, observations_batch, multi_x_given_g, graphSize):


    TIMESTEPS = 10    
    batch_size = observations_batch.shape[0]


    num_samples = 1000
    sample1 = diffusion_sample_graphs(model, TIMESTEPS, num_samples, graphSize)
    argBest, _ = VAE_inferGivenSample(sample1, observations_batch, model, multi_x_given_g, doSimple=True, doRandom=False)
    sample1 = sample1[argBest]
    sample1 = sample1.data.numpy()


    return sample1




