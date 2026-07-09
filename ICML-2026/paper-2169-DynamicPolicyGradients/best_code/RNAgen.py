import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
import time

import math
import os

from sharedGen import *
#from baselines import *
import pandas as pd


# Parameters
#num_simulations = 20
#num_data_points = 100
#num_nodes = 10  # Reduced to 10 nodes for faster runtimes
#num_nodes = 3
#num_paths_per_graph = 10


#Epoch 541/5000, Average Reward: -2.018215434968489




import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


np.random.seed(1)
torch.manual_seed(0)


import warnings
warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",  # regex match on that specific message
    category=FutureWarning
)



class ToySpliceClass(gClass):


    def __init__(self, finalProbSize, Njunction, Nsample):
        super().__init__(finalProbSize=finalProbSize)  # call parent init to set self.value
        self.finalProbSize = finalProbSize
        self.Njunction = Njunction
        self.Nsample = Nsample
        #self.name = name  


    def graphRules(self, adjacency_matrices):

        boolUse = torch.sum(adjacency_matrices, axis=1)
        graphNew = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]+1))
        graphNew[boolUse == 1, :-1] = -float('inf')
        graphNew[boolUse == 0, -1] = -float('inf')

        finalProbAllow = torch.zeros((adjacency_matrices.shape[0], self.finalProbSize))

        return graphNew, finalProbAllow

    def offPolicyRule(self, adjacency_matrices, arange1):

        junction = arange1 % self.Njunction
        boolHasNeeded = adjacency_matrices[arange1, junction]

        graphNew = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]+1))
        graphNew[:] = -float('inf')
        graphNew[boolHasNeeded == 0, junction[boolHasNeeded==0]] = 0
        graphNew[boolHasNeeded == 1, -1] = 0

        finalProbAllow = torch.zeros((adjacency_matrices.shape[0], self.finalProbSize))
        
        return graphNew, finalProbAllow
    

    def rewardFunction(self, adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, finalProb, giveInfo=False):


        pr_x_given_g_matrix = torch.log(adjacency_matrices + 1e-10) - np.log(adjacency_matrices.shape[1])
        pr_x_given_g_matrix = pr_x_given_g_matrix.reshape((pr_x_given_g_matrix.shape[0], pr_x_given_g_matrix.shape[1], 1)) #batchSize, Junction, Sample
        #prob_mult = probs.reshape((probs.shape[0], probs.shape[1], 1)) * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))

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
        
        if False:
            batchCorrection = np.log(batchSize - 1) - np.log(observations_all.shape[0] - 1)
            pr_x_given_g_matrix_adjusted[np.arange(batchSize*dupGen), np.arange(batchSize).repeat(dupGen)] -= batchCorrection

        #print ('pr_x_given_g_matrix_adjusted')
        #print (pr_x_given_g_matrix_adjusted)

        #print ('rrr')

        #print (pr_x_given_g_matrix_adjusted.shape)

        #print ('pr_x_given_g_matrix_adjusted')
        #print (pr_x_given_g_matrix_adjusted)
        #quit()

        #pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix_adjusted * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1])) #Adds weighting based on number of reads for each

        #rewards = torch.logsumexp(pr_x_given_g_matrix_adjusted, dim=1) #IS THIS LINE OK????
        #rewards = torch.exp(rewards)

        rewards = torch.exp(pr_x_given_g_matrix_adjusted)
        rewards = rewards * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1])) / torch.mean(obs_matrix)# Adds weighting based on number of reads for each
        rewards = torch.sum(rewards, dim=1)
        
        rewards = rewards * (pr_x_given_g_matrix_adjusted.shape[0] / pr_x_given_g_matrix_adjusted.shape[1])

        #print ('rewards', rewards)
        #quit()

        if giveInfo:
            info = [pr_x_given_g_matrix]
            return X_probs, rewards, info
        else:
            return X_probs, rewards




#sample_gClass
class SpliceClass(sample_gClass):


    def __init__(self, finalProbSize, Njunction, Nsample, edgeMatrix):
        super().__init__(finalProbSize=finalProbSize)  # call parent init to set self.value
        self.finalProbSize = finalProbSize
        self.Njunction = Njunction
        self.Nsample = Nsample
        self.edgeMatrix = edgeMatrix

        edgeMatrix_log = torch.tensor(edgeMatrix).float()
        edgeMatrix_log[edgeMatrix_log == 0] = -float('inf')
        edgeMatrix_log[edgeMatrix_log == 1] = 0

        self.edgeMatrix_log = edgeMatrix_log


        if False:
            #Calculating the off policy 
            graphPolicy_mini = torch.tensor(edgeMatrix).long()
            graphPolicy_mini[np.arange(graphPolicy_mini.shape[0]), np.arange(graphPolicy_mini.shape[0])] = 1
            sum1 = 0
            while torch.sum(graphPolicy_mini) != sum1:
                sum1 = torch.sum(graphPolicy_mini)
                graphPolicy_mini = torch.mm(graphPolicy_mini, graphPolicy_mini)
                graphPolicy_mini[graphPolicy_mini >= 1] = 1
                
            
            graphPolicy_mini = graphPolicy_mini.float().T
            graphPolicy_mini[graphPolicy_mini == 0] = -float('inf')
            graphPolicy_mini[graphPolicy_mini == 1] = 0

            graphPolicy = torch.zeros((self.Njunction, self.Njunction+1))
            graphPolicy[:, :-1] = graphPolicy_mini
            self.graphPolicy = graphPolicy

        #self.name = name  


    def OLD_graphRules(self, adjacency_matrices):

        #print ('')

        #rint ('adjacency_matrices', adjacency_matrices)

        edgeMatrix = np.copy(self.edgeMatrix)
        sum1 = np.sum(edgeMatrix, axis=0)

        #plt.imshow(edgeMatrix)
        #plt.show()
        #quit()

        #print ('sum1', sum1)
        assert 0 in sum1
        
        introJunctions = np.argwhere(sum1 == 0)[:, 0] 
        sum2 = np.sum(edgeMatrix, axis=1)
        #endJunctions = np.argwhere(sum2 == 0)[:, 0] 


        #edgeMatrix[np.arange(edgeMatrix.shape[0]), np.arange(edgeMatrix.shape[0])] = edgeMatrix[np.arange(edgeMatrix.shape[0]), np.arange(edgeMatrix.shape[0])] + 2

        #plt.imshow(edgeMatrix)
        #plt.show()
        #quit()

        boolUse = torch.sum(adjacency_matrices, axis=1)

        #print ('boolUse', boolUse)
        
        argUse = torch.argwhere(boolUse >= 1)[:, 0]
        argNoUse = torch.argwhere(boolUse == 0)[:, 0]
        graphNew = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]+1))
        graphNew[:] = -float('inf')

        M = adjacency_matrices.size(1)
        indices = torch.arange(M).expand_as(adjacency_matrices)
        # Mask with X (so only positions with 1 keep their index; others become -1)
        masked = torch.where(adjacency_matrices.bool(), indices, torch.full_like(adjacency_matrices, -1))
        # Take max along each row — gives last index where value is 1
        lastJunction = masked.max(dim=1).values
        lastJunction = lastJunction[argUse].data.numpy().astype(int)

        nextJunction = torch.tensor(edgeMatrix).float()[lastJunction]
        nextJunction[nextJunction == 0] = -float('inf')
        nextJunction[nextJunction == 1] = 0

        #print ('argUse', argUse)
        #print ('lastJunction', lastJunction)

        
        
        argUse[sum2[lastJunction] == 0]

        endJunctions = argUse[sum2[lastJunction] == 0]

        #print (boolUse)
        #print (introJunctions)

        graphIntro = torch.zeros((argNoUse.shape[0], adjacency_matrices.shape[1]+1))
        graphIntro[:] = -float('inf')
        graphIntro[:, introJunctions] = 0

        #print ('introJunctions', introJunctions)

        assert not 0 in graphNew[:, :-1][adjacency_matrices == 1]
        
        graphNew[boolUse == 0] = graphIntro


        assert not 0 in graphNew[:, :-1][adjacency_matrices == 1]

        graphNew[argUse, :-1] = nextJunction

        assert not 0 in graphNew[:, :-1][adjacency_matrices == 1]

        graphNew[endJunctions, -1] = 0


        assert not 0 in graphNew[:, :-1][adjacency_matrices == 1]

        graphExist = graphNew[:, :-1][adjacency_matrices == 1]
        assert not 0 in graphExist

        for a in range(graphNew.shape[0]):
            if not 0 in graphNew[a]:
                print (torch.sum(adjacency_matrices[a]))
                print (adjacency_matrices[a])
                quit()
            assert 0 in graphNew[a]

        #print ('endJunctions', endJunctions)

        
        #print ('graphNew', graphNew)

        finalProbAllow = torch.zeros((adjacency_matrices.shape[0], self.finalProbSize))

        return graphNew, finalProbAllow
    


    def graphRules(self, adjacency_matrices):

        

        time1 = time.time()

        
        #edgeMatrix = np.copy(self.edgeMatrix)
        

        #plt.imshow(edgeMatrix)
        #plt.show()

        

        graphNew = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]+1))
        graphNew[:] = -float('inf')

        

        

        M = adjacency_matrices.size(1)
        indices = torch.arange(M).expand_as(adjacency_matrices)
        # Mask with X (so only positions with 1 keep their index; others become -1)
        masked = torch.where(adjacency_matrices.bool(), indices, torch.full_like(adjacency_matrices, -1))
        # Take max along each row — gives last index where value is 1
        lastJunction = masked.max(dim=1).values.long()
        #print (lastJunction)
        #quit()

        time1 = time.time() - time1 
        time2 = time.time()

        

        #graphNew = torch.tensor(edgeMatrix).float()[lastJunction]
        graphNew = self.edgeMatrix_log[lastJunction]

        time2 = time.time() - time2
        time3 = time.time()


        #graphNew[graphNew == 0] = -float('inf')
        #graphNew[graphNew == 1] = 0

        
        #Removed for speed
        #assert not 0 in graphNew[:, :-1][adjacency_matrices == 1]

        

        #This assertion is very slow!!
        #for a in range(graphNew.shape[0]):
        #    assert 0 in graphNew[a]

        

        


        #finalProbAllow = torch.zeros((adjacency_matrices.shape[0], self.finalProbSize))
        finalProbAllow = torch.zeros((adjacency_matrices.shape[0], 1))

        
        time3 = time.time() - time3
        

        #print ('miniTime', time1, time2, time3)

        return graphNew, finalProbAllow

    def offPolicyRule(self, adjacency_matrices, arange1):

        #print (adjacency_matrices.shape)
        #print (np.max(arange1), self.Njunction)

        junction = arange1 % self.Njunction
        boolHasNeeded = adjacency_matrices[np.arange(adjacency_matrices.shape[0]), junction].data.numpy()

        

        graphNew = torch.zeros((arange1.shape[0], self.Njunction+1))
        #graphNew[:] = -float('inf')
        #graphNew[boolHasNeeded == 0, junction[boolHasNeeded==0]] = 0
        #graphNew[boolHasNeeded == 1, -1] = 0

        #Temporary solution only for testing!
        #graphNew[0, 0] = 0
        #graphNew[0, 1] = 0
        #graphNew[1, 0] = 0
        #graphNew[1, 1] = 0
        #graphNew[2, 2] = 0
        #graphNew[:, -1] = 0

        

        graphPolicy = self.graphPolicy
        graphNew[boolHasNeeded == 0] = graphPolicy[junction[boolHasNeeded == 0]]

        #print ('graphPolicy', graphNew)

        finalProbAllow = torch.zeros((adjacency_matrices.shape[0], self.finalProbSize))
        
        return graphNew, finalProbAllow
    

    def rewardFunction(self, adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, finalProb, giveInfo=False):

        #import seaborn as sns

        #from matplotlib.colors import LogNorm
        #sns.heatmap(obs_matrix.data.numpy() , norm=LogNorm())
        #plt.show()
        #quit()

        timeList = []

        leakage = 0.1

        

        #pr_x_given_g_matrix = torch.log(adjacency_matrices + 1e-10) - np.log(adjacency_matrices.shape[1])
        #pr_x_given_g_matrix = torch.log(adjacency_matrices + 1e-10) - torch.log(torch.sum(adjacency_matrices + 1e-10, axis=1)).reshape((-1, 1))
        #pr_x_given_g_matrix = torch.log(adjacency_matrices + 1e-10) - torch.log(torch.sum(adjacency_matrices + 1e-10, axis=1)).reshape((-1, 1))

        timeList.append(time.time())

        #sizeNormalize = torch.log(  ((1.0 - leakage)* torch.sum(adjacency_matrices + 1e-10, axis=1)) + (leakage * adjacency_matrices.shape[1] )      ).reshape((-1, 1))
        sizeNormalize = torch.log( torch.sum(adjacency_matrices + 1e-10, axis=1)).reshape((-1, 1))
        pr_x_given_g_matrix = torch.log(adjacency_matrices + 1e-10) - sizeNormalize 

        pr_x_given_g_matrix[adjacency_matrices == 0] = -500



        pr_x_given_g_matrix = pr_x_given_g_matrix.reshape((pr_x_given_g_matrix.shape[0], pr_x_given_g_matrix.shape[1], 1)) #batchSize, Junction, Sample
        #prob_mult = probs.reshape((probs.shape[0], probs.shape[1], 1)) * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))

        if self.finalProbSize >= 2:
            importance_weights_log = (log_prob_pi - log_prob_pi_prime.reshape(((-1, 1)))  ).detach()
        else:
            importance_weights_log = (log_prob_pi - log_prob_pi_prime).detach()
        
        #print (self.finalProbSize)
        #print (importance_weights_log.shape)
        #quit()

        timeList.append(time.time())
        
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
        
        if False:
            batchCorrection = np.log(batchSize - 1) - np.log(observations_all.shape[0] - 1)
            pr_x_given_g_matrix_adjusted[np.arange(batchSize*dupGen), np.arange(batchSize).repeat(dupGen)] -= batchCorrection

        #print ('pr_x_given_g_matrix_adjusted')
        #print (pr_x_given_g_matrix_adjusted)

        #print ('X_probs')
        #print (X_probs.shape)
        #print (obs_matrix.shape)
        #quit()
        timeList.append(time.time())

        #sns.heatmap(observations_batch.data.numpy())
        #plt.show()
        if False:#int(time.time()) % 30 == 0:
            X_probs_copy = np.copy(X_probs.data.numpy())
            X_probs_copy[X_probs_copy < -50] = -50
            sns.heatmap(X_probs_copy)
            plt.show()
        

        
        X_probs = X_probs * obs_matrix

        timeList.append(time.time())

        #print ('rrr')

        #print (pr_x_given_g_matrix_adjusted.shape)

        #print ('pr_x_given_g_matrix_adjusted')
        #print (pr_x_given_g_matrix_adjusted)
        #quit()

        #pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix_adjusted * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1])) #Adds weighting based on number of reads for each

        #rewards = torch.logsumexp(pr_x_given_g_matrix_adjusted, dim=1) #IS THIS LINE OK????
        #rewards = torch.exp(rewards)

        rewards = torch.exp(pr_x_given_g_matrix_adjusted)
        #print (rewards.shape, obs_matrix.shape)
        #timeList.append(time.time())

        #print (rewards.shape)
        #print (obs_matrix.shape)

        

        #rewards_mod = rewards.clone()

        timeList.append(time.time())

        #rewards_mod[:, obs_matrix!=0] = rewards_mod[:, obs_matrix!=0] * obs_matrix[obs_matrix!=0].reshape((1, -1)) / torch.mean(obs_matrix)
        #rewards_mod[:, obs_matrix==0] = 0
        #rewards_mod = torch.sum(rewards_mod, dim=1)

        timeList.append(time.time())

        rewards = rewards * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1])) / torch.mean(obs_matrix)# Adds weighting based on number of reads for each
        rewards = torch.sum(rewards, dim=1)

        #print (torch.mean(torch.abs(rewards - rewards_mod)))

        timeList.append(time.time())

        

        #obs_matrix = (obs_matrix / obs_matrix.mean()).contiguous()        # (T, F)
        #rewards = torch.einsum('btf,tf->bf', rewards, obs_matrix)


        #timeList.append(time.time())
        
        rewards = rewards * (pr_x_given_g_matrix_adjusted.shape[0] / pr_x_given_g_matrix_adjusted.shape[1])

        
        

        if giveInfo:
            info = [pr_x_given_g_matrix]
            return X_probs, rewards, info
        else:
            return X_probs, rewards
        




class biasSpliceClass(SpliceClass):


    def __init__(self, finalProbSize, Njunction, Nsample, edgeMatrix):
        super().__init__(finalProbSize, Njunction, Nsample, edgeMatrix)  # call parent init to set self.value
        self.finalProbSize = finalProbSize
        self.Njunction = Njunction
        self.Nsample = Nsample
        self.edgeMatrix = edgeMatrix

    

    def rewardFunction(self, pr_x_given_g_matrix, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=False):

        

        timeList = []


        timeList.append(time.time())

        
        


        pr_x_given_g_matrix = pr_x_given_g_matrix.reshape((pr_x_given_g_matrix.shape[0], pr_x_given_g_matrix.shape[1], 1)) #batchSize, Junction, Sample
        #prob_mult = probs.reshape((probs.shape[0], probs.shape[1], 1)) * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))

        if self.finalProbSize >= 2:
            importance_weights_log = (log_prob_pi - log_prob_pi_prime.reshape(((-1, 1)))  ).detach()
        else:
            importance_weights_log = (log_prob_pi - log_prob_pi_prime).detach()
        
        #print (self.finalProbSize)
        #print (importance_weights_log.shape)
        #quit()

        timeList.append(time.time())
        
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
        
        if False:
            batchCorrection = np.log(batchSize - 1) - np.log(observations_all.shape[0] - 1)
            pr_x_given_g_matrix_adjusted[np.arange(batchSize*dupGen), np.arange(batchSize).repeat(dupGen)] -= batchCorrection

        #print ('pr_x_given_g_matrix_adjusted')
        #print (pr_x_given_g_matrix_adjusted)

        #print ('X_probs')
        #print (X_probs.shape)
        #print (obs_matrix.shape)
        #quit()
        timeList.append(time.time())

        #sns.heatmap(observations_batch.data.numpy())
        #plt.show()
        if False:#int(time.time()) % 30 == 0:
            X_probs_copy = np.copy(X_probs.data.numpy())
            X_probs_copy[X_probs_copy < -50] = -50
            sns.heatmap(X_probs_copy)
            plt.show()
        

        
        X_probs = X_probs * obs_matrix

        timeList.append(time.time())

        #print ('rrr')

        #print (pr_x_given_g_matrix_adjusted.shape)

        #print ('pr_x_given_g_matrix_adjusted')
        #print (pr_x_given_g_matrix_adjusted)
        #quit()

        #pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix_adjusted * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1])) #Adds weighting based on number of reads for each

        #rewards = torch.logsumexp(pr_x_given_g_matrix_adjusted, dim=1) #IS THIS LINE OK????
        #rewards = torch.exp(rewards)

        rewards = torch.exp(pr_x_given_g_matrix_adjusted)
        #print (rewards.shape, obs_matrix.shape)
        #timeList.append(time.time())

        #print (rewards.shape)
        #print (obs_matrix.shape)

        

        #rewards_mod = rewards.clone()

        timeList.append(time.time())

        #rewards_mod[:, obs_matrix!=0] = rewards_mod[:, obs_matrix!=0] * obs_matrix[obs_matrix!=0].reshape((1, -1)) / torch.mean(obs_matrix)
        #rewards_mod[:, obs_matrix==0] = 0
        #rewards_mod = torch.sum(rewards_mod, dim=1)

        timeList.append(time.time())

        rewards = rewards * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1])) / torch.mean(obs_matrix)# Adds weighting based on number of reads for each

        rewards = rewards * (pr_x_given_g_matrix_adjusted.shape[0] / pr_x_given_g_matrix_adjusted.shape[1])
        #rewards = rewards * pr_x_given_g_matrix_adjusted.shape[0] #Attempted JAN 31 2026

        
        

        if giveInfo:
            info = [pr_x_given_g_matrix]
            return X_probs, rewards, info
        else:
            return X_probs, rewards
        
    


    def lossFunction(self, adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=False):


        time1 = time.time()

        if (self.finalProbSize >= 2) and (adjacency_matrices.shape[0] >= 1):
            _, finalProbAllow = self.graphRules(adjacency_matrices)
            finalProb = self.model.finalProb(adjacency_matrices)
            finalProb = finalProb + finalProbAllow
            finalProb = nn.LogSoftmax(dim=1)(finalProb)
        else:
            finalProb = torch.zeros((adjacency_matrices.shape[0], 1))
    
        bias = self.model.giveBias()


        #leakage = torch.sum(adjacency_matrices, axis=1)
        #leakage = torch.log(leakage) - torch.log(leakage + 1)

        #print (leakage)
        #quit()

        pr_x_given_g_matrix = torch.log(adjacency_matrices + 1e-10)
        pr_x_given_g_matrix[adjacency_matrices == 0] = -500

        pr_x_given_g_matrix = pr_x_given_g_matrix + bias.reshape((1, -1))
        pr_x_given_g_matrix = torch.nn.LogSoftmax(dim=1)(pr_x_given_g_matrix)

        leakage, _ = torch.max(pr_x_given_g_matrix, axis=1)
        
        #argsort1 = np.argsort(pr_x_given_g_matrix.data.numpy(), axis=1 )
        #print (argsort1.shape)
        #quit()
        #leakage1 = pr_x_given_g_matrix[np.arange(argsort1.shape[0]),  argsort1[:, -1] ]
        #leakage2 = pr_x_given_g_matrix[np.arange(argsort1.shape[0]),  argsort1[:, -2] ]
        #leakage = torch.logaddexp(leakage1, leakage2)


        leakage = -1 * torch.logaddexp(leakage, torch.zeros(leakage.shape))
        #leakage = torch.log(1.0 - (0.95 * torch.exp(leakage)))

        #leakage2 = 

        #leakage = torch.logsumexp(pr_x_given_g_matrix * 2, axis=1)
        #leakage = 1.0 - (0.1 * torch.exp(leakage))
        #leakage = torch.log(leakage)


        #leakage = leakage * 0.5
        leakage = leakage * 2.0
        #leakage = leakage * 5.0

        #leakage = leakage * 3.0


        pr_x_given_g_matrix = pr_x_given_g_matrix  + leakage.reshape((-1, 1))

        time1 = time.time() - time1
        time2 = time.time()



        log_prob_pi = log_prob_pi.reshape((-1, 1)) + finalProb
        
        with torch.no_grad():
            if giveInfo:
                X_probs, rewards, info = self.rewardFunction(pr_x_given_g_matrix, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=giveInfo)
                info.append(X_probs)
                info.append(rewards)
            else:
                X_probs, rewards = self.rewardFunction(adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, finalProb, bias, giveInfo=giveInfo)

                #rewards = rewards - 1 #Changed to mean 0. Attempted JAN 31 2026

            rewards_mean = torch.mean(rewards, axis=1)


        time2 = time.time()  - time2 
        time3 = time.time()

        #print (pr_x_given_g_matrix.shape)
        #print (log_prob_pi.shape)
        #print (rewards.shape)
        #quit()

        #print (torch.sum(rewards_mean))
        #print (torch.sum(rewards))
        #print (torch.mean(rewards) )
        #print (rewards.shape)
        #quit()

        loss_bias = - (rewards *  pr_x_given_g_matrix.reshape(( pr_x_given_g_matrix.shape[0], pr_x_given_g_matrix.shape[1], 1 ))  ).mean()

        loss = - ( log_prob_pi * rewards_mean ).mean()

        loss = loss + loss_bias

        time3 = time.time() - time3

        

        #print ('minitime', time1, time2, time3)

        if giveInfo:
            return loss, info
        else:
            return loss
        






class baselineBiasSpliceClass(SpliceClass):


    def __init__(self, finalProbSize, Njunction, Nsample, edgeMatrix):
        super().__init__(finalProbSize, Njunction, Nsample, edgeMatrix)  # call parent init to set self.value
        self.finalProbSize = finalProbSize
        self.Njunction = Njunction
        self.Nsample = Nsample
        self.edgeMatrix = edgeMatrix


        self.baselineType = 'naive'
        #self.baselineType = 'GFlowNet'

    

    def rewardFunction(self, pr_x_given_g_matrix, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=False):


        
        

        timeList = []


        timeList.append(time.time())

        
        


        pr_x_given_g_matrix = pr_x_given_g_matrix.reshape((pr_x_given_g_matrix.shape[0], pr_x_given_g_matrix.shape[1], 1)) #batchSize, Junction, Sample
        #prob_mult = probs.reshape((probs.shape[0], probs.shape[1], 1)) * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))

        if self.finalProbSize >= 2:
            importance_weights_log = (log_prob_pi - log_prob_pi_prime.reshape(((-1, 1)))  ).detach()
        else:
            importance_weights_log = (log_prob_pi - log_prob_pi_prime).detach()
        
        #print (self.finalProbSize)
        #print (importance_weights_log.shape)
        #quit()

        timeList.append(time.time())
        
        importance_weights = torch.exp(importance_weights_log)  # Detach to prevent gradient flow    

        if self.finalProbSize >= 2:
            pr_x_given_g_matrix = pr_x_given_g_matrix + importance_weights_log.reshape((importance_weights_log.shape[0], 1, importance_weights_log.shape[1]))
        else:
            pr_x_given_g_matrix = pr_x_given_g_matrix + importance_weights_log.reshape((-1, 1))


        
        

        
        X_probs = torch.logsumexp(pr_x_given_g_matrix, dim=0)

        if self.baselineType == 'ours': #NOT NAIVE
            if self.finalProbSize >= 2:
                pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix - X_probs.reshape((1, X_probs.shape[0], X_probs.shape[1]))

            else:
                pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix - X_probs.reshape((1, -1))



        else:
            pr_x_given_g_matrix_adjusted = pr_x_given_g_matrix
        
        
        X_probs = X_probs * obs_matrix



        if not 'GFlowNet' in self.baselineType:
            rewards = torch.exp(pr_x_given_g_matrix_adjusted)
        
            rewards = rewards * obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1])) / torch.mean(obs_matrix)# Adds weighting based on number of reads for each

            rewards = rewards * (pr_x_given_g_matrix_adjusted.shape[0] / pr_x_given_g_matrix_adjusted.shape[1])

        else:

            rewards = pr_x_given_g_matrix_adjusted
            rewards = rewards + torch.log(obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1])) / torch.mean(obs_matrix) + 1e-100)

            rewards[rewards < -5000] = -5000



        if giveInfo:
            info = [pr_x_given_g_matrix]
            return X_probs, rewards, info
        else:
            return X_probs, rewards
        
    


    def lossFunction(self, adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=False):


        time1 = time.time()

        if (self.finalProbSize >= 2) and (adjacency_matrices.shape[0] >= 1):
            _, finalProbAllow = self.graphRules(adjacency_matrices)
            finalProb = self.model.finalProb(adjacency_matrices)
            finalProb = finalProb + finalProbAllow
            finalProb = nn.LogSoftmax(dim=1)(finalProb)
        else:
            finalProb = torch.zeros((adjacency_matrices.shape[0], 1))
    
        bias = self.model.giveBias()

        pr_x_given_g_matrix = torch.log(adjacency_matrices + 1e-10)
        pr_x_given_g_matrix[adjacency_matrices == 0] = -500

        pr_x_given_g_matrix = pr_x_given_g_matrix + bias.reshape((1, -1))
        pr_x_given_g_matrix = torch.nn.LogSoftmax(dim=1)(pr_x_given_g_matrix)

        leakage, _ = torch.max(pr_x_given_g_matrix, axis=1)


        leakage = -1 * torch.logaddexp(leakage, torch.zeros(leakage.shape))

        leakage = leakage * 2.0


        pr_x_given_g_matrix = pr_x_given_g_matrix  + leakage.reshape((-1, 1))

        log_prob_pi = log_prob_pi.reshape((-1, 1)) + finalProb
        
        with torch.no_grad():
            if giveInfo:
                X_probs, rewards, info = self.rewardFunction(pr_x_given_g_matrix, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=giveInfo)
                info.append(X_probs)
                info.append(rewards)
            #else:
            #    X_probs, rewards = self.rewardFunction(adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, finalProb, bias, giveInfo=giveInfo)

            rewards_mean = torch.mean(rewards, axis=1)



        if not 'GFlowNet' in self.baselineType:


            loss_bias = - (rewards *  pr_x_given_g_matrix.reshape(( pr_x_given_g_matrix.shape[0], pr_x_given_g_matrix.shape[1], 1 ))  ).mean()

            loss = - ( log_prob_pi * rewards_mean ).mean()

            loss = loss + loss_bias



        else:

            diff1 =  log_prob_pi.reshape(( log_prob_pi.shape[0], 1, log_prob_pi.shape[1] )) + pr_x_given_g_matrix.reshape(( pr_x_given_g_matrix.shape[0], pr_x_given_g_matrix.shape[1], 1 )) - rewards
            #diff1 = log_prob_pi.reshape(( log_prob_pi.shape[0], 1, log_prob_pi.shape[1] )) - rewards

            #diff_bias = diff_bias - torch.mean(diff_bias)
            #diff_bias = diff_bias - torch.mean(diff_bias)


            diff1 = diff1 - torch.mean(diff1)
            loss = torch.mean(diff1 ** 2)




        #time3 = time.time() - time3

        

        #print ('minitime', time1, time2, time3)

        if giveInfo:
            return loss, info
        else:
            return loss
        







class biasExonClass(biasSpliceClass):


    def __init__(self, finalProbSize, Njunction, Nsample, edgeMatrix):
        super().__init__(finalProbSize, Njunction, Nsample, edgeMatrix)  # call parent init to set self.value
        self.finalProbSize = finalProbSize
        self.Njunction = Njunction
        self.Nsample = Nsample
        self.edgeMatrix = edgeMatrix

    


    def exonCounter(self, adjacency_matrices):

        edgeMatrix = self.edgeMatrix
        edges = np.argwhere(edgeMatrix >= 1)
        Njunction = edgeMatrix.shape[0] - 1

        edgeMatrix_index = np.zeros(edgeMatrix.shape, dtype=int)
        edgeMatrix_index[edges[:, 0], edges[:, 1]] = np.arange(edges.shape[0])

        #print (edgeMatrix_index)



        exon_matrix = torch.zeros((  adjacency_matrices.shape[0],  edges.shape[0] ))

        for index1 in range(adjacency_matrices.shape[0]):

            adj_now = adjacency_matrices[index1].data.numpy()
            juncNow = np.argwhere(adj_now != 0)[:, 0]
            juncNow = [Njunction] + list(juncNow) + [Njunction]
            juncNow = np.array(juncNow, dtype=int)
            exonNow = np.array( [ juncNow[:-1], juncNow[1:] ] ).T
            exonNow_index = edgeMatrix_index[exonNow[:, 0], exonNow[:, 1]]
            exonNow_dup = edges[exonNow_index]

            #if index1 == 0:
            #    print (exonNow)
            #    print (exonNow_dup)
            #    quit()


            exon_matrix[index1, exonNow_index] = 1

        transMatrix = self.transMatrix

        #print (edges)

        #print (adjacency_matrices[0])
        #print (exon_matrix[0])
        #print (edges[exon_matrix[0] != 0])

        #print (transMatrix)


        exon_matrix = torch.matmul(exon_matrix, transMatrix)

        #print (exon_matrix[0])
        #quit()
        
        return exon_matrix



    def multi_x_given_g(self, adjacency_matrices):
        
        
        bias = self.model.giveBias()

        exon_matrix = self.exonCounter(adjacency_matrices)


        edgeMatrix = self.edgeMatrix
        edges = np.argwhere(edgeMatrix >= 1)
        
        #quit()

        size1 = adjacency_matrices.shape[1]

        adjacency_matrices = torch.cat(( adjacency_matrices, exon_matrix ), axis=1)

        pr_x_given_g_matrix = torch.log(adjacency_matrices + 1e-10)
        pr_x_given_g_matrix[adjacency_matrices == 0] = -500
        pr_x_given_g_matrix = pr_x_given_g_matrix + bias.reshape((1, -1))


        #leakage, _ = torch.max(   torch.nn.LogSoftmax(dim=1)(pr_x_given_g_matrix[:, :size1])  , axis=1)
        #leakage2, _ = torch.max(   torch.nn.LogSoftmax(dim=1)(pr_x_given_g_matrix[:, size1:])  , axis=1)
        #leakage = torch.logaddexp(leakage, leakage2)

        #pr_x_given_g_matrix = torch.nn.LogSoftmax(dim=1)(pr_x_given_g_matrix)

        pr_x_given_g_matrix[:, :size1] = torch.nn.LogSoftmax(dim=1)(pr_x_given_g_matrix[:, :size1])
        pr_x_given_g_matrix[:, size1:] = torch.nn.LogSoftmax(dim=1)(pr_x_given_g_matrix[:, size1:])

        leakage, _ = torch.max( pr_x_given_g_matrix[:, :size1]  , axis=1)
        leakage2, _ = torch.max( pr_x_given_g_matrix[:, size1:]  , axis=1)
        leakage = -1 * torch.logaddexp(leakage, torch.zeros(leakage.shape))
        leakage2 = -1 * torch.logaddexp(leakage2, torch.zeros(leakage2.shape))


        leakage = leakage * 5.0
        #leakage = leakage * 2.0
        leakage2 = leakage2 * 5.0
        #leakage = leakage * 20.0

        pr_x_given_g_matrix[:, :size1] = pr_x_given_g_matrix[:, :size1]  + leakage.reshape((-1, 1)) 
        pr_x_given_g_matrix[:, size1:] = pr_x_given_g_matrix[:, size1:] + leakage2.reshape((-1, 1)) 


        #pr_x_given_g_matrix = pr_x_given_g_matrix  + leakage.reshape((-1, 1)) 

        return pr_x_given_g_matrix


    def lossFunction(self, adjacency_matrices, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=False):


        _, finalProbAllow = self.graphRules(adjacency_matrices)
        finalProb = self.model.finalProb(adjacency_matrices)
        finalProb = finalProb + finalProbAllow
        finalProb = nn.LogSoftmax(dim=1)(finalProb)


        
        pr_x_given_g_matrix = self.multi_x_given_g(adjacency_matrices)




        #print (pr_x_given_g_matrix.shape)
        #print (obs_matrix.shape)


        log_prob_pi = log_prob_pi.reshape((-1, 1)) + finalProb

        
        
        with torch.no_grad():
            if giveInfo:
                X_probs, rewards, info = self.rewardFunction(pr_x_given_g_matrix, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=giveInfo)
                info.append(X_probs)
                info.append(rewards)
            else:
                X_probs, rewards = self.rewardFunction(pr_x_given_g_matrix, obs_matrix, log_prob_pi, log_prob_pi_prime, giveInfo=giveInfo)

            rewards_mean = torch.mean(rewards, axis=1)


        #print (pr_x_given_g_matrix.shape)
        #print (log_prob_pi.shape)
        #print (rewards.shape)
        #quit()

        loss_bias = - (rewards *  pr_x_given_g_matrix.reshape(( pr_x_given_g_matrix.shape[0], pr_x_given_g_matrix.shape[1], 1 ))  ).mean()

        loss = - ( log_prob_pi * rewards_mean ).mean()

        loss = loss + loss_bias

        

        if giveInfo:
            return loss, info
        else:
            return loss
        




class SpliceNet(nn.Module):
    def __init__(self, graphSize, finalProbSize, Nhidden, biasSize=0):
        super(SpliceNet, self).__init__()
        self.input_size = graphSize
        self.hidden_size = 50#200#50#100# Nhidden
        self.finalProbSize = finalProbSize
        self.output_size = graphSize + 1  # All edges + stop action

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        
        self.fc_finalProb1 = nn.Linear(self.input_size, 30)
        self.fc_finalProb2 = nn.Linear(30, self.finalProbSize)

        #self.fc_finalProb1 = nn.Linear(self.input_size, 200)
        #self.fc_finalProb2 = nn.Linear(200, self.finalProbSize)

        #self.fc_finalProb1 = nn.Linear(self.input_size, 1000)
        #self.fc_finalProb2 = nn.Linear(1000, self.finalProbSize)

        #self.fc_finalProb = nn.Linear(self.hidden_size, self.finalProbSize)

        if biasSize == 0:
            biasSize = graphSize
        self.bias = nn.Parameter(torch.zeros(biasSize))


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
    
    def giveBias(self):
        bias = self.bias #* 10 #* 10 #30#10 #* 0.1
        bias = torch.tanh(bias) * (np.log(10) / 2.0)
        return bias


    
    def finalProb(self, x):
        x = F.leaky_relu(self.fc_finalProb1(x))
        #print (x.shape)
        logits = self.fc_finalProb2(x)

        return logits





class ExonNet(nn.Module):
    def __init__(self, graphSize, finalProbSize, Nhidden, biasSize1, biasSize2):
        super(ExonNet, self).__init__()
        self.input_size = graphSize
        self.hidden_size = 50#200#50#100# Nhidden
        self.finalProbSize = finalProbSize
        self.output_size = graphSize + 1  # All edges + stop action

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        
        self.fc_finalProb1 = nn.Linear(self.input_size, 30)
        self.fc_finalProb2 = nn.Linear(30, self.finalProbSize)

        self.biasSize1 = biasSize1
        self.biasSize2 = biasSize2

        self.bias = nn.Parameter(torch.zeros(biasSize1+biasSize2))

        self.ExonBias = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
    
    def giveBias(self):


        bias = self.bias #* 10 #* 10 #30#10 #* 0.1
        bias = torch.tanh(bias) * (np.log(10) / 2.0)

        bias[self.biasSize1:] = bias[self.biasSize1:] + self.ExonBias

        return bias


    
    def finalProb(self, x):
        x = F.leaky_relu(self.fc_finalProb1(x))
        #print (x.shape)
        logits = self.fc_finalProb2(x)

        return logits






def baselinesSplice():

    gene_unique = loadnpz('./data/temp/gene_unique_perm.npz')

    badGenes = loadnpz('./data/temp/badGenes.npz')

    np.random.seed(1)

    runTimesList = []

    for gene_index in range(0, gene_unique.shape[0]):# gene_unique.shape[0]): #15

        geneNow = gene_unique[gene_index]

        if True:
            learning_rate = 1e-3
            Nhidden = 50
            #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_timingReview.pt'
            #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_GFlow2.pt'
            model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_Naive2.pt'

            existingModels = os.listdir('./data/real/splicing/geneFiles/geneModels/')
            alreadyExist = (geneNow + '_sampleGen_fast.pt' in existingModels)

            
            if True:

                junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
                isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')

                multiCount = np.sum(isoformCounts_long, axis=1)
                multiCount = np.sort(multiCount)[-1::-1]

                #print (multiCount)
                GoodNow = False 
                if multiCount.shape[0] >= 5:
                    if multiCount[4] >= 1000:
                        GoodNow = True
                        #print (multiCount)
                        

                #print ("A")

                strandDirection = junctionNow[0, 3]

                #if True:#GoodNow:#(strandDirection == '+') and (np.sum(isoformCounts_long) > 10000):
                if True:#

                    #print ('geneNow', geneNow)

                    for q in range(10):
                        print ('')
                    print (geneNow)
                    print ("Run Times")
                    print (np.mean(np.array(runTimesList)), len(runTimesList))
                    #print (runTimesList)
                    for q in range(10):
                        print ('')


                    #print ('multiCount', multiCount)

                
                    countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                    edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')


                    if True:
                        sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                        sampleLong = loadnpz('./data/real/splicing/input/samples_longRead.npz')
                        inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
                        inverse1 = uniqueValMaker(inverse1)
                        #argGood = np.argwhere(np.isin( inverse1[:sampleLong.shape[0]]   , inverse1[sampleLong.shape[0]:] ))[:, 0]
                        argGood = np.argwhere(np.isin( inverse1[sampleLong.shape[0]:], inverse1[:sampleLong.shape[0]]  ))[:, 0]

                        countNow = countNow[:, argGood]
                        

                    

                    countNow = torch.tensor(countNow).float()

                    Nsample = countNow.shape[1]
                    Njunction = countNow.shape[0]

                    #print ('Nsample', Nsample)
                    #quit()

                    observations_batch = countNow

                    edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                    edgeMatrix[edges[:, 0], edges[:, 1]] = 1
                    edgeSum = np.sum(edgeMatrix, axis=1)


                    if torch.sum(countNow) >= 100:

                        startPos = junctionNow[:, 1].astype(int)
                        if startPos.shape[0] >= 2:
                            assert np.min(startPos[1:] - startPos[:-1]) >= 0

                        #if goodCount:

                        #print ("HI")
                        

                        goodMatrix = True 
                        if (0 in np.sum(edgeMatrix, axis=0)) or (0 in np.sum(edgeMatrix, axis=1)):
                            goodMatrix = False 

                        if goodMatrix:


                            
                            


                            graphSize = Njunction
                            finalProbSize = Nsample 
                            #batchSize = Njunction
                            
                            ruleObject = baselineBiasSpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                            ruleObject.graphSize = graphSize
                            ruleObject.observations_batch = observations_batch
                            ruleObject.model = SpliceNet(graphSize, finalProbSize, Nhidden)



                            #batchSize = 200
                            #batchSize = 5000
                            batchSize = 500
                            #batchSize = 100
                            #batchSize = 10000


                            offPolicy = False
                            #dupGen = 1

                            #num_epochs = 5000
                            #num_epochs = 500
                            num_epochs = 1000

                            timeStart = time.time()

                            #train_model_off_policy(Nhidden, ruleObject, learning_rate, observations_batch,  batchSize, dupGen, offPolicy, num_epochs=5000, model_filename=model_filename, rewardType='')
                            train_model_off_policy(ruleObject, learning_rate,  batchSize, offPolicy, num_epochs=num_epochs, model_filename=model_filename, rewardType='')

                            #train_model_off_policy(ruleObject, learning_rate, batchSize, offPolicy, num_epochs=10000, model_filename='', rewardType='', giveTrajectory=False):

                            runTimesList.append(time.time() - timeStart)

                            #print ('time total', runTimesList[-1])

                            #quit()


                    np.savez_compressed('./data/temp/runtimes_Naive.npz', np.array(runTimesList) )


baselinesSplice()
quit()





def trySplice():

    #model_filename = './data/sims/causal/model/graph_' + 'spliceMini' + '_' + 'ours' + '_4.pt'

    #model_filename = './data/sims/causal/model/graph_' + 'spliceMini' + '_' + 'ours' + '_down_0.01.pt'

    
    #ENSG00000022267, ENSG00000067225, ENSG00000070081, ENSG00000104529, ENSG00000113648, ENSG00000114416, ENSG00000129197, ENSG00000140416 (great), ENSG00000143549 (great!), ENSG00000163110
    #ENSG00000168958 (ok), ENSG00000178104 (extra complex splicing), ENSG00000183091 (insane number of junctions), ENSG00000185787 (ok), ENSG00000187109 (ok), ENSG00000196756,
    #ENSG00000196776, ENSG00000196923, ENSG00000198363, ENSG00000198467, ENSG00000214548
    #quit()

    #geneValid = ['ENSG00000022267', 'ENSG00000067225', 'ENSG00000070081', 'ENSG00000104529', 'ENSG00000113648', 'ENSG00000114416', 'ENSG00000129197', 'ENSG00000140416', 'ENSG00000143549', 
    #             'ENSG00000163110', 'ENSG00000168958']#, ENSG00000178104 (extra complex splicing), ENSG00000183091 (insane number of junctions), ENSG00000185787 (ok), ENSG00000187109 (ok), ENSG00000196756,
    #ENSG00000196776, ENSG00000196923, ENSG00000198363, ENSG00000198467, ENSG00000214548


    #gene_unique = loadnpz('./data/real/splicing/input/longRead_geneList.npz')
    #gene_unique = gene_unique[np.random.permutation(gene_unique.shape[0])]
    #np.savez_compressed('./data/temp/gene_unique_perm.npz', gene_unique)
    gene_unique = loadnpz('./data/temp/gene_unique_perm.npz')

    #print (gene_unique[:5])
    #quit()

    badGenes = loadnpz('./data/temp/badGenes.npz')

    np.random.seed(1)


    #print (np.argwhere(gene_unique == 'ENSG00000003509'))
    #quit()

    runTimesList = []

    for gene_index in range(0, gene_unique.shape[0]):# gene_unique.shape[0]): #15

        geneNow = gene_unique[gene_index]

        if True:# geneNow in badGenes:# 'ENSG00000180210':#geneNow == 'ENSG00000067225':# 'ENSG00000022267':# 'ENSG00000067225':# :# 'ENSG00000070081':# in geneValid:# == 'ENSG00000114416':

        
            
            #timeStart = time.time()
    
            #dupGen = 5
            #batchSize = 100
            #batchSize = 3
            learning_rate = 1e-3 #DEFAULT
            #learning_rate = 1e-4 #SLOW!!
            #Nhidden = 2
            #Nhidden = 5
            #Nhidden = 10
            Nhidden = 50
            #Nhidden = 100
            

            #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_all.pt'
            #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_size500_leak2.pt' #Up to date and Good, Jan 10 2026
            #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_fast.pt'

            #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_RewardNormalized.pt'
            model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_Slow.pt'

            existingModels = os.listdir('./data/real/splicing/geneFiles/geneModels/')
            alreadyExist = (geneNow + '_sampleGen_fast.pt' in existingModels)

            #print (gene_index)
            
            if True:#not alreadyExist:

                #print ("New Gene")
                #quit()


                #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_size100.pt'
                #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_modifyLeak.pt'

                junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')

                #print (junctionNow)
                #quit()

                isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')

                multiCount = np.sum(isoformCounts_long, axis=1)
                multiCount = np.sort(multiCount)[-1::-1]

                #print (multiCount)
                GoodNow = False 
                if multiCount.shape[0] >= 5:
                    if multiCount[4] >= 1000:
                        GoodNow = True
                        #print (multiCount)
                        

                #print ("A")

                strandDirection = junctionNow[0, 3]

                #if True:#GoodNow:#(strandDirection == '+') and (np.sum(isoformCounts_long) > 10000):
                if True:#

                    #print ('geneNow', geneNow)

                    for q in range(10):
                        print ('')
                    print (geneNow)
                    print ("Run Times")
                    print (np.mean(np.array(runTimesList)), len(runTimesList))
                    #print (runTimesList)
                    for q in range(10):
                        print ('')


                    #print ('multiCount', multiCount)

                
                    countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                    edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')


                    if True:
                        sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                        sampleLong = loadnpz('./data/real/splicing/input/samples_longRead.npz')
                        inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
                        inverse1 = uniqueValMaker(inverse1)
                        #argGood = np.argwhere(np.isin( inverse1[:sampleLong.shape[0]]   , inverse1[sampleLong.shape[0]:] ))[:, 0]
                        argGood = np.argwhere(np.isin( inverse1[sampleLong.shape[0]:], inverse1[:sampleLong.shape[0]]  ))[:, 0]

                        countNow = countNow[:, argGood]
                        

                    

                    countNow = torch.tensor(countNow).float()

                    Nsample = countNow.shape[1]
                    Njunction = countNow.shape[0]

                    #print ('Nsample', Nsample)
                    #quit()

                    observations_batch = countNow

                    edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                    edgeMatrix[edges[:, 0], edges[:, 1]] = 1
                    edgeSum = np.sum(edgeMatrix, axis=1)


                    if torch.sum(countNow) >= 100:#np.argwhere(edgeSum >= 2).shape[0] > np.argwhere(edgeSum == 1).shape[0]:
                        #if True:

                        startPos = junctionNow[:, 1].astype(int)
                        if startPos.shape[0] >= 2:
                            assert np.min(startPos[1:] - startPos[:-1]) >= 0

                        #if goodCount:

                        #print ("HI")
                        

                        goodMatrix = True 
                        if (0 in np.sum(edgeMatrix, axis=0)) or (0 in np.sum(edgeMatrix, axis=1)):
                            goodMatrix = False 

                        if goodMatrix:


                            
                            


                            graphSize = Njunction
                            finalProbSize = Nsample 
                            #batchSize = Njunction
                            
                            ruleObject = biasSpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                            ruleObject.graphSize = graphSize
                            ruleObject.observations_batch = observations_batch
                            ruleObject.model = SpliceNet(graphSize, finalProbSize, Nhidden)



                            #batchSize = 200
                            #batchSize = 5000
                            batchSize = 500
                            #batchSize = 100
                            #batchSize = 10000


                            offPolicy = False
                            #dupGen = 1

                            #num_epochs = 5000
                            #num_epochs = 500
                            num_epochs = 1000

                            timeStart = time.time()
                            saveTrainingLoss = './data/real/splicing/geneFiles/loss/TrainingLoss_' + geneNow + '_median.npz'



                            #train_model_off_policy(Nhidden, ruleObject, learning_rate, observations_batch,  batchSize, dupGen, offPolicy, num_epochs=5000, model_filename=model_filename, rewardType='')
                            train_model_off_policy(ruleObject, learning_rate,  batchSize, offPolicy, num_epochs=num_epochs, model_filename=model_filename, rewardType='', saveTrainingLoss=saveTrainingLoss)

                            #train_model_off_policy(ruleObject, learning_rate, batchSize, offPolicy, num_epochs=10000, model_filename='', rewardType='', giveTrajectory=False):

                            runTimesList.append(time.time() - timeStart)

                            #print ('time total', runTimesList[-1])

                            #quit()


                    #np.savez_compressed('./data/temp/runtimes_March.npz', np.array(runTimesList) )


trySplice()
quit()


#model_filename = './data/real/splicing/geneFiles/loss/TrainingLoss_' + geneNow + '.pt'

#quit()





def OLD_learnExonSplice():


    def reworkExonCounts(exon_pos, exon_counts):

        #print (exon_pos[:4])
        continue1 = True
        while continue1:
            #print (exon_pos[:4])
            matchMatrix = exon_pos.reshape(( exon_pos.shape[0], 1, 2 )) - exon_pos.reshape(( 1, exon_pos.shape[0], 2 ))
            matchMatrix = np.min(np.abs(matchMatrix), axis=2)
            matchMatrix[matchMatrix!=0] = 1
            matchMatrix = 1 - matchMatrix
            matchMatrix[np.arange(matchMatrix.shape[0]), np.arange(matchMatrix.shape[0])] = 0
            matchNum = np.sum(matchMatrix, axis=0)
            #print (matchNum)
            argMatch = np.argwhere(matchNum >= 1)[:, 0]

            if argMatch.shape[0] == 0:
                continue1 = False
            else:
                
                lengths1 = exon_pos[:, 1] - exon_pos[:, 0]
                argBest = argMatch[np.argmin(lengths1[argMatch])]
                argBest_ar = np.zeros(1, dtype=int)+argBest

                argModify = np.argwhere(matchMatrix[argBest] == 1)[:, 0]
                exon_counts[argModify] = exon_counts[argModify] - exon_counts[argBest_ar]


                argSameStart = np.argwhere(exon_pos[:, 0] == exon_pos[argBest, 0])[:, 0]
                argSameStart = argSameStart[argSameStart!=argBest]
                argSameEnd = np.argwhere(exon_pos[:, 1] == exon_pos[argBest, 1])[:, 0]
                argSameEnd = argSameEnd[argSameEnd!=argBest]

                
                exon_pos[argSameStart, 0] = exon_pos[argBest, 1]
                exon_pos[argSameEnd, 1] = exon_pos[argBest, 0]

        return exon_pos, exon_counts

    def giveExonTranslator(exon_pos, junctionPos, edgeMatrix, exon_counts):


        exon_pos, exon_counts = reworkExonCounts(exon_pos, exon_counts)

        #print (exon_pos)

        overlap = np.argwhere(  exon_pos[:-1, 1] > exon_pos[1:, 0]  )[:, 0]
        subset = np.argwhere(  np.logical_or(exon_pos[:-1, 1] >= exon_pos[1:, 1], exon_pos[:-1, 0] == exon_pos[1:, 0])  )[:, 0]
        #print (subset)
        #print (overlap)
        
        


        
        edges = np.argwhere(edgeMatrix >= 1)
        Njunction = junctionPos.shape[0]

        
        transMatrix = torch.zeros((  edges.shape[0],  exon_counts.shape[0]  )).float()


        lengths = exon_pos[:, 1] - exon_pos[:, 0]
        lengthDefault = 1
        lengthBooster = (lengths + lengthDefault) / lengthDefault

        exon_counts = exon_counts / lengthBooster.reshape((-1, 1))
        


        for edge_index in range(edges.shape[0]):
            edge = edges[edge_index]
            if Njunction in edge:

                if Njunction == edge[1]:
                    exonStart = junctionPos[edge[0], 1]
                    argGood = np.argwhere(np.abs(exon_pos[:, 1] - exonStart) <= 200)[:, 0]
                if Njunction == edge[0]:
                    exonEnd = junctionPos[edge[1], 0]
                    argGood = np.argwhere(np.abs(exon_pos[:, 0] - exonEnd) <= 200)[:, 0]
                transMatrix[edge_index, argGood] = 1
            else:
                #print (edge, Njunction)
                exonStart = junctionPos[edge[0], 1]
                exonEnd = junctionPos[edge[1], 0]

                #for countExon_index in range(exon_pos.shape[0]):
                #    countExonStart = exon_pos[countExon_index, 0]
                #    countExonEnd = exon_pos[countExon_index, 1]

                #argGood = np.argwhere(  np.logical_and( exon_pos[:, 0] <= exonEnd + 1, exon_pos[:, 1] >= exonStart - 1 ) )[:, 0]
                #argGood = np.argwhere(  np.logical_and( exon_pos[:, 0] <= exonEnd - 2 , exon_pos[:, 1] >= exonStart + 2  ) )[:, 0]

                exon_pos_mod = np.copy(exon_pos)
                exon_pos_mod[ exon_pos_mod[:, 0] < exonStart  , 0] = exonStart
                exon_pos_mod[ exon_pos_mod[:, 1] > exonEnd  , 1] = exonEnd

                overlap = exon_pos_mod[:, 1] - exon_pos_mod[:, 0]
                overlap[overlap<0] = 0
                overlap = overlap / (exonEnd - exonStart)
                #print (overlap)
                #quit()
                transMatrix[edge_index, :] = torch.tensor(overlap)
                

                #if countExonStart <= exonEnd + 1:
                #    if countExonEnd >= exonStart - 1:
                #        #Overlapping 
                #        transMatrix[edge_index, countExon_index] = 1

        

        #edges

        junctionPos_mod = np.concatenate(( junctionPos , np.zeros((1, 2))-1 ), axis=0)
        edges_pos = np.array([  junctionPos_mod[edges[:, 0], 1], junctionPos_mod[edges[:, 1], 0] ], dtype=int).T

        #print (transMatrix)
        #argTrans = np.argwhere(transMatrix.data.numpy() != 0)
        argTrans = np.argwhere(transMatrix.data.numpy() >= 0.9)
        #print (argTrans)

        allPos = np.concatenate( ( edges_pos[argTrans[:, 0]], exon_pos[argTrans[:, 1]]  ), axis=1 )
        #print (edges_pos)
        #print (exon_pos)
        #print (allPos)
        #quit()
        

        return transMatrix, exon_counts
    


    
    gene_unique = loadnpz('./data/temp/gene_unique_perm.npz')
    for gene_index in range(0, 30):# gene_unique.shape[0]): #15

        geneNow = gene_unique[gene_index]

        if True:#geneNow == 'ENSG00000112294':
            learning_rate = 1e-3
            Nhidden = 50
            model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_scaledExon.pt'
            junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
            junctionPos = junctionNow[:, 1:3].astype(int)
            isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')
            multiCount = np.sum(isoformCounts_long, axis=1)
            multiCount = np.sort(multiCount)[-1::-1]
            if True:
                print ('geneNow', geneNow)
                countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')

                exon_counts = loadnpz('./data/real/splicing/geneFiles/exonCounts_longSample/exonCounts_' + str(geneNow) + '.npz')
                exon_pos = loadnpz('./data/real/splicing/geneFiles/exonInfo_longSample/exonInfo_' + str(geneNow) + '.npz')
                exon_pos = exon_pos.astype(int)
                exon_sample = loadnpz('./data/real/splicing/input/samples_RailLongRead.npz')
                #exon_counts = torch.tensor(exon_counts).float()
                

                sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                inverse1 = np.concatenate((  exon_sample, sampleList ), axis=0)
                inverse1 = uniqueValMaker(inverse1)
                argGood = np.zeros(exon_sample.shape[0], dtype=int)
                for exon_sample_index in range(exon_sample.shape[0]):
                    arg1 = np.argwhere( inverse1 == inverse1[exon_sample_index] )[:, 0]
                    argGood[exon_sample_index] = arg1[1] - exon_sample.shape[0]
                countNow = countNow[:, argGood]


                Nsample = countNow.shape[1]
                Njunction = junctionPos.shape[0]

                #print ('Nsample', Nsample)
                #quit()

                

                edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                edgeMatrix[edges[:, 0], edges[:, 1]] = 1
                edgeSum = np.sum(edgeMatrix, axis=1)

                #print (edgeMatrix)
                #quit()


                transMatrix, exon_counts = giveExonTranslator(exon_pos, junctionPos, edgeMatrix, exon_counts)


                
                print (countNow.shape, exon_counts.shape)
                print (np.sum(countNow), np.sum(exon_counts))
                #quit()

                exon_counts = exon_counts * 0.1 * (np.sum(countNow) / np.sum(exon_counts)) #TODO REMOVE
                #countNow = countNow * 0

                print (np.sum(countNow), np.sum(exon_counts))
                

                countNow = np.concatenate( (countNow, exon_counts), axis=0 )


                #model = torch.load(model_filename)
                #print (model.ExonBias)
                #plt.plot(  np.sum(countNow, axis=1))
                #plt.show()
                #quit()
                countNow = torch.tensor(countNow).float()
                


                if True:#np.argwhere(edgeSum >= 2).shape[0] > np.argwhere(edgeSum == 1).shape[0]:
                    #if True:

                    startPos = junctionNow[:, 1].astype(int)
                    if startPos.shape[0] >= 2:
                        assert np.min(startPos[1:] - startPos[:-1]) >= 0

                    goodMatrix = True 
                    if (0 in np.sum(edgeMatrix, axis=0)) or (0 in np.sum(edgeMatrix, axis=1)):
                        goodMatrix = False 

                    

                    if goodMatrix:


                        
                        
                        observations_batch = countNow #, transMatrix, exon_counts]

                        graphSize = Njunction
                        finalProbSize = Nsample 
                        #batchSize = Njunction
                        
                        ruleObject = biasExonClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                        ruleObject.graphSize = graphSize
                        ruleObject.observations_batch = observations_batch
                        ruleObject.model = ExonNet(graphSize, finalProbSize, Nhidden, countNow.shape[0] - exon_counts.shape[0], exon_counts.shape[0] )
                        ruleObject.transMatrix = transMatrix


                        
                        #batchSize = 200
                        #batchSize = 5000
                        batchSize = 500
                        #batchSize = 10000


                        offPolicy = False
                        dupGen = 1

                        #num_epochs = 5000
                        #num_epochs = 500
                        num_epochs = 1000

                        #train_model_off_policy(Nhidden, ruleObject, learning_rate, observations_batch,  batchSize, dupGen, offPolicy, num_epochs=5000, model_filename=model_filename, rewardType='')
                        train_model_off_policy(ruleObject, learning_rate,  batchSize, dupGen, offPolicy, num_epochs=num_epochs, model_filename=model_filename, rewardType='')



#learnExonSplice()
#quit()




def processPredictionCounts(adjacency_matrices, finalProb, normalizeJunc=False):

        adjacency_matrices = adjacency_matrices.data.numpy()

        inverse1 = uniqueValMaker(adjacency_matrices)
        _, count1 = np.unique(inverse1, return_counts=True)
        _, index1 = np.unique(inverse1, return_index=True)

        numUse = 5
        if count1.shape[0] >= numUse:
            cutOff = np.sort(count1)[-numUse]
        else:
            cutOff = 0

        #adjacency_matrices = adjacency_matrices[count1[inverse1] >= 200]
        #finalProb = finalProb[count1[inverse1] >= 200]
        #adjacency_matrices = adjacency_matrices[count1[inverse1] >= cutOff]
        #finalProb = finalProb[count1[inverse1] >= cutOff]
        
        finalProb_exp = np.exp(finalProb.data.numpy())


        inverse1 = uniqueValMaker(adjacency_matrices)
        _, index1 = np.unique(inverse1, return_index=True)
        adjacency_matrices = adjacency_matrices[index1]

        numJunction = np.sum(adjacency_matrices, axis=1)

        max1 = int(np.max(inverse1)+1) 
        finalProb_exp_sum = np.zeros((  max1, finalProb.shape[1] ))
        for a in range(inverse1.shape[0]):
            finalProb_exp_sum[inverse1[a]] = finalProb_exp_sum[inverse1[a]] + finalProb_exp[a]


        if normalizeJunc:
            finalProb_exp_sum = finalProb_exp_sum / numJunction.reshape((-1, 1))

        return adjacency_matrices, finalProb_exp_sum


def testTypes():

    

    from statsmodels.stats.oneway import anova_oneway
    import statsmodels.formula.api as smf
    import pandas as pd

    files = os.listdir('./data/real/splicing/geneFiles/geneModels')
    geneList = []
    for file1 in files:
        if '.pt' in file1:
            if '_1' in file1:
                geneList.append(file1.split('_')[0])
    geneList = np.unique(np.array(geneList))


    geneList = loadnpz('./data/real/splicing/input/longRead_geneList.npz')

    allScores_our = []
    allScore_short = []


    sampleInfo = np.loadtxt('./data/real/splicing/info/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt', dtype=str, delimiter='\t')

    tissueInfo = np.loadtxt('./data/real/splicing/info/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt', dtype=str, delimiter='\t')
    tissueInfo = tissueInfo[:, np.array([0, 5, 6]) ]

    #print (tissueInfo[:5, 5:7])
    #quit()

    #print (sampleInfo)
    #quit()

    #DTHHRDY is GTEx’s Hardy death classification (0–4) describing circumstances and terminal interval:
    #0 = Ventilator case (on mechanical ventilation immediately before death). 
    #1 = Violent & fast death (accident, blunt trauma, suicide; terminal phase <10 min). 
    #2 = Fast natural death (sudden, reasonably healthy; terminal phase <1 hr, e.g., MI). 
    #3 = Intermediate death (terminal phase 1–24 hr; ill but death unexpected). 
    #4 = Slow death (long illness; terminal phase >1 day, e.g., cancer, COPD). 


    geneValid = ['ENSG00000022267', 'ENSG00000067225', 'ENSG00000070081', 'ENSG00000104529', 'ENSG00000113648', 'ENSG00000114416', 'ENSG00000129197', 'ENSG00000140416', 'ENSG00000143549', 
                 'ENSG00000163110', 'ENSG00000168958']
    with torch.no_grad():
    

        for gene_index in range(0, len(geneList)):# range(40):

            #print ('gene_index', gene_index)

            geneNow = geneList[gene_index]
            #geneNow = geneValid[2]

            countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz').astype(float)
            edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
            Njunction = countNow.shape[0]

            edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
            edgeMatrix[edges[:, 0], edges[:, 1]] = 1
            if (not 0 in np.sum(edgeMatrix, axis=0)) and (not 0 in np.sum(edgeMatrix, axis=1)):
                avgCount = np.mean(countNow)
                if avgCount < 1:
                    print (geneNow)
                    print ('avgCount', avgCount)
            
            #ENSG00000002079
            if geneNow ==  'ENSG00000004846':# 'ENSG00000114416':#:#:

                
                model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_all.pt'
                model = torch.load(model_filename, weights_only=False)
                


                junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
                countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz').astype(float)
                edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
                junctionPos = junctionNow[:, 1:3].astype(int)

                #print (countNow.shape)
                #quit()

                validSamples = np.argwhere( np.sum(countNow, axis=0) >= 1 )[:, 0]

                

                sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                

                if True:
                    Njunction = junctionNow.shape[0]
                    isoformJunctions = loadnpz('./data/real/splicing/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                    isoformJunctionPos = loadnpz('./data/real/splicing/geneFiles/isoformJunctionsPos/' + str(geneNow) + '.npz')
                    isoformCounts = loadnpz('./data/real/splicing/geneFiles/isoform_counts/' + str(geneNow) + '.npz')


                    isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                    isoformJunctionPos_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctionPos/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                    isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')

                    validIsoform = np.sum(isoformJunctions_long, axis=1)
                    isoformJunctions_long = isoformJunctions_long[validIsoform >= 1]
                    isoformCounts_long = isoformCounts_long[validIsoform >= 1]

                    
                    

                    Nsample = countNow.shape[1]
                    Njunction = countNow.shape[0]

                    observations_batch =  torch.tensor(countNow).float()

                    edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                    edgeMatrix[edges[:, 0], edges[:, 1]] = 1


                    graphSize = Njunction
                    #finalProbSize = 19788# Nsample 
                    finalProbSize = countNow.shape[1] 
                    batchSize = Njunction
                    ruleObject = SpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                    ruleObject.graphSize = graphSize
                    ruleObject.observations_batch = observations_batch

                    

                    #batchSize = 200
                    batchSize = 2000
                    #batchSize = 500
                    offPolicy = False
                    model = torch.load(model_filename)
                    adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories, finalProb = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batchSize)

                    adjacency_matrices, finalProb_exp_sum = processPredictionCounts(adjacency_matrices, finalProb)


                #print (finalProb_exp_sum.shape)

                count_pred = np.matmul(adjacency_matrices.T, finalProb_exp_sum)
                #)

                #quit()

                    

                totalIsoformProps = np.sum(finalProb_exp_sum, axis=1)
                totalIsoformProps = totalIsoformProps / np.sum(totalIsoformProps)
                #adjacency_matrices = adjacency_matrices[totalIsoformProps > 0.005]
                #print (finalProb_exp_sum.shape)
                #quit()

                #finalProb_exp_sum = countNow

                


                patientInfo = getPatientInfo(sampleList, sampleInfo)

                tissueInfo2 = getTissueInfo(sampleList, tissueInfo)

                tissueInfo2 = tissueInfo2[validSamples]



                pValLists_count = subsetPValues(countNow[:, validSamples], tissueInfo2)
                pValLists_ours = subsetPValues(finalProb_exp_sum[:, validSamples], tissueInfo2)
                pValLists_baseline = subsetPValues(isoformCounts[:, validSamples], tissueInfo2)

                index1 = 1

                #pValLists_count = getPValues(countNow[:, validSamples], tissueInfo2)[index1]
                #pValLists_ours = getPValues(finalProb_exp_sum[:, validSamples], tissueInfo2)[index1]
                #pValLists_baseline = getPValues(isoformCounts[:, validSamples], tissueInfo2)[index1]





                

                print ("T")
                if True:

                    print (countNow.shape)
                    print (tissueInfo.shape)

                    #pValLists_ours = getPValues(finalProb_exp_sum, tissueInfo)
                    #pValLists_count = getPValues(countNow, tissueInfo)
                    #pValLists_count = getPValues(isoformCounts, tissueInfo)

                    #for phen_index in range(tissueInfo.shape[1]):
                    Y = tissueInfo2[:, index1]
                    unique_groups = np.unique(Y)
                    for group_index in range(unique_groups.shape[0]):
                        group_now = unique_groups[group_index]

                        print (group_now)

                        p_ours = getAdjustedP(pValLists_ours, group_index, finalProb_exp_sum.shape[0] * unique_groups.shape[0])
                        p_baseline = getAdjustedP(pValLists_baseline, group_index, isoformCounts.shape[0] * unique_groups.shape[0])
                        p_count = getAdjustedP(pValLists_count, group_index, countNow.shape[0] * unique_groups.shape[0])

                        p_all = np.array([p_ours, p_baseline, p_count])

                        #p_ours = pValLists_ours[group_index, :, 0]
                        #p_count = pValLists_count[group_index, :, 0]
                        #p_ours[np.isnan(p_ours)] = 1
                        #_count[np.isnan(p_count)] = 1

                        #p_ours = np.min(  p_ours ) * finalProb_exp_sum.shape[0] * unique_groups.shape[0]
                        #p_count = np.min( p_count ) * countNow.shape[0] * unique_groups.shape[0]
                        if np.min(p_all) < 0.05:
                            print (p_ours, p_baseline, p_count)


                countNow = countNow[:, validSamples]
                finalProb_exp_sum = finalProb_exp_sum[:, validSamples]

                print (countNow.shape)
                print (count_pred.shape)
                

                sns.heatmap(  (countNow  / np.mean(countNow, axis=0).reshape((1, -1)))[:, np.argsort(tissueInfo2[:, 0])]   )
                plt.show()

                sns.heatmap(  (count_pred  / np.mean(count_pred, axis=0).reshape((1, -1)))[:, np.argsort(tissueInfo2[:, 0])]   )
                plt.show()

                quit()
                
                    
#testTypes()
#quit()


def analyzeSplice():

    import numpy as np

    def compute_overlap_np(e1, e2):
        """Compute overlap in base pairs between two exons given as arrays."""
        start = np.maximum(e1[0], e2[0])
        end = np.minimum(e1[1], e2[1])
        return max(0, end - start)

    def jaccard_distance_exons_np(exons1, exons2):
        """
        Compute Jaccard distance between two exon sets.
        exons1 and exons2 are NumPy arrays of shape (n, 2) with [start, end] positions.
        """
        # Sort by start position
        exons1 = exons1[np.argsort(exons1[:, 0])]
        exons2 = exons2[np.argsort(exons2[:, 0])]

        total_bp1 = np.sum(exons1[:, 1] - exons1[:, 0])
        total_bp2 = np.sum(exons2[:, 1] - exons2[:, 0])

        # Two-pointer overlap computation
        i = j = 0
        intersection_bp = 0

        while i < len(exons1) and j < len(exons2):
            e1 = exons1[i]
            e2 = exons2[j]

            # Add overlapping base pairs
            overlap = compute_overlap_np(e1, e2)
            intersection_bp += overlap

            # Advance the interval with the smaller end
            if e1[1] < e2[1]:
                i += 1
            else:
                j += 1

        union_bp = total_bp1 + total_bp2 - intersection_bp
        return 1 - (intersection_bp / union_bp) if union_bp > 0 else 0.0


    #-134.5021209716797

    import seaborn as sns

    #model_filename = './data/sims/causal/model/graph_' + 'spliceMini' + '_' + 'ours' + '_3.pt'

    #geneNow = 'ENSG00000075413'
    #geneNow = 'ENSG00000048052'
    #geneNow = 'ENSG00000142621'
    #geneNow = 'ENSG00000124721'
    #geneNow = 'ENSG00000142449'
    #geneNow = 'ENSG00000005810'
    #geneNow = 'ENSG00000006283'
    #geneNow = 'ENSG00000103657'
    #geneNow = 'ENSG00000003393'


    geneNow = 'ENSG00000143549'



    #model_filename = './data/real/splicing/model/' + geneNow + '_1.pt'
    model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_1.pt'
    model = torch.load(model_filename)
    



    #exons = loadnpz('./data/real/splicing/input/exons.npz')
    #exons_now = exons[exons[:, 3] == geneNow]
    #exonLength = exons_now[:, 2].astype(int) - exons_now[:, 1].astype(int)
    #plt.hist(exonLength, bins=100)
    #plt.show()
    #quit()


    junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
    countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
    edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')


    #print (np.mean(countNow))
    #quit()

    Njunction = junctionNow.shape[0]

    isoformJunctions = loadnpz('./data/real/splicing/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')[:, :Njunction] #Todo remove this subsetting
    isoformCounts = loadnpz('./data/real/splicing/geneFiles/isoform_counts/' + str(geneNow) + '.npz')

    isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')[:, :Njunction] #Todo remove this subsetting
    isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')



    sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
    sampleLong = loadnpz('./data/real/splicing/input/samples_longRead.npz')
    #inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
    #inverse1 = uniqueValMaker(inverse1)
    #argGood = np.argwhere(np.isin( inverse1[:sampleLong.shape[0]]   , inverse1[sampleLong.shape[0]:] ))[:, 0]
    #countNow = countNow[:, argGood]
    #isoformCounts = isoformCounts[:, argGood]

    #if True:
    inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
    inverse1 = uniqueValMaker(inverse1)
    argGood = np.argwhere(np.isin(  inverse1[sampleLong.shape[0]:], inverse1[:sampleLong.shape[0]]   ))[:, 0]
    sampleList = sampleList[argGood]
    countNow = countNow[:, argGood]
    isoformCounts = isoformCounts[:, argGood]
    print (countNow.shape)
    print (isoformCounts.shape)
    #quit()

    #print (np.mean(countNow))
    #quit()

    

    

    
    #print (edges[:5, :5])
    #quit()
    #plt.imshow(isoformJunctions)
    #plt.show()
    #quit()


    #argsortJunction = np.argsort(junctionNow[:, 1].astype(float))
    #junctionNow = junctionNow[argsortJunction]
    #countNow = countNow[argsortJunction]
    #isoformJunctions = isoformJunctions[:, argsortJunction]
    countNow = torch.tensor(countNow).float()
    Nsample = countNow.shape[1]
    Njunction = countNow.shape[0]

    observations_batch = countNow

    edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
    edgeMatrix[edges[:, 0], edges[:, 1]] = 1

    #print (edgeMatrix[-1])
    #quit()

    #edgeMatrix = edgeMatrix[argsortJunction][:, argsortJunction]

    graphSize = Njunction
    #finalProbSize = 19788# Nsample 
    #finalProbSize = 88 
    finalProbSize = isoformCounts.shape[1]
    batchSize = Njunction
    ruleObject = SpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
    ruleObject.graphSize = graphSize
    ruleObject.observations_batch = observations_batch

    #dupGen = 5
    dupGen = 10
    offPolicy = False
    model = torch.load(model_filename)
    adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories, finalProb = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, dupGen)
    adjacency_matrices = adjacency_matrices.data.numpy()


    inverse1 = uniqueValMaker(adjacency_matrices)
    _, count1 = np.unique(inverse1, return_counts=True)
    count1_inverse = count1[inverse1]
    #print (count1_inverse.shape)
    #print (np.argwhere(count1_inverse >= 2).shape)
    #print (adjacency_matrices.shape)
    #adjacency_matrices = adjacency_matrices[count1_inverse >= 3]
    #print (adjacency_matrices.shape)



    #print (isoformCounts.shape)
    isoformCounts_total = np.sum(isoformCounts, axis=1)
    isoformCounts_long_total = np.sum(isoformCounts_long, axis=1)
    #quit()
    #print (adjacency_matrices.shape)
    #print (isoformJunctions.shape)
    #quit()

    inverse1 = uniqueValMaker(adjacency_matrices)
    _, count1 = np.unique(inverse1, return_counts=True)
    _, index1 = np.unique(inverse1, return_index=True)
    index1 = index1[np.argsort(count1)[-1::-1]][:20]

    #print (np.sort(count1)[-1::-1][:30])
    #print (np.sort(isoformCounts_total)[-1::-1][:30])
    #quit()
    

    #sns.clustermap(adjacency_matrices, col_cluster=False)
    #plt.show()  
    #print (np.sort(isoformCounts_total)[-1::-1][:10])

    #cat = np.concatenate(( isoformJunctions[ np.argsort(isoformCounts_total * -1)[:20] ],   adjacency_matrices[index1]  ), axis=0)

    #plt.imshow(cat)
    #plt.show()
    #quit()

    #sns.heatmap(isoformJunctions[ np.argsort(isoformCounts_total * -1)[:10] ])
    #plt.show()
    #quit()

    junction_true = np.sum(isoformJunctions * isoformCounts_total.reshape((-1, 1)), axis=0)
    junction_long = np.sum(isoformJunctions_long * isoformCounts_long_total.reshape((-1, 1)), axis=0)
    junction_pred = np.sum(adjacency_matrices, axis=0)
    junction_observe = np.sum(countNow.data.numpy(), axis=1)


    #plt.plot(junction_observe)
    #plt.plot(junction_long)
    #plt.plot(junction_true)
    #plt.show()
    #quit()

    
    #plt.plot(  junction_pred / np.mean(junction_pred) )#, alpha=0.5)
    #plt.plot(  junction_true / np.mean(junction_true)  )#, alpha=0.5  )
    #plt.plot(  junction_observe / np.mean(junction_observe)  )#, alpha=0.5 )
    #plt.show()
    #quit()

    finalProb_exp = np.exp(finalProb.data.numpy())
    #sfinalProb_exp[:] = 1

    
    combineIsoforms = np.concatenate(( isoformJunctions,  isoformJunctions_long, adjacency_matrices ), axis=0)
    combineIsoforms_inverse = uniqueValMaker(combineIsoforms)
    _, combineIsoforms_index = np.unique(combineIsoforms_inverse, return_index=True)
    combineIsoforms_unique = combineIsoforms[combineIsoforms_index]


    distance_matrix = np.zeros((combineIsoforms_unique.shape[0], combineIsoforms_unique.shape[0]))
    if False:
        for a in range(distance_matrix.shape[0]):
            for b in range(distance_matrix.shape[0]):
                sum1 = combineIsoforms_unique[a] + combineIsoforms_unique[b]
                sum1[sum1 > 1] = 1
                intersect1 = combineIsoforms_unique[a] * combineIsoforms_unique[b]
                dist1 = np.sum(intersect1) / np.sum(sum1)
                dist1 = 1.0 - dist1
                distance_matrix[a, b] = dist1
    else:
        for a in range(distance_matrix.shape[0]):
            for b in range(distance_matrix.shape[0]):
                if b > a:
                    junc1 = junctionNow[combineIsoforms_unique[a] > 0, 1:3].astype(int)
                    junc2 = junctionNow[combineIsoforms_unique[b] > 0, 1:3].astype(int)

                    exons1 = np.array([  junc1[:-1, 1], junc1[1:, 0]   ]).T
                    exons2 = np.array([  junc2[:-1, 1], junc2[1:, 0]   ]).T
                    
                    #print (exons1)
                    #print (exons2)

                    #exons1 = np.array([[100, 200], [300, 400]])
                    #exons2 = np.array([[150, 250], [300, 390]])

                    dist1 = jaccard_distance_exons_np(exons1, exons2)

                    distance_matrix[a, b] = dist1
                    distance_matrix[b, a] = dist1

    

    maxInverse = int(np.max(combineIsoforms_inverse)+1)

    
    true_inverse = combineIsoforms_inverse[:isoformJunctions.shape[0]]
    true_inverse_long = combineIsoforms_inverse[isoformJunctions.shape[0]:][:isoformJunctions_long.shape[0]]
    pred_inverse = combineIsoforms_inverse[-adjacency_matrices.shape[0]:]

    #true_inverse,  true_inverse_long, pred_inverse = combineIsoforms_inverse[:isoformJunctions.shape[0]], combineIsoforms_inverse[isoformJunctions.shape[0]:]

    


    #pastedTrueCounts = np.zeros(( maxInverse, finalProb_exp.shape[1] ))
    pastedTrueCounts = np.zeros(( maxInverse, isoformCounts.shape[1] ))
    pastedTrueCounts[true_inverse] = isoformCounts

    print ('isoformCounts', isoformCounts.shape)

    

    pastedTrueCounts_long = np.zeros(( maxInverse, isoformCounts_long.shape[1] ))
    pastedTrueCounts_long[true_inverse_long] = isoformCounts_long


    pastedPredCounts = np.zeros(( maxInverse, finalProb_exp.shape[1] ))
    for a in range(pred_inverse.shape[0]):
        pastedPredCounts[pred_inverse[a]] = pastedPredCounts[pred_inverse[a]] + finalProb_exp[a]

    import seaborn as sns

    
    pastedPredCounts_sum = np.sum(pastedPredCounts, axis=1) / np.sum(np.sum(pastedPredCounts, axis=1))
    pastedTrueCounts_sum = np.sum(pastedTrueCounts, axis=1) / np.sum(np.sum(pastedTrueCounts, axis=1))
    pastedTrueCounts_long_sum = np.sum(pastedTrueCounts_long, axis=1) / np.sum(np.sum(pastedTrueCounts_long, axis=1))

    print (pastedPredCounts.shape)
    print (pastedTrueCounts.shape)
    print (pastedTrueCounts_long.shape)
    quit()



    import ot
    # Compute optimal transport plan
    print (pastedPredCounts_sum.shape, pastedTrueCounts_long_sum.shape, distance_matrix.shape)
    T_pred = ot.emd(pastedPredCounts_sum, pastedTrueCounts_long_sum, distance_matrix)
    T_true = ot.emd(pastedTrueCounts_sum, pastedTrueCounts_long_sum, distance_matrix)

    #print (np.sum(  T_pred, axis=1  ) - pastedPredCounts_sum)
    #print (np.sum(  T_pred, axis=0  ) - pastedTrueCounts_long_sum)
    #print (np.sum(T_true, axis=0))
    #quit()

    # Compute similarity score
    score_pred = np.sum(T_pred * distance_matrix)
    score_true = np.sum(T_true * distance_matrix)
    print ('score ours', score_pred)
    print ('score original', score_true)
    #quit()


    
    

    pastedPredCounts_sum_J = np.sum(pastedPredCounts_sum.reshape((-1, 1)) * combineIsoforms_unique, axis=0)
    pastedTrueCounts_sum_J = np.sum(pastedTrueCounts_sum.reshape((-1, 1)) * combineIsoforms_unique, axis=0)
    pastedLongCounts_sum_J = np.sum(pastedTrueCounts_long_sum.reshape((-1, 1)) * combineIsoforms_unique, axis=0)
    pastedPredCounts_sum_J = pastedPredCounts_sum_J / np.mean(pastedPredCounts_sum_J)
    pastedTrueCounts_sum_J = pastedTrueCounts_sum_J / np.mean(pastedTrueCounts_sum_J)
    pastedLongCounts_sum_J = pastedLongCounts_sum_J / np.mean(pastedLongCounts_sum_J)

    #print (pastedPredCounts_sum_J)
    #print (pastedTrueCounts_sum_J)

    #plt.plot( pastedPredCounts_sum_J  )
    #plt.plot(  junction_pred / np.mean(junction_pred) )
    #plt.show()
    #quit()

    if True:
        junction_observe = junction_observe / np.mean(junction_observe)
        

        print (junctionNow[ np.logical_and(pastedLongCounts_sum_J > 2, junction_observe < 0.5)  ])
        print ('')
        print (junctionNow[ np.logical_and(pastedLongCounts_sum_J < 0.5, junction_observe > 2)  ])

        plt.plot( pastedPredCounts_sum_J  )
        plt.plot( pastedTrueCounts_sum_J  )
        plt.plot(pastedLongCounts_sum_J)
        #plt.plot(  junction_observe  )
        plt.show()
        quit()

    plt.plot( pastedPredCounts_sum  )
    plt.plot( pastedTrueCounts_sum  )
    plt.plot(pastedTrueCounts_long_sum)
    plt.show()

    quit()

    cutOff = np.sort(pastedPredCounts.reshape((-1,)))[-pastedPredCounts.size // 100]
    pastedPredCounts[pastedPredCounts > cutOff] = cutOff

    cutOff = np.sort(pastedTrueCounts.reshape((-1,)))[-pastedTrueCounts.size // 100]
    pastedTrueCounts[pastedTrueCounts > cutOff] = cutOff

    #sns.clustermap(pastedPredCounts)
    #plt.show()

    #sns.clustermap(pastedTrueCounts)
    #plt.show()

    #print (adjacency_matrices_inverse.shape)
    #pred_isoforms, pred_counts = np.unique(pred_inverse, return_counts=True)


    #print (pred_isoforms, pred_counts)
    #print (true_inverse)
    quit()


    print (adjacency_matrices.shape)
    print (isoformJunctions.shape)


    quit()

    attribution = adjacency_matrices.reshape((adjacency_matrices.shape[0], adjacency_matrices.shape[1], 1)) * torch.exp(finalProb.reshape((finalProb.shape[0], 1, finalProb.shape[1])))
    attribution = torch.mean(attribution, axis=0)
    

    import seaborn as sns

    observations_batch[observations_batch > 300] = 300

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    distance_matrix = pdist(observations_batch.T, metric='euclidean')  # or 'correlation', etc.
    linkage_matrix = linkage(distance_matrix, method='average')  # or 'ward', 'single', etc.
    sorted_indices = leaves_list(linkage_matrix)


    sns.heatmap(observations_batch.data.numpy()[:, sorted_indices])
    plt.show()

    sns.heatmap(attribution.data.numpy()[:, sorted_indices])
    plt.show()

    attribution_replot = attribution *   ( torch.mean(observations_batch, axis=1) / (torch.mean(attribution, axis=1) + 1e-10) ).reshape((-1, 1))
    print (torch.mean(attribution_replot, axis=1))
    print (torch.mean(observations_batch, axis=1))

    attribution_replot[attribution_replot > 300] = 300

    sns.heatmap(attribution_replot.data.numpy()[:, sorted_indices])
    plt.show()
    quit()


#analyzeSplice()
#quit()



def NANCheck():

    files = os.listdir('./data/real/splicing/geneFiles/geneModels')
    geneList = []
    for file1 in files:
        if '.pt' in file1:
            if '_sampleGen_fast' in file1:
            #if '_sampleGen_size500_leak2' in file1:
                #if  '_sample' in file1:
                geneList.append(file1.split('_')[0])
    geneList = np.unique(np.array(geneList))

    allScores_our = []
    allScore_short = []
    
    gene_unique = loadnpz('./data/real/splicing/eval/gene_unique_perm.npz')

    useExonModel = False

    scoreList = np.zeros(( len(gene_unique), 2 ))
    usedList = np.zeros( len(gene_unique) )
    meanJunctions = np.zeros( len(gene_unique) )
    meanJunctions_long = np.zeros(len(gene_unique))



    badGenes = []
    with torch.no_grad():
    

        for gene_index in range(0, len(gene_unique)):# range(40): 354

            #print ('gene_index', gene_index)

            geneNow = gene_unique[gene_index]

            
            #

            if geneNow in geneList:# ['ENSG00000112294']:# ['ENSG00000164181', 'ENSG00000112294', 'ENSG00000134042', 'ENSG00000115257']:#geneList:# == 'ENSG00000067225':#'ENSG00000067225':# 'ENSG00000067225':# 'ENSG00000067225':# 'ENSG00000022267':# 'ENSG00000070081':

                #print ("PASS")

                #print (gene_index, geneNow)

                


                if useExonModel:
                    model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_scaledExon.pt'
                else:
                    #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_size500_leak2.pt' #Good
                    model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_fast.pt'
                model = torch.load(model_filename, weights_only=False)
                


                junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
                countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
                junctionPos = junctionNow[:, 1:3].astype(int)

                #print (junctionPos[:, 1] - junctionPos[:, 0])

                #print (junctionPos)
                #quit()
            


                Njunction = junctionNow.shape[0]
                isoformJunctions = loadnpz('./data/real/splicing/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos = loadnpz('./data/real/splicing/geneFiles/isoformJunctionsPos/' + str(geneNow) + '.npz')
                isoformCounts = loadnpz('./data/real/splicing/geneFiles/isoform_counts/' + str(geneNow) + '.npz')

                #print ('isoformJunctions')
                #print (isoformJunctions)


                isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctionPos/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')
                

                validIsoform = np.sum(isoformJunctions_long, axis=1)
                isoformJunctions_long = isoformJunctions_long[validIsoform >= 1]
                isoformCounts_long = isoformCounts_long[validIsoform >= 1]

                #print (isoformCounts_long.shape)
                #quit()

                if np.sum(isoformCounts_long) >= 1:


                    #print ('isoformCounts_long total', np.sum(isoformCounts_long))
                    #print ('Junction total', np.sum(countNow))

                    

                    sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                    sampleLong = loadnpz('./data/real/splicing/input/samples_longRead.npz')



                    inv1 = uniqueValMaker(sampleList[:, :3])
                    inv2 = uniqueValMaker(sampleLong[:, :3])

                    #print (np.unique(inv1).shape)
                    #print (np.intersect1d(inv1, inv2).shape)
                    #quit()
                    
                    
                    if not useExonModel:
                        inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
                        inverse1 = uniqueValMaker(inverse1)
                        argGood = np.argwhere(np.isin(  inverse1[sampleLong.shape[0]:], inverse1[:sampleLong.shape[0]]   ))[:, 0]
                        sampleList = sampleList[argGood]
                        countNow = countNow[:, argGood]
                        isoformCounts = isoformCounts[:, argGood]

                    if useExonModel:
                        exon_sample = loadnpz('./data/real/splicing/input/samples_RailLongRead.npz')
                        sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                        inverse1 = np.concatenate((  exon_sample, sampleList ), axis=0)
                        inverse1 = uniqueValMaker(inverse1)
                        argGood = np.zeros(exon_sample.shape[0], dtype=int)
                        for exon_sample_index in range(exon_sample.shape[0]):
                            arg1 = np.argwhere( inverse1 == inverse1[exon_sample_index] )[:, 0]
                            argGood[exon_sample_index] = arg1[1] - exon_sample.shape[0]
                        countNow = countNow[:, argGood]
                        sampleList = sampleList[argGood]



                    #countNow_mod = countNow / np.sum(countNow, axis=0).reshape((1, -1))
                    #sns.clustermap(countNow)
                    #plt.show()
                    #quit()


                    Nsample = countNow.shape[1]
                    Njunction = countNow.shape[0]

                    observations_batch =  torch.tensor(countNow).float()

                    edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                    edgeMatrix[edges[:, 0], edges[:, 1]] = 1

                    #plt.imshow(edgeMatrix)
                    #plt.show()

                    edgeSum = np.sum(edgeMatrix, axis=1)

                    goodCount = False
                    
                    #goodCount = False 

                    #if goodCount: #ONes not done yet.
                    #if (np.argwhere(edgeSum >= 2).shape[0] > np.argwhere(edgeSum == 1).shape[0]) and np.mean(countNow) > 10 and isoformJunctionPos_long.shape[0] == junctionPos.shape[0]:
                    #if (isoformJunctionPos_long.shape[0] == junctionPos.shape[0]) and (np.sum( isoformCounts_long) > 200):# geneNow == 'ENSG00000004487':
                    if True:# (isoformJunctionPos_long.shape[0] == junctionPos.shape[0]) and (np.sum( isoformCounts_long) > 10000) and (np.argwhere(edgeSum >= 2).shape[0] > np.argwhere(edgeSum == 1).shape[0]):
                        
                        
                        


                        graphSize = Njunction
                        #finalProbSize = 19788# Nsample 
                        finalProbSize = countNow.shape[1] 
                        batchSize = Njunction
                        ruleObject = SpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                        ruleObject.graphSize = graphSize
                        ruleObject.observations_batch = observations_batch
                        ruleObject.model = model

                        

                        #batchSize = 200
                        batchSize = 2000
                        #batchSize = 500
                        offPolicy = False
                        model = torch.load(model_filename)
                        adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batchSize)
                        _, finalProbAllow = ruleObject.graphRules(adjacency_matrices)
                        finalProb = model.finalProb(adjacency_matrices)

                        mean1 = np.mean(finalProb.data.numpy())

                        if np.isnan(mean1):
                            print ("Bad one")
                            print (geneNow)
                            badGenes.append(geneNow)
                            print (len(badGenes))

                            np.savez_compressed('./data/temp/badGenes2.npz', np.array(badGenes))





#badGenes = loadnpz('./data/temp/badGenes.npz')

#print (badGenes[1])

#print (len(badGenes))
#quit()

#NANCheck()
#quit()


def saveReadDepth():


    def findValidSamples(sampleList1, sampleList2, columnUse):
        inverse1 = np.concatenate(( sampleList1[:, columnUse], sampleList2[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_unique, inverse_index = np.unique(inverse1[:sampleList1.shape[0]], return_index=True)
        inverse_index = inverse_index[  np.isin(inverse_unique , inverse1[sampleList1.shape[0]:]  ) ]
        sampleList_new = sampleList1[inverse_index]
        return sampleList_new

    

    def processSamples(sampleList, sampleList_include, columnUse, countList):

        inverse1 = np.concatenate(( sampleList_include[:, columnUse], sampleList[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_include, inverse_samples = inverse1[:sampleList_include.shape[0]], inverse1[sampleList_include.shape[0]:]

        countList_new = np.zeros(( countList.shape[0],  sampleList_include.shape[0]  ))
        for a in range(sampleList_include.shape[0]):
            args1 = np.argwhere(inverse_samples == inverse_include[a])[:, 0]
            countList_new[:, a] = np.sum(countList[:, args1], axis=1)
        return countList_new
        
        


    #-134.5021209716797

    import seaborn as sns

    #geneNow = 'ENSG00000001461'

    files = os.listdir('./data/real/splicing/geneFiles/geneModels')
    geneList = []
    for file1 in files:
        if '.pt' in file1:
            if '_sampleGen_fast' in file1:
            #if '_sampleGen_size500_leak2' in file1:
                #if  '_sample' in file1:
                geneList.append(file1.split('_')[0])
    geneList = np.unique(np.array(geneList))

    gene_unique = loadnpz('./data/real/splicing/eval/gene_unique_perm.npz')

    useExonModel = False

    readCountShort = []
    readCountLong = []



    with torch.no_grad():
        for gene_index in range(0, len(gene_unique)):
            geneNow = gene_unique[gene_index]
            if geneNow in geneList:
                print (gene_index, geneNow)

                countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                
                isoformCounts = loadnpz('./data/real/splicing/geneFiles/isoform_counts/' + str(geneNow) + '.npz')
                isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')
                isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')
                

                validIsoform = np.sum(isoformJunctions_long, axis=1)
                isoformJunctions_long = isoformJunctions_long[validIsoform >= 1]
                isoformCounts_long = isoformCounts_long[validIsoform >= 1]

                if np.sum(isoformCounts_long) >= 1:

                    sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                    sampleLong = loadnpz('./data/real/splicing/input/samples_longRead.npz')

                    if not useExonModel:
                        inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
                        inverse1 = uniqueValMaker(inverse1)
                        argGood = np.argwhere(np.isin(  inverse1[sampleLong.shape[0]:], inverse1[:sampleLong.shape[0]]   ))[:, 0]
                        sampleList = sampleList[argGood]
                        countNow = countNow[:, argGood]
                        isoformCounts = isoformCounts[:, argGood]
                                        
                    if True:
                        columnUse = np.arange(3)
                        sampleList_include = findValidSamples(sampleList, sampleLong, columnUse)

                        countNow = processSamples(sampleList, sampleList_include, columnUse, countNow)
                        isoformCounts_long = processSamples(sampleLong, sampleList_include, columnUse, isoformCounts_long)

                        if np.sum(isoformCounts_long) >= 1:

                            longSum = np.sum(isoformCounts_long)
                            shortSum = np.sum(countNow)
                            readCountShort.append(shortSum)
                            readCountLong.append(longSum)
                            

    #plt.hist(readCountShort, bins=100)
    #plt.show()

    #plt.hist(readCountLong, bins=100)
    #plt.show()

    readCountShort = np.array(readCountShort)
    readCountLong = np.array(readCountLong)

    np.savez_compressed(  './data/real/splicing/eval/readCountShort.npz', readCountShort)
    np.savez_compressed(  './data/real/splicing/eval/readCountLong.npz', readCountLong)


#saveReadDepth()
#quit()


#plt.axvline(x=0, c='black')
#plt.axvline(x=  np.mean(improve) , c='red')#, linestyle='dashed')
#plt.xlabel('GReinSS error - RSEM error')
#plt.ylabel('number of genes')
#plt.gcf().set_size_inches(3.2, 3)
#plt.tight_layout()
#plt.savefig('./images/splicing/errorHist.pdf')
#plt.show()





def evaluateSplice():

    import numpy as np

    def compute_overlap_np(e1, e2):
        """Compute overlap in base pairs between two exons given as arrays."""
        start = np.maximum(e1[0], e2[0])
        end = np.minimum(e1[1], e2[1])
        return max(0, end - start)

    def jaccard_distance_exons_np(exons1, exons2):
        """
        Compute Jaccard distance between two exon sets.
        exons1 and exons2 are NumPy arrays of shape (n, 2) with [start, end] positions.
        """
        # Sort by start position
        exons1 = exons1[np.argsort(exons1[:, 0])]
        exons2 = exons2[np.argsort(exons2[:, 0])]

        #print ('exons2', exons2)

        total_bp1 = np.sum(exons1[:, 1] - exons1[:, 0])
        total_bp2 = np.sum(exons2[:, 1] - exons2[:, 0])

        # Two-pointer overlap computation
        i = j = 0
        intersection_bp = 0

        while i < len(exons1) and j < len(exons2):
            e1 = exons1[i]
            e2 = exons2[j]

            # Add overlapping base pairs
            overlap = compute_overlap_np(e1, e2)
            intersection_bp += overlap

            # Advance the interval with the smaller end
            if e1[1] < e2[1]:
                i += 1
            else:
                j += 1

        union_bp = total_bp1 + total_bp2 - intersection_bp

        #print (total_bp1 , total_bp2 , intersection_bp)

        return 1 - (intersection_bp / union_bp) if union_bp > 0 else 0.0


    def findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='j'):

        distance_matrix = np.zeros((isoJunction_pred.shape[0], isoJunction_true.shape[0]))

        if distType in ['junction']:
            for a in range(distance_matrix.shape[0]):
                for b in range(distance_matrix.shape[1]):
                    sum1 = combineIsoforms_unique[a] + combineIsoforms_unique[b]
                    sum1[sum1 > 1] = 1
                    intersect1 = combineIsoforms_unique[a] * combineIsoforms_unique[b]
                    if np.sum(sum1) == 0:
                        dist1 = 0.0
                    else:
                        dist1 = np.sum(intersect1) / np.sum(sum1)
                        dist1 = 1.0 - dist1
                    distance_matrix[a, b] = dist1
        
        if distType in ['exact']:
            for a in range(distance_matrix.shape[0]):
                for b in range(distance_matrix.shape[1]):
                    if True:#b != a:
                        junc1 = junctionPos_pred[isoJunction_pred[a] > 0]
                        junc2 = junctionPos_true[isoJunction_true[b] > 0]
                        junc1 = junc1[np.argsort(junc1[:, 0])]
                        junc2 = junc2[np.argsort(junc2[:, 0])]
                        if np.array_equal(junc1, junc2):
                            distance_matrix[a, b] = 0
                        else:
                            distance_matrix[a, b] = 1



        
        if distType in ['j', 'jaccard']:
            for a in range(distance_matrix.shape[0]):
                for b in range(distance_matrix.shape[1]):
                    if True:#b != a:
                        junc1 = junctionPos_pred[isoJunction_pred[a] > 0]
                        junc2 = junctionPos_true[isoJunction_true[b] > 0]

                        junc1 = junc1[np.argsort(junc1[:, 0])]
                        junc2 = junc2[np.argsort(junc2[:, 0])]

                        exons1 = np.array([  junc1[:-1, 1], junc1[1:, 0]   ]).T
                        exons2 = np.array([  junc2[:-1, 1], junc2[1:, 0]   ]).T

                        dist1 = jaccard_distance_exons_np(exons1, exons2)

                        distance_matrix[a, b] = dist1
                        #distance_matrix[b, a] = dist1

        return distance_matrix


    def evaluator(isoJunction_pred, isoCount_pred, isoJunction_true, isoCount_true, junctionPos_pred, junctionPos_true):


        

        distance_matrix = findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='j')
        #distance_matrix = findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='exact')

        

        if False:#distance_matrix.shape[0] <= 2:
            print (junctionPos_pred[ isoJunction_pred[0] > 0 ])
            print (junctionPos_pred[ isoJunction_pred[1] > 0 ])
            print (np.sum(isoCount_pred, axis=1) / np.sum(isoCount_pred))
            print (np.sum(isoCount_true, axis=1) / np.sum(np.sum(isoCount_true, axis=1)))
            print (distance_matrix)
            quit()

        if False:
            print ('dist')
            print (distance_matrix)
            print (isoJunction_pred)
            print (junctionPos_pred)
            #print (np.mean(isoCount_pred, axis=1))
            #print (np.mean(isoCount_true, axis=1))

        
        #print (distance_matrix.shape)
        

        #print (isoCount_true.shape, isoCount_pred.shape)

        #print (isoCount_true.shape)
        #quit()

        weights = np.sum(isoCount_true, axis=0)
        weights = weights[weights > 0]
        weights = weights / np.mean(weights)

        #print (isoJunction_pred[1])
        #print (isoJunction_true[:5])
        #print (distance_matrix[1])
        #quit()

        import ot
        scoreList_ours = []
        for sample_index in range(isoCount_pred.shape[1]):
            longCounts = isoCount_true[:, sample_index]
            predCounts = isoCount_pred[:, sample_index]

            #print ('longCounts', longCounts)

            #predCounts[:] = 1

            

            #print (np.sum(predCounts))

            if np.sum(predCounts) == 0:
                predCounts[:] = 1
                #Trigger mod!

            if np.sum(longCounts) > 0:
                longCounts = longCounts / np.sum(longCounts)
                predCounts = predCounts / np.sum(predCounts)

                #print ('a')
                #print (np.round(longCounts*100))
                #print (np.round(predCounts*100))

                #print (longCounts)
                
                T_pred = ot.emd(predCounts, longCounts, distance_matrix) #TODO: Look into if optimal transport is miminizing or maximizing
                #print (T_pred)
                score_pred = np.sum(T_pred * distance_matrix)
                #print (score_pred)
                #quit()
                scoreList_ours.append(score_pred)
        scoreList_ours = np.array(scoreList_ours)

        #print (scoreList_ours)
        #print (np.argwhere( np.isnan(scoreList_ours) ))
        #print (scoreList_ours.shape)
        
        score = np.mean(scoreList_ours * weights)

        #print ('scoreList_ours', scoreList_ours)
        #quit()
        
        return score
    
    def existEvaluator(isoJunction_pred, isoCount_pred, isoJunction_true, isoCount_true, junctionPos_pred, junctionPos_true):


        distance_matrix = findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='j')
        

        #plt.imshow(distance_matrix)
        #plt.show()
        
        scoreList_ours = []
        for sample_index in range(isoCount_pred.shape[1]):
            longCounts = isoCount_true[:, sample_index]
            predCounts = isoCount_pred[:, sample_index]

            if np.sum(predCounts) == 0:
                predCounts[:] = 1
            if np.sum(longCounts) > 0:
                #longCounts = longCounts / np.sum(longCounts)
                predCounts = predCounts / np.sum(predCounts)

                argLong = np.argwhere(longCounts > 0)[:, 0]

                argPred = np.argsort( predCounts * -1 )[:20]

                closest = distance_matrix[argPred][:, argLong]
                #closest = np.min(closest, axis=0)

                closest_mod = np.ones(20)
                for size_include in range(argPred.shape[0]):
                    closest_mod[size_include] = np.mean( np.min(closest[:size_include+1], axis=0) )
                
                closest_mod[argPred.shape[0]:] = closest_mod[argPred.shape[0]-1]

                #print (closest_mod)


                scoreList_ours.append( np.copy(closest_mod) )
        



                if False:

                    longProps = np.zeros(argLong.shape[0])
                    for a in range(argLong.shape[0]):
                        if 0 in distance_matrix[:, argLong[a]]:
                            arg1 = np.argmin(distance_matrix[:, argLong[a]])
                            longProps[a] = predCounts[arg1]

                    #print (longProps)

                    scoreList_ours.append(np.median(longProps))


       #quit()

        scoreList_ours = np.array(scoreList_ours)
        scoreList_ours = np.mean(scoreList_ours, axis=0)
        #print (scoreList_ours)
        #score = np.mean(scoreList_ours * scoreList_ours)

        
        return scoreList_ours
    

    def logProbEval(isoJunction_pred, isoCount_pred, isoJunction_true, isoCount_true, junctionPos_pred, junctionPos_true):

        distance_matrix = findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='exact')
        scoreList_ours = []
        for sample_index in range(isoCount_pred.shape[1]):
            longCounts = isoCount_true[:, sample_index]
            predCounts = isoCount_pred[:, sample_index]

            if np.sum(predCounts) == 0:
                predCounts[:] = 1

            if np.sum(longCounts) > 0:
                longCounts = longCounts / np.sum(longCounts)
                predCounts = predCounts / np.sum(predCounts)
                predCounts = np.sum(predCounts.reshape((-1, 1)) * (1-distance_matrix), axis=0)
                
                
                score_pred = -1 * np.sum(np.log(longCounts + 1e-2) * predCounts)
                scoreList_ours.append(score_pred)
                
        scoreList_ours = np.array(scoreList_ours)
        return scoreList_ours


    def findValidSamples(sampleList1, sampleList2, columnUse):
        inverse1 = np.concatenate(( sampleList1[:, columnUse], sampleList2[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_unique, inverse_index = np.unique(inverse1[:sampleList1.shape[0]], return_index=True)
        inverse_index = inverse_index[  np.isin(inverse_unique , inverse1[sampleList1.shape[0]:]  ) ]
        sampleList_new = sampleList1[inverse_index]
        return sampleList_new

    

    def processSamples(sampleList, sampleList_include, columnUse, countList):

        inverse1 = np.concatenate(( sampleList_include[:, columnUse], sampleList[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_include, inverse_samples = inverse1[:sampleList_include.shape[0]], inverse1[sampleList_include.shape[0]:]

        countList_new = np.zeros(( countList.shape[0],  sampleList_include.shape[0]  ))
        for a in range(sampleList_include.shape[0]):
            args1 = np.argwhere(inverse_samples == inverse_include[a])[:, 0]
            countList_new[:, a] = np.sum(countList[:, args1], axis=1)
        return countList_new
        
        


    #-134.5021209716797

    import seaborn as sns

    #geneNow = 'ENSG00000001461'

    files = os.listdir('./data/real/splicing/geneFiles/geneModels')
    geneList = []
    for file1 in files:
        if '.pt' in file1:
            #_sampleGen_RewardNormalized
            #if '_sampleGen_fast' in file1:
            if '_sampleGen_Naive' in file1:
            #if '_sampleGen_GFlow' in file1:
            #if '_sampleGen_size500_leak2' in file1:
                #if  '_sample' in file1:
                geneList.append(file1.split('_')[0])
    geneList = np.unique(np.array(geneList))

    allScores_our = []
    allScore_short = []

    #print ("T")

    #gene_unique = loadnpz('./data/temp/gene_unique_perm.npz')#[:30]  #[:6]
    gene_unique = loadnpz('./data/real/splicing/eval/gene_unique_perm.npz')#[:30]  #[:6]

    #print (np.argwhere(gene_unique == 'ENSG00000166343'))
    #quit()

    #print (gene_unique[:15]) #ENSG00000102755
    #quit()

    #geneInclude = ['ENSG00000159173', 'ENSG00000165948', 'ENSG00000196247', 'ENSG00000153140']#, 'ENSG00000164181', 'ENSG00000166343']


    useExonModel = False

    scoreList = np.zeros(( len(gene_unique), 2 ))
    usedList = np.zeros( len(gene_unique) )
    meanJunctions = np.zeros( len(gene_unique) )
    meanJunctions_long = np.zeros(len(gene_unique))




    with torch.no_grad():
    

        for gene_index in range(0, len(gene_unique)):# range(40): 354

            #print ('gene_index', gene_index)

            geneNow = gene_unique[gene_index]

            
            #

            if geneNow in geneList:# ['ENSG00000112294']:# ['ENSG00000164181', 'ENSG00000112294', 'ENSG00000134042', 'ENSG00000115257']:#geneList:# == 'ENSG00000067225':#'ENSG00000067225':# 'ENSG00000067225':# 'ENSG00000067225':# 'ENSG00000022267':# 'ENSG00000070081':

                #print ("PASS")

                print (gene_index, geneNow)

                


                if useExonModel:
                    model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_scaledExon.pt'
                else:
                    #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_size500_leak2.pt' #Good
                    #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_fast.pt'
                    #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_RewardNormalized.pt'
                    #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_fast_RerunTime.pt'
                    model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_Naive.pt'
                    #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_GFlow.pt'

                    
                    
                
                model = torch.load(model_filename, weights_only=False)
                


                junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
                countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
                junctionPos = junctionNow[:, 1:3].astype(int)

                #print (junctionPos[:, 1] - junctionPos[:, 0])

                #print (junctionPos)
                #quit()
            


                Njunction = junctionNow.shape[0]
                isoformJunctions = loadnpz('./data/real/splicing/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos = loadnpz('./data/real/splicing/geneFiles/isoformJunctionsPos/' + str(geneNow) + '.npz')
                isoformCounts = loadnpz('./data/real/splicing/geneFiles/isoform_counts/' + str(geneNow) + '.npz')

                #print ('isoformJunctions')
                #print (isoformJunctions)


                isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctionPos/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')
                

                validIsoform = np.sum(isoformJunctions_long, axis=1)
                isoformJunctions_long = isoformJunctions_long[validIsoform >= 1]
                isoformCounts_long = isoformCounts_long[validIsoform >= 1]

                #print (isoformCounts_long.shape)
                #quit()

                if np.sum(isoformCounts_long) >= 1:


                    print ('isoformCounts_long total', np.sum(isoformCounts_long))
                    print ('Junction total', np.sum(countNow))

                    

                    sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                    sampleLong = loadnpz('./data/real/splicing/input/samples_longRead.npz')



                    inv1 = uniqueValMaker(sampleList[:, :3])
                    inv2 = uniqueValMaker(sampleLong[:, :3])

                    #print (np.unique(inv1).shape)
                    #print (np.intersect1d(inv1, inv2).shape)
                    #quit()
                    
                    
                    if not useExonModel:
                        inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
                        inverse1 = uniqueValMaker(inverse1)
                        argGood = np.argwhere(np.isin(  inverse1[sampleLong.shape[0]:], inverse1[:sampleLong.shape[0]]   ))[:, 0]
                        sampleList = sampleList[argGood]
                        countNow = countNow[:, argGood]
                        isoformCounts = isoformCounts[:, argGood]


                        
                        

                        if False:
                            exon_sample = loadnpz('./data/real/splicing/input/samples_RailLongRead.npz')
                            inverse1 = np.concatenate((  exon_sample, sampleList ), axis=0)
                            inverse1 = uniqueValMaker(inverse1)
                            argGood2 = np.zeros(exon_sample.shape[0], dtype=int)
                            for exon_sample_index in range(exon_sample.shape[0]):
                                arg1 = np.argwhere( inverse1 == inverse1[exon_sample_index] )[:, 0]
                                argGood2[exon_sample_index] = arg1[1] - exon_sample.shape[0]
                            countNow = countNow[:, argGood2]
                            sampleList = sampleList[argGood2]


                    if useExonModel:
                        exon_sample = loadnpz('./data/real/splicing/input/samples_RailLongRead.npz')
                        sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                        inverse1 = np.concatenate((  exon_sample, sampleList ), axis=0)
                        inverse1 = uniqueValMaker(inverse1)
                        argGood = np.zeros(exon_sample.shape[0], dtype=int)
                        for exon_sample_index in range(exon_sample.shape[0]):
                            arg1 = np.argwhere( inverse1 == inverse1[exon_sample_index] )[:, 0]
                            argGood[exon_sample_index] = arg1[1] - exon_sample.shape[0]
                        countNow = countNow[:, argGood]
                        sampleList = sampleList[argGood]



                    #countNow_mod = countNow / np.sum(countNow, axis=0).reshape((1, -1))
                    #sns.clustermap(countNow)
                    #plt.show()
                    #quit()


                    Nsample = countNow.shape[1]
                    Njunction = countNow.shape[0]

                    observations_batch =  torch.tensor(countNow).float()

                    edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                    edgeMatrix[edges[:, 0], edges[:, 1]] = 1

                    #plt.imshow(edgeMatrix)
                    #plt.show()

                    edgeSum = np.sum(edgeMatrix, axis=1)

                    goodCount = False
                    
                    #goodCount = False 

                    #if goodCount: #ONes not done yet.
                    #if (np.argwhere(edgeSum >= 2).shape[0] > np.argwhere(edgeSum == 1).shape[0]) and np.mean(countNow) > 10 and isoformJunctionPos_long.shape[0] == junctionPos.shape[0]:
                    #if (isoformJunctionPos_long.shape[0] == junctionPos.shape[0]) and (np.sum( isoformCounts_long) > 200):# geneNow == 'ENSG00000004487':
                    if True:# (isoformJunctionPos_long.shape[0] == junctionPos.shape[0]) and (np.sum( isoformCounts_long) > 10000) and (np.argwhere(edgeSum >= 2).shape[0] > np.argwhere(edgeSum == 1).shape[0]):
                        
                        
                        


                        graphSize = Njunction
                        #finalProbSize = 19788# Nsample 
                        finalProbSize = countNow.shape[1] 
                        batchSize = Njunction
                        ruleObject = SpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                        ruleObject.graphSize = graphSize
                        ruleObject.observations_batch = observations_batch
                        ruleObject.model = model

                        

                        #batchSize = 200
                        batchSize = 2000
                        #batchSize = 500
                        offPolicy = False
                        model = torch.load(model_filename)
                        adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batchSize)
                        

                        #plt.plot(np.sum(countNow, axis=1))
                        #plt.show()
                        if False:
                            matrix1_short= np.zeros((adjacency_matrices.shape[1]+1, adjacency_matrices.shape[1]+1))
                            #isoformJunctions_long, isoformCounts_long
                            for a in range(isoformJunctions.shape[0]):
                                junc1 = np.argwhere(isoformJunctions[a] == 1)[:, 0]
                                sum1 = np.sum(isoformCounts[a])
                                if junc1.shape[0] >= 1:
                                    matrix1_short[-1, junc1[0]] += sum1
                                    matrix1_short[junc1[-1], -1] += sum1
                                    for b in range(junc1.shape[0] - 1):
                                        matrix1_short[junc1[b], junc1[b+1]] += sum1
                            matrix1_short = 2 * matrix1_short / np.max(matrix1_short)
                            matrix1_cat_short = matrix1_short + edgeMatrix
                            #plt.imshow(matrix1_cat)
                            #plt.show()

                            matrix1_long = np.zeros((adjacency_matrices.shape[1]+1, adjacency_matrices.shape[1]+1))

                            #isoformJunctions_long, isoformCounts_long
                            for a in range(isoformJunctions_long.shape[0]):
                                junc1 = np.argwhere(isoformJunctions_long[a] == 1)[:, 0]
                                sum1 = np.sum(isoformCounts_long[a])
                                if junc1.shape[0] >= 1:
                                    matrix1_long[-1, junc1[0]] += sum1
                                    matrix1_long[junc1[-1], -1] += sum1
                                    for b in range(junc1.shape[0] - 1):
                                        matrix1_long[junc1[b], junc1[b+1]] += sum1
                            matrix1_long = 2 * matrix1_long / np.max(matrix1_long)
                            matrix1_cat_long = matrix1_long + edgeMatrix
                            #plt.imshow(matrix1_cat)
                            #plt.show()

                            matrix1 = np.zeros((adjacency_matrices.shape[1]+1, adjacency_matrices.shape[1]+1))
                            for a in range(adjacency_matrices.shape[0]):
                                junc1 = np.argwhere(adjacency_matrices[a].data.numpy() == 1)[:, 0]
                                if junc1.shape[0] >= 1:
                                    matrix1[-1, junc1[0]] += 1
                                    matrix1[junc1[-1], -1] += 1
                                    for b in range(junc1.shape[0] - 1):
                                        matrix1[junc1[b], junc1[b+1]] += 1

                            #print ('matrix1', matrix1)
                            #matrix1_cat = np.concatenate((matrix1, edgeMatrix* np.max(matrix1)), axis=0)
                            matrix1 = 2 * matrix1 / np.max(matrix1)
                            matrix1_cat_pred = matrix1 + edgeMatrix
                            #plt.imshow(matrix1_cat)
                            #plt.show()
                            #quit()


                        #adjacency_matrices, finalProb_exp_sum = processPredictionCounts(adjacency_matrices, finalProb)
                        _, finalProbAllow = ruleObject.graphRules(adjacency_matrices)
                        finalProb = model.finalProb(adjacency_matrices)

                        #print (finalProb)
                        #quit()

                        #if not useExonModel:
                        #    finalProb = finalProb[:, argGood2]


                        finalProb = finalProb + finalProbAllow
                        finalProb = nn.LogSoftmax(dim=1)(finalProb)

                        


                        


                        adjacency_matrices, finalProb_exp_sum = processPredictionCounts(adjacency_matrices, finalProb, normalizeJunc=True)


                        

                        #np.savez_compressed('./data/temp/adjacency_matrices.npz',adjacency_matrices )
                        #np.savez_compressed('./data/temp/finalProb_exp_sum.npz', finalProb_exp_sum)
                        #quit()


                        if False:
                            bias = model.giveBias()
                            adjacency_matrices_weighted = torch.log(torch.tensor(adjacency_matrices).float() + 1e-10)
                            adjacency_matrices_weighted[adjacency_matrices == 0] = -500
                            adjacency_matrices_weighted = adjacency_matrices_weighted + bias.reshape((1, -1))
                            adjacency_matrices_weighted = torch.nn.Softmax(dim=1)(adjacency_matrices_weighted)
                            adjacency_matrices_weighted = adjacency_matrices_weighted.data.numpy()

                        

                        #columnUse = np.arange(5)
                        columnUse = np.arange(3)
                        sampleList_include = findValidSamples(sampleList, sampleLong, columnUse)

                        print (countNow.shape)

                        countNow = processSamples(sampleList, sampleList_include, columnUse, countNow)
                        finalProb_exp_sum = processSamples(sampleList, sampleList_include, columnUse, finalProb_exp_sum)
                        isoformCounts = processSamples(sampleList, sampleList_include, columnUse, isoformCounts)
                        isoformCounts_long = processSamples(sampleLong, sampleList_include, columnUse, isoformCounts_long)

                        if np.sum(isoformCounts_long) >= 1:


                            #print (isoformJunctions_long.shape, isoformCounts_long.shape)
                            impliedJunc = np.matmul(isoformJunctions_long.T, isoformCounts_long)
                            impliedJunc_short = np.matmul(isoformJunctions.T, isoformCounts)
                            #predJunc = np.matmul(adjacency_matrices_weighted.T, finalProb_exp_sum)
                            predJunc = np.matmul(adjacency_matrices.T, finalProb_exp_sum)
                            #impliedPred = np.

                            

                            impliedJunc_mod = impliedJunc / np.mean(impliedJunc, axis=0).reshape((1, -1))
                            countNow_mod = countNow / np.mean(countNow, axis=0).reshape((1, -1))
                            predJunc_mod = predJunc / np.mean(predJunc, axis=0).reshape((1, -1))

                            


                            
                            #print (finalProb_exp_sum.shape)
                            predCount_sum = np.sum(finalProb_exp_sum, axis=1)
                            trueCount_sum = np.sum(isoformCounts_long, axis=1)

                            
                            predCount_print = predCount_sum[predCount_sum > np.sum(predCount_sum) / 10]
                            adj_print  = adjacency_matrices[predCount_sum > np.sum(predCount_sum) / 10].astype(int)
                            trueCount_print  = trueCount_sum[trueCount_sum> np.sum(trueCount_sum) / 10].astype(int)
                            iso_print = isoformJunctions_long[trueCount_sum > np.sum(trueCount_sum) / 10]
                            isoformJunctionPos_long_print = np.concatenate((  np.arange(isoformJunctionPos_long.shape[0]).reshape((-1, 1)), isoformJunctionPos_long   ), axis=1)    

                            if True:
                                print (isoformJunctionPos_long_print)
                                for a in range(predCount_print.shape[0]):
                                    print (adj_print[a], int(predCount_print[a]))
                                print ('')
                                for a in range(iso_print.shape[0]):
                                    print (iso_print[a], int(trueCount_print[a]))

                            #quit()
                            

                            #score_ours = existEvaluator(adjacency_matrices, finalProb_exp_sum, isoformJunctions_long, isoformCounts_long, junctionPos, isoformJunctionPos_long)
                            #score_short = existEvaluator(isoformJunctions, isoformCounts, isoformJunctions_long, isoformCounts_long, isoformJunctionPos, isoformJunctionPos_long)

                            

                            score_ours = evaluator(adjacency_matrices, np.copy(finalProb_exp_sum), isoformJunctions_long, isoformCounts_long, junctionPos, isoformJunctionPos_long)
                            score_short = evaluator(isoformJunctions, np.copy(isoformCounts), isoformJunctions_long, isoformCounts_long, isoformJunctionPos, isoformJunctionPos_long)

                            #scoreList_ours = logProbEval(adjacency_matrices, finalProb_exp_sum, isoformJunctions_long, isoformCounts_long, junctionPos, isoformJunctionPos_long)
                            #scoreList_short = logProbEval(isoformJunctions, isoformCounts, isoformJunctions_long, isoformCounts_long, isoformJunctionPos, isoformJunctionPos_long)

                            assert not np.isnan(score_ours)
                            assert not np.isnan(score_short)
                            

                            #score_ours = np.mean(scoreList_ours)
                            #score_short = np.mean(scoreList_short)

                            allScores_our.append(score_ours)
                            allScore_short.append(score_short)

                            #print (geneNow)
                            print (score_ours, score_short)

                            scoreList[gene_index, 0] = score_ours
                            scoreList[gene_index, 1] = score_short
                            usedList[gene_index] = 1


                            meanJunctions[gene_index] = np.mean( np.sum(predJunc, axis=1)  / np.sum(finalProb_exp_sum)  )
                            meanJunctions_long[gene_index] = np.mean(   np.sum(impliedJunc, axis=1) / np.sum(isoformCounts_long)   )

                            if False:#score_ours > score_short + 0.2:

                                plt.plot(np.sum(predJunc, axis=1)  / np.sum(finalProb_exp_sum))
                                plt.plot(np.sum(countNow, axis=1) /  np.max(np.sum(countNow, axis=1)) ) # / np.sum(countNow))
                                plt.plot(np.sum(impliedJunc, axis=1) / np.sum(isoformCounts_long))
                                plt.plot(np.sum(impliedJunc_short, axis=1) / np.sum(isoformCounts))
                                plt.show()




                            #quit()

                            #print ('isoform total', np.sum( isoformCounts_long))
                            #print (isoformCounts_long.shape)

                            #if score_ours > score_short + 0.1:
                            if False:

                                print (junctionPos)
                                print (countNow.shape)
                                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                                axs[0].imshow(matrix1_cat_short)
                                axs[1].imshow(matrix1_cat_pred)
                                axs[2].imshow(matrix1_cat_long)
                                plt.show()


                                #plt.plot(np.sum(matrix1_cat_short, axis=0))
                                plt.plot(np.sum(matrix1, axis=0)  / np.sum(matrix1), alpha=0.5)
                                plt.plot(np.sum(matrix1_long, axis=0) / np.sum(matrix1_long), alpha=0.5)
                                plt.plot(np.sum(countNow, axis=1) / np.sum(countNow), alpha=0.5)
                                plt.legend(['ours', 'true', 'count'])
                                plt.show()


                                plt.imshow(isoformCounts_long)
                                plt.show()


                                
                            #quit()


    #quit()
    

    #np.savez_compressed('./data/real/splicing/eval/fast_scoreList.npz', scoreList)
    #np.savez_compressed('./data/real/splicing/eval/fast_usedList.npz', usedList)


    np.savez_compressed('./data/real/splicing/eval/scoreList_Naive.npz', scoreList)


    #np.savez_compressed('./data/real/splicing/eval/rewardNorm_scoreList.npz', scoreList)
    #np.savez_compressed('./data/real/splicing/eval/rewardNorm_usedList.npz', usedList)

    #np.savez_compressed('./data/temp/meanJunctions.npz', meanJunctions)
    #np.savez_compressed('./data/temp/meanJunctions_long.npz', meanJunctions_long)
    
    allScores_our, allScore_short = np.array(allScores_our), np.array(allScore_short)

    
    print ('ours', np.mean(allScores_our) )
    print ('short', np.mean(allScore_short) )

    plt.scatter(  allScore_short, allScores_our ) 
    plt.plot([0, 1], [0, 1], c='black')
    plt.xlabel('baseline error')
    plt.ylabel('our error')
    plt.show()
    
    

    #ours 0.23082808508827252
    #short 0.3479232026301234

    #ours 0.23958565975263052
    #short 0.3412720781782155

    #ours 0.26145167058624236 #5 colUse 
    #short 0.3643891021721837


    #ours 0.1758531611232415
    #short 0.3297498289663494


    #Reward normalized
    #ours 0.2845706652916118
    #short 0.34218850193549755

    #Fast
    #ours 0.29308505089390746
    #short 0.3424423090798629
    


evaluateSplice()
quit()




def entropyCor():

    

    def findValidSamples(sampleList1, sampleList2, columnUse):
        inverse1 = np.concatenate(( sampleList1[:, columnUse], sampleList2[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_unique, inverse_index = np.unique(inverse1[:sampleList1.shape[0]], return_index=True)
        inverse_index = inverse_index[  np.isin(inverse_unique , inverse1[sampleList1.shape[0]:]  ) ]
        sampleList_new = sampleList1[inverse_index]
        return sampleList_new

    

    def processSamples(sampleList, sampleList_include, columnUse, countList):

        inverse1 = np.concatenate(( sampleList_include[:, columnUse], sampleList[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_include, inverse_samples = inverse1[:sampleList_include.shape[0]], inverse1[sampleList_include.shape[0]:]

        countList_new = np.zeros(( countList.shape[0],  sampleList_include.shape[0]  ))
        for a in range(sampleList_include.shape[0]):
            args1 = np.argwhere(inverse_samples == inverse_include[a])[:, 0]
            countList_new[:, a] = np.sum(countList[:, args1], axis=1)
        return countList_new
        
        

    def miniEntropyCor(predCounts, trueCounts):

        argGood = np.argwhere(np.sum(trueCounts, axis=0) >= 1)[:, 0]

        predCounts, trueCounts = predCounts[:, argGood], trueCounts[:, argGood]

        argBad = np.argwhere(np.sum(predCounts, axis=0) == 0)
        predCounts[:, argBad] = 1e-5

        trueCounts = trueCounts / np.sum(trueCounts, axis=0).reshape((1, -1))

        predCounts = predCounts / np.sum(predCounts, axis=0).reshape((1, -1))

        entropyTrue = np.sum(trueCounts * np.log(trueCounts + 1e-10) * -1, axis=0)
        entropyPred = np.sum(predCounts * np.log(predCounts + 1e-10) * -1, axis=0)

        cor1 = scipy.stats.pearsonr(entropyPred, entropyTrue)

        return cor1[0]



    #-134.5021209716797

    import seaborn as sns

    #geneNow = 'ENSG00000001461'

    files = os.listdir('./data/real/splicing/geneFiles/geneModels')
    geneList = []
    for file1 in files:
        if '.pt' in file1:
            if '_sampleGen_fast' in file1:
            #if '_sampleGen_size500_leak2' in file1:
                #if  '_sample' in file1:
                geneList.append(file1.split('_')[0])
    geneList = np.unique(np.array(geneList))

    allScores_our = []
    allScore_short = []

    #print ("T")

    #gene_unique = loadnpz('./data/temp/gene_unique_perm.npz')#[:30]  #[:6]
    gene_unique = loadnpz('./data/real/splicing/eval/gene_unique_perm.npz')#[:30]  #[:6]

    #print (np.argwhere(gene_unique == 'ENSG00000166343'))
    #quit()

    #print (gene_unique[:15]) #ENSG00000102755
    #quit()

    #geneInclude = ['ENSG00000159173', 'ENSG00000165948', 'ENSG00000196247', 'ENSG00000153140']#, 'ENSG00000164181', 'ENSG00000166343']


    useExonModel = False

    scoreList = np.zeros(( len(gene_unique), 2 ))
    usedList = np.zeros( len(gene_unique) )
    meanJunctions = np.zeros( len(gene_unique) )
    meanJunctions_long = np.zeros(len(gene_unique))


    corList_ours = []
    corList_other = []


    with torch.no_grad():
    

        for gene_index in range(0, len(gene_unique)):# range(40): 354

            #print ('gene_index', gene_index)

            geneNow = gene_unique[gene_index]

            
            #

            if geneNow in geneList:# ['ENSG00000112294']:# ['ENSG00000164181', 'ENSG00000112294', 'ENSG00000134042', 'ENSG00000115257']:#geneList:# == 'ENSG00000067225':#'ENSG00000067225':# 'ENSG00000067225':# 'ENSG00000067225':# 'ENSG00000022267':# 'ENSG00000070081':

                #print ("PASS")

                print (gene_index, geneNow)

                


                if useExonModel:
                    model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_scaledExon.pt'
                else:
                    #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_size500_leak2.pt' #Good
                    model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_fast.pt'
                model = torch.load(model_filename, weights_only=False)
                


                junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
                countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
                junctionPos = junctionNow[:, 1:3].astype(int)

                #print (junctionPos[:, 1] - junctionPos[:, 0])

                #print (junctionPos)
                #quit()
            


                Njunction = junctionNow.shape[0]
                isoformJunctions = loadnpz('./data/real/splicing/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos = loadnpz('./data/real/splicing/geneFiles/isoformJunctionsPos/' + str(geneNow) + '.npz')
                isoformCounts = loadnpz('./data/real/splicing/geneFiles/isoform_counts/' + str(geneNow) + '.npz')

                #print ('isoformJunctions')
                #print (isoformJunctions)


                isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctionPos/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')
                

                validIsoform = np.sum(isoformJunctions_long, axis=1)
                isoformJunctions_long = isoformJunctions_long[validIsoform >= 1]
                isoformCounts_long = isoformCounts_long[validIsoform >= 1]

                #print (isoformCounts_long.shape)
                #quit()

                if np.sum(isoformCounts_long) >= 1:


                    print ('isoformCounts_long total', np.sum(isoformCounts_long))
                    print ('Junction total', np.sum(countNow))

                    

                    sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                    sampleLong = loadnpz('./data/real/splicing/input/samples_longRead.npz')

                    
                    if not useExonModel:
                        inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
                        inverse1 = uniqueValMaker(inverse1)
                        argGood = np.argwhere(np.isin(  inverse1[sampleLong.shape[0]:], inverse1[:sampleLong.shape[0]]   ))[:, 0]
                        sampleList = sampleList[argGood]
                        countNow = countNow[:, argGood]
                        isoformCounts = isoformCounts[:, argGood]


                    Nsample = countNow.shape[1]
                    Njunction = countNow.shape[0]

                    observations_batch =  torch.tensor(countNow).float()

                    edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                    edgeMatrix[edges[:, 0], edges[:, 1]] = 1

                    
                    if True:# (isoformJunctionPos_long.shape[0] == junctionPos.shape[0]) and (np.sum( isoformCounts_long) > 10000) and (np.argwhere(edgeSum >= 2).shape[0] > np.argwhere(edgeSum == 1).shape[0]):
                        
                        
                        


                        graphSize = Njunction
                        #finalProbSize = 19788# Nsample 
                        finalProbSize = countNow.shape[1] 
                        batchSize = Njunction
                        ruleObject = SpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                        ruleObject.graphSize = graphSize
                        ruleObject.observations_batch = observations_batch
                        ruleObject.model = model

                        

                        #batchSize = 200
                        batchSize = 2000
                        #batchSize = 500
                        offPolicy = False
                        model = torch.load(model_filename)
                        adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batchSize)
                        

                        _, finalProbAllow = ruleObject.graphRules(adjacency_matrices)
                        finalProb = model.finalProb(adjacency_matrices)

                        finalProb = finalProb + finalProbAllow
                        finalProb = nn.LogSoftmax(dim=1)(finalProb)

                        adjacency_matrices, finalProb_exp_sum = processPredictionCounts(adjacency_matrices, finalProb, normalizeJunc=True)

                        #columnUse = np.arange(5)
                        columnUse = np.arange(3)
                        sampleList_include = findValidSamples(sampleList, sampleLong, columnUse)

                        print (countNow.shape)

                        countNow = processSamples(sampleList, sampleList_include, columnUse, countNow)
                        finalProb_exp_sum = processSamples(sampleList, sampleList_include, columnUse, finalProb_exp_sum)
                        isoformCounts = processSamples(sampleList, sampleList_include, columnUse, isoformCounts)
                        isoformCounts_long = processSamples(sampleLong, sampleList_include, columnUse, isoformCounts_long)

                        if np.sum(isoformCounts_long) >= 1:


                            cor_ours = miniEntropyCor(finalProb_exp_sum, isoformCounts_long)
                            cor_other = miniEntropyCor(isoformCounts, isoformCounts_long)

                            corList_ours.append(cor_ours)
                            corList_other.append(cor_other)
                            print (np.mean(np.array(corList_ours)), np.mean(np.array(corList_other)))
                            
                            




#entropyCor()
#quit()




def searchPlotExon():

    import numpy as np

    def compute_overlap_np(e1, e2):
        """Compute overlap in base pairs between two exons given as arrays."""
        start = np.maximum(e1[0], e2[0])
        end = np.minimum(e1[1], e2[1])
        return max(0, end - start)

    def jaccard_distance_exons_np(exons1, exons2):
        """
        Compute Jaccard distance between two exon sets.
        exons1 and exons2 are NumPy arrays of shape (n, 2) with [start, end] positions.
        """
        # Sort by start position
        exons1 = exons1[np.argsort(exons1[:, 0])]
        exons2 = exons2[np.argsort(exons2[:, 0])]

        #print ('exons2', exons2)

        total_bp1 = np.sum(exons1[:, 1] - exons1[:, 0])
        total_bp2 = np.sum(exons2[:, 1] - exons2[:, 0])

        # Two-pointer overlap computation
        i = j = 0
        intersection_bp = 0

        while i < len(exons1) and j < len(exons2):
            e1 = exons1[i]
            e2 = exons2[j]

            # Add overlapping base pairs
            overlap = compute_overlap_np(e1, e2)
            intersection_bp += overlap

            # Advance the interval with the smaller end
            if e1[1] < e2[1]:
                i += 1
            else:
                j += 1

        union_bp = total_bp1 + total_bp2 - intersection_bp

        #print (total_bp1 , total_bp2 , intersection_bp)

        return 1 - (intersection_bp / union_bp) if union_bp > 0 else 0.0


    def findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='j'):

        distance_matrix = np.zeros((isoJunction_pred.shape[0], isoJunction_true.shape[0]))

        if distType in ['junction']:
            for a in range(distance_matrix.shape[0]):
                for b in range(distance_matrix.shape[1]):
                    sum1 = combineIsoforms_unique[a] + combineIsoforms_unique[b]
                    sum1[sum1 > 1] = 1
                    intersect1 = combineIsoforms_unique[a] * combineIsoforms_unique[b]
                    if np.sum(sum1) == 0:
                        dist1 = 0.0
                    else:
                        dist1 = np.sum(intersect1) / np.sum(sum1)
                        dist1 = 1.0 - dist1
                    distance_matrix[a, b] = dist1
        
        if distType in ['exact']:
            for a in range(distance_matrix.shape[0]):
                for b in range(distance_matrix.shape[1]):
                    if True:#b != a:
                        junc1 = junctionPos_pred[isoJunction_pred[a] > 0]
                        junc2 = junctionPos_true[isoJunction_true[b] > 0]
                        junc1 = junc1[np.argsort(junc1[:, 0])]
                        junc2 = junc2[np.argsort(junc2[:, 0])]
                        if np.array_equal(junc1, junc2):
                            distance_matrix[a, b] = 0
                        else:
                            distance_matrix[a, b] = 1



        
        if distType in ['j', 'jaccard']:
            for a in range(distance_matrix.shape[0]):
                for b in range(distance_matrix.shape[1]):
                    if True:#b != a:
                        junc1 = junctionPos_pred[isoJunction_pred[a] > 0]
                        junc2 = junctionPos_true[isoJunction_true[b] > 0]

                        junc1 = junc1[np.argsort(junc1[:, 0])]
                        junc2 = junc2[np.argsort(junc2[:, 0])]

                        exons1 = np.array([  junc1[:-1, 1], junc1[1:, 0]   ]).T
                        exons2 = np.array([  junc2[:-1, 1], junc2[1:, 0]   ]).T

                        dist1 = jaccard_distance_exons_np(exons1, exons2)

                        distance_matrix[a, b] = dist1
                        #distance_matrix[b, a] = dist1

        return distance_matrix


    def evaluator(isoJunction_pred, isoCount_pred, isoJunction_true, isoCount_true, junctionPos_pred, junctionPos_true):


        

        distance_matrix = findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='j')
        #distance_matrix = findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='exact')
        

        if False:#distance_matrix.shape[0] <= 2:
            print (junctionPos_pred[ isoJunction_pred[0] > 0 ])
            print (junctionPos_pred[ isoJunction_pred[1] > 0 ])
            print (np.sum(isoCount_pred, axis=1) / np.sum(isoCount_pred))
            print (np.sum(isoCount_true, axis=1) / np.sum(np.sum(isoCount_true, axis=1)))
            print (distance_matrix)
            quit()

        if False:
            print ('dist')
            print (distance_matrix)
            print (isoJunction_pred)
            print (junctionPos_pred)
            #print (np.mean(isoCount_pred, axis=1))
            #print (np.mean(isoCount_true, axis=1))

        
        #print (distance_matrix.shape)
        

        #print (isoCount_true.shape, isoCount_pred.shape)

        #print (isoCount_true.shape)
        #quit()

        weights = np.sum(isoCount_true, axis=0)
        weights = weights[weights > 0]
        weights = weights / np.mean(weights)

        #print (isoJunction_pred[1])
        #print (isoJunction_true[:5])
        #print (distance_matrix[1])
        #quit()

        import ot
        scoreList_ours = []
        for sample_index in range(isoCount_pred.shape[1]):
            longCounts = isoCount_true[:, sample_index]
            predCounts = isoCount_pred[:, sample_index]

            #print ('longCounts', longCounts)

            #predCounts[:] = 1

            #print (np.sum(predCounts))

            if np.sum(predCounts) == 0:
                predCounts[:] = 1
                #Trigger mod!

            if np.sum(longCounts) > 0:
                longCounts = longCounts / np.sum(longCounts)
                predCounts = predCounts / np.sum(predCounts)

                #print ('a')
                #print (np.round(longCounts*100))
                #print (np.round(predCounts*100))

                #print (longCounts)

                
                
                T_pred = ot.emd(predCounts, longCounts, distance_matrix) #TODO: Look into if optimal transport is miminizing or maximizing
                #print (T_pred)
                score_pred = np.sum(T_pred * distance_matrix)

                

                #print (score_pred)
                #quit()
                scoreList_ours.append(score_pred)

                if sample_index == 40:
                    print ("Score Now", score_pred)
                    print (predCounts, longCounts, distance_matrix)
        scoreList_ours = np.array(scoreList_ours)

        #print (scoreList_ours)
        #print (np.argwhere( np.isnan(scoreList_ours) ))
        #print (scoreList_ours.shape)
        
        score = np.mean(scoreList_ours * weights)

        #print ('scoreList_ours', scoreList_ours)
        #quit()
        
        return score
    

    def findValidSamples(sampleList1, sampleList2, columnUse):
        inverse1 = np.concatenate(( sampleList1[:, columnUse], sampleList2[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_unique, inverse_index = np.unique(inverse1[:sampleList1.shape[0]], return_index=True)
        inverse_index = inverse_index[  np.isin(inverse_unique , inverse1[sampleList1.shape[0]:]  ) ]
        sampleList_new = sampleList1[inverse_index]
        return sampleList_new

    

    def processSamples(sampleList, sampleList_include, columnUse, countList):

        inverse1 = np.concatenate(( sampleList_include[:, columnUse], sampleList[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_include, inverse_samples = inverse1[:sampleList_include.shape[0]], inverse1[sampleList_include.shape[0]:]

        countList_new = np.zeros(( countList.shape[0],  sampleList_include.shape[0]  ))
        for a in range(sampleList_include.shape[0]):
            args1 = np.argwhere(inverse_samples == inverse_include[a])[:, 0]
            countList_new[:, a] = np.sum(countList[:, args1], axis=1)
        return countList_new
        
    


    def isoformPlotter(ours_plot, ours_pos, short_plot, short_pos, true_plot, true_pos):



        labels = []
        fullBinary = []
        for a in range(true_plot.shape[0]):
            labels.append(0)
            fullBinary.append(  np.copy(true_plot[a]))
        for a in range(ours_plot.shape[0]):
            labels.append(1)
            fullBinary.append(  np.copy(ours_plot[a]))
        for a in range(short_plot.shape[0]):
            labels.append(2)
            fullBinary.append(  np.copy(short_plot[a]))
        


        min1 = np.min(np.array([np.min(ours_pos), np.min(short_pos), np.min(true_pos)]))
        ours_pos, short_pos, true_pos = ours_pos - min1, short_pos - min1, true_pos - min1
        max1 = np.max(np.array([np.max(ours_pos), np.max(short_pos), np.max(true_pos)]))

        pasted = np.zeros((  len(fullBinary), max1+1 ), dtype=int)

        fullPos = []
        for a in range(true_plot.shape[0]):
            fullPos.append(  np.copy(true_pos))
        for a in range(ours_plot.shape[0]):
            fullPos.append(  np.copy(ours_pos))
        for a in range(short_plot.shape[0]):
            fullPos.append(  np.copy(short_pos))
        


        colorList = ['blue', 'red', 'orange']
        for a in range(len(fullBinary)):

            posLevel = a * -1

            color_index = labels[a]
            color = colorList[color_index]

            binaryNow = np.copy(fullBinary[a])
            posNow = np.copy(fullPos[a])
            posNow2 = posNow[binaryNow == 1]

            posNow2 = posNow2[np.argsort(posNow2[:, 0])]
            #posFlat = posNow2.reshape((-1,))
            #yPos = np.zeros(posFlat.shape[0]) + a


            for b in range(posNow2.shape[0]-1):
                plt.plot(  [posNow2[b][1], posNow2[b+1][0]] , [ posLevel,  posLevel ]   , c=color)
                
                assert posNow2[b][1] <= posNow2[b+1][0]

            for b in range(posNow2.shape[0]):
                arange1 = np.arange(101) / 100
                #arange1 = np.arange(2)
                start1, end1 = posNow2[b][0], posNow2[b][1]
                length1 = end1 - start1
                yPos = (arange1 * 2) - 1
                #yPos = posLevel + ((1 - (yPos ** 2)) * 0.3)
                yPos = posLevel + ((1 - (yPos ** 2)) * 0.6)


                assert start1 <= end1


                plt.plot(  start1 + (arange1 * length1) , yPos   , c=color, linestyle=':')
                #plt.plot(  [start1, end1], [a, a])

                #print (arange1, length1)
                #print ( start1 + (arange1 * length1)  ,  (arange1*0) + a)
                #plt.plot(   start1 + (arange1 * length1)  ,  (arange1*0) + a  )
                plt.scatter(  [start1, end1], [posLevel, posLevel] , c=color, edgecolors='black', s=10) #, marker="^"
            #for b in range(len(posNow2) - 1):
            #    pasted[a, posNow2[b, 1]:posNow[b+1, 0]] = 1

            #if a == 0:
            #    print ("B")
            #    print (binaryNow)
            #    print (posNow)
            #    plt.plot(pasted[0])
            #    plt.show()


        #sns.heatmap(pasted)
        plt.show()

        #quit()


    #-134.5021209716797

    import seaborn as sns

    #geneNow = 'ENSG00000001461'

    files = os.listdir('./data/real/splicing/geneFiles/geneModels')
    geneList = []
    for file1 in files:
        if '.pt' in file1:
            #if '_exon_leak10' in file1:
            if '_sampleGen_size500_leak2' in file1:
                #if  '_sample' in file1:
                geneList.append(file1.split('_')[0])
    geneList = np.unique(np.array(geneList))

    allScores_our = []
    allScore_short = []

    #print ("T")

    gene_unique = loadnpz('./data/temp/gene_unique_perm.npz')#[:30]  #[:6]

    useExonModel = False

    scoreList = np.zeros(( len(gene_unique), 2 ))
    usedList = np.zeros( len(gene_unique) )
    meanJunctions = np.zeros( len(gene_unique) )
    meanJunctions_long = np.zeros(len(gene_unique))



    with torch.no_grad():
    

        for gene_index in range(0, len(gene_unique)):

            geneNow = gene_unique[gene_index]

            

            #genesCheck0 = ['PRELID1', 'ARSB', 'DUSP13', 'PTBP1', 'SLC1A5', 'PPA2', 'NDUFS4', 'CCDC69', 'ACSSL3']


            #
            #genesCheck = ['ENSG00000169230', 'ENSG00000113273', 'ENSG00000079393', 'ENSG00000011304', 'ENSG00000105281', 'ENSG00000138777', 'ENSG00000164258', 'ENSG00000198624', 'ENSG00000123983']
            #SRP14
            #genesCheck = ['ENSG00000140319', 'ENSG00000135218', 'ENSG00000185201', 'ENSG00000170291']
            #Ensembl:ENSG00000170291 MIM:615019; AllianceGenome:HGNC:30617
            #if geneNow in geneList:
            if geneNow in ['ENSG00000134046']:
                print (gene_index, geneNow)
                #model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_size500_leak2.pt'
                model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_fast.pt'
                model = torch.load(model_filename, weights_only=False)
                


                junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
                countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
                junctionPos = junctionNow[:, 1:3].astype(int)


                Njunction = junctionNow.shape[0]
                isoformJunctions = loadnpz('./data/real/splicing/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos = loadnpz('./data/real/splicing/geneFiles/isoformJunctionsPos/' + str(geneNow) + '.npz')
                isoformCounts = loadnpz('./data/real/splicing/geneFiles/isoform_counts/' + str(geneNow) + '.npz')


                isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctionPos/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')
                

                validIsoform = np.sum(isoformJunctions_long, axis=1)
                isoformJunctions_long = isoformJunctions_long[validIsoform >= 1]
                isoformCounts_long = isoformCounts_long[validIsoform >= 1]

                sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                sampleLong = loadnpz('./data/real/splicing/input/samples_longRead.npz')
                inv1 = uniqueValMaker(sampleList[:, :3])
                inv2 = uniqueValMaker(sampleLong[:, :3])                    
                if not useExonModel:
                    inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
                    inverse1 = uniqueValMaker(inverse1)
                    argGood = np.argwhere(np.isin(  inverse1[sampleLong.shape[0]:], inverse1[:sampleLong.shape[0]]   ))[:, 0]
                    sampleList = sampleList[argGood]
                    countNow = countNow[:, argGood]
                    isoformCounts = isoformCounts[:, argGood]



                if np.sum(isoformCounts_long) >= 1:


                    print ('isoformCounts_long total', np.sum(isoformCounts_long))
                    print ('Junction total', np.sum(countNow))

                    

                    


                    Nsample = countNow.shape[1]
                    Njunction = countNow.shape[0]

                    observations_batch =  torch.tensor(countNow).float()

                    edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                    edgeMatrix[edges[:, 0], edges[:, 1]] = 1

                    edgeSum = np.sum(edgeMatrix, axis=1)

                    goodCount = False
                    
                    if np.sum( isoformCounts_long) > 1:
                        graphSize = Njunction
                        finalProbSize = countNow.shape[1] 
                        batchSize = Njunction
                        ruleObject = SpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                        ruleObject.graphSize = graphSize
                        ruleObject.observations_batch = observations_batch
                        ruleObject.model = model

                        batchSize = 2000
                        offPolicy = False
                        model = torch.load(model_filename)
                        adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batchSize)
                        

                        _, finalProbAllow = ruleObject.graphRules(adjacency_matrices)
                        finalProb = model.finalProb(adjacency_matrices)
                        finalProb = finalProb + finalProbAllow
                        finalProb = nn.LogSoftmax(dim=1)(finalProb)

                        adjacency_matrices, finalProb_exp_sum = processPredictionCounts(adjacency_matrices, finalProb, normalizeJunc=True)

                        columnUse = np.arange(3)
                        sampleList_include = findValidSamples(sampleList, sampleLong, columnUse)

                        

                        countNow = processSamples(sampleList, sampleList_include, columnUse, countNow)
                        finalProb_exp_sum = processSamples(sampleList, sampleList_include, columnUse, finalProb_exp_sum)
                        isoformCounts = processSamples(sampleList, sampleList_include, columnUse, isoformCounts)
                        isoformCounts_long = processSamples(sampleLong, sampleList_include, columnUse, isoformCounts_long)


                        if np.sum(isoformCounts_long) >= 1:
                            impliedJunc = np.matmul(isoformJunctions_long.T, isoformCounts_long)
                            predJunc = np.matmul(adjacency_matrices.T, finalProb_exp_sum)
                            predCount_sum = np.sum(finalProb_exp_sum, axis=1)
                            trueCount_sum = np.sum(isoformCounts_long, axis=1)
                            shortCount_sum = np.sum(isoformCounts, axis=1)
                            predCount_sum = predCount_sum / np.sum(predCount_sum)
                            trueCount_sum = trueCount_sum / np.sum(trueCount_sum)
                            shortCount_sum = shortCount_sum / np.sum(shortCount_sum)


                            doPlot = True 
                            predCount_sort = np.sort(predCount_sum)[-1::-1] 
                            if np.sum(predCount_sort[5:]) > 0.1: #3:
                                doPlot = False 
                            trueCount_sort = np.sort(trueCount_sum)[-1::-1] 
                            if np.sum(trueCount_sort[5:]) > 0.1:
                                doPlot = False 
                            shortCount_sort = np.sort(shortCount_sum)[-1::-1] 
                            if np.sum(shortCount_sort[5:]) > 0.1:
                                doPlot = False 

                            doPlot = True

                            if doPlot:
                                predCount_print = predCount_sum[predCount_sum > np.sum(predCount_sum) / 10]
                                adj_print  = adjacency_matrices[predCount_sum > np.sum(predCount_sum) / 10].astype(int)
                                trueCount_print  = trueCount_sum[trueCount_sum> np.sum(trueCount_sum) / 10].astype(int)
                                iso_print = isoformJunctions_long[trueCount_sum > np.sum(trueCount_sum) / 10]
                                isoformJunctionPos_long_print = np.concatenate((  np.arange(isoformJunctionPos_long.shape[0]).reshape((-1, 1)), isoformJunctionPos_long   ), axis=1)    

                                if False:
                                    print (isoformJunctionPos_long_print)
                                    for a in range(predCount_print.shape[0]):
                                        print (adj_print[a], int(predCount_print[a]))
                                    print ('')
                                    for a in range(iso_print.shape[0]):
                                        print (iso_print[a], int(trueCount_print[a]))

                                score_ours = evaluator(adjacency_matrices, np.copy(finalProb_exp_sum), isoformJunctions_long, isoformCounts_long, junctionPos, isoformJunctionPos_long)
                                score_short = evaluator(isoformJunctions, np.copy(isoformCounts), isoformJunctions_long, isoformCounts_long, isoformJunctionPos, isoformJunctionPos_long)
                                assert not np.isnan(score_ours)
                                assert not np.isnan(score_short)

                                allScores_our.append(score_ours)
                                allScore_short.append(score_short)

                                #print (geneNow)
                                print (score_ours, score_short)

                                scoreList[gene_index, 0] = score_ours
                                scoreList[gene_index, 1] = score_short
                                usedList[gene_index] = 1


                                if (score_ours + 0.1 < score_short):# and (score_ours < 0.05):

                                    numInclude = 4
                                    ours_plot = adjacency_matrices[np.argsort(predCount_sum*-1)[:numInclude]]
                                    short_plot = isoformJunctions[np.argsort(shortCount_sum*-1)[:numInclude]]
                                    true_plot = isoformJunctions_long[np.argsort(trueCount_sum*-1)[:numInclude]]


                                    
                                    allPredCount_sort = finalProb_exp_sum[np.argsort(predCount_sum*-1)]
                                    allShortCount_sort = isoformCounts[np.argsort(shortCount_sum*-1)]  
                                    allTrueCount_sort = isoformCounts_long[np.argsort(trueCount_sum*-1)]

                                    #print (allTrueCount_sort.shape)

                                    print ('all')

                                    #argMostLong = np.argmax(np.sum(allPredCount_sort, axis=0))
                                    argMostLong = np.argmax(np.sum(allTrueCount_sort, axis=0))

                                    print ('argMostLong', argMostLong)


                                    print (sampleList_include.shape)
                                    #print (sampleList_include[argMostLong])
                                    print (allTrueCount_sort[:, argMostLong])
                                    print (allTrueCount_sort[:, argMostLong] / np.sum(allTrueCount_sort[:, argMostLong]))
                                    print (allPredCount_sort[:, argMostLong] / np.sum(allPredCount_sort[:, argMostLong]))
                                    print (allShortCount_sort[:, argMostLong] / np.sum(allShortCount_sort[:, argMostLong]))

                                    #ours_plot = ours_plot[predCount_sort[:3] > 0.05]
                                    #short_plot = short_plot[shortCount_sort[:3] > 0.05]
                                    #true_plot = true_plot[trueCount_sort[:3] > 0.05]
                                    ours_plot = ours_plot[predCount_sort[:numInclude] > 0.01]
                                    short_plot = short_plot[shortCount_sort[:numInclude] > 0.01]
                                    true_plot = true_plot[trueCount_sort[:numInclude] > 0.01]


                                    




                                    print ("A")
                                    print (trueCount_sort[:true_plot.shape[0]])
                                    print (predCount_sort[:ours_plot.shape[0]])
                                    print (shortCount_sort[:short_plot.shape[0]])
                                    #print (true_plot[0])
                                    #print (isoformJunctionPos_long)

                                    #print (np.sort(trueCount_sum)[-1::-1])
                                    #print (np.sort(predCount_sum)[-1::-1])
                                    #print (np.sort(shortCount_sum)[-1::-1])

                                    isoformPlotter(ours_plot, junctionPos, short_plot, isoformJunctionPos, true_plot, isoformJunctionPos_long)
                                    #quit()

                                    for spacer in range(10):
                                        print ('')







                            meanJunctions[gene_index] = np.mean( np.sum(predJunc, axis=1)  / np.sum(finalProb_exp_sum)  )
                            meanJunctions_long[gene_index] = np.mean(   np.sum(impliedJunc, axis=1) / np.sum(isoformCounts_long)   )
    
    allScores_our, allScore_short = np.array(allScores_our), np.array(allScore_short)

    
    print ('ours', np.mean(allScores_our) )
    print ('short', np.mean(allScore_short) )

    plt.scatter(  allScore_short, allScores_our ) 
    plt.plot([0, 1], [0, 1], c='black')
    plt.xlabel('baseline error')
    plt.ylabel('our error')
    plt.show()

    #14 ENSG00000102755
    #146 ENSG00000163349
    #293 ENSG00000173372
    #615 ENSG00000110195 #looks great!
    #2221 ENSG00000105447
    #2537 ENSG00000028839 #Seems very good
    #2567 ENSG00000221986 #3 isoforms each
    #2781 ENSG00000135185
    #2865 ENSG00000134046 #Seems great!
    #3607 ENSG00000138660
    

searchPlotExon()
quit()


def analyzeScoresSplice():


    #-134.5021209716797

    import seaborn as sns

    #geneNow = 'ENSG00000001461'

    files = os.listdir('./data/real/splicing/geneFiles/geneModels')
    geneList = []
    for file1 in files:
        if '.pt' in file1:
            #if '_exon_leak10' in file1:
            if '_sampleGen_size500_leak2' in file1:
                #if  '_sample' in file1:
                geneList.append(file1.split('_')[0])
    geneList = np.unique(np.array(geneList))

    allScores_our = []
    allScore_short = []

    #print ("T")

    gene_unique = loadnpz('./data/temp/gene_unique_perm.npz')#[:30]  #[:6]

    useExonModel = False


    scoreList = loadnpz('./data/temp/scoreList.npz')

    print (gene_unique.shape)
    print (scoreList.shape)
    argLast = np.max(np.argwhere( scoreList!=0 ))
    print (argLast)
    quit()
    scoreList = scoreList[:argLast+1]

    scoreList = scoreList[scoreList[:, 0] != 0]

    improvement = scoreList[:, 0] - scoreList[:, 1]

    #plt.boxplot(improvement)
    plt.hist(improvement, bins=10, range=(-0.8, 0.8))
    plt.axvline(x=0, c='black')
    plt.show()

    plt.scatter(scoreList[:, 0], scoreList[:, 1])
    plt.show()
    #quit()

    print (scoreList.shape)
    print (np.argwhere( scoreList[:, 0] == 0 ).shape)
    quit()


    meanJunctions = loadnpz('./data/temp/meanJunctions.npz')[:argLast+1]
    meanJunctions_long = loadnpz('./data/temp/meanJunctions_long.npz')[:argLast+1]


    

    totalCounts = np.zeros(scoreList.shape[0])
    totalCounts_long = np.zeros(scoreList.shape[0])
    numIsoforms = np.zeros(scoreList.shape[0])
    numJunctions = np.zeros(scoreList.shape[0])
    numEdges = np.zeros(scoreList.shape[0])




    



    with torch.no_grad():
    

        for gene_index in range(len(gene_unique)):
            geneNow = gene_unique[gene_index]
            if geneNow in geneList:

                print (gene_index, geneNow)
                model_filename = './data/real/splicing/geneFiles/geneModels/' + geneNow + '_sampleGen_size500_leak2.pt'
                model = torch.load(model_filename, weights_only=False)
                


                junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
                countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
                junctionPos = junctionNow[:, 1:3].astype(int)

                numEdges[gene_index] = edges.shape[0]

                
                Njunction = junctionNow.shape[0]
                isoformJunctions = loadnpz('./data/real/splicing/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos = loadnpz('./data/real/splicing/geneFiles/isoformJunctionsPos/' + str(geneNow) + '.npz')
                isoformCounts = loadnpz('./data/real/splicing/geneFiles/isoform_counts/' + str(geneNow) + '.npz')

                numJunctions[gene_index] = countNow.shape[0]
                numIsoforms[gene_index] = isoformCounts.shape[0]

                isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformJunctionPos_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctionPos/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')
                

                validIsoform = np.sum(isoformJunctions_long, axis=1)
                isoformJunctions_long = isoformJunctions_long[validIsoform >= 1]
                isoformCounts_long = isoformCounts_long[validIsoform >= 1]

                if np.sum(isoformCounts_long) >= 1:
                    #print ('isoformCounts_long total', np.sum(isoformCounts_long))
                    #print ('Junction total', np.sum(countNow))
                    sampleList = loadnpz('./data/real/splicing/input/samples_isoforms.npz')
                    sampleLong = loadnpz('./data/real/splicing/input/samples_longRead.npz')
                    
                    
                    if not useExonModel:
                        inverse1 = np.concatenate(( sampleLong[:, 0:3], sampleList[:, 0:3] ), axis=0)
                        inverse1 = uniqueValMaker(inverse1)
                        argGood = np.argwhere(np.isin(  inverse1[sampleLong.shape[0]:], inverse1[:sampleLong.shape[0]]   ))[:, 0]
                        sampleList = sampleList[argGood]
                        countNow = countNow[:, argGood]
                        isoformCounts = isoformCounts[:, argGood]

                    Nsample = countNow.shape[1]
                    Njunction = countNow.shape[0]

                    observations_batch =  torch.tensor(countNow).float()

                    edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                    edgeMatrix[edges[:, 0], edges[:, 1]] = 1

                    totalCounts[gene_index] = np.sum(countNow)
                    totalCounts_long[gene_index] = np.sum(isoformCounts_long)

                    
    print (improvement.shape)
    print (totalCounts.shape)
    print (totalCounts_long.shape)


    plt.scatter(meanJunctions * numJunctions, improvement)
    plt.show()

    plt.scatter(meanJunctions_long * numJunctions, improvement)
    plt.show()

    plt.scatter(meanJunctions / meanJunctions_long, improvement)
    plt.show()

    #plt.scatter( numJunctions, improvement )
    #plt.show()

    #plt.scatter( numIsoforms, improvement )
    #plt.show()
    quit()

    plt.scatter( totalCounts, improvement )
    plt.xscale('log')
    plt.show()

    plt.scatter( totalCounts_long, improvement )
    plt.xscale('log')
    plt.show()
    


#analyzeScoresSplice()
#quit()


def countLongReadGenes():

    
    gene_unique = loadnpz('./data/temp/gene_unique_perm.npz')#[:30]  #[:6]

    geneCount = 0
    with torch.no_grad():
        for gene_index in range(len(gene_unique)):
            geneNow = gene_unique[gene_index]    
            isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')
            if np.sum(isoformCounts_long) >= 1:
                geneCount += 1

    print (geneCount)

#countLongReadGenes()
#quit()

def analyzeExon():

    geneNow = 'ENSG00000023839'
    countData = loadnpz('./data/real/splicing/geneFiles/exonCounts_longSample/exonCounts_' + str(geneNow) + '.npz')
    positionInfo = loadnpz('./data/real/splicing/geneFiles/exonInfo_longSample/exonInfo_' + str(geneNow) + '.npz')
    positionInfo = positionInfo.astype(int)
    

    junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
    countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
    edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
    junctionPos = junctionNow[:, 1:3].astype(int)

    #First, junction pair extractor function
    #then Matrix mult converts from junction pair exons to measurement exons. 


    print (np.sum(countData))
    print (np.sum(countNow))
    quit()

    print (edges)

    print (junctionPos)
    print (positionInfo)
    


    print (countData.shape)
    #print (countData)
    quit()

    geneNow = 'ENSG00000103226'
    outputFile = './data/real/splicing/geneFiles/rawExon/' + geneNow + '.tsv'
    exonData = np.loadtxt(outputFile, dtype=str, delimiter='\t')

    exonStartEnd = exonData[1:, 3:5].astype(int)

    #plt.scatter( exonStartEnd[:, 0], exonStartEnd[:, 1] )
    #plt.show()
    #plt.plot(exonStartEnd[:, 0])
    #plt.show()
    intervalLengths = (exonStartEnd[:, 1] - exonStartEnd[:, 0]) + 100
    
    sampleData = exonData[1:, 12]
    totalCounts = np.zeros( exonStartEnd.shape[0], dtype=float )

    for a in range(sampleData.shape[0]):
        sampleData_mini = sampleData[a]
        sampleData_mini = sampleData_mini[1:]
        sampleData_mini = sampleData_mini.replace(',', ':')
        sampleData_mini = sampleData_mini.split(':')
        sampleData_mini = np.array(sampleData_mini)
        sampleData_mini = sampleData_mini.reshape(( sampleData_mini.shape[0] // 2, 2 ))
        countTotal = np.sum( sampleData_mini[:, 1].astype(int) )
        totalCounts[a] = countTotal

    exonCounts_adj = totalCounts / intervalLengths.astype(float)
    #print (exonCounts_adj)
    #quit()
    exonCounts_adj = exonCounts_adj / np.max(exonCounts_adj)
    

    #plt.plot(totalCounts / intervalLengths)
    #plt.show()
    #quit()

    adjacency_matrices = loadnpz('./data/temp/adjacency_matrices.npz').astype(float)
    finalProb_exp_sum = loadnpz('./data/temp/finalProb_exp_sum.npz').astype(float)
    junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
    #finalProb_exp_sum = np.sum(finalProb_exp_sum, axis=1)
    junctionPred = np.matmul(  adjacency_matrices.T, finalProb_exp_sum  )
    junctionPred = np.sum(junctionPred, axis=1)
    junctionPred = junctionPred/np.sum(finalProb_exp_sum)
    
    #plt.plot(finalProb_exp_sum)
    #plt.show() 

    #ENSG00000103226
    #quit()
    


    isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
    isoformJunctionPos_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctionPos/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
    isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')

    junctionLong = np.matmul(isoformJunctions_long.T, isoformCounts_long)
    junctionLong = np.sum(junctionLong, axis=1)

    junctionLong = junctionLong/np.sum(isoformCounts_long)

    #plt.plot(junctionLong)
    plt.plot(junctionNow[:, 1].astype(int), junctionPred)
    plt.plot(exonStartEnd[:, 0], exonCounts_adj)
    plt.show()
    #quit()

    #longCountSum = np.sum(isoformCounts_long, axis=1)

    argTop = np.argwhere( longCountSum / np.sum(longCountSum) > 0.2  )[:, 0]

    print (argTop)
    print (isoformJunctions_long[argTop])
    quit()

    print (longCountSum)
    plt.plot(longCountSum)
    plt.show()
    quit()

    



#analyzeExon()
#quit()




def cancerSplice():

    #ENSG00000000457

    import seaborn as sns

    
    #gene_unique = loadnpz('./data/real/splicing/cancer/input/geneIntersect.npz')
    gene_unique0 = os.listdir('./data/real/splicing/cancer/geneFiles/shortJunctionPos/')
    gene_unique = []
    for a in range(len(gene_unique0)):
        if '.npz' in gene_unique0[a]:
            gene_unique.append( gene_unique0[a].split('.')[0] )
    gene_unique = np.array(gene_unique)

    #print (np.argwhere(gene_unique == 'ENSG00000003509'))
    #quit()

    


    for gene_index in range(0, gene_unique.shape[0]):

        geneNow = gene_unique[gene_index]

        if geneNow == 'ENSG00000249464':
            
            learning_rate = 1e-3
            #learning_rate = 1e-4
            Nhidden = 50
            

            model_filename = './data/real/splicing/cancer/geneFiles/geneModels/' + geneNow + '_2.pt'


            junctionNow = loadnpz('./data/real/splicing/cancer/geneFiles/shortJunctionPos/' + geneNow + '.npz')
            #junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
            #isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')

            #print (junctionNow[:4])
            #quit()

            strandDirection = junctionNow[0, 3]

            #print (strandDirection)

            if True:#(strandDirection == '+'):

                print ('geneNow', geneNow)


                #countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                #edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')

                edges = loadnpz('./data/real/splicing/cancer/geneFiles/shortEdges/' + geneNow + '.npz')
                countNow = loadnpz('./data/real/splicing/cancer/geneFiles/shortCounts/' + geneNow + '.npz').T

                #print (countNow.shape)

                #quit()


                isoJunction_long = loadnpz('./data/real/splicing/cancer/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')
                #countNow = np.zeros(countNow.shape, dtype=int)
                #countNow = countNow + (isoJunction_long[:1].T * np.arange(countNow.shape[1]).reshape((1, -1)))
                #countNow = countNow + (isoJunction_long[-1:].T * (countNow.shape[1] - 1 - np.arange(countNow.shape[1]).reshape((1, -1)) ))
                
                #countNow[:, :1] = countNow[:, :1] + isoJunction_long[:1].T * 20
                #countNow[:, 1:] = countNow[:, 1:] + isoJunction_long[-1:].T * 20
                #sns.heatmap(countNow)
                #plt.show()
                #quit()

                

                countNow = torch.tensor(countNow).float()


                Nsample = countNow.shape[1]
                Njunction = countNow.shape[0]

                observations_batch = countNow



                edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                edgeMatrix[edges[:, 0], edges[:, 1]] = 1
                edgeSum = np.sum(edgeMatrix, axis=1)


                if True:#np.argwhere(edgeSum >= 2).shape[0] > np.argwhere(edgeSum == 1).shape[0]:
                    #if True:


                    startPos = junctionNow[:, 1].astype(int)
                    assert np.min(startPos[1:] - startPos[:-1]) >= 0

                    edgeMatrix[-1, np.sum(edgeMatrix, axis=0) == 0 ] = 1
                    edgeMatrix[np.sum(edgeMatrix, axis=1) == 0, -1 ] = 1

                    goodMatrix = True 
                    if (0 in np.sum(edgeMatrix, axis=0)) or (0 in np.sum(edgeMatrix, axis=1)):
                        goodMatrix = False 

                        #plt.imshow(edgeMatrix)
                        #plt.show()

                        argBad1 = np.argwhere(   np.sum(edgeMatrix[:, :], axis=0) == 0 )[:, 0]
                        argBad2 = np.argwhere(   np.sum(edgeMatrix[:, :], axis=1) == 0 )[:, 0]

                        plt.imshow(edgeMatrix)
                        plt.show()

                        print ('bad', argBad1, argBad2)

                        sum = np.sum(countNow.data.numpy(),axis=1 )

                        print (sum)

                        print (sum[argBad1])
                        print (sum[argBad2])
                        
                        quit()

                    print ('goodMatrix', goodMatrix)

                    #plt.imshow(edgeMatrix)
                    #plt.show()

                    if goodMatrix:
                        
                        graphSize = Njunction
                        finalProbSize = Nsample 
                        #batchSize = Njunction
                        
                        ruleObject = SpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                        ruleObject.graphSize = graphSize
                        ruleObject.observations_batch = observations_batch

                        #batchSize = 200
                        #batchSize = 1000
                        #batchSize = 3000
                        batchSize = 5000
                        #batchSize = 10000



                        offPolicy = False
                        dupGen = 1

                        train_model_off_policy(Nhidden, ruleObject, learning_rate, observations_batch,  batchSize, dupGen, offPolicy, num_epochs=5000, model_filename=model_filename, rewardType='')



#cancerSplice()
#quit()



def analyzeCancerResult():

    import numpy as np

    def compute_overlap_np(e1, e2):
        """Compute overlap in base pairs between two exons given as arrays."""
        start = np.maximum(e1[0], e2[0])
        end = np.minimum(e1[1], e2[1])
        return max(0, end - start)

    def jaccard_distance_exons_np(exons1, exons2):
        """
        Compute Jaccard distance between two exon sets.
        exons1 and exons2 are NumPy arrays of shape (n, 2) with [start, end] positions.
        """
        # Sort by start position
        exons1 = exons1[np.argsort(exons1[:, 0])]
        exons2 = exons2[np.argsort(exons2[:, 0])]

        #print ('exons2', exons2)

        total_bp1 = np.sum(exons1[:, 1] - exons1[:, 0])
        total_bp2 = np.sum(exons2[:, 1] - exons2[:, 0])

        

        # Two-pointer overlap computation
        i = j = 0
        intersection_bp = 0

        while i < len(exons1) and j < len(exons2):
            e1 = exons1[i]
            e2 = exons2[j]

            # Add overlapping base pairs
            overlap = compute_overlap_np(e1, e2)
            intersection_bp += overlap

            # Advance the interval with the smaller end
            if e1[1] < e2[1]:
                i += 1
            else:
                j += 1

        union_bp = total_bp1 + total_bp2 - intersection_bp

        #print (total_bp1 , total_bp2 , intersection_bp)

        score = 1 - (intersection_bp / union_bp) if union_bp > 0 else 0.0

        #if score == 1:
        #    print ("A")
        #    print (exons1)
        #    print (exons2)

        return score 


    def findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='j'):

        distance_matrix = np.zeros((isoJunction_pred.shape[0], isoJunction_true.shape[0]))

        if distType in ['junction']:
            for a in range(distance_matrix.shape[0]):
                for b in range(distance_matrix.shape[1]):
                    sum1 = combineIsoforms_unique[a] + combineIsoforms_unique[b]
                    sum1[sum1 > 1] = 1
                    intersect1 = combineIsoforms_unique[a] * combineIsoforms_unique[b]
                    if np.sum(sum1) == 0:
                        dist1 = 0.0
                    else:
                        dist1 = np.sum(intersect1) / np.sum(sum1)
                        dist1 = 1.0 - dist1
                    distance_matrix[a, b] = dist1
        
        if distType in ['exact']:
            for a in range(distance_matrix.shape[0]):
                for b in range(distance_matrix.shape[1]):
                    if True:#b != a:
                        junc1 = junctionPos_pred[isoJunction_pred[a] > 0]
                        junc2 = junctionPos_true[isoJunction_true[b] > 0]
                        junc1 = junc1[np.argsort(junc1[:, 0])]
                        junc2 = junc2[np.argsort(junc2[:, 0])]
                        if np.array_equal(junc1, junc2):
                            distance_matrix[a, b] = 0
                        else:
                            distance_matrix[a, b] = 1



        
        if distType in ['j', 'jaccard']:
            for a in range(distance_matrix.shape[0]):
                for b in range(distance_matrix.shape[1]):
                    if True:#b != a:
                        junc1 = junctionPos_pred[isoJunction_pred[a] > 0]
                        junc2 = junctionPos_true[isoJunction_true[b] > 0]

                        junc1 = junc1[np.argsort(junc1[:, 0])]
                        junc2 = junc2[np.argsort(junc2[:, 0])]

                        exons1 = np.array([  junc1[:-1, 1], junc1[1:, 0]   ]).T
                        exons2 = np.array([  junc2[:-1, 1], junc2[1:, 0]   ]).T

                        dist1 = jaccard_distance_exons_np(exons1, exons2)

                        distance_matrix[a, b] = dist1
                        #distance_matrix[b, a] = dist1

        return distance_matrix


    def evaluator(isoJunction_pred, isoCount_pred, isoJunction_true, isoCount_true, junctionPos_pred, junctionPos_true):


        

        distance_matrix = findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='j')
        #distance_matrix = findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='exact')
        


        weights = np.sum(isoCount_true, axis=0)
        weights = weights[weights > 0]
        weights = weights / np.mean(weights)

        import ot
        scoreList_ours = []
        for sample_index in range(isoCount_pred.shape[1]):
            longCounts = isoCount_true[:, sample_index]
            predCounts = isoCount_pred[:, sample_index]

            #print ('longCounts', longCounts)

            if np.sum(predCounts) == 0:
                predCounts[:] = 1

            if np.sum(longCounts) > 0:
                longCounts = longCounts / np.sum(longCounts)
                predCounts = predCounts / np.sum(predCounts)
                
                T_pred = ot.emd(predCounts, longCounts, distance_matrix) #TODO: Look into if optimal transport is miminizing or maximizing
                #print (T_pred)
                score_pred = np.sum(T_pred * distance_matrix)
                scoreList_ours.append(score_pred)
        scoreList_ours = np.array(scoreList_ours)


        

        
        score = np.mean(scoreList_ours * weights)

        #print ('scoreList_ours', scoreList_ours)
        
        return score
    

    def existEvaluator(isoJunction_pred, isoCount_pred, isoJunction_true, isoCount_true, junctionPos_pred, junctionPos_true):


        distance_matrix = findDistMatrix(isoJunction_pred, isoJunction_true, junctionPos_pred, junctionPos_true, distType='j')
        

        #plt.imshow(distance_matrix)
        #plt.show()
        
        scoreList_ours = []
        for sample_index in range(isoCount_pred.shape[1]):
            longCounts = isoCount_true[:, sample_index]
            predCounts = isoCount_pred[:, sample_index]

            if np.sum(predCounts) == 0:
                predCounts[:] = 1
            if np.sum(longCounts) > 0:
                #longCounts = longCounts / np.sum(longCounts)
                predCounts = predCounts / np.sum(predCounts)

                argLong = np.argwhere(longCounts > 0)[:, 0]

                argPred = np.argsort( predCounts * -1 )[:20]


                print (isoJunction_pred[argPred[4]].astype(int))
                print (isoJunction_true[argLong])
                #quit()

                closest = distance_matrix[argPred][:, argLong]
                #closest = np.min(closest, axis=0)

                print (closest)
                quit()

                closest_mod = np.ones(20)
                for size_include in range(argPred.shape[0]):
                    closest_mod[size_include] = np.mean( np.min(closest[:size_include+1], axis=0) )
                
                closest_mod[argPred.shape[0]:] = closest_mod[argPred.shape[0]-1]

                #print (closest_mod)


                scoreList_ours.append( np.copy(closest_mod) )
        



                if False:

                    longProps = np.zeros(argLong.shape[0])
                    for a in range(argLong.shape[0]):
                        if 0 in distance_matrix[:, argLong[a]]:
                            arg1 = np.argmin(distance_matrix[:, argLong[a]])
                            longProps[a] = predCounts[arg1]

                    #print (longProps)

                    scoreList_ours.append(np.median(longProps))


       #quit()

        scoreList_ours = np.array(scoreList_ours)
        scoreList_ours = np.mean(scoreList_ours, axis=0)
        #print (scoreList_ours)
        #score = np.mean(scoreList_ours * scoreList_ours)

        
        return scoreList_ours


    

    def findValidSamples(sampleList1, sampleList2, columnUse):
        inverse1 = np.concatenate(( sampleList1[:, columnUse], sampleList2[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_unique, inverse_index = np.unique(inverse1[:sampleList1.shape[0]], return_index=True)
        inverse_index = inverse_index[  np.isin(inverse_unique , inverse1[sampleList1.shape[0]:]  ) ]
        sampleList_new = sampleList1[inverse_index]
        return sampleList_new

    

    def processSamples(sampleList, sampleList_include, columnUse, countList):

        inverse1 = np.concatenate(( sampleList_include[:, columnUse], sampleList[:, columnUse] ), axis=0)
        inverse1 = uniqueValMaker(inverse1)
        inverse_include, inverse_samples = inverse1[:sampleList_include.shape[0]], inverse1[sampleList_include.shape[0]:]

        countList_new = np.zeros(( countList.shape[0],  sampleList_include.shape[0]  ))
        for a in range(sampleList_include.shape[0]):
            args1 = np.argwhere(inverse_samples == inverse_include[a])[:, 0]
            countList_new[:, a] = np.sum(countList[:, args1], axis=1)
        return countList_new
        
    

    def getPValues(finalProb_exp_sum, tissueInfo):

        #print (np.sum(finalProb_exp_sum))

        finalProb_exp_sum = finalProb_exp_sum[np.sum(finalProb_exp_sum, axis=1) > 0]

        finalProb_exp_sum = finalProb_exp_sum.astype(float)
        finalProb_exp_sum = finalProb_exp_sum / (np.sum(finalProb_exp_sum, axis=0).reshape((1, -1)) + 1e-8)

        #print (finalProb_exp_sum.shape)



        

        #print (finalProb_exp_sum.shape)
        #quit()

        pValLists = []

        for phen_index in range(tissueInfo.shape[1]):
            Y = tissueInfo[:, phen_index]
            unique_groups = np.unique(Y)

            #print (unique_groups)

            pValuesPaste = np.zeros((  unique_groups.shape[0],  finalProb_exp_sum.shape[0] , 2))

            #print (pValuesPaste.shape)
            #quit()

            for group_index in range(unique_groups.shape[0]):
                group_now = unique_groups[group_index]

                #print (group_now)

                for isoform_index in range(finalProb_exp_sum.shape[0]):
                    X = finalProb_exp_sum[isoform_index]
                    t_statistic, p_value = scipy.stats.ttest_ind(X[Y==group_now], X[Y!=group_now])

                    pValuesPaste[group_index, isoform_index, 0] = p_value
                    pValuesPaste[group_index, isoform_index, 1] = t_statistic

                    #print (t_statistic, p_value)
                    #plt.hist(X[Y==group_now], bins=100, density=True, alpha=0.5)
                    #plt.hist(X[Y!=group_now], bins=100, density=True, alpha=0.5)
                    #plt.show()

                    
                pBest = np.min(pValuesPaste[group_index, :, 0]) * finalProb_exp_sum.shape[0]
                pBest = pBest * unique_groups.shape[0]

                #print (pBest)

            pValLists.append( np.copy(pValuesPaste) )
        
        return pValLists

    #-134.5021209716797

    import seaborn as sns

    #geneNow = 'ENSG00000001461'

    gene_unique0 = os.listdir('./data/real/splicing/cancer/geneFiles/shortJunctionPos/')
    gene_unique = []
    for a in range(len(gene_unique0)):
        if '.npz' in gene_unique0[a]:
            gene_unique.append( gene_unique0[a].split('.')[0] )
    geneList = np.array(gene_unique)


    
    seqList_short = loadnpz('./data/real/splicing/cancer/info/short_seqList.npz')[:, 0]
    seqList_long = loadnpz('./data/real/splicing/cancer/info/long_seqList.npz')[:, 0]
    
    metaTableLong = np.loadtxt('./data/real/splicing/raw/SraRunTable_mod.csv', dtype=str, delimiter=',', skiprows=1)
    metaTableShort = np.loadtxt('./data/real/splicing/raw/SraRunTable.csv', dtype=str, delimiter=',', skiprows=1)
    metaTableShort = metaTableShort[np.isin(metaTableShort[:, 0], seqList_short )]
    cancer_long = metaTableLong[:, -1]
    cancer_long[cancer_long==''] = 'Normal' #There was one blank which is actually normal sample. Can see from "NAT" vs "CRC" column
    identifiers_long = metaTableLong[:, 23]
    identifiers_short = metaTableShort[:, 24]
    cancer_short_other = metaTableShort[:,  metaTableShort[0] == 'CRC_T_10' ][:, 0]
    age_short = metaTableShort[:, 1]
    sex_short = metaTableShort[:, -4]
    for a in range(cancer_short_other.shape[0]):
        cancer_short_other[a] = cancer_short_other[a].split('_')[1]


    sampleInfo = np.array([ cancer_short_other, age_short, sex_short ]).T


    indices_short = []
    for a in range(metaTableShort.shape[0]):
        arg1 = np.argwhere( seqList_short == metaTableShort[a, 0]   )[0, 0]
        indices_short.append(arg1)
    indices_short = np.array(indices_short, dtype=int)
    #indent_short_pasta = 
    

    argInLong = []
    for a in range(seqList_long.shape[0]):
        indent_long = identifiers_long[ metaTableLong[:, 0] == seqList_long[a] ][0]
        arg1 = np.argwhere(identifiers_short == indent_long)[0, 0]
        arg1 = indices_short[arg1]
        argInLong.append(arg1)
    argInLong = np.array(argInLong, dtype=int)

    

    with torch.no_grad():
    

        for gene_index in range(len(geneList)):

            #print ('gene_index', gene_index)

            geneNow = geneList[gene_index]

            


            countNow = loadnpz('./data/real/splicing/cancer/geneFiles/shortCounts/' + geneNow + '.npz').T

            if False:#countNow.shape[1] == 87:
                #print (countNow.shape)
                avgCount = np.mean(countNow)

                if avgCount < 1:
                    print ('geneNow', geneNow)
                    print ('avgCount', avgCount)
                #quit()

            

            if  geneNow == 'ENSG00000249464':# 'ENSG00000087460':# 'ENSG00000222005':# 'ENSG00000087460':# 'ENSG00000131236':#
                
                print (geneNow)


                junctionNow = loadnpz('./data/real/splicing/cancer/geneFiles/shortJunctionPos/' + geneNow + '.npz')
                edges = loadnpz('./data/real/splicing/cancer/geneFiles/shortEdges/' + geneNow + '.npz')
                countNow = loadnpz('./data/real/splicing/cancer/geneFiles/shortCounts/' + geneNow + '.npz').T


                junctionNow[:, -1] = np.arange(junctionNow.shape[0])
                
                #from matplotlib.colors import LogNorm    
                #sns.clustermap(countNow, norm=LogNorm(), row_cluster=False)
                #plt.show()
                #quit()


                isoJunction_long = loadnpz('./data/real/splicing/cancer/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')
                junctionPos_long = loadnpz('./data/real/splicing/cancer/geneFiles/isoformJunctionsPos/' + str(geneNow) + '.npz')
                countNow_long = loadnpz('./data/real/splicing/cancer/geneFiles/isoformJunctions_countData/' + str(geneNow) + '.npz')


                countNow = countNow[:, argInLong]
                #print (countNow.shape)
                #print (countNow_long.shape)
                #print (seqList_short.shape)
                #print (seqList_long.shape)
                #quit()

                #countNow = np.zeros(countNow.shape, dtype=int)
                #countNow = countNow + (isoJunction_long[:1].T * np.arange(countNow.shape[1]).reshape((1, -1)))
                #countNow = countNow + (isoJunction_long[-1:].T * (countNow.shape[1] - 1 - np.arange(countNow.shape[1]).reshape((1, -1)) ))

                model_filename = './data/real/splicing/cancer/geneFiles/geneModels/' + geneNow + '_2.pt'
                model = torch.load(model_filename, weights_only=False)


                seqList_long = loadnpz('./data/real/splicing/cancer/info/long_seqList.npz')[:, 1]
                


                #junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
                #countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                #edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
                junctionPos = junctionNow[:, 1:3].astype(int)

                


                #isoformJunctions_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctions/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                #isoformJunctionPos_long = loadnpz('./data/real/splicing/geneFiles/longRead_isoformJunctionPos/' + str(geneNow) + '.npz')#[:, :Njunction] #Todo remove this subsetting
                #isoformCounts_long = loadnpz('./data/real/splicing/geneFiles/longRead_countData/' + str(geneNow) + '.npz')

                #validIsoform = np.sum(isoformJunctions_long, axis=1)
                #isoformJunctions_long = isoformJunctions_long[validIsoform >= 1]
                #isoformCounts_long = isoformCounts_long[validIsoform >= 1]

                Nsample = countNow.shape[1]
                Njunction = countNow.shape[0]

                observations_batch =  torch.tensor(countNow).float()

                edgeMatrix = np.zeros(( Njunction+1, Njunction+1 ), dtype=int)
                edgeMatrix[edges[:, 0], edges[:, 1]] = 1

                edgeMatrix[-1, np.sum(edgeMatrix, axis=0) == 0 ] = 1
                edgeMatrix[np.sum(edgeMatrix, axis=1) == 0, -1 ] = 1

                edgeSum = np.sum(edgeMatrix, axis=1)

                if True:

                    graphSize = Njunction
                    #finalProbSize = 19788# Nsample 
                    finalProbSize = 87 # countNow.shape[1] 
                    batchSize = Njunction
                    ruleObject = SpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                    ruleObject.graphSize = graphSize
                    ruleObject.observations_batch = observations_batch

                    

                    #batchSize = 200
                    batchSize = 2000
                    #batchSize = 20000
                    #batchSize = 500
                    offPolicy = False
                    model = torch.load(model_filename)
                    adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories, finalProb = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batchSize)

                    adjacency_matrices, finalProb_exp_sum = processPredictionCounts(adjacency_matrices, finalProb)
                    finalProb_exp_sum = finalProb_exp_sum[:, argInLong]

                    print (countNow.shape)
                    #print (isoformCounts.shape)
                    print (finalProb_exp_sum.shape)


                    #print (countNow_long.shape)
                    #quit()

                    print (countNow_long.shape)

                    score_ours = existEvaluator(adjacency_matrices, finalProb_exp_sum, isoJunction_long, countNow_long[:, :, 0], junctionPos, junctionPos_long)
                    print (score_ours)
                    quit()
                    score_short = evaluator(isoformJunctions, isoformCounts, isoformJunctions_long, isoformCounts_long, isoformJunctionPos, isoformJunctionPos_long)


                    #sns.heatmap(countNow )#, norm=LogNorm())
                    #plt.show()

                    #sns.heatmap(isoformCounts )#, norm=LogNorm())
                    #plt.show()

                    #sns.heatmap(finalProb_exp_sum )#, norm=LogNorm())
                    #plt.show()



                    #print (adjacency_matrices.shape)

                    values1 = np.mean(adjacency_matrices, axis=0)
                    values2 = np.mean(countNow, axis=1)

                    #print (values1)
                    #print (values2)

                    values1, values2 = values1 / np.sum(values1), values2 / np.sum(values2)

                    print (countNow.shape)
                    print ("A")

                    pVal_count = getPValues(countNow[:, indices_short], np.copy(sampleInfo))
                    #print (pVal_count[0][0, :, 0])#[:, 0])
                    print (np.min(pVal_count[0][0, :, 0])   * countNow.shape[0] )#[:, 0])
                    #countSelect = np.argmin(np.min(pVal_count[0][0, :, 0]))
                    #pVal_baseline = getPValues(isoformCounts[:, indices_short], np.copy(sampleInfo))
                    #print (np.min(pVal_baseline[0][0, :, 0])  *isoformCounts.shape[0]  )#[:, 0])

                    pVal_pred = getPValues(finalProb_exp_sum[:, indices_short], np.copy(sampleInfo))
                    print (np.min(pVal_pred[0][0, :, 0]) * finalProb_exp_sum.shape[0]  )#[:, 0])


                    if False:
                        adjacency_matrices_top = adjacency_matrices[np.sum(finalProb_exp_sum, axis=1) > 50].astype(int)
                        score_top = np.sum(finalProb_exp_sum, axis=1)[np.sum(finalProb_exp_sum, axis=1) > 50]

                        for a in range(adjacency_matrices_top.shape[0]):
                            print (score_top[a])
                            #print (isoJunction_long[-2])
                            #error1 = np.sum(np.abs(  isoJunction_long[-2] - adjacency_matrices_top[a] ))
                            #print (error1)
                            print (adjacency_matrices_top[a])

                        countSum = np.sum(countNow_long, axis=1)

                        
                        print (isoJunction_long.shape)

                        print (isoJunction_long[countSum[:, 0] > 100][:, :junctionPos.shape[0]]  )
                        print (countSum[:, 0][countSum[:, 0] > 100])

                        sns.heatmap(countNow_long[:, :, 0], norm=LogNorm())
                        plt.show()

                        sns.heatmap(edgeMatrix)
                        plt.show()
                        


                        #isoJunction_long = loadnpz('./data/real/splicing/cancer/geneFiles/isoformJunctions/' + str(geneNow) + '.npz')
                        #junctionPos_long = loadnpz('./data/real/splicing/cancer/geneFiles/isoformJunctionsPos/' + str(geneNow) + '.npz')
                        #countNow_long = loadnpz('./data/real/splicing/cancer/geneFiles/isoformJunctions_countData/' + str(geneNow) + '.npz')

                        junctCount = np.matmul(adjacency_matrices.T, finalProb_exp_sum)

                        


                        subsetLong = np.zeros( seqList_long.shape[0], dtype=int )
                        for a in range(seqList_long.shape[0]):
                            arg1 = np.argwhere(seqList_short == seqList_long[a])[0, 0]
                            subsetLong[a] = arg1

                        #from matplotlib.colors import LogNorm

                        countNow_mod = np.mean(countNow) * countNow / np.mean(countNow, axis=0).reshape((1, -1))

                        sns.heatmap(countNow_mod[:, :])# , norm=LogNorm())
                        plt.show()

                        #sns.heatmap(junctCount[:, :], norm=LogNorm())
                        #plt.show()

                        junctCount_mod = np.mean(junctCount) * junctCount / np.mean(junctCount, axis=0).reshape((1, -1))

                        sns.heatmap(junctCount_mod)
                        plt.show()

                        print (junctCount.shape)
                        print (countNow.shape)
                        quit()

                    
                    

            
    allScores_our, allScore_short = np.array(allScores_our), np.array(allScore_short)
    print ('ours', np.mean(allScores_our) )
    print ('short', np.mean(allScore_short) )
    plt.scatter(allScores_our, allScore_short )
    plt.plot([0, 1], [0, 1])
    plt.show()




#analyzeCancerResult()
#quit()


def trianACTG():

    #countFile = './data/real/splicing/tcga/geneFiles/counts/' + gene_now + '.npz'
    #samplesFile = './data/real/splicing/tcga/geneFiles/samples/' + gene_now + '.npz'
    #edgeFile = './data/real/splicing/tcga/geneFiles/edges/' + gene_now + '.npz'
    #junctionFile = './data/real/splicing/tcga/geneFiles/junctionPos/' + gene_now + '.npz'

    #np.savez_compressed(countFile, countArray) #smaller, surprisingly
    #np.savez_compressed(samplesFile, samples_unique)
    #np.savez_compressed(edgeFile, edgeMatrix)
    #np.savez_compressed(junctionFile, data_now)


    #gene_unique = loadnpz('./data/real/splicing/input/longRead_geneList.npz')

    #print (np.argwhere(gene_unique == 'ENSG00000003509'))
    #quit()


    #['BLCA' '433']
    #['BRCA' '1256']
    #['COAD' '546']
    #['HNSC' '548']
    #['KIRC' '618']
    #['LGG' '532']
    #['LIHC' '424']
    #['LUAD' '601']
    #['LUSC' '555']
    #['OV' '430']
    #['PRAD' '558']
    #['SKCM' '473']
    #['STAD' '453']
    #['THCA' '572']


    sampleStudy = loadnpz('./data/real/splicing/tcga/input/sampleStudy.npz')

    #gene_unique = np.array([ 'ENSG00000070886', 'ENSG00000120949', 'ENSG00000008118', 'ENSG00000009709', 'ENSG00000010072', 'ENSG00000033122', 'ENSG00000040487', 'ENSG00000117411', 'ENSG00000000971'])

    gene_unique = np.array([ 'ENSG00000000971', 'ENSG00000004487', 'ENSG00000065978', 'ENSG00000016490', 'ENSG00000031698'])


    tissueType = 'BRCA'

    #ENSG00000016490
    #tissueType = 'COAD'
    #tissueType = 'BLCA'

    #Survival
    #BRCA (better), KIRC (better), LGG (great!), SKCM (great!), 

    #Treatment:
    #LGG (great!), STAD(great!), 

    


    for tissueType in ['BRCA']: #['BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PRAD', 'SKCM', 'STAD', 'THCA']:
        for a in range(5):
            print ('')
        print (tissueType)
        for a in range(5):
            print ('')

        for gene_index in range(0, gene_unique.shape[0]):

            geneNow = gene_unique[gene_index]

            if geneNow == 'ENSG00000000971': #'ENSG00000010072':

                print ('geneNow', geneNow)

        
                #dupGen = 5
                #batchSize = 100
                #batchSize = 3
                learning_rate = 1e-3 #Typical
                #learning_rate = 5e-4 #Low learning rate
                #learning_rate = 1e-4
                #learning_rate = 1e-2
                #learning_rate = 1e-5
                #Nhidden = 2
                #Nhidden = 5
                #Nhidden = 10
                Nhidden = 50
                #Nhidden = 100
                

                model_filename = './data/real/splicing/tcga/geneFiles/geneModels/' + geneNow + '_' + tissueType + '_9.pt'
                #junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')


                if True:#GoodNow:#(strandDirection == '+') and (np.sum(isoformCounts_long) > 10000):

                    #BRCA, 

                    samples_valid = sampleStudy[sampleStudy[:, 1] == tissueType, 0]
                    
                    
                    samples = loadnpz('./data/real/splicing/tcga/geneFiles/samples/' + geneNow + '.npz')
                    countNow = loadnpz('./data/real/splicing/tcga/geneFiles/counts/' + geneNow + '.npz')
                    countNow = countNow[:, np.isin(samples, samples_valid)]

                    for sample_index in range(countNow.shape[1]):
                        sum1 = np.sum(countNow[:, sample_index])
                        if sum1 > 200:
                            countNow[:, sample_index] = countNow[:, sample_index] * 200 / sum1 



                    

                    print ('countMean', np.mean(countNow))

                    count_prop = np.sum(countNow, axis=1) / np.sum(countNow) 
                    if True:
                        #goodJunction = np.argwhere(count_prop > 0.0001)[:, 0]
                        #goodJunction = np.argwhere(count_prop > 0.00001)[:, 0]
                        goodJunction = np.argwhere(count_prop > 0.000001)[:, 0]
                    else:
                        goodJunction = np.arange(count_prop.shape[0])
                        
                    #print (np.sum(count_prop[goodJunction]))
                    #print (goodJunction.shape)

                    #quit()
                    

                    #countNow_sum = np.sum(countNow, axis=1)
                    #goodJunction = np.argwhere(countNow_sum >= 10)[:, 0]
                    countNow = countNow[goodJunction]

                    print (countNow.shape)

                    #plt.plot( np.sum(countNow, axis=0) )
                    #plt.show()
                    #quit()
                    

                    #print (countNow.shape)
                    #quit()


                    edgeMatrix = loadnpz('./data/real/splicing/tcga/geneFiles/edges/' + geneNow + '.npz')
                    edgeMatrix[edgeMatrix>1] = 1

                    
                    



                    goodJunction_plus = np.concatenate((goodJunction,  np.zeros(1, dtype=int) + edgeMatrix.shape[0] - 1 ))
                    edgeMatrix = edgeMatrix[goodJunction_plus][:, goodJunction_plus]
                    edgeMatrix_sum1 = np.sum(edgeMatrix, axis=0)
                    edgeMatrix_sum2 = np.sum(edgeMatrix, axis=1)
                    edgeMatrix[-1, edgeMatrix_sum1 == 0] = 1
                    edgeMatrix[edgeMatrix_sum2 == 0, -1] = 1

                    #plt.imshow(edgeMatrix)
                    #plt.show()

                    #sns.heatmap(edgeMatrix)
                    #plt.show()


                    countNow = torch.tensor(countNow).float()

                    Nsample = countNow.shape[1]
                    Njunction = countNow.shape[0]

                    observations_batch = countNow

                    
                    


                    graphSize = Njunction
                    finalProbSize = Nsample# + Njunction
                    
                    ruleObject = biasSpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                    ruleObject.graphSize = graphSize
                    ruleObject.observations_batch = observations_batch
                    ruleObject.model = SpliceNet(graphSize, finalProbSize, Nhidden)

                    #batchSize = 100
                    #batchSize = 200
                    #batchSize = 5000
                    batchSize = 10000
                    #batchSize = 1000
                    #batchSize = 2000
                    #batchSize = 500


                    offPolicy = False
                    dupGen = 1


                    train_model_off_policy(ruleObject, learning_rate, batchSize, dupGen, offPolicy, num_epochs=5000, model_filename=model_filename)

                    #train_model_off_policy(Nhidden, ruleObject, learning_rate, observations_batch,  batchSize, dupGen, offPolicy, num_epochs=5000, model_filename=model_filename, rewardType='')



#trianACTG()
#quit()




def evaluateACTG():


    def getPValues(finalProb_exp_sum, tissueInfo):

        #print (np.sum(finalProb_exp_sum))

        finalProb_exp_sum = finalProb_exp_sum[np.sum(finalProb_exp_sum, axis=1) > 0]

        finalProb_exp_sum = finalProb_exp_sum.astype(float)
        finalProb_exp_sum = finalProb_exp_sum / (np.sum(finalProb_exp_sum, axis=0).reshape((1, -1)) + 1e-8)

        pValLists = []

        for phen_index in range(tissueInfo.shape[1]):
            Y = tissueInfo[:, phen_index]
            unique_groups = np.unique(Y)

            #print (unique_groups)

            pValuesPaste = np.zeros((  unique_groups.shape[0],  finalProb_exp_sum.shape[0] , 2))

            #print (pValuesPaste.shape)
            #quit()

            for group_index in range(unique_groups.shape[0]):
                group_now = unique_groups[group_index]

                #print (group_now)

                for isoform_index in range(finalProb_exp_sum.shape[0]):
                    X = finalProb_exp_sum[isoform_index]
                    Y_bool = np.zeros(X.shape[0])
                    Y_bool[Y==group_now] = 1
                    #t_statistic, p_value = scipy.stats.ttest_ind(X[Y==group_now], X[Y!=group_now])
                    t_statistic, p_value = scipy.stats.pearsonr(X, Y_bool)

                    pValuesPaste[group_index, isoform_index, 0] = p_value
                    pValuesPaste[group_index, isoform_index, 1] = t_statistic

                    #print (t_statistic, p_value)
                    #plt.hist(X[Y==group_now], bins=100, density=True, alpha=0.5)
                    #plt.hist(X[Y!=group_now], bins=100, density=True, alpha=0.5)
                    #plt.show()

                    
                pBest = np.min(pValuesPaste[group_index, :, 0]) * finalProb_exp_sum.shape[0]
                pBest = pBest * unique_groups.shape[0]

                #print (pBest)

            pValLists.append( np.copy(pValuesPaste) )
        
        return pValLists


    def getPcor(finalProb_exp_sum, tissueInfo):

        #print (np.sum(finalProb_exp_sum))

        finalProb_exp_sum = finalProb_exp_sum[np.sum(finalProb_exp_sum, axis=1) > 0]

        finalProb_exp_sum = finalProb_exp_sum.astype(float)
        finalProb_exp_sum = finalProb_exp_sum / (np.sum(finalProb_exp_sum, axis=0).reshape((1, -1)) + 1e-8)

        pValLists = []

        for phen_index in range(tissueInfo.shape[1]):
            Y = tissueInfo[:, phen_index]
            

            pValuesPaste = np.zeros((  finalProb_exp_sum.shape[0] , 2))
            pValuesPaste[:, 0] = 1

            argGood = np.argwhere(np.isnan(Y) == False)[:, 0]
            Y = Y[argGood]

            if argGood.shape[0] >= 2:

                for isoform_index in range(finalProb_exp_sum.shape[0]):
                    X = finalProb_exp_sum[isoform_index][argGood]
                    if np.sum(np.abs(X)) > 1e-6:
                        
                    
                        t_statistic, p_value = scipy.stats.pearsonr(X, Y)

                        #print (np.mean(np.abs(X)), np.mean(np.abs(Y)) )
                        #print (t_statistic, p_value)

                        pValuesPaste[isoform_index, 0] = p_value
                        pValuesPaste[isoform_index, 1] = t_statistic


            pValLists.append( np.copy(pValuesPaste) )

        pValLists = np.array(pValLists)
        
        return pValLists


    def removeBackground(patientInfo, gene_pcs):

        
        gene_pcs = gene_pcs.T 
        gene_pcs = gene_pcs / np.max(gene_pcs)
        '''
        gene_pcs = gene_pcs - np.mean(gene_pcs, axis=0).reshape((1, -1))
        gene_pcs = gene_pcs / np.mean(gene_pcs**2, axis=0).reshape((1, -1))

        for a in range(patientInfo.shape[1]):
            patientInfo0 = patientInfo[:, a]
            argGood = np.argwhere(np.isnan(patientInfo0) == False)[:, 0]
            patientInfo0 = patientInfo0[argGood]#.reshape((-1, 1))
            patientInfo0 = patientInfo0 - np.sum(np.mean(patientInfo0.reshape((-1, 1)) * gene_pcs[argGood], axis=0) * gene_pcs[argGood], axis=1)
            patientInfo[argGood, a] = patientInfo0#[:, 0]
        '''


        

        for a in range(patientInfo.shape[1]):
            patientInfo0 = patientInfo[:, a]
            argGood = np.argwhere(np.isnan(patientInfo0) == False)[:, 0]
            patientInfo0 = patientInfo0[argGood]

            patientInfo0 = patientInfo0 - np.mean(patientInfo0)

            
            if argGood.shape[0] >= 2:
                # Step 1. Fit gene-only model
                ridge = RidgeCV(alphas=[0.001, 0.01, 0.01, 1.0], cv=5).fit(gene_pcs[argGood], patientInfo0)
                #print ("A")
                #print (np.mean(np.abs(patientInfo0)))
                patientInfo0 = patientInfo0 - ridge.predict(gene_pcs[argGood])
                #print (np.mean(np.abs(patientInfo0)))
                patientInfo[argGood, a] = patientInfo0

            #print("Chosen alpha:", ridge.alpha_)
            #print("Coefficient of determination R^2:", ridge.score(gene_pcs[argGood], patientInfo0))
            #quit()

        return patientInfo

    

    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV

    df = pd.read_csv('./data/real/splicing/tcga/input/tcga_samples.tsv', sep='\t', dtype=str)


    'gdc_cases.diagnoses.vital_status'
    'cgc_case_primary_therapy_outcome_success'

    survival = df['gdc_cases.diagnoses.vital_status'].to_numpy().astype(str)
    treatmentResponse = df['cgc_case_primary_therapy_outcome_success'].to_numpy().astype(str)

    patientInfo_all = np.array([survival, treatmentResponse]).T


    #colList = df.columns
    #colList = np.array(list(colList))
    #for a in range(colList.shape[0]):
    #    print ('')
    #    print (colList[a])
    #    print (df[colList[a]][0])
    #quit()

    sampleIDs = df['rail_id'].to_numpy().astype(str)
    studyNames = df['study'].to_numpy().astype(str)
    sampleNames = df['tcga_barcode'].to_numpy().astype(str)
    sampleStudy = np.array([sampleIDs,studyNames , sampleNames]).T

    #print (sampleStudy.shape)
    #quit()




    #sampleStudy = loadnpz('./data/real/splicing/tcga/input/sampleStudy.npz')

    #gene_unique = np.array([  'ENSG00000070886'])
    #gene_unique = np.array(['ENSG00000016490'])
    #gene_unique = np.array([  'ENSG00000065978'])
    #gene_unique = np.array(['ENSG00000004487'])
    gene_unique = np.array(['ENSG00000000971'])



    #['BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PRAD', 'SKCM', 'STAD', 'THCA']

    #tissueType = 'STAD'
    tissueType = 'BRCA'
    #tissueType = 'OV'


    #Survival
    #BRCA (better), KIRC (better), LGG (great!), SKCM (great!), 
    #Treatment:
    #LGG (great!), STAD(great!), 

    #ENSG00000004487
    # BRCA (great), KIRC (good), STAD (good, treatment)

    
    #ENSG00000120949
    #['STAD', 'BRCA', 'KIRC']
    
    with torch.no_grad():
    

        for gene_index in range(0, len(gene_unique)):# range(40):

            #print ('gene_index', gene_index)

            geneNow = gene_unique[gene_index]
            #geneNow = geneValid[2]

            
            #unique1, count1 = np.unique(sampleStudy[:, 1], return_counts=True)
            #ar = np.array([unique1, count1]).T
            #print (ar)
            #quit()


            samples_valid = sampleStudy[sampleStudy[:, 1] == tissueType]
            #print (samples_valid.shape)

            
            samples_gene = loadnpz('./data/real/splicing/tcga/input/samples_all.npz')
            geneCount = loadnpz('./data/real/splicing/tcga/input/geneCount.npz').astype(float)
            #geneCount = geneCount / np.mean(geneCount, axis=1).reshape((-1, 1))
            #geneList = loadnpz('./data/real/splicing/tcga/input/geneList.npz')
            #pca = PCA(n_components=50)   # keep ~80-90% variance
            #gene_pcs = pca.fit_transform(geneCount.T).T
            #print(pca.explained_variance_ratio_)
            #quit()

            junctionPos = loadnpz('./data/real/splicing/tcga/geneFiles/junctionPos/' + geneNow + '.npz')
            
            samples = loadnpz('./data/real/splicing/tcga/geneFiles/samples/' + geneNow + '.npz')
            countNow = loadnpz('./data/real/splicing/tcga/geneFiles/counts/' + geneNow + '.npz')
            countNow = countNow[:, np.isin(samples, samples_valid[:, 0] )]

            for sample_index in range(countNow.shape[1]):
                sum1 = np.sum(countNow[:, sample_index])
                if sum1 > 200:
                    countNow[:, sample_index] = countNow[:, sample_index] * 200 / sum1

            
            samples = samples[ np.isin(samples, samples_valid[:, 0])]
            modSampleName = []
            for a in range(samples.shape[0]):
                arg1 = np.argwhere(samples_valid[:, 0] == samples[a])[0, 0]
                modSampleName.append(arg1)
            modSampleName = samples_valid[np.array(modSampleName, dtype=int)]

            geneCount = geneCount[:, np.isin(samples_gene, samples_valid[:, 0])]
            #gene_pcs = gene_pcs[:, np.isin(samples_gene, samples_valid[:, 0])]
            samples_gene = samples_gene[np.isin(samples_gene, samples_valid[:, 0])]

            #print (modSampleName.shape)
            #quit()


            assert np.array_equal(modSampleName[:, 0], samples)

            subsetUse = []
            for a in range(samples.shape[0]):
                arg1 = np.argwhere(sampleStudy[:, 0] == samples[a] )[0, 0]
                subsetUse.append(arg1)
            subsetUse = np.array(subsetUse).astype(int)

            #print (patientInfo_all.shape)

            patientInfo = patientInfo_all[subsetUse]

            #print (patientInfo.shape)

            

            count_prop = np.sum(countNow, axis=1) / np.sum(countNow) 
            #goodJunction = np.argwhere(count_prop > 0.0001)[:, 0]
            #goodJunction = np.argwhere(count_prop > 0.00001)[:, 0]
            goodJunction = np.argwhere(count_prop > 0.000001)[:, 0]
            #print (np.sum(count_prop[goodJunction]))
            #print (goodJunction.shape)

            countNow = countNow[goodJunction]
            count_prop_subset = count_prop[goodJunction]
            junctionPos = junctionPos[goodJunction]


            edgeMatrix = loadnpz('./data/real/splicing/tcga/geneFiles/edges/' + geneNow + '.npz')
            edgeMatrix[edgeMatrix>1] = 1

            goodJunction_plus = np.concatenate((goodJunction,  np.zeros(1, dtype=int) + edgeMatrix.shape[0] - 1 ))
            edgeMatrix = edgeMatrix[goodJunction_plus][:, goodJunction_plus]
            edgeMatrix_sum1 = np.sum(edgeMatrix, axis=0)
            edgeMatrix_sum2 = np.sum(edgeMatrix, axis=1)
            edgeMatrix[-1, edgeMatrix_sum1 == 0] = 1
            edgeMatrix[edgeMatrix_sum2 == 0, -1] = 1



            Njunction = countNow.shape[0]


            if True:#geneNow ==  'ENSG00000072694':

                
                #model_filename = './data/real/splicing/tcga/geneFiles/geneModels/' + geneNow + '_3.pt'
                model_filename = './data/real/splicing/tcga/geneFiles/geneModels/' + geneNow + '_' + tissueType + '_9.pt'
                model = torch.load(model_filename, weights_only=False)


                bias = model.giveBias()

                #plt.plot( np.exp(bias.data.numpy() ) )
                #plt.show()
                #quit()
                

                
                Nsample = countNow.shape[1]
                Njunction = countNow.shape[0]

                observations_batch =  torch.tensor(countNow).float()

                geneTotalCount = np.sum(countNow, axis=0)



                graphSize = Njunction
                #finalProbSize = 19788# Nsample 
                finalProbSize = countNow.shape[1] 
                batchSize = Njunction
                ruleObject = SpliceClass(finalProbSize, Njunction, Nsample, edgeMatrix)
                ruleObject.graphSize = graphSize
                ruleObject.observations_batch = observations_batch
                ruleObject.model = model

                

                #batchSize = 200
                #batchSize = 2000
                batchSize = 10000
                #batchSize = 500
                offPolicy = False
                model = torch.load(model_filename)
                adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories = generate_graph_batch_with_modified_policy(model, ruleObject, offPolicy, batchSize)
                #adjacency_matrices, log_prob_pi, log_prob_pi_prime, trajectories = spread_graph_batch(model, ruleObject, offPolicy, batchSize) #TODO set back to original 


                _, finalProbAllow = ruleObject.graphRules(adjacency_matrices)
                finalProb = ruleObject.model.finalProb(adjacency_matrices)
                finalProb = finalProb + finalProbAllow
                finalProb = nn.LogSoftmax(dim=1)(finalProb)

                adjacency_matrices, finalProb_exp_sum = processPredictionCounts(adjacency_matrices, finalProb)

                adjacency_matrices_weighted = torch.log(torch.tensor(adjacency_matrices).float() + 1e-10)
                adjacency_matrices_weighted[adjacency_matrices == 0] = -500
                adjacency_matrices_weighted = adjacency_matrices_weighted + bias.reshape((1, -1))
                adjacency_matrices_weighted = torch.nn.Softmax(dim=1)(adjacency_matrices_weighted)
                adjacency_matrices_weighted = adjacency_matrices_weighted.data.numpy()


                #sns.heatmap(adjacency_matrices)
                #plt.show()

                #sns.heatmap(adjacency_matrices_weighted)
                #plt.show()

                

                print ('adjacency_matrices', adjacency_matrices.shape)
                print ('adjacency_matrices_weighted', adjacency_matrices_weighted.shape)

                finalProb_exp_sum_normed = finalProb_exp_sum / (np.sum(finalProb_exp_sum, axis=0)+1e-10).reshape((1, -1))

                #print (np.min(finalProb_exp_sum_normed), np.max(finalProb_exp_sum_normed))
                #
                #entropy = -1 * np.sum(finalProb_exp_sum_normed * np.log(finalProb_exp_sum_normed+1e-10), axis=0 )
                #print (np.mean(entropy))
                #quit()

                #plt.hist (np.sum(adjacency_matrices, axis=1), bins=100)
                #plt.show()
                #quit()

                #print (finalProb_exp_sum.shape)

                #print (np.unique(patientInfo[:, 0], return_counts=True))
                #print (np.unique(patientInfo[:, 1], return_counts=True))


                samples_RSEM = loadnpz('./data/real/splicing/tcga/input/samplesFromRSEM.npz')
                RSEM = loadnpz('./data/real/splicing/tcga/geneFiles/isoform_counts/' + geneNow + '.npz')
                RSEM = np.power(2, RSEM) - 0.001
                RSEM[RSEM < 1e-6] = 0
                
                RSEM_index = np.zeros(samples_RSEM.shape[0], dtype=int) - 1
                for a in range(samples_RSEM.shape[0]):
                    for b in range(samples.shape[0]):
                        if samples_RSEM[a] in modSampleName[b, 2]:
                            RSEM_index[a] = b
                RSEM = RSEM[:, RSEM_index!=-1]
                samples_RSEM = samples_RSEM[RSEM_index!=-1]
                RSEM_index = RSEM_index[RSEM_index!=-1]
                #print (RSEM.shape)
                #print (RSEM_index.shape)
                #print (np.min(RSEM_index))
                #print (RSEM_index)
                #quit()
                


                patientInfo[np.isin(patientInfo[:, 0], np.array(['alive', 'dead'])) == False, 0] = float('nan')
                patientInfo[patientInfo[:, 0] == 'alive', 0] = 1
                patientInfo[patientInfo[:, 0] == 'dead', 0] = 0

                #print (patientInfo.shape)
                #quit()

                patientInfo[np.isin(patientInfo[:, 1], np.array(['Complete Remission/Response', 'Partial Remission/Response', 'Progressive Disease', 'Stable Disease'])) == False, 1] = float('nan')
                patientInfo[patientInfo[:, 1] == 'Complete Remission/Response', 1] = 3
                patientInfo[patientInfo[:, 1] == 'Partial Remission/Response', 1] = 2
                patientInfo[patientInfo[:, 1] == 'Stable Disease', 1] = 1
                patientInfo[patientInfo[:, 1] == 'Progressive Disease', 1] = 0

                patientInfo = patientInfo.astype(float)

                print (patientInfo[np.isnan(patientInfo[:, 0]) == False, 0].shape)
                print (patientInfo[np.isnan(patientInfo[:, 1]) == False, 1].shape)

                patientInfo = removeBackground(patientInfo, geneCount)

                

                


                patientInfo_RSEM = patientInfo[RSEM_index]

                #print (np.mean(patientInfo[:, 0]))
                #print (np.mean(patientInfo_RSEM[:, 0]))
                #quit()


                #print (scipy.stats.pearsonr(entropy[np.isnan(patientInfo[:, 0])==False],patientInfo[ np.isnan(patientInfo[:, 0])==False , 0] ))
                #print ("Gene cor")
                #print (scipy.stats.pearsonr(geneTotalCount[np.isnan(patientInfo[:, 0])==False],patientInfo[ np.isnan(patientInfo[:, 0])==False , 0] ))


                #print (scipy.stats.pearsonr(entropy[np.isnan(patientInfo[:, 0])==False],patientInfo[ np.isnan(patientInfo[:, 0])==False , 0] ))
                #print (scipy.stats.pearsonr(entropy[np.isnan(patientInfo[:, 1])==False],patientInfo[ np.isnan(patientInfo[:, 1])==False, 1] ))

                #print (finalProb_exp_sum.shape)
                #print (RSEM.shape)

                count_pred = np.matmul(adjacency_matrices_weighted.T, finalProb_exp_sum)

                print (count_pred.shape)
                print (countNow.shape)

                print (scipy.stats.pearsonr(  count_pred.reshape((-1,)) ,  countNow.reshape((-1,)) ))

                #plt.scatter( count_pred.reshape((-1,)) ,  countNow.reshape((-1,)) )
                #plt.show()
                #quit()

                

                print (count_pred.shape)
                print (patientInfo.shape)
                
                pOurs = getPcor(finalProb_exp_sum, patientInfo)
                pCount = getPcor(countNow, patientInfo)
                pCountPred = getPcor(count_pred, patientInfo)
                pOther = getPcor(RSEM, patientInfo_RSEM)

                pGene = getPcor(geneCount, patientInfo)

                print (pCountPred.shape)
                #quit()
                #print (np.sum(countNow, axis=1))
                #sns.heatmap(countNow, norm=LogNorm())
                #plt.show()

                print (countNow.shape)
                print (finalProb_exp_sum.shape)

                corMatrix = np.zeros((countNow.shape[0],   countNow.shape[0]))
                selfCorMatrix = np.zeros((countNow.shape[0],   countNow.shape[0]))
                for a in range(corMatrix.shape[0]):
                    for b in range(count_pred.shape[0]):
                        cor1 = scipy.stats.pearsonr(countNow[a],  count_pred[b] )[0]
                        corMatrix[a, b] = cor1

                        selfCorMatrix[a, b] = scipy.stats.pearsonr(count_pred[a],  count_pred[b] )[0]

                print (scipy.stats.pearsonr(countNow[64],  count_pred[63] ))
                sns.heatmap(selfCorMatrix)
                plt.show()

                sns.heatmap(corMatrix)
                plt.show()

                #quit()

                highP = np.argmin(pCount[0, :, 0])
                #print (pCount[0, highP, 0])


                #print ('')
                #print (np.sum(countNow[highP]))
                #print (np.sum(countNow))
                #print ('')
                #print (np.sum(finalProb_exp_sum))
                #print (np.sum( finalProb_exp_sum[  adjacency_matrices[:, highP] == 1 ] ))
                #plt.plot(adjacency_matrices[:, highP])
                #plt.show()

                #quit()

                
                
                #print (pGene[0])

                #print (geneCount.shape)
                #print (patientInfo.shape)

                #plt.scatter( geneCount[0], patientInfo[:, 0] )
                #plt.show()
                #quit()


                #print (pOurs[0].shape)

                #print (pOurs.shape)

                #print (pOurs)
                #print (pCount[0, :, 0])

                #plt.plot( np.log10(pCount[0, :, 0])*-1 )
                #plt.plot( count_prop_subset * 10 )
                #plt.plot( np.max(corMatrix, axis=1) )
                #plt.show()

                pVals = np.log10(pCount[0, :, 0])*-1

                #print (pVals.shape)
                #print (junctionPos.shape)

                if False:
                    junctionSig =  junctionPos[pVals > 3]

                    print (junctionSig[0])
                    juncClose = junctionPos[ np.abs(junctionPos[:, 1].astype(int) - int(junctionSig[0, 1])) < 200  ]
                    print (juncClose)
                    print ('')
                    print (junctionSig[0])
                    juncClose = junctionPos[ np.abs(junctionPos[:, 2].astype(int) - int(junctionSig[0, 2])) < 200  ]
                    print (juncClose)
                    print ('')
                    print (junctionSig[1])
                    juncClose = junctionPos[ np.abs(junctionPos[:, 1].astype(int) - int(junctionSig[1, 1])) < 200  ]
                    print (juncClose)
                    print ('')
                    print (junctionSig[1])
                    juncClose = junctionPos[ np.abs(junctionPos[:, 2].astype(int) - int(junctionSig[1, 2])) < 200  ]
                    print (juncClose)
                    print ('')



                if True:
                    print (np.min(pOurs[0, :, 0], axis=0) * pOurs.shape[1])
                    print (np.min(pCountPred[0, :, 0], axis=0) * pOurs.shape[1])
                    print (np.min(pGene[0, :, 0], axis=0) * pGene.shape[1])
                    print (np.min(pCount[0, :, 0], axis=0) * pCount.shape[1])
                    print (np.min(pOther[0, :, 0], axis=0) * pOther.shape[1])

                    #print (np.max(pOurs[0, :, 1], axis=0), np.min(pOurs[0, :, 1], axis=0) )
                    #print (np.max(pCount[0, :, 1], axis=0), np.min(pCount[0, :, 1], axis=0) )
                    #print (np.max(pOther[0, :, 1], axis=0), np.min(pOther[0, :, 1], axis=0) )
                    #quit()

                    print ('')
                    print (np.min(pOurs[1, :, 0], axis=0) * pOurs.shape[1])
                    print (np.min(pCountPred[1, :, 0], axis=0) * pOurs.shape[1])
                    print (np.min(pGene[1, :, 0], axis=0) * pGene.shape[1])
                    print (np.min(pCount[1, :, 0], axis=0) * pCount.shape[1])
                    print (np.min(pOther[1, :, 0], axis=0) * pOther.shape[1])

                #print (np.max(pOurs[1, :, 1], axis=0), np.min(pOurs[1, :, 1], axis=0) )
                #print (np.max(pCount[1, :, 1], axis=0), np.min(pCount[1, :, 1], axis=0) )
                #quit()

                #print (np.min(pOurs[0][:, :, 0], axis=1) * pOurs[0].shape[1])
                #print (np.min(pCount[0][:, :, 0], axis=1) * pCount[0].shape[1])
                #print ('')
                #print (np.min(pOurs[1][:, :, 0], axis=1) * pOurs[0].shape[1])
                #print (np.min(pCount[1][:, :, 0], axis=1) * pCount[0].shape[1])
                #quit()

                #print (finalProb_exp_sum.shape)

                
                

                edgeMatrix_mod = np.copy(edgeMatrix).astype(float)
                edgeMatrix_mod[highP] += 0.2
                edgeMatrix_mod[:, highP] += 0.2

                


                #quit()
                print (highP)
                print (np.sum(countNow, axis=1)[highP-1:highP+2] / np.sum(countNow))
                print (np.sum(count_pred, axis=1)[highP-1:highP+2] / np.sum(count_pred))

                #sns.heatmap(edgeMatrix_mod)
                #plt.show()


                print ("junction proportions")
                plt.plot( np.sum(countNow, axis=1) / np.sum(countNow) )
                plt.plot( np.sum(count_pred, axis=1) / np.sum(count_pred) )
                plt.show()

                plt.plot(bias)
                plt.show()

                quit()
                #quit()
                

                countNow_mod = (countNow  / np.sum(countNow, axis=0).reshape((1, -1)))

                #print (np.max(countNow_mod))

                #from matplotlib.colors import LogNorm

                sns.heatmap(  countNow_mod )#, norm=LogNorm()    )
                plt.show()

                sns.heatmap(  (count_pred  / np.mean(count_pred, axis=0).reshape((1, -1)))   )#, norm=LogNorm()    )
                plt.show()

                quit()


evaluateACTG()
quit()