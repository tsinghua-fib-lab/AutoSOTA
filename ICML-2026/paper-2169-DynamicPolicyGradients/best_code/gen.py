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
from baselines import *
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




def inferDistributions():


    modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'diffusion'
    #modelType = 'localSolver'
    ####modelType = 'metropolas'

    
    #graphSize = 100
    #graphSize = 90

    simList = [] #data points, nodes, paths
    simList.append([100, 10, 100])  #default 

    #simList.append([1, 10, 100])  #modified number of data points 
    #simList.append([5, 10, 100]) 
    #simList.append([10, 10, 100]) 
    #simList.append([25, 10, 100]) 
    #simList.append([250, 10, 100])
    #simList.append([1000, 10, 100])


    #simList.append([100, 5, 100])  #modified number of nodes
    #simList.append([100, 15, 100])  #modified number of nodes
    #simList.append([100, 20, 100])  #modified number of nodes


    #simList.append([100, 10, 1])  #modified number of paths
    #simList.append([100, 10, 10])  #modified number of paths
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

            
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

            
            print (simPart, modelType)

            
            observations_batch = loadnpz('./data/sims/initial/' + simPart + '_obs.npz')
            adjacency_matrices = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')

            #print (observations_batch[0])
            #quit()

            

            graphSize = (num_nodes - 1) * num_nodes
            modelFile = './data/sims/startEnd/model/graph_' + simPart + '_' + modelType + '.pt' #1
            model = torch.load(modelFile)

            multi_x_given_g, log_calculate_pr_x_given_g = sim1_fast_multi, sim1_log_calculate_pr_x_given_g  
            finalProbSize = 1

            ruleObject = gClass()
            def graphRules(graph):
                graphAllow = torch.zeros((graph.shape[0], graphSize+1))
                finalProbAllow = torch.zeros((graph.shape[0], finalProbSize))
                return graphAllow, finalProbAllow

            def offPolicyRule(graphList, arange1):

                probList = np.zeros(graphList.shape[0])

                for a in range(probList.shape[0]):
                    graphNow = graphList[a]
                    prob1 = log_calculate_pr_x_given_g(graphNow, observations_batch[arange1[a]]  )
                    probList[a] = prob1

                argNotDone = np.argwhere(probList ==  -float('inf'))[:, 0]
                

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

            batch_size = 10000
            #batch_size = 10
            
            if modelType in ['ours', 'naiveReward', 'GFlowReward']:
                sampled_graphs, _, _, _  = generate_graph_batch_with_modified_policy(model, ruleObject, False, batch_size)
                sampled_graphs = sampled_graphs.data.numpy()

            if modelType == 'VAE':
                sampled_graphs = sample_graphs(model, num_samples=batch_size)
                sampled_graphs = sampled_graphs.data.numpy()

            if modelType == 'autoreg':
                sampled_graphs = model.generate(batch_size=batch_size)
                sampled_graphs = sampled_graphs.data.numpy()


            print (sampled_graphs.shape)
            

            #generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, doTrain, batchSize, offPolicy)

            np.savez_compressed('./data/sims/startEnd/model/sampledGraphs_' + simPart + '_' + modelType + '.npz', sampled_graphs)
    

#inferDistributions()
#quit()


def generic_TrainModel(ruleObject, simPart, modelType, predFile, model_filename, sampleFile, doTrain, batchSize, offPolicy, giveTrajectory=False, saveTrainingLoss=''):



    #saveModelBackup(model_filename)

    learning_rate = 1e-3


    adjacency_matrices = ruleObject.adjacency_matrices
    multi_x_given_g = ruleObject.multi_x_given_g
    log_calculate_pr_x_given_g = ruleObject.log_calculate_pr_x_given_g
    observations_batch = ruleObject.observations_batch
    graphSize = ruleObject.graphSize
    

    print (adjacency_matrices.shape)
    #reward_true = multi_x_given_g( torch.tensor(adjacency_matrices).float() , observations_batch)
    #reward_true = reward_true.data.numpy()
    #reward_true = reward_true[np.arange(reward_true.shape[0]), np.arange(reward_true.shape[0])]

    # -103.8510971069336
    #print ('reward_true', np.mean(reward_true))
    #quit()

    Nhidden = 50



    sampleSize = 100000
    
    if modelType == 'ours':
        if doTrain:
            #if graphSize <= 300:
            #    learning_rate = 1e-2
            
            
            #learning_rate = 1e-2
            #learning_rate = 5e-3 
            learning_rate = 1e-3 #GOOD

            
            #learning_rate = 2e-4
            
            
            #print ("NOTE, Learning rate modified!!!!!!")


            print ('learning_rate', learning_rate)
            print ('batchSize', batchSize)

            #sizeSampler = duplicationNum * observations_batch.shape[0]
            #M_sampler = sizeSampler // observations_batch.shape[0]
            #observations_batch = observations_batch[np.arange(sizeSampler) // M_sampler]
            #observations_batch = observations_batch[np.arange(observations_batch.shape[0] * 4) // 4]
            #observations_batch = observations_batch[np.arange(observations_batch.shape[0] * 10) // 10]

            #print (np.arange(sizeSampler) // M_sampler)
            #quit()
            
            #model_filename="./data/model/our_" + simPart + "_3.pt"

            #finalProbSize, graphRules

            ruleObject.observations_batch = observations_batch

            train_model_off_policy(ruleObject, learning_rate, batchSize, offPolicy, num_epochs=50000, model_filename=model_filename, rewardType='', giveTrajectory=giveTrajectory, saveTrainingLoss=saveTrainingLoss)
            #train_model_off_policy(Nhidden, ruleObject, learning_rate, observations_batch,  batchSize, dupGen, num_epochs=5000, model_filename=model_filename, rewardType='')
            #quit()
        else:
            #model_filename="./data/model/our_N10_P100.pt"
            #model_filename="./data/model/our_" + simPart + "_3.pt"
            policyModel = torch.load(model_filename)
            #predicted_graphs = batch_predict_graphs(policyModel, ruleObject, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g, 1000, offPolicy, giveTrajectory=giveTrajectory)
            adjacency_matrices, log_prob_pi, log_prob_prime, trajectory = batchSamplerOurs(policyModel, ruleObject, offPolicy, sampleSize)
            adjProb = []
            if offPolicy:
                adjProb = [ log_prob_pi - log_prob_prime ]
            predicted_graphs = simpleGeneralPredictor(adjacency_matrices, observations_batch, multi_x_given_g, adjProb=adjProb)


    
    if modelType == 'naiveReward':
        if doTrain:
            learning_rate = 5e-3

            ruleObject.observations_batch = observations_batch
            #ruleObject.model = GraphGeneratorNet(ruleObject.graphSize, 1, Nhidden)

            #model_filename="./data/model/ourNaive_" + simPart + "_1.pt"
            train_model_off_policy(ruleObject, learning_rate, batchSize, offPolicy, num_epochs=50000, model_filename=model_filename, rewardType='easy')
            #train_model_off_policy(Nhidden, graphSize, finalProbSize, graphRules, offPolicyRule,  multi_x_given_g, log_calculate_pr_x_given_g, learning_rate, observations_batch,  batchSize, dupGen, num_epochs=5000, model_filename=model_filename, rewardType='easy')
            #quit()
        else:
            #model_filename="./data/model/ourNaive_" + simPart + "_1.pt" 
            #policyModel = torch.load(model_filename)
            #predicted_graphs = batch_predict_graphs(policyModel, graphSize, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g, 100)
            policyModel = torch.load(model_filename)
            #predicted_graphs = batch_predict_graphs(policyModel, ruleObject, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g, 1000, offPolicy)
            adjacency_matrices, log_prob_pi, log_prob_prime, trajectory = batchSamplerOurs(policyModel, ruleObject, offPolicy, sampleSize)
            adjProb = []
            if offPolicy:
                adjProb = [ log_prob_pi - log_prob_prime ]
            predicted_graphs = simpleGeneralPredictor(adjacency_matrices, observations_batch, multi_x_given_g, adjProb=adjProb)
    

    if modelType == 'GFlowReward':
        if doTrain:

            #learning_rate = 1e-4
            #learning_rate = 1e-3
            learning_rate = 5e-3
            #model_filename="./data/model/GFlowPol_" + simPart + "_1.pt"
            #train_model_off_policy(Nhidden, graphSize, finalProbSize, graphRules, offPolicyRule,  multi_x_given_g, log_calculate_pr_x_given_g, learning_rate, observations_batch,  batchSize, dupGen, num_epochs=5000, model_filename=model_filename, rewardType='GFlow')
            #quit()
            ruleObject.observations_batch = observations_batch
            #ruleObject.model = GraphGeneratorNet(ruleObject.graphSize, 1, Nhidden)
            train_model_off_policy(ruleObject, learning_rate, batchSize, offPolicy, num_epochs=50000, model_filename=model_filename, rewardType='GFlow')
        else:
            #model_filename="./data/model/GFlowPol_" + simPart + "_1.pt"
            policyModel = torch.load(model_filename)
            #predicted_graphs = batch_predict_graphs(policyModel, ruleObject, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g, 1000, offPolicy)
            #predicted_graphs = batch_predict_graphs(policyModel, graphSize, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g, 100)
            adjacency_matrices, log_prob_pi, log_prob_prime, trajectory = batchSamplerOurs(policyModel, ruleObject, offPolicy, sampleSize)
            adjProb = []
            if offPolicy:
                adjProb = [ log_prob_pi - log_prob_prime ]
            predicted_graphs = simpleGeneralPredictor(adjacency_matrices, observations_batch, multi_x_given_g, adjProb=adjProb)
    

    if modelType == 'FlowMatch':
        if doTrain:
            #model_filename="./data/model/FlowMatch_" + simPart + "_1.pt"
            train_GFlowNet(learning_rate, num_nodes, num_data_points, observations_batch, 50000, model_filename, rewardType='')
        else:
            #model_filename="./data/model/FlowMatch_" + simPart + "_1.pt"
            policyModel = torch.load(model_filename)
            predicted_graphs = predict_graphs(policyModel, graphSize, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g)
    

    if modelType == 'VAE':
        if doTrain:
            #model_filename="./data/model/VAE_" + simPart + "_1.pt"
            learning_rate = 1e-3
            #learning_rate = 5e-3
            train_graph_vae(ruleObject, Nhidden, graphSize, observations_batch, multi_x_given_g, num_epochs=50000, batch_size=32, learning_rate=learning_rate, model_filename=model_filename)
        else:
            #model_filename="./data/model/VAE_" + simPart + "_1.pt"
            model = torch.load(model_filename)
            #predicted_graphs = VAE_predict(observations_batch, model, multi_x_given_g)
            adjacency_matrices = sampleVAE(model, sampleSize)
            predicted_graphs = simpleGeneralPredictor(adjacency_matrices, observations_batch, multi_x_given_g)

    
    if modelType == 'autoreg':
        if doTrain:
            #model_filename="./data/model/autoReg_" + simPart + "_1.pt"

            learning_rate = 1e-3 #Default
            #learning_rate = 1e-2
            #learning_rate = 1e-4

            #learning_rate = 3e-4
            autoregressive_train_model(Nhidden, ruleObject, observations_batch, learning_rate, model_filename, multi_x_given_g, num_epochs=50000)

        else:
            #model_filename="./data/model/autoReg_" + simPart + "_1.pt"
            model = torch.load(model_filename)
            #predicted_graphs = autoregressive_pred(model, observations_batch, multi_x_given_g)
            adjacency_matrices = sampleAutoreg(model, sampleSize)
            predicted_graphs = simpleGeneralPredictor(adjacency_matrices, observations_batch, multi_x_given_g)

           

    
    if modelType == 'diffusion':
        #TIMESTEPS = 10
        TIMESTEPS = 20
        if doTrain:
            diffusion_train(Nhidden, ruleObject, multi_x_given_g, log_calculate_pr_x_given_g, observations_batch, model_filename, TIMESTEPS)

        else:
            model = torch.load(model_filename)
            #predicted_graphs = diffusion_pred(model, observations_batch, multi_x_given_g, graphSize)
            
            adjacency_matrices = diffusion_sample_graphs(model, TIMESTEPS, batchSize, graphSize)
            adjacency_matrices = adjacency_matrices.data.numpy()
            predicted_graphs = simpleGeneralPredictor(adjacency_matrices, observations_batch, multi_x_given_g)




    if modelType == 'metropolas':
        predicted_graphs = metropolas(graphSize, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g)
        adjacency_matrices = predicted_graphs

    if modelType == 'localSolver':
        predicted_graphs = localSolver(graphSize, observations_batch, multi_x_given_g, log_calculate_pr_x_given_g)
        adjacency_matrices = predicted_graphs



    print (adjacency_matrices.shape)
    inverse1 = uniqueValMaker(adjacency_matrices)
    _, count1 = np.unique(inverse1, return_counts=True)
    print (np.unique(inverse1).shape)


    print (1, np.argwhere(count1 == 1).shape)
    print ('2+', np.argwhere(count1 > 1).shape)

    #plt.plot(np.sum(adjacency_matrices, axis=0))
    #plt.show()

    #plt.hist(count1, bins=100)
    #plt.show()
    #quit()
    

    if not doTrain:

        #print ('./data/pred/graphs/' + modelType + '_' + simPart + '.npz')
        #np.savez_compressed('./data/pred/graphs/' + modelType + '_' + simPart + '.npz', predicted_graphs)
        np.savez_compressed(predFile, predicted_graphs)
        np.savez_compressed(sampleFile, adjacency_matrices)

        #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_' + simPart + '.npz')

        #reward_new = multi_x_given_g( torch.tensor(predicted_graphs).float() , observations_batch)
        #reward_new = reward_new.data.numpy()
        #reward_new = reward_new[np.arange(reward_new.shape[0]), np.arange(reward_new.shape[0])]

        #print ('reward_new', np.mean(reward_new))




        #Fscore = checkFscore(predicted_graphs, adjacency_matrices)
        #print (np.mean(Fscore))
        #error1 = np.mean(np.abs(predicted_graphs - adjacency_matrices), axis=(1, 2)  )
        #print (np.mean(error1))
        #print (scipy.stats.pearsonr(  predicted_graphs.reshape((-1,)),   adjacency_matrices.reshape((-1,)) ))
        #quit()
        True
    



def fullTrainModel():


    doTrain = True
    #doTrain = False

    #np.random.seed(2) #Attempt
    #torch.manual_seed(1)

    
    modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'diffusion'
    #modelType = 'localSolver'
    ####modelType = 'metropolas'

    
    #graphSize = 100
    #graphSize = 90

    simList = [] #data points, nodes, paths
    #simList.append([10, 10, 100])  
    #simList.append([100, 10, 1000])  


    #simList.append([10, 10, 100])  
    #simList.append([10000, 10, 100])  
    #simList.append([1000, 10, 100])  

    #simList.append([1000, 10, 10])  
    simList.append([1000, 10, 100])  
    simList.append([1000, 10, 1000])  
    

    #simList.append([100, 10, 100])  #default 

    #Not used:
    #simList.append([5, 10, 100]) 
    #simList.append([25, 10, 100]) 
    #simList.append([250, 10, 100])



    #simList.append([1, 10, 100])  #modified number of data points 
    #simList.append([10, 10, 100]) 
    #simList.append([1000, 10, 100])


    #simList.append([100, 10, 100])


    #simList.append([100, 5, 100])  #modified number of nodes
    #simList.append([100, 15, 100])  #modified number of nodes
    #simList.append([100, 20, 100])  #modified number of nodes


    #simList.append([100, 15, 10])
    #simList.append([100, 20, 10])


    #simList.append([100, 10, 1])  #modified number of paths
    #simList.append([100, 10, 10])  #modified number of paths
    #simList.append([100, 10, 1000])  #modified number of paths
    #simList.append([100, 10, 10000])  #modified number of paths
    
    

    for simParamIndex in range(0, len(simList)):
        print ('simParamIndex', simParamIndex)
        for simIndex in range(1):# range(2, 3):

            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            num_paths_per_graph =  simList[simParamIndex][2]

            for a in range(5):
                print ('')
            print ('Sim Index ' + str(simIndex))
            for a in range(5):
                print ('')

            
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

            
            print (simPart, modelType)

            
            observations_batch = loadnpz('./data/sims/new/' + simPart + '_obs.npz')
            adjacency_matrices = loadnpz('./data/sims/new/' + simPart + '_graphs.npz')

            print (observations_batch.shape)

            eyeBlank = 1 - np.eye(10)

            #sum1 = np.sum(observations_batch * eyeBlank.reshape((1, 10, 10)), axis=(1, 2)  )
            #print (sum1)
            #quit()

            #print (adjacency_matrices.shape)
            N_edge = np.max(np.sum(adjacency_matrices, axis= (1, 2) ))
            print ('N_edge', N_edge)
            #quit()

            N_edge_max = adjacency_matrices.shape[1] * 3

            #print (observations_batch[0])
            #quit()
            

            graphSize = (num_nodes - 1) * num_nodes
            predFile =   './data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '_temp.npz'
            modelFile = './data/sims/startEnd/model/graph_' + simPart + '_' + modelType + '_temp.pt' #1
            sampleFile = './data/sims/startEnd/samples/graph_' + simPart + '_' + modelType + '_temp.npz'

            multi_x_given_g, log_calculate_pr_x_given_g = sim1_fast_multi, sim1_log_calculate_pr_x_given_g  


            finalProbSize = 1

            ruleObject = gClass()
            def graphRules(graph):
                graphAllow = torch.zeros((graph.shape[0], graphSize+1))
                finalProbAllow = torch.zeros((graph.shape[0], finalProbSize))

                numEdge = np.sum(graphAllow.data.numpy(), axis=1)
                argDone = np.argwhere(numEdge >= N_edge_max)[:, 0]

                graphAllow[argDone, :-1] =  -float('inf')


                return graphAllow, finalProbAllow

            def offPolicyRule(graphList, arange1):

                probList = np.zeros(graphList.shape[0])

                for a in range(probList.shape[0]):
                    graphNow = graphList[a]
                    prob1 = log_calculate_pr_x_given_g(graphNow, observations_batch[arange1[a]]  )
                    probList[a] = prob1

                argNotDone = np.argwhere(probList ==  -float('inf'))[:, 0]
                

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
            


            
            offPolicy = False
            #offPolicy = True
            
            Nhidden = 50
            ruleObject.model = GraphGeneratorNet(ruleObject.graphSize, 1, Nhidden)


            saveTrainingLoss = './data/sims/startEnd/loss/TrainingLoss_' + simPart + '_' + modelType + '.npz'


            generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, sampleFile, doTrain, batchSize, offPolicy, saveTrainingLoss=saveTrainingLoss)



            alsoPred = True
            if doTrain and alsoPred:
                generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, sampleFile, False, batchSize, offPolicy)



#fullTrainModel()
#quit()



#simPart = 'D' + str(1000) +  '_N' + str(10) + '_P' + str(10) + '_sim' + str(0)
#saveTrainingLoss = './data/sims/startEnd/loss/TrainingLoss_' + simPart + '_' + 'ours' + '.npz'
#loss = loadnpz(saveTrainingLoss)

#print (loss)

#plt.plot(loss)
#plt.show()
#quit()



def fullTrainConv():


    doTrain = True
    #doTrain = False

    #np.random.seed(2) #Attempt
    #torch.manual_seed(1)

    
    modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'diffusion'
    #modelType = 'localSolver'
    ####modelType = 'metropolas'

    
    simList = [] #data points, nodes, paths
    simList.append([1000, 100, 0.5])
    




    for simParamIndex in range(0, len(simList)):
        print ('simParamIndex', simParamIndex)
        for simIndex in range(1):

            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            noise_level =  simList[simParamIndex][2]

            for a in range(5):
                print ('')
            print ('Sim Index ' + str(simIndex))
            for a in range(5):
                print ('')

            
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(noise_level) + '_sim' + str(simIndex)

            #simPart = 'fake'

            
            print (simPart, modelType)

            #simpleSet, vector
            observations_batch = loadnpz('./data/sims/convSet/input/' + simPart + '_obs.npz')
            convMatrix = loadnpz('./data/sims/convSet/input/' + simPart + '_convMatrix.npz')
            convMatrix = torch.tensor(convMatrix).float()
            #adjacency_matrices = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')


            
            #graphSize = observations_batch.shape[1]
            graphSize = num_nodes
            predFile =   './data/sims/convSet/pred/graph_' + simPart + '_' + modelType + '.npz' #_strictOffPolicy
            modelFile = './data/sims/convSet/model/graph_' + simPart + '_' + modelType + '.pt' #1#_onPolicy

            #multi_x_given_g, log_calculate_pr_x_given_g = sim1_fast_multi, sim1_log_calculate_pr_x_given_g  


            finalProbSize = 1

            def OLD_log_calculate_pr_x_given_g(graphNow, obs_now):

                #print (graphNow.shape, obs_now.shape)

                diff1 = 0.5 * (obs_now - graphNow) ** 2 

                logProb = -1 * (np.sum(diff1) / (noise_level ** 2))

                return logProb
            

            def multi_x_given_g(adjacency_matrices, obs_matrix):

                with torch.no_grad():
                    adjacency_matrices = torch.matmul(adjacency_matrices.float(), convMatrix)

                    obs_matrix = torch.tensor(obs_matrix).float().to(adjacency_matrices.device)
                    
                    adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))
                    obs_matrix = obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))

                    diff1 = 0.5 * torch.sum((adjacency_matrices - obs_matrix) ** 2 , axis=2)
                    prob_mult = -1 * diff1 / (noise_level ** 2)


                return prob_mult

            
            #def multi_x_given_g(adjacency_matrices, obs_matrix):
                
            #    prob_mult = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]))
            #    return prob_mult

            ruleObject = gClass()
            def graphRules(graph):
                graphAllow = torch.zeros((graph.shape[0], graphSize+1))
                finalProbAllow = torch.zeros((graph.shape[0], finalProbSize))
                return graphAllow, finalProbAllow


            ruleObject.graphRules = graphRules
            #ruleObject.offPolicyRule = offPolicyRule
            ruleObject.adjacency_matrices = observations_batch #TODO this should no longer be needed for setting the size of vectors. 
            ruleObject.multi_x_given_g = multi_x_given_g
            #ruleObject.log_calculate_pr_x_given_g = log_calculate_pr_x_given_g
            ruleObject.graphSize = graphSize
            ruleObject.observations_batch = observations_batch
            ruleObject.batchSize = 1000
            #if observations_batch.shape[0] > ruleObject.batchSize:
            #    ruleObject.batchSize = observations_batch.shape[0]
            batchSize = ruleObject.batchSize


            


            
            Nhidden = 50
            ruleObject.model = GraphGeneratorNet(ruleObject.graphSize, 1, Nhidden, endingBias=0)


            offPolicy = False
            generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, doTrain, batchSize, offPolicy)#, giveTrajectory=giveTrajectory)



            alsoPred = True
            if doTrain and alsoPred:
                generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, False, batchSize, offPolicy)#, giveTrajectory=giveTrajectory)



#fullTrainConv()
#quit()




def fullTrainSet():

    #1415
    #1435
    #22041

    
    doTrain = True
    #doTrain = False

    #np.random.seed(2) #Attempt
    #torch.manual_seed(1)

    
    #modelType = 'ours'

    #modelType = 'ours_offPolicy_2x'
    #modelType = 'ours_offPolicy_05x'
    modelType = 'ours_offPolicy_15x'
    #modelType = 'ours_onPolicy'
    #modelType = 'ours_offPolicy'
    #modelType = 'ours_offPolicy2'
    #modelType = 'naiveReward_offPolicy'
    #modelType = 'naiveReward_onPolicy'
    #modelType = 'GFlowReward_offPolicy'
    #modelType = 'GFlowReward_onPolicy'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'diffusion'
    #modelType = 'localSolver'
    ####modelType = 'metropolas'

    
    simList = [] #data points, nodes, paths
    #simList.append([1000, 100, 0.5])
    #simList.append([1000, 100, 0.25])
    #simList.append([1000, 100, 0.4])

    #simList.append([1000, 100, 0.1])

    #simList.append([100, 100, 0.5])

    #simList.append([100, 100, 0.1])
    #simList.append([100, 100, 0.2])
    #simList.append([100, 100, 0.3])

    simList.append([100, 100, 0.1])
    simList.append([100, 100, 0.2])
    simList.append([100, 100, 0.3])
    simList.append([100, 100, 0.4])
    simList.append([100, 100, 0.5])

    #simList.append([100, 1000, 0.3])



    #simList.append([100, 100, 0.75])
    #simList.append([100, 100, 1.0])

    #simList.append([100, 10, 0.5])
    #simList.append([100, 1000, 0.3])
    #simList.append([100, 1000, 0.5])


    #simList.append([10000, 100, 0.5])
    #simList.append([1, 100, 0.5])


    #simList.append([10, 100, 0.5])


    #simList.append([1000, 100, 1.0])
    #simList.append([100, 100, 1.0])
    #simList.append([100, 100, 1.0])
    #simList.append([100, 1000, 0.5])
    




    for simParamIndex in range(0, len(simList)):
        print ('simParamIndex', simParamIndex)
        for simIndex in range(1):
            
            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            noise_level =  simList[simParamIndex][2]

            for a in range(5):
                print ('')
            print ('Sim Index ' + str(simIndex))
            for a in range(5):
                print ('')

            
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(noise_level) + '_sim' + str(simIndex)

            #simPart = 'fake'

            
            print (simPart, modelType)

            #simpleSet, vector
            observations_batch = loadnpz('./data/sims/simpleSet/input/' + simPart + '_obs.npz')
            adjacency_matrices = loadnpz('./data/sims/simpleSet/input/' + simPart + '_latent.npz')

            #print (adjacency_matrices.shape)
            #inverse1 = uniqueValMaker(adjacency_matrices)
            #print (np.unique(inverse1).shape)
            #quit()

            N_edge_max = np.max(np.sum( observations_batch, axis=0 ))

            #print (N_edge_max)
            #quit()

            
            graphSize = observations_batch.shape[1]
            predFile =   './data/sims/simpleSet/pred/graph_' + simPart + '_' + modelType + '.npz' #_strictOffPolicy
            modelFile = './data/sims/simpleSet/model/graph_' + simPart + '_' + modelType + '.pt' #1#_onPolicy
            sampleFile = './data/sims/simpleSet/sample/graph_' + simPart + '_' + modelType + '.pt' #1#_onPolicy

            #multi_x_given_g, log_calculate_pr_x_given_g = sim1_fast_multi, sim1_log_calculate_pr_x_given_g  


            finalProbSize = 1

            def log_calculate_pr_x_given_g(graphNow, obs_now):

                #print (graphNow.shape, obs_now.shape)

                diff1 = 0.5 * (obs_now - graphNow) ** 2 

                logProb = -1 * (np.sum(diff1) / (noise_level ** 2))

                return logProb
            

            def multi_x_given_g(adjacency_matrices, obs_matrix):

                #print (adjacency_matrices.shape, obs_matrix.shape)
                #quit()

                
                
                if True:
                    obs_matrix = torch.tensor(obs_matrix).float().to(adjacency_matrices.device)
                    
                    adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))
                    obs_matrix = obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))

                    diff1 = 0.5 * torch.sum((adjacency_matrices - obs_matrix) ** 2 , axis=2)
                    prob_mult = -1 * diff1 / (noise_level ** 2) 


                if False:

                    obs_matrix = torch.tensor(obs_matrix).float().to(adjacency_matrices.device)
                    adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))
                    obs_matrix = obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))
                    obs_matrix[obs_matrix==2] = 10

                    diff1 = adjacency_matrices - obs_matrix
                    diff1[diff1 <= -2] = 0
                    diff1 = torch.sum(torch.abs(diff1), axis=2)

                    prob_mult = -10 * diff1

                    if False:
                        prob_mult[prob_mult <= -1] = -1000#-float('inf')
                

                return prob_mult

            
            #def multi_x_given_g(adjacency_matrices, obs_matrix):
                
            #    prob_mult = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]))
            #    return prob_mult

            ruleObject = gClass()
            def graphRules(graph):
                graphAllow = torch.zeros((graph.shape[0], graphSize+1))
                finalProbAllow = torch.zeros((graph.shape[0], finalProbSize))
                return graphAllow, finalProbAllow

            def offPolicyRule(graphList, arange1):

                #print ('off policy')
                #print (graphList[0])

                

                if True:

                    #print (arange1.shape[0])
                    #arange1 = arange1[arange1 < observations_batch.shape[0]]
                    arange1 = arange1 % (2 * observations_batch.shape[0])
                    subsetGood = np.argwhere(arange1 < observations_batch.shape[0])[:, 0]
                    arange1 = arange1[subsetGood]

                    
                    graphAllow = torch.zeros((graphList.shape[0], graphSize+1))
                    obsList = np.copy(observations_batch[arange1])
                    probRatio = ((obsList - 1) ** 2) - (obsList ** 2)
                    probRatio = probRatio * 0.5
                    #noise_level_mod = ((1.0 - noise_level) ** 3.0) / (noise_level ** 2)
                    noise_level_mod = 1.0 / (noise_level ** 2)
                    if noise_level_mod < 0:
                        noise_level_mod = 0.0
                    #probRatio = probRatio / (noise_level ** 2)
                    probRatio = probRatio * noise_level_mod
                    probRatio = probRatio * -1

                    if '2x' in modelType:
                        probRatio = probRatio * 2
                    if '05x' in modelType:
                        probRatio = probRatio * 0.5
                    if '15x' in modelType:
                        probRatio = probRatio * 1.5

                    #graphAllow[:, :-1] = torch.tensor(probRatio).float() 
                    graphAllow[subsetGood, :-1] = torch.tensor(probRatio).float() 

                    #graphAllow = graphAllow * 0.5 #Smaller


                if False:
                    graphAllow = torch.zeros((graphList.shape[0], graphSize+1))
                    obsList = np.copy(observations_batch[arange1])
                    probRatio = ((obsList - 1) ** 2) - (obsList ** 2)
                    probRatio = probRatio * 0.5
                    probRatio = probRatio / (noise_level ** 2)
                    probRatio = probRatio * -1
                    max1 = np.max(probRatio, axis=1)

                    #probRatio = probRatio + 1 #TODO Remove
                    probRatio[probRatio > 0] = 0
                    #probRatio[probRatio > 1] = 1
                    graphAllow[:, :-1] = torch.tensor(probRatio).float()
                    graphAllow[:, -1] = -1 * torch.tensor(max1).float()


                    #graphAllow = graphAllow * 0.2


                if False:


                    probMult = multi_x_given_g(graphList, np.copy(observations_batch))
                    probMult = nn.LogSoftmax(dim=1)(probMult)


                    graphAllow = torch.zeros((graphList.shape[0], graphSize+1))
                    #obsList = np.copy(observations_batch[arange1])
                    probRatio = ((observations_batch - 1) ** 2) - (observations_batch ** 2)
                    probRatio = probRatio * 0.5
                    probRatio = probRatio / (noise_level ** 2)
                    probRatio = probRatio * -1
                    probRatio = torch.tensor(probRatio)

                    #probRatio = probRatio - torch.logaddexp(probRatio , torch.zeros(probRatio.shape))


                    probMult = probMult.reshape((probMult.shape[0], probMult.shape[1], 1))
                    probRatio = probRatio.reshape((1, probRatio.shape[0], probRatio.shape[1] ))

                    probSum = torch.logsumexp(probMult + probRatio  , axis=1  )

                    graphAllow[:, :-1] = probSum


                if False:

                    graphAllow = torch.zeros((graphList.shape[0], graphSize+1))

                    obsMean = np.mean(observations_batch, axis=0)

                    probRatio = ((obsMean - 1) ** 2) - (obsMean ** 2)
                    probRatio = probRatio * 0.5
                    probRatio = probRatio / (noise_level ** 2)
                    probRatio = probRatio * -1
                    probRatio = torch.tensor(probRatio)


                    graphAllow[:, :-1] = graphAllow[:, :-1] + probRatio.reshape((1, -1))
                
                if False:

                    graphAllow = torch.zeros((graphList.shape[0], graphSize+1))

                    obsList = np.copy(observations_batch[arange1])
                    missing = np.copy(obsList)
                    missing[graphList == 1] = 0
                    missing[missing!=1] = 0
                    missing = np.sum(missing, axis=1)
                    
                    

                    graphAllow_mini = torch.zeros((graphList.shape[0], graphSize))
                    graphAllow_mini[obsList == 0] = -float('inf')
                    graphAllow[:, :-1] = graphAllow_mini

                    graphAllow[missing>=1, -1] = -float('inf')


                
                if False:

                    graphAllow = torch.zeros((graphList.shape[0], graphSize+1))

                    obsList = np.copy(observations_batch[arange1])
                    missing = np.copy(obsList)
                    missing[graphList == 1] = 0
                    missing[missing!=1] = 0
                    missing = np.sum(missing, axis=1)
                    
                    

                    graphAllow_mini = torch.zeros((graphList.shape[0], graphSize))
                    graphAllow_mini[obsList == 0] = -float('inf')
                    graphAllow_mini[obsList == 1] = 5


                finalProbAllow = torch.zeros((graphList.shape[0], finalProbSize))
                return graphAllow, finalProbAllow


            ruleObject.graphRules = graphRules
            ruleObject.offPolicyRule = offPolicyRule
            ruleObject.adjacency_matrices = observations_batch #TODO Change!!
            ruleObject.multi_x_given_g = multi_x_given_g
            ruleObject.log_calculate_pr_x_given_g = log_calculate_pr_x_given_g
            ruleObject.graphSize = graphSize
            ruleObject.observations_batch = observations_batch
            #ruleObject.batchSize = 1100
            #ruleObject.batchSize = 2000
            ruleObject.batchSize = 1000
            #ruleObject.batchSize = 10000
            #if observations_batch.shape[0] > ruleObject.batchSize:
            #    ruleObject.batchSize = observations_batch.shape[0]
            batchSize = ruleObject.batchSize


            


            
            Nhidden = 50
            ruleObject.model = GraphGeneratorNet(ruleObject.graphSize, 1, Nhidden, endingBias=0)


            
            offPolicy = False
            #offPolicy = True
            #giveTrajectory = False

            if 'onPolicy' in modelType:
                offPolicy = False
            if 'offPolicy' in modelType:
                offPolicy = True




            print ('offPolicy', offPolicy)
            
            modelTypePart = modelType.split('_')[0]
            
            generic_TrainModel(ruleObject, simPart, modelTypePart, predFile, modelFile, sampleFile, doTrain, batchSize, offPolicy)#, giveTrajectory=giveTrajectory)



            alsoPred = True
            if doTrain and alsoPred:
                generic_TrainModel(ruleObject, simPart, modelTypePart, predFile, modelFile, sampleFile, False, batchSize, offPolicy)#, giveTrajectory=giveTrajectory)



fullTrainSet()
quit()



def fullTrainVector():


    doTrain = True

    modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'diffusion'
    #modelType = 'localSolver'
    ####modelType = 'metropolas'

    
    simList = [] 
    #simList.append([100, 1000, 0.5])

    #simList.append([100, 100, 0.5])
    #simList.append([100, 50, 0.5])
    simList.append([100, 25, 0.5])
    #simList.append([100, 10, 0.5])
    
    

    offPolicy = False
    #offPolicy = True

    nameAdder = '_onPolicy'
    if offPolicy:
        nameAdder = '_offPolicy'



    for simParamIndex in range(0, len(simList)):
        print ('simParamIndex', simParamIndex)
        for simIndex in range(1):

            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            noise_level =  simList[simParamIndex][2]

            for a in range(5):
                print ('')
            print ('Sim Index ' + str(simIndex))
            for a in range(5):
                print ('')

            
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(noise_level) + '_sim' + str(simIndex)            
            print (simPart, modelType)

            #simpleSet, vector
            observations_batch = loadnpz('./data/sims/vector/input/' + simPart + '_obs.npz')
            
            graphSize = observations_batch.shape[1]
            predFile =   './data/sims/vector/pred/graph_' + simPart + '_' + modelType + nameAdder + '.npz' #_strictOffPolicy
            modelFile = './data/sims/vector/model/graph_' + simPart + '_' + modelType + nameAdder + '.pt' #1#_onPolicy


            finalProbSize = 1

            def log_calculate_pr_x_given_g(graphNow, obs_now):

                #print (graphNow.shape, obs_now.shape)

                diff1 = 0.5 * (obs_now - graphNow) ** 2 

                logProb = -1 * (np.sum(diff1) / (noise_level ** 2))

                return logProb
            

            def multi_x_given_g(adjacency_matrices, obs_matrix):
                
                obs_matrix = torch.tensor(obs_matrix).float().to(adjacency_matrices.device)
                adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))
                obs_matrix = obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))
                obs_matrix[obs_matrix==2] = 10

                diff1 = adjacency_matrices - obs_matrix
                diff1[diff1 <= -2] = 0
                diff1 = torch.sum(torch.abs(diff1), axis=2)

                prob_mult = -10 * diff1

                if True:
                    prob_mult[prob_mult <= -1] = -1000#-float('inf')

                return prob_mult


            ruleObject = gClass()
            def graphRules(graph):
                graphAllow = torch.zeros((graph.shape[0], graphSize+1))
                finalProbAllow = torch.zeros((graph.shape[0], finalProbSize))
                return graphAllow, finalProbAllow

            def offPolicyRule(graphList, arange1):

                
                graphAllow = torch.zeros((graphList.shape[0], graphSize+1))

                obsList = np.copy(observations_batch[arange1])
                missing = np.copy(obsList)
                missing[graphList == 1] = 0
                missing[missing!=1] = 0
                missing = np.sum(missing, axis=1)
                
                

                graphAllow_mini = torch.zeros((graphList.shape[0], graphSize))
                graphAllow_mini[obsList == 0] = -float('inf')
                graphAllow[:, :-1] = graphAllow_mini

                graphAllow[missing>=1, -1] = -float('inf')


                finalProbAllow = torch.zeros((graphList.shape[0], finalProbSize))
                return graphAllow, finalProbAllow


            ruleObject.graphRules = graphRules
            ruleObject.offPolicyRule = offPolicyRule
            ruleObject.adjacency_matrices = observations_batch #TODO Change!!
            ruleObject.multi_x_given_g = multi_x_given_g
            ruleObject.log_calculate_pr_x_given_g = log_calculate_pr_x_given_g
            ruleObject.graphSize = graphSize
            ruleObject.observations_batch = observations_batch
            ruleObject.batchSize = 1000
            batchSize = ruleObject.batchSize

            Nhidden = 50
            ruleObject.model = GraphGeneratorNet(ruleObject.graphSize, 1, Nhidden, endingBias=0)

            print (ruleObject.model)


            print ('offPolicy', offPolicy)
            
            generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, doTrain, batchSize, offPolicy)#, giveTrajectory=giveTrajectory)



            alsoPred = True
            if doTrain and alsoPred:
                generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, False, batchSize, offPolicy)#, giveTrajectory=giveTrajectory)



fullTrainVector()
quit()



def fullTrainTemporal():


    doTrain = True
    #doTrain = False

    #np.random.seed(2) #Attempt
    #torch.manual_seed(1)

    
    modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'diffusion'
    #modelType = 'localSolver'
    ####modelType = 'metropolas'

    

    simList = [] #data points, nodes, paths
    #simList.append([100, 100, 20])  #modified number of nodes
    simList.append([1000, 50, 20])
    #simList.append([1000, 10, 10])

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

            
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

            #simPart = 'fake'

            
            print (simPart, modelType)

            
            observations_batch = loadnpz('./data/sims/temporal/' + simPart + '_obs.npz')
            #adjacency_matrices = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')

            N_edge_max = np.max(np.sum( observations_batch, axis=0 ))

            #print (N_edge_max)
            #quit()

            
            graphSize = observations_batch.shape[1]
            predFile =   './data/sims/temporal/pred/graph_' + simPart + '_' + modelType + '_mod.npz'
            modelFile = './data/sims/temporal/model/graph_' + simPart + '_' + modelType + '_mod.pt' #1

            #multi_x_given_g, log_calculate_pr_x_given_g = sim1_fast_multi, sim1_log_calculate_pr_x_given_g  


            finalProbSize = 1

            def log_calculate_pr_x_given_g(graphNow, obs_now):

                #print (graphNow.shape, obs_now.shape)
                mult1 = graphNow * obs_now
                #print (mult1.shape)
                #print (type(mult1))
                min1 = np.min(mult1)#, axis=1)
                max1 = np.max(mult1)#, axis=1)
                sum1 = np.sum(obs_now)#, axis=1)

                #perfect1 = np.zeros(graphNow.shape, dtype=int)
                #perfect1[ (sum1 - max1) == 1 ] = 1
                #perfect1[min1 == -1] = 0

                #print (np.sum(perfect1))

                if (sum1 - 1 == max1) and (min1 != -1):
                    
                    logProb = 0
                else:

                    argPred = np.argwhere(graphNow < sum1 )[:, 0]
                    argTrue = np.argwhere(obs_now== 1 )[:, 0]

                    intersect1 = np.intersect1d(argPred, argTrue)

                    error1 = (argPred.shape[0] - intersect1.shape[0]) + (argTrue.shape[0] - intersect1.shape[0])

                    logProb = -10 * (error1 + 1)

                return logProb

            
            #def multi_x_given_g(adjacency_matrices, obs_matrix):
                
            #    prob_mult = torch.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]))
            #    return prob_mult

            ruleObject = gClass()
            def graphRules(graph):
                graphAllow = torch.zeros((graph.shape[0], graphSize+1))
                finalProbAllow = torch.zeros((graph.shape[0], finalProbSize))

                numEdge = np.sum(graph.data.numpy(), axis=1)
                argDone = np.argwhere(numEdge >= N_edge_max)[:, 0]

                #graphAllow[argDone, :-1] =  -float('inf')

                


                return graphAllow, finalProbAllow

            def offPolicyRule(graphList, arange1):

                #print ('off policy')
                #print (graphList[0])

                graphAllow = torch.zeros((graphList.shape[0], graphSize+1))
                graphAllow[:] = -float('inf')

                graphList_np = graphList.data.numpy()
                obsList = np.copy(observations_batch[arange1])

                #print ('graphList', np.argwhere(graphList_np[0] == 1)[:, 0] )
                #print ('obsList', np.argwhere(obsList[0] == 1)[:, 0])

                max1 = np.sum(graphList_np, axis=1)
                max2 = np.sum(obsList, axis=1)

                argBefore = np.argwhere(max1 - max2 < 0)[:, 0]
                argAfter = np.argwhere(max1 - max2 >= 0)[:, 0]
                graphAllow[argAfter, :] = 0

                #print (0 in argAfter)

                
                graphAllow_mini = np.copy(graphAllow[:, :-1])
                graphAllow_mini[obsList == 1] = 0
                #graphAllow_mini[argAfter, :] = 0

                graphAllow_mini[graphList_np == 1] =  -float('inf')
                graphAllow[:, :-1] = torch.tensor(graphAllow_mini)

                max1 = np.max(graphAllow_mini, axis=1)
                #graphAllow[max1 != 0, -1] = 0


                argIssue1 = np.argwhere(max1 != 0)[:, 0]
                min2 = np.min(graphList_np, axis=1)
                argIssue1 = argIssue1[min2[argIssue1] == 0]
                if argIssue1.shape[0] >= 1:
                    print (graphList[argIssue1[0]])
                    print (obsList[argIssue1[0]])
                    print (graphAllow[argIssue1[0]])
                    quit()

                #print ('graphAllow', np.argwhere(graphAllow[0] == 0))

                #if np.max(graphList_np) == 1:
                #    quit()

                if False:



                    for a in range(graphList.shape[0]):
                        graphNow = graphList_np[a]
                        obs1 = observations_batch[arange1[a]]

                        #print (obs1.shape, graphNow.shape)

                        argAllow = np.argwhere(np.logical_and(obs1 == 1, graphNow == 0 ))
                        #print (graphNow)
                        #print (obs1)
                        #print (argAllow)

                        if argAllow.shape[0] >= 1:
                            #print (argAllow)
                            graphAllow[a, argAllow[:, 0]  ] = 0
                        else:
                            graphAllow[a, -1] = 0

                        assert torch.max(graphAllow[a]) == 0


                finalProbAllow = torch.zeros((graphList.shape[0], finalProbSize))
                return graphAllow, finalProbAllow


            ruleObject.graphRules = graphRules
            ruleObject.offPolicyRule = offPolicyRule
            ruleObject.adjacency_matrices = observations_batch #TODO Change!!
            #ruleObject.multi_x_given_g = multi_x_given_g
            ruleObject.log_calculate_pr_x_given_g = log_calculate_pr_x_given_g
            ruleObject.graphSize = graphSize
            ruleObject.observations_batch = observations_batch
            ruleObject.batchSize = 1000
            batchSize = ruleObject.batchSize
            
            Nhidden = 50
            ruleObject.model = GraphGeneratorNet(ruleObject.graphSize, 1, Nhidden, endingBias=0)


            
            #offPolicy = False
            offPolicy = True
            giveTrajectory = True
            


            generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, doTrain, batchSize, offPolicy, giveTrajectory=giveTrajectory)



            alsoPred = True
            if doTrain and alsoPred:
                generic_TrainModel(ruleObject, simPart, modelType, predFile, modelFile, False, batchSize, offPolicy, giveTrajectory=giveTrajectory)



#fullTrainTemporal()
#quit()