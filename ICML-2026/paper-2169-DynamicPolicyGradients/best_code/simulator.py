import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
import time

import math
import hashlib

# Parameters
#num_simulations = 20
#num_data_points = 100
#num_nodes = 10  # Reduced to 10 nodes for faster runtimes
#num_nodes = 3
#num_paths_per_graph = 10


from sharedGen import *

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

np.random.seed(1)
torch.manual_seed(0)


def seed_from_three(a, b, c):
    s = f"{a},{b},{c}".encode("utf-8")
    h = hashlib.sha256(s).digest()
    seed = int.from_bytes(h[:8], "big")  # 64-bit seed
    seed = seed % (2**32)
    #seed = seed % (2**10)
    return seed


def observationSampler(adjacency_matrix, num_paths_per_graph, num_nodes, doMatrix=False):

    if doMatrix:
        observations = np.zeros((num_nodes, num_nodes), dtype=int)
    else:
        observations = []
    
    for _ in range(num_paths_per_graph):
        start_node = np.random.choice(num_nodes)
        current_node = start_node
        
        while True:
            out_edges = np.where(adjacency_matrix[current_node] == 1)[0]
            total_options = len(out_edges) + 1
            choice = np.random.choice(np.append(-1, out_edges), p=[1/total_options] + [1/total_options]*len(out_edges))
            
            if choice == -1:
                if doMatrix:
                    observations[start_node, current_node] += 1
                else:
                    observations.append((start_node, current_node))
                    
                break
            else:
                current_node = choice

    return observations

# Function to generate a single simulation instance
def generate_simulation_instance(num_nodes, num_data_points, num_paths_per_graph, doMatrix=False):

    #probNoEdge = 1.0 - (2.0 / float(num_nodes - 1))

    #probNoEdge = 1.0 - (2.0 / float(num_nodes - 1))
    #probNoEdge = 0.75
    probNoEdge = 0.5
    #probNoEdge = 0.0

    if False:
        thresholds = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:             
                    delta = 0 if np.random.rand() < probNoEdge else np.random.uniform(0, 1)  # 90% chance of threshold 0
                    thresholds[i, j] = delta
    else:
        #NbaseGraph = 20
        NbaseGraph = 1
        thresholds = np.zeros((NbaseGraph, num_nodes, num_nodes))
        for graph_index in range(thresholds.shape[0]):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:             
                        delta = 0 if np.random.rand() < probNoEdge else np.random.uniform(0, 1)  # 90% chance of threshold 0

                        if delta != 0: #TODO Remove 
                            delta = 0.25 + (0.75 * delta)

                        thresholds[graph_index, i, j] = delta



    adjacency_matrices = np.zeros((num_data_points, num_nodes, num_nodes))
    

    if True:
        a_values = np.random.random(adjacency_matrices.shape[0])
        #a_values = np.floor(np.random.random(adjacency_matrices.shape[0]) * 5).astype(float) / 5
        #a_values[:] = 0.0001 #TODO REMOVE TEST



        randomSelection = np.random.randint(NbaseGraph, size=adjacency_matrices.shape[0])
        a_values_mod = thresholds[randomSelection] - a_values.reshape((-1, 1, 1))
    if False:
        a_values = np.random.random(adjacency_matrices.shape[0])
        a_values_mod = thresholds.reshape((1, num_nodes, num_nodes)) - a_values.reshape((-1, 1, 1))
    if False:
        a_values = np.random.random(adjacency_matrices.shape)
        a_values_mod = thresholds.reshape((1, num_nodes, num_nodes)) - a_values
    
    adjacency_matrices[a_values_mod > 0] = 1

    observations_list = np.zeros((num_data_points, num_nodes, num_nodes))
    #observations_list = []
    

    for datapoint_index in range(num_data_points):

        adjacency_matrix = adjacency_matrices[datapoint_index]
        
        observations = observationSampler(adjacency_matrix, num_paths_per_graph, num_nodes, doMatrix=doMatrix)
        
        
        observations_list[datapoint_index] = observations
        #observations_list.append(observations)

    

    #argsort1 = np.argsort(np.sum(adjacency_matrices, axis=(1, 2)))

    #for a in range(argsort1.shape[0]):
    #    plt.imshow(adjacency_matrices[argsort1[a]])
    #    plt.show()
    #quit()



    return np.array(adjacency_matrices), observations_list, thresholds




def original_generateSims():

    #np.random.seed(1) #num_data_points = 100
    #torch.manual_seed(0) #num_data_points = 100

    

    simList = [] #data points, nodes, paths
    #simList.append([1000, 10, 10])
    #simList.append([1000, 10, 100])
    
    
    #simList.append([10, 10, 100])
    simList.append([10000, 10, 100])



    #simList.append([100, 10, 10])  #default 

    #simList.append([1, 10, 10])  #modified number of data points 
    #simList.append([5, 10, 10]) 
    #simList.append([10, 10, 10]) 
    #simList.append([25, 10, 10]) 
    #simList.append([250, 10, 10])
    #simList.append([1000, 10, 10])


    #simList.append([100, 5, 10])  #modified number of nodes
    #simList.append([100, 15, 10])  #modified number of nodes
    #simList.append([100, 20, 10])  #modified number of nodes


    #simList.append([1000, 10, 1])  #modified number of paths
    #simList.append([1000, 10, 10])  #modified number of paths
    #simList.append([1000, 10, 100])  #modified number of paths
    #simList.append([1000, 10, 1000])  #modified number of paths
    #simList.append([1000, 10, 10000])  #modified number of paths



    #simList.append([100, 5, 100])
    #simList.append([100, 15, 100])
    #simList.append([100, 20, 100])


    #simList.append([1, 10, 100])  #modified number of data points 
    #simList.append([5, 10, 100]) 
    #simList.append([10, 10, 100]) 
    #simList.append([25, 10, 100]) 
    #simList.append([250, 10, 100])
    #simList.append([1000, 10, 100])



    for simParamIndex in range(len(simList)):
        print ('simParamIndex', simParamIndex)


        seed1 = seed_from_three(simList[simParamIndex][0], simList[simParamIndex][1], simList[simParamIndex][2])
        np.random.seed(seed1) #num_data_points = 100
        torch.manual_seed(seed1) #num_data_points = 100


        for simIndex in range(10):

            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            num_paths_per_graph =  simList[simParamIndex][2]
            
            #simPart = 'N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_' + str(simIndex)
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

            # Generate a single simulation instance
            adjacency_matrices, observations_batch, thresholds = generate_simulation_instance(num_nodes, num_data_points, num_paths_per_graph, doMatrix=True)

            print (thresholds)

            np.savez_compressed('./data/sims/new/' + simPart + '_obs.npz', observations_batch)
            np.savez_compressed('./data/sims/new/' + simPart + '_graphs.npz', adjacency_matrices)
            np.savez_compressed('./data/sims/new/' + simPart + '_thresholds.npz', thresholds)


#original_generateSims()
#quit()


def generateTemporalSim(num_data_points, num_nodes, DAG_size):

    #import networkx 

    #tree1 = networkx.random_tree(DAG_size)
    #tree1 = list(tree1.edges())
    #print (tree1)
    #quit()
    #4

    DAG_matrix0 = np.random.random( DAG_size * DAG_size )
    #cutoff =  (4.0 / float(DAG_size))
    #cutoff =  (2.0 / float(DAG_size))

    #cutoff =  (2.0 / 5.0)
    cutoff =  (2.0 / 4.0)
    cutoff = 1 - cutoff
    DAG_matrix0[DAG_matrix0 > cutoff ] = 1
    DAG_matrix0[DAG_matrix0 < cutoff ] = 0
    DAG_matrix0 = DAG_matrix0.reshape( (DAG_size , DAG_size) )
    argAll = np.argwhere(DAG_matrix0 > -1)
    #print (argAll.shape)
    argAll = argAll[argAll[:, 1] - argAll[:, 0] <= 4]
    #print (argAll.shape)
    argAll = argAll[argAll[:, 1] - argAll[:, 0] >= 1]
    

    DAG_matrix = np.zeros(DAG_matrix0.shape)
    DAG_matrix[argAll[:, 0], argAll[:, 1]] = DAG_matrix0[argAll[:, 0], argAll[:, 1]]
    
    nodesCorrespond = np.random.randint(DAG_size, size=num_nodes)
    while np.unique(nodesCorrespond).shape[0] < DAG_size:
        nodesCorrespond = np.random.randint(DAG_size, size=num_nodes)

    DAG_big = DAG_matrix[nodesCorrespond][:, nodesCorrespond]

    argStart = np.argwhere(  np.sum(DAG_big, axis=0)  == 0)[:, 0]

    listAll = np.zeros(( num_data_points, num_nodes  ), dtype=int) - 1
    randomViews = np.zeros(( num_data_points, num_nodes  ), dtype=int)# - 1

    indexNow = 0
    while indexNow < num_data_points:
        list1 = []

        start1 = argStart[np.random.randint(argStart.shape[0])]
        list1.append(start1)
        #print (DAG_big[start1])
        while np.sum(DAG_big[start1]) != 0:
            argNow = np.argwhere( DAG_big[start1] == 1 )[:, 0]
            start1 = argNow[np.random.randint(argNow.shape[0])]
            list1.append(start1)
        #print (list1)

        if len(list1) >= 2:
            list1 = np.array(list1)
            view1 = list1[:np.random.randint(list1.shape[0])+1]

            listAll[indexNow][:list1.shape[0]] = list1
            randomViews[indexNow][view1] = 1

            indexNow += 1
            #print (list1)
            #quit()

    return DAG_big, listAll, randomViews



def temporalSim():
    np.random.seed(2) #num_data_points = 100
    torch.manual_seed(2) #num_data_points = 100

    simList = [] #data points, set size, DAG size
    #simList.append([100, 100, 20])  #default 
    #simList.append([1000, 100, 20])  #default 
    #simList.append([1000, 50, 20])
    simList.append([1000, 10, 10])

    for simParamIndex in range(len(simList)):
        print ('simParamIndex', simParamIndex)
        for simIndex in range(10):

            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            DAG_size =  simList[simParamIndex][2]
            
            ############simPart = 'N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_' + str(simIndex)
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(DAG_size) + '_sim' + str(simIndex)

            # Generate a single simulation instance
            #adjacency_matrices, observations_batch, thresholds = generate_simulation_instance(num_nodes, num_data_points, num_paths_per_graph, doMatrix=True)
            DAG_big, listAll, randomViews = generateTemporalSim(num_data_points, num_nodes, DAG_size)


            np.savez_compressed('./data/sims/temporal/' + simPart + '_obs.npz', randomViews)
            np.savez_compressed('./data/sims/temporal/' + simPart + '_DAG.npz', DAG_big)
            np.savez_compressed('./data/sims/temporal/' + simPart + '_lists.npz', listAll)



#randomViews = np.eye(10)
#randomViews = np.cumsum(randomViews, axis=1)[:, -1::-1][-1::-1]
#print (randomViews)


#np.savez_compressed('./data/sims/temporal/' + 'fake' + '_obs.npz', randomViews)
#quit()



#temporalSim()
#quit()


def generateSetSim(num_data_points, num_nodes, DAG_size, noise_size, doMask=False):


    DAG_matrix0 = np.random.random( DAG_size * DAG_size )
    #cutoff =  1 - (4.0 / float(DAG_size))

    #cutoff =  (4.0 / float(DAG_size))
    #cutoff = 0.25
    cutoff = 0.1
    
    
    cutoff = 1.0 - cutoff
    #cutoff =  (2.0 / float(DAG_size))

    #cutoff =  (2.0 / 5.0)
    #cutoff =  (2.0 / 4.0)
    #cutoff = 1 - cutoff
    DAG_matrix0[DAG_matrix0 > cutoff ] = 1
    DAG_matrix0[DAG_matrix0 < cutoff ] = 0
    DAG_matrix0 = DAG_matrix0.reshape( (DAG_size , DAG_size) )
    argAll = np.argwhere(DAG_matrix0 > -1)
    #print (argAll.shape)
    #argAll = argAll[argAll[:, 1] - argAll[:, 0] <= 4]
    #print (argAll.shape)
    argAll = argAll[argAll[:, 1] - argAll[:, 0] >= 1]
    

    DAG_matrix = np.zeros(DAG_matrix0.shape)
    DAG_matrix[argAll[:, 0], argAll[:, 1]] = DAG_matrix0[argAll[:, 0], argAll[:, 1]]

    

    if DAG_size == num_nodes:
        nodesCorrespond = np.random.permutation(DAG_size)
    else:

        nodesCorrespond = np.random.randint(DAG_size, size=num_nodes)
        while np.unique(nodesCorrespond).shape[0] < DAG_size:
            nodesCorrespond = np.random.randint(DAG_size, size=num_nodes)

    #DAG_big = DAG_matrix[nodesCorrespond][:, nodesCorrespond]

    argStart = np.argwhere(  np.sum(DAG_matrix, axis=0)  == 0)[:, 0]

    listAll = np.zeros(( num_data_points, num_nodes  ), dtype=int) - 1
    randomViews = np.zeros(( num_data_points, num_nodes  ), dtype=int)# - 1

    
    for indexNow in range(num_data_points):
        list1 = []

        start1 = argStart[np.random.randint(argStart.shape[0])]
        list1.append(start1)
        #print (DAG_big[start1])
        while np.sum(DAG_matrix[start1]) != 0:
            argNow = np.argwhere( DAG_matrix[start1] == 1 )[:, 0]
            start1 = argNow[np.random.randint(argNow.shape[0])]
            list1.append(start1)

        list1 = np.array(list1)
        list2 = np.argwhere( np.isin(nodesCorrespond, list1) )[:, 0]
        listAll[indexNow][:list2.shape[0]] = list2
        randomViews[indexNow, list2] = 1


    #plt.plot(np.sum(randomViews,axis=0))
    #plt.plot(np.sum(DAG_matrix, axis=0))
    #plt.plot(np.sum(DAG_matrix, axis=1))
    #plt.show()
    #quit()




    if doMask:

        randomMask = np.random.randint(2, size=randomViews.size).reshape(randomViews.shape)
        randomViews[randomMask == 1] = 2

    else:
        randomNoise = np.random.normal(size=randomViews.size).reshape(randomViews.shape) * noise_size
        randomViews = randomViews + randomNoise

    return DAG_matrix, nodesCorrespond, listAll, randomViews




def generateSetSim2(num_data_points, num_nodes, DAG_size, noise_size, doMask=False):

    if False:
        probNoEdge = 1.0 - (2.0 / float(10 - 1))
        thresholds = np.zeros(num_nodes)
        for i in range(num_nodes):
            delta = 0 if np.random.rand() < probNoEdge else np.random.uniform(0, 1)  # 90% chance of threshold 0
            thresholds[i] = delta
        adjacency_matrices = np.zeros((num_data_points, num_nodes))
        a_values = np.random.random(adjacency_matrices.shape[0])
        a_values_mod = thresholds.reshape((1, num_nodes)) - a_values.reshape((-1, 1))
        adjacency_matrices[a_values_mod > 0] = 1

    
    if False:

        perm1 = np.random.permutation(num_nodes)
        num_nodes0 = 20
        perm2 = perm1 // ( num_nodes // num_nodes0  )
        randomMini = np.random.randint(2, size=num_nodes0 * num_data_points ).reshape(( num_data_points, num_nodes0 ))
        adjacency_matrices = randomMini[:, perm2]
        thresholds = 0

    if False:

        import seaborn as sns

        numChunk = 10
        chunkSize = 25
        randomPart = np.random.randint(2, size= num_nodes * numChunk ).reshape(( numChunk, num_nodes ))
        adjacency_matrices = np.zeros( (num_data_points , num_nodes) )

        for a in range(num_nodes // chunkSize):
            arg1 = np.arange(chunkSize) + (chunkSize * a)
            rand1 = np.random.randint(10, size=num_data_points)
            adjacency_matrices[:, arg1] = randomPart[rand1][:, arg1]

            #sns.clustermap(adjacency_matrices[:, arg1], col_cluster=False)
            #plt.show()
        #quit()
        thresholds = 0


    if True:
        nodeRootFloat = float(num_nodes) ** 0.5
        nodesRootInt = int(np.floor(nodeRootFloat))

        dictionarySize = nodesRootInt
        
        dictionary = np.random.random((dictionarySize * 10, num_nodes))
        #cutOff = 1 - (nodeRootFloat / num_nodes)
        cutOff = 1 - ( 2 * nodeRootFloat / num_nodes)
        argGood0 = np.argwhere( np.max(dictionary, axis=1) > cutOff )[:, 0][:dictionarySize]
        dictionary = dictionary[argGood0]
        dictionary[dictionary >  cutOff ] = 1
        dictionary[dictionary <= cutOff ] = 0
        dictionary = dictionary.astype(int)

        #plt.plot(np.sum(dictionary, axis=0))
        #plt.show()
        #quit()

        #inclusionCutff = 1.0 - ((  nodeRootFloat / dictionarySize) * 0.25)
        #inclusionCutff = 1.0 - 0.25
        inclusionCutff = 1.0 - 0.1
        randomUsage = np.random.random((num_data_points * 10, dictionarySize))
        argGood = np.argwhere( np.max(randomUsage, axis=1) > inclusionCutff )[:, 0][:num_data_points]
        randomUsage = randomUsage[argGood]
        randomUsage[randomUsage >  inclusionCutff ] = 1
        randomUsage[randomUsage <= inclusionCutff ] = 0
        randomUsage = randomUsage.astype(int)

        adjacency_matrices = np.matmul(randomUsage, dictionary)
        adjacency_matrices[adjacency_matrices >= 1] = 1

        #print (np.mean(adjacency_matrices))
        #inverse1 = uniqueValMaker(adjacency_matrices)
        #print (np.unique(inverse1).shape)

        #import seaborn as sns
        #sns.clustermap(adjacency_matrices)
        #plt.show()
        #quit()

        thresholds = 0


    if doMask:
        randomMask = np.random.randint(2, size=randomViews.size).reshape(randomViews.shape)
        randomViews[randomMask == 1] = 2
    else:
        randomNoise = np.random.normal(size=adjacency_matrices.size).reshape(adjacency_matrices.shape) * noise_size
        randomViews = adjacency_matrices + randomNoise

    return thresholds, 0, adjacency_matrices, randomViews



def simpleSetSim():

    

    simList = [] #data points, set size, DAG size
    #simList.append([1000, 100, 0.33])
    #simList.append([1000, 1000, 0.33])
    #simList.append([1000, 100, 0.25])
    #simList.append([1000, 100, 0.5])

    #simList.append([1000, 100, 0.2])


    #simList.append([100, 100, 0.5])

    #simList.append([25, 100, 0.5])
    

    #simList.append([100, 1000, 0.5])


    #simList.append([100, 1000, 0.5])
    #simList.append([100, 100, 0.5])
    #simList.append([100, 50, 0.5])
    #simList.append([100, 25, 0.5])
    #simList.append([100, 10, 0.5])


    #simList.append([100, 100, 0.01])

    

    #True
    #simList.append([100, 100, 0.6])
    #simList.append([100, 100, 0.5])
    #simList.append([100, 100, 0.4])
    #simList.append([100, 100, 0.3])
    #simList.append([100, 100, 0.2])
    #simList.append([100, 100, 0.1])

    #simList.append([100, 10, 0.3])
    simList.append([100, 1000, 0.5])


    

    doMask = False


    for simParamIndex in range(len(simList)):
        print ('simParamIndex', simParamIndex)
        for simIndex in range(10):
            
            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            noise_size =  simList[simParamIndex][2]
            num_paths_per_graph = noise_size

            seed1 = seed_from_three(simList[simParamIndex][0], simList[simParamIndex][1], simList[simParamIndex][2])
            np.random.seed(seed1) #num_data_points = 100
            torch.manual_seed(seed1) #num_data_points = 100

            
            ############simPart = 'N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_' + str(simIndex)
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(noise_size) + '_sim' + str(simIndex)

            # Generate a single simulation instance
            #adjacency_matrices, observations_batch, thresholds = generate_simulation_instance(num_nodes, num_data_points, num_paths_per_graph, doMatrix=True)

            #DAG_size = 10

            DAG_size = 10
            DAG_matrix, _, listAll, randomViews = generateSetSim2(num_data_points, num_nodes, DAG_size, noise_size, doMask=doMask)
            #DAG_matrix, _, listAll, randomViews = generateSetSim(num_data_points, num_nodes, DAG_size, noise_size, doMask=doMask)

            #print (simPart)
            #print (listAll[0])
            #print (randomViews[0])
            #quit()

            if not doMask:#simIndex != 0:
                np.savez_compressed('./data/sims/simpleSet/input/' + simPart + '_obs.npz', randomViews)
                np.savez_compressed('./data/sims/simpleSet/input/' + simPart + '_DAG.npz', DAG_matrix)
                #np.savez_compressed('./data/sims/simpleSet/input/' + simPart + '_DAGnodes.npz', nodesCorrespond)
                np.savez_compressed('./data/sims/simpleSet/input/' + simPart + '_latent.npz', listAll)

            if doMask:#simIndex != 0:
                np.savez_compressed('./data/sims/vector/input/' + simPart + '_obs.npz', randomViews)
                np.savez_compressed('./data/sims/vector/input/' + simPart + '_DAG.npz', DAG_matrix)
                #np.savez_compressed('./data/sims/vector/input/' + simPart + '_DAGnodes.npz', nodesCorrespond)
                np.savez_compressed('./data/sims/vector/input/' + simPart + '_latent.npz', listAll)


#simpleSetSim()
#quit()




def generateConvSetSim(num_data_points, num_nodes, DAG_size, noise_size, convSize):


    DAG_matrix0 = np.random.random( DAG_size * DAG_size )
    cutoff =  (4.0 / float(DAG_size))
    #cutoff =  (2.0 / float(DAG_size))

    #cutoff =  (2.0 / 5.0)
    #cutoff =  (2.0 / 4.0)
    #cutoff = 1 - cutoff
    DAG_matrix0[DAG_matrix0 > cutoff ] = 1
    DAG_matrix0[DAG_matrix0 < cutoff ] = 0
    DAG_matrix0 = DAG_matrix0.reshape( (DAG_size , DAG_size) )
    argAll = np.argwhere(DAG_matrix0 > -1)
    #print (argAll.shape)
    #argAll = argAll[argAll[:, 1] - argAll[:, 0] <= 4]
    #print (argAll.shape)
    argAll = argAll[argAll[:, 1] - argAll[:, 0] >= 1]
    

    DAG_matrix = np.zeros(DAG_matrix0.shape)
    DAG_matrix[argAll[:, 0], argAll[:, 1]] = DAG_matrix0[argAll[:, 0], argAll[:, 1]]
    
    nodesCorrespond = np.random.randint(DAG_size, size=num_nodes)
    while np.unique(nodesCorrespond).shape[0] < DAG_size:
        nodesCorrespond = np.random.randint(DAG_size, size=num_nodes)

    #DAG_big = DAG_matrix[nodesCorrespond][:, nodesCorrespond]

    argStart = np.argwhere(  np.sum(DAG_matrix, axis=0)  == 0)[:, 0]

    listAll = np.zeros(( num_data_points, num_nodes  ), dtype=int) - 1
    randomViews = np.zeros(( num_data_points, num_nodes  ), dtype=int)# - 1

    
    for indexNow in range(num_data_points):
        list1 = []

        start1 = argStart[np.random.randint(argStart.shape[0])]
        list1.append(start1)
        #print (DAG_big[start1])
        while np.sum(DAG_matrix[start1]) != 0:
            argNow = np.argwhere( DAG_matrix[start1] == 1 )[:, 0]
            start1 = argNow[np.random.randint(argNow.shape[0])]
            list1.append(start1)

        
        
        list1 = np.array(list1)

        list2 = np.argwhere( np.isin(nodesCorrespond, list1) )[:, 0]

        listAll[indexNow][:list2.shape[0]] = list2
        
        randomViews[indexNow, list2] = 1

    states = np.copy(randomViews)

    randomViews = randomViews.astype(float)

    #convMatrix = np.random.normal( (randomViews.shape[1], convSize)  )
    #convMatrix = np.random.normal( size = randomViews.shape[1] * convSize  ).reshape(  (randomViews.shape[1], convSize)   )
    #convMatrix = np.random.random( size = randomViews.shape[1] * convSize  ).reshape(  (randomViews.shape[1], convSize)   )
    convMatrix = np.random.randint(2,  size = randomViews.shape[1] * convSize  ).reshape(  (randomViews.shape[1], convSize)   )

    randomViews = np.matmul(randomViews, convMatrix)

    randomNoise = np.random.normal(size=randomViews.size).reshape(randomViews.shape) * noise_size
    randomViews = randomViews + randomNoise

    return DAG_matrix, nodesCorrespond, states, randomViews, convMatrix




def convSetSim():

    np.random.seed(2) #num_data_points = 100
    torch.manual_seed(2) #num_data_points = 100

    simList = [] #data points, set size, DAG size
    
    simList.append([1000, 100, 0.1])

    doMask = False


    for simParamIndex in range(len(simList)):
        print ('simParamIndex', simParamIndex)
        for simIndex in range(10):
            
            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            noise_size =  simList[simParamIndex][2]

            
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(noise_size) + '_sim' + str(simIndex)

            
            DAG_size = 10
            convSize = 100
            DAG_matrix, nodesCorrespond, states, randomViews, convMatrix = generateConvSetSim(num_data_points, num_nodes, DAG_size, noise_size, convSize)# doMask=True)


            
            np.savez_compressed('./data/sims/convSet/input/' + simPart + '_obs.npz', randomViews)
            np.savez_compressed('./data/sims/convSet/input/' + simPart + '_DAG.npz', DAG_matrix)
            np.savez_compressed('./data/sims/convSet/input/' + simPart + '_DAGnodes.npz', nodesCorrespond)
            np.savez_compressed('./data/sims/convSet/input/' + simPart + '_latent.npz', states)
            np.savez_compressed('./data/sims/convSet/input/' + simPart + '_convMatrix.npz', convMatrix)


#convSetSim()
#quit()




def generateCausalNetworkGraphs(num_nodes, num_data_points, num_measure, num_subgraphs ):



    numPoints = num_nodes * 3
    argPerm = np.random.permutation(num_nodes * num_nodes)[:numPoints]
    argPerm = argPerm.reshape(( num_subgraphs,  argPerm.shape[0] //  num_subgraphs ))

    graphList = np.zeros(( num_subgraphs, num_nodes * num_nodes ))
    for subgraph_index in range(num_subgraphs):
        graphList[subgraph_index, argPerm[subgraph_index]] = 1


    usedSubgraphs = np.random.randint(2, size=num_data_points*num_subgraphs  ).reshape(( num_data_points, num_subgraphs ))


    #true_graphs = np.zeros(( num_data_points, num_nodes*num_nodes  ))
    true_graphs = np.matmul(usedSubgraphs, graphList)
    true_graphs = true_graphs.reshape((num_data_points, num_nodes, num_nodes))


    #for a in range(10):
    #    print (np.sum(true_graphs[a]))
    #quit()
    

    X_data = np.random.normal(size=(  num_data_points, num_measure, num_nodes ))
    err_Y = np.random.normal(size=(  num_data_points, num_measure, num_nodes ))


    #print (X_data.shape)
    #print (X_data[0, 0])

    Y_data = np.matmul(X_data, true_graphs)
    Y_data = Y_data + err_Y


    obs_data = np.zeros((num_data_points, 2, num_measure, num_nodes ))
    obs_data[:, 0] = X_data
    obs_data[:, 1] = Y_data


    return true_graphs, obs_data

    




def saveCausalSim():

    for simIndex in range(10):

        num_subgraphs = 5
        #num_nodes, num_data_points, num_measure = 10, 100, 5
        num_nodes, num_data_points, num_measure = 10, 500, 1
        true_graphs, obs_data = generateCausalNetworkGraphs(num_nodes, num_data_points, num_measure, num_subgraphs )

        #simPart = 'causal_1'
        #simPart = 'causal_3'

        simPart = f'causal_N{num_nodes}_D{num_data_points}_M{num_measure}_{simIndex}'
        print (simPart)
        #quit()

        np.savez_compressed('./data/sims/initial/' + simPart + '_obs.npz', obs_data)
        np.savez_compressed('./data/sims/initial/' + simPart + '_graphs.npz', true_graphs)



#true_graphs = torch.tensor(true_graphs).float()


#causal_x_given_g(true_graphs[0], obs_data[0])

#saveCausalSim()
#quit()



# Function to generate a single simulation instance
def generate_graph(num_nodes, num_data_points, num_paths_per_graph, doMatrix=False):

    num_edge = num_nodes * (num_nodes - 1)

    
    nodeRootFloat = float(num_edge) ** 0.5
    nodesRootInt = int(np.floor(nodeRootFloat))

    dictionarySize = nodesRootInt
    
    dictionary = np.random.random((dictionarySize * 10, num_edge))
    #cutOff = 1 - (nodeRootFloat / num_nodes)
    cutOff = 1 - ( 2 * nodeRootFloat / num_edge)
    argGood0 = np.argwhere( np.max(dictionary, axis=1) > cutOff )[:, 0][:dictionarySize]
    dictionary = dictionary[argGood0]
    dictionary[dictionary >  cutOff ] = 1
    dictionary[dictionary <= cutOff ] = 0
    dictionary = dictionary.astype(int)

    #plt.plot(np.sum(dictionary, axis=0))
    #plt.show()
    #quit()

    #inclusionCutff = 1.0 - ((  nodeRootFloat / dictionarySize) * 0.25)
    #inclusionCutff = 1.0 - 0.25
    inclusionCutff = 1.0 - 0.1
    randomUsage = np.random.random((num_data_points * 10, dictionarySize))
    argGood = np.argwhere( np.max(randomUsage, axis=1) > inclusionCutff )[:, 0][:num_data_points]
    randomUsage = randomUsage[argGood]
    randomUsage[randomUsage >  inclusionCutff ] = 1
    randomUsage[randomUsage <= inclusionCutff ] = 0
    randomUsage = randomUsage.astype(int)

    adjacency_matrices = np.matmul(randomUsage, dictionary)
    adjacency_matrices[adjacency_matrices >= 1] = 1

    

    adjacency_matrices0 = np.zeros((num_data_points, num_nodes, num_nodes))
    argValid = np.argwhere(np.eye(num_nodes) == 0)
    adjacency_matrices0[:, argValid[:, 0], argValid[:, 1]] = adjacency_matrices
    adjacency_matrices = adjacency_matrices0

    
    observations_list = np.zeros((  num_data_points, num_nodes, num_nodes))
    

    for datapoint_index in range(num_data_points):

        adjacency_matrix = adjacency_matrices[datapoint_index]
        
        observations = observationSampler(adjacency_matrix, num_nodes, num_nodes, doMatrix=doMatrix)
        
        
        observations_list[datapoint_index] = observations
    


    return np.array(adjacency_matrices), observations_list, ''


def wackyGraphSim():

    #np.random.seed(1) #num_data_points = 100
    #torch.manual_seed(0) #num_data_points = 100

    

    simList = [] #data points, nodes, paths
    
    simList.append([100, 10, 1000])

    for simParamIndex in range(len(simList)):
        print ('simParamIndex', simParamIndex)


        seed1 = seed_from_three(simList[simParamIndex][0], simList[simParamIndex][1], simList[simParamIndex][2])
        np.random.seed(seed1) #num_data_points = 100
        torch.manual_seed(seed1) #num_data_points = 100


        for simIndex in range(1):

            print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            num_paths_per_graph =  simList[simParamIndex][2]
            
            #simPart = 'N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_' + str(simIndex)
            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

            # Generate a single simulation instance
            adjacency_matrices, observations_batch, thresholds = generate_graph(num_nodes, num_data_points, num_paths_per_graph, doMatrix=True)

            print (thresholds)

            np.savez_compressed('./data/sims/wacky/graph/' + simPart + '_obs.npz', observations_batch)
            np.savez_compressed('./data/sims/wacky/graph/' + simPart + '_graphs.npz', adjacency_matrices)


wackyGraphSim()
quit()