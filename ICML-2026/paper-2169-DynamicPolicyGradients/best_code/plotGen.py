import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns

import scipy
import time

import math

from sharedGen import *
import pandas as pd
import os
# Parameters
#num_simulations = 20
#num_data_points = 100
#num_nodes = 10  # Reduced to 10 nodes for faster runtimes
#num_nodes = 3
#num_paths_per_graph = 10





np.random.seed(1)
torch.manual_seed(0)


def overviewExample():



    Y = [1, -0.5, -1, -0.4, 1, 1.5]
    Y = np.array(Y)
    X = np.arange(len(Y))
    plt.plot(X, np.zeros(X.shape) , c='black')
    plt.plot(X, Y)
    plt.scatter(X, Y)
    #plt.ylim(-2, 2)
    plt.ylim(np.min(Y) - 1, np.max(Y) + 1)
    plt.xlim(np.min(X) - 1, np.max(X) + 1)
    plt.show()
    
    
    Y1 = [0.3, 0.2, 0.12, 0.08, 0.05]
    Y2 = [0.25, 0.26, 0.1, 0.11, 0.04]
    X = np.arange(len(Y1))
    plt.bar(X, Y1, alpha=0.5)
    plt.bar(X, Y2, alpha=0.5)
    plt.xticks([])
    #plt.legend(['$\Pr^*(S)$', r'$\Pr(S \mid \theta)$' ])
    plt.show()


#overviewExample()
#quit()


def makeBreakPlot(X, Ylist, lowY, midY1, midY2, topY, hspace=0.0, palette=[], xscale='', figFile='', xlabel='', vertline=''):


    #methods = [0.1, 0.2, 0.3, 0.4, 0.5]
    #acc = np.array([0.995, 0.982, 0.74, 0.51, 0.12])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(6,4),
        gridspec_kw={"height_ratios": [1, 1]}
    )

    

    print (palette)
    print (len(Ylist))
    # Plot on both axes
    for a in range(len(Ylist)):
        ax_top.plot(X, Ylist[a], marker="o", c=palette[a])
        ax_bot.plot(X, Ylist[a], marker="o", c=palette[a])
    # Set limits
    ax_top.set_ylim(midY2, topY)
    ax_bot.set_ylim(lowY, midY1)

    # Hide spines between axes
    ax_top.spines.bottom.set_visible(False)
    ax_bot.spines.top.set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bot.xaxis.tick_bottom()

    

    # Diagonal break marks
    #d = 0.015
    d = 0.03
    #d = 0.001
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax_bot.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    #plt.

    #ax_bot.set_ylabel("$F_1$ score")

    if vertline != '':
        ax_top.axvline(x=vertline, color='grey', linestyle=':', linewidth=2)
        ax_bot.axvline(x=vertline, color='grey', linestyle=':', linewidth=2)
    plt.ylabel("$F_1$ score")
    plt.xticks(X, X)
    if xscale == 'log':
        ax_top.set_xscale("log")
        ax_bot.set_xscale("log")
    if xlabel != '':
        plt.xlabel(xlabel)
    
    plt.gcf().set_size_inches(2.2, 3)
    plt.tight_layout()
    fig.subplots_adjust(hspace=hspace) 
    if figFile != '':
        plt.savefig(figFile)
    plt.show()






def plotGraphExample():

    num_data_points = 1000
    num_nodes = 10
    num_paths_per_graph = 100
    


    for simIndex in range(1):

        

        simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)
        

        graph_true = loadnpz('./data/sims/new/' + simPart + '_graphs.npz').astype(int)
        print (graph_true[0])

        import networkx as nx

        G = nx.from_numpy_array(graph_true[0], create_using=nx.DiGraph)

        pos = nx.spring_layout(G, seed=0)



        #highlight = {(1, 9), (3, 7), (6, 0)}
        #highlight = {(9, 6), (6, 4), (4, 8), (8, 2)}

        for plotIndex in range(3):
            
            edges = list(G.edges())
            if plotIndex == 1:
                #highlight = {(9, 4), (4, 8), (8, 2)}
                #highlight_nodes = {2, 9}

                highlight = {(9, 4), (4, 8), (8, 2), (5, 3), (3, 7)}


                highlight_nodes = {2, 9, 5, 7}

            if plotIndex == 2:
                highlight_nodes = {2, 9, 5, 7}
                highlight = {}

                
            if plotIndex == 0:
                highlight_nodes = {}
                highlight = edges
                

            

            

            if plotIndex in [1, 2]:
                edge_colors = [
                    "black" if (u, v) in highlight else "black"
                    for (u, v) in edges
                ]
            else:
                edge_colors = [
                    "black" if (u, v) in highlight else "black"
                    for (u, v) in edges
                ]

            node_colors = [
                "red" if n in highlight_nodes else "lightgray"
                for n in G.nodes()
            ]
        
            
            nx.draw_networkx_nodes(G, pos, node_size=600, node_color = node_colors)
            nx.draw_networkx_labels(G, pos)

            nx.draw_networkx_edges(
                G, pos,
                edgelist = highlight,
                arrows=True,
                arrowstyle='-|>',
                arrowsize=18,
                width=2.0, #edge_color=edge_colors,
                connectionstyle='arc3,rad=0.1'  # slight curvature helps readability
            )

            plt.axis("off")

            ax = plt.gca()
            ax.set_aspect("equal")

            xy = np.array(list(pos.values()))
            xmin, ymin = xy.min(axis=0)
            xmax, ymax = xy.max(axis=0)

            pad = 0.15
            ax.set_xlim(xmin - pad*(xmax-xmin), xmax + pad*(xmax-xmin))
            ax.set_ylim(ymin - pad*(ymax-ymin), ymax + pad*(ymax-ymin))

            plt.gcf().set_size_inches(3, 2.6)
            
            #plt.tight_layout()
            if plotIndex == 0:
                plt.savefig('./images/diagram/graph_example.pdf')
            if plotIndex == 1:
                plt.savefig('./images/diagram/path_example.pdf')
            if plotIndex == 2:
                plt.savefig('./images/diagram/emptyGraph_example.pdf')
            plt.show()


#plotGraphExample()
#quit()


def plotSetExample():
    #num_nodes = 100

    num_nodes = 100
    num_data_points = 100
    noise_level = 0.3 

    simType = 'simpleSet'

    for simIndex in range(1):
        simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(noise_level) + '_sim' + str(simIndex)
        latent = loadnpz('./data/sims/' + simType + '/input/' + simPart + '_latent.npz')
        observations_batch = loadnpz('./data/sims/' + simType + '/input/' + simPart + '_obs.npz')
        if np.max(latent) != 1:
            graph_true = processLatent(latent) 
        else:
            graph_true = latent


            if False:
                plt.scatter(latent[0],  np.arange(100)+1, s=20, c='black', alpha=0.5)#, marker='|')
                plt.scatter(observations_batch[0], np.arange(100)+1, c='red', s=20, alpha=0.5)#, marker='|')
                plt.ylabel('universe $|\mathcal{U}|$')
                
                Ytick = np.array([1, 20, 40, 60, 80, 100])
                plt.yticks(100 - Ytick, Ytick)
                plt.xticks([-1, 0, 1, 2], [-1, 0, 1, 2])
                #plt.gcf().set_size_inches(2, 1.4)

                plt.legend(['$S^*_i$', '$X_i$'])
                plt.gcf().set_size_inches(1.8, 2.6)
                plt.tight_layout()
                plt.savefig('./images/diagram/gauss_example_tall.pdf')
                plt.show()


            print (latent[0])

            plt.scatter(np.arange(num_nodes), latent[0], s=20, c='black', alpha=0.5)#, marker='|')
            plt.scatter(np.arange(num_nodes), observations_batch[0], c='red', s=20, alpha=0.5)#, marker='|')
            plt.xlabel('universe $\mathcal{U}$')
            plt.xticks([1, num_nodes / 2, num_nodes]) #, [1, 20, 40, 60, 80, 100]
            #plt.gcf().set_size_inches(2, 1.4)
            min1 = min(np.min(observations_batch[0]),  np.min(latent) )
            plt.ylim( min1 - 0.1, np.max(observations_batch[0]) + 1.5 )

            plt.yticks([-1, 0, 1, 2])

            plt.legend(['$S^*_i$', '$X_i$'])
            plt.gcf().set_size_inches(1.6, 2.6)
            plt.tight_layout()
            #plt.savefig('./images/diagram/gauss_example.pdf')
            plt.savefig('./images/diagram/gauss_example.pdf')
            plt.show()


        if False:
            for plotIndex in range(2):
                print (graph_true[0].shape)
                print (observations_batch[0].shape)

                plt.scatter(np.arange(100), latent[0], s=20, c='grey')#, marker='|')
                if plotIndex == 1:
                    plt.scatter(np.arange(100), observations_batch[0], c='red', s=20)#, marker='|')
                else:
                    plt.ylim(-0.1, 1.1)
                plt.xlabel('element')

                #plt.gcf().set_size_inches(2.3, 1.3)
                ##plt.gcf().set_size_inches(2, 1.3)
                #plt.gcf().set_size_inches(2, 1.5)
                
                #plt.gcf().set_size_inches(2.4, 1.3)
                plt.gcf().set_size_inches(2, 1.4)
                plt.xticks([1, 20, 40, 60, 80, 100]), [1, 20, 40, 60, 80, 100]
                plt.tight_layout()
                if plotIndex == 0:
                    plt.savefig('./images/diagram/set_example.pdf')
                else:
                    plt.savefig('./images/diagram/gauss_example.pdf')
                plt.show()

        #plt.plot(observations_batch[0])
        #plt.show()

        #plt.plot(observations_batch[0])
        #plt.show()
        quit()
    
            
#plotSetExample()
#quit()


def plotIsoformExample():


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
        
    

    def isoformPlotter(juncList, isoform1):


        highlight_index = 4
        for plotIndex in range(2):
            color = 'blue'

            posNow2 = juncList[isoform1 == 1]
            posNow2 = posNow2[np.argsort(posNow2[:, 0])]

            posNow2 = posNow2.astype(float) / 1000000

            posLevel = 0
            for b in range(posNow2.shape[0]-1):
                #plt.plot(  [posNow2[b][1], posNow2[b+1][0]] , [ posLevel,  posLevel ]   , c=color)
                
                assert posNow2[b][1] <= posNow2[b+1][0]


            pointColor = 'blue'
            if plotIndex == 1:  
                pointColor = 'red'
            plt.scatter(  posNow2, posNow2*0 , c=pointColor, edgecolors='black', s=10)

            for b in range(posNow2.shape[0]):
                arange1 = np.arange(101) / 100
                start1, end1 = posNow2[b][0], posNow2[b][1]
                length1 = end1 - start1
                yPos = (arange1 * 2) - 1
                yPos = posLevel + ((1 - (yPos ** 2)) * 0.6)


                assert start1 <= end1

                colorNow = color 
                pointColor = 'blue'
                if plotIndex == 1:
                    colorNow = 'lightgrey'
                    #pointColor = 'lightgrey'
                    if b == highlight_index:
                        colorNow = 'red'
                        #pointColor = 'red'

                    

                plt.plot(  start1 + (arange1 * length1) , yPos   , c=colorNow, linestyle=':', linewidth=1.5)

                if plotIndex == 1:
                    if b == highlight_index:
                        print ('hi')
                        plt.scatter(  [start1, end1], [posLevel, posLevel] , c='red', edgecolors='black', s=10)

            plt.ylim(-0.2, 0.7)
            plt.gcf().set_size_inches(2.5, 1.1)
            plt.tight_layout()
            if plotIndex == 0:
                plt.savefig('./images/diagram/isoform_example.pdf')
            else:
                plt.savefig('./images/diagram/read_example.pdf')
            plt.show()





            

    
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

    print ("A")

    with torch.no_grad():
    

        for gene_index in range(0, len(gene_unique)):

            geneNow = gene_unique[gene_index]

            #if geneNow in geneList:
            if geneNow in ['ENSG00000134046']:
                print (gene_index, geneNow)

                junctionNow = loadnpz('./data/real/splicing/geneFiles/junctions/junctions_' + str(geneNow) + '.npz')
                countNow = loadnpz('./data/real/splicing/geneFiles/counts/counts_' + str(geneNow) + '.npz')
                edges = loadnpz('./data/real/splicing/geneFiles/edges/edges_' + str(geneNow) + '.npz')
                junctionPos = junctionNow[:, 1:3].astype(int)

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

                    #print (np.unique(inverse1[sampleLong.shape[0]:][argGood]).shape)
                    #quit()


                if np.sum(isoformCounts_long) >= 1000:
                    
                    if True:
                        
                        
                        columnUse = np.arange(3)
                        sampleList_include = findValidSamples(sampleList, sampleLong, columnUse)
                        countNow = processSamples(sampleList, sampleList_include, columnUse, countNow)
                        isoformCounts_long = processSamples(sampleLong, sampleList_include, columnUse, isoformCounts_long)
                        
                        if np.sum(isoformCounts_long) >= 1:

                            trueCount_sum = np.sum(isoformCounts_long, axis=1)
                            argmax1 = np.argmax(trueCount_sum)
                            isoform1 = isoformJunctions_long[argmax1]
                            highlight_index = 1
                            isoformPlotter(isoformJunctionPos_long, isoform1)
                            
                            quit()

                            impliedJunc = np.matmul(isoformJunctions_long.T, isoformCounts_long)
                            trueCount_sum = np.sum(isoformCounts_long, axis=1)
                            shortCount_sum = np.sum(isoformCounts, axis=1)
                            trueCount_sum = trueCount_sum / np.sum(trueCount_sum)
                            shortCount_sum = shortCount_sum / np.sum(shortCount_sum)


                            doPlot = True 
                            predCount_sort = np.sort(predCount_sum)[-1::-1] 
                            if np.sum(predCount_sort[3:]) > 0.1:
                                doPlot = False 
                            trueCount_sort = np.sort(trueCount_sum)[-1::-1] 
                            if np.sum(trueCount_sort[3:]) > 0.1:
                                doPlot = False 
                            shortCount_sort = np.sort(shortCount_sum)[-1::-1] 
                            if np.sum(shortCount_sort[3:]) > 0.1:
                                doPlot = False 

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
                                    ours_plot = adjacency_matrices[np.argsort(predCount_sum*-1)[:3]]
                                    short_plot = isoformJunctions[np.argsort(shortCount_sum*-1)[:3]]
                                    true_plot = isoformJunctions_long[np.argsort(trueCount_sum*-1)[:3]]

                                    ours_plot = ours_plot[predCount_sort[:3] > 0.05]
                                    short_plot = short_plot[shortCount_sort[:3] > 0.05]
                                    true_plot = true_plot[trueCount_sort[:3] > 0.05]




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



#plotIsoformExample()
#quit()




def processLatent(latent):

    latent2 = np.zeros(latent.shape, dtype=int)

    for a in range(latent.shape[0]):
        arg1 = latent[a]
        arg1 = arg1[arg1!=-1]
        latent2[a, arg1] = 1
    
    return latent2 



def multi_x_given_g(adjacency_matrices, obs_matrix):

    obs_matrix = torch.tensor(obs_matrix).float().to(adjacency_matrices.device)
    
    #time2 = time.time()
    adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))
    obs_matrix = obs_matrix.reshape((1, obs_matrix.shape[0], obs_matrix.shape[1]))

    diff1 = 0.5 * torch.sum((adjacency_matrices - obs_matrix) ** 2 , axis=2)
    prob_mult = -1 * diff1 / (0.5 ** 2)

    return prob_mult



def evaluateSet():

    
    num_data_points = 100
    #num_nodes = 100
    num_nodes = 1000
    
    #num_data_points = 1000
    #num_data_points = 10000
    #noise_level = 0.5
     #Not ready
    noise_level = 0.3
    #noise_level = 0.75
    #noise_level = 1.0
    #noise_level = 0.4 #Do ours
    #noise_level = 0.2 #Do ours
    #noise_level = 0.1


    def log_calculate_pr_x_given_g(graphNow, obs_now):

        #print (graphNow.shape, obs_now.shape)

        diff1 = 0.5 * (obs_now - graphNow) ** 2 

        logProb = -1 * (np.sum(diff1) / (noise_level ** 2))

        return logProb
    
    


    learning_rate = 1e-3

    #simType = 'convSet'
    simType = 'simpleSet'



    methodList = []
    #methodList.append('ground truth')
    methodList.append('ours_offPolicy')
    #methodList.append('ours_onPolicy')
    
    #methodList.append('ours')
    methodList.append('VAE')
    methodList.append('autoreg')
    methodList.append('diffusion')
    methodList.append('localSolver')
    ########methodList.append('metropolas')
    methodList.append('naiveReward_offPolicy')
    methodList.append('GFlowReward_offPolicy')
    #methodList.append('zero')

    #methodList.append('FlowMatch')
    


    #modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'FlowMatch'
    #modelType = 'localSolver'
    #modelType = 'metropolas'

    methodName1 = 'method'
    #errorName = 'error'
    #errorName = 'graph edit distance'
    errorName = '$F_1$ score'
    probName = '$\log(\Pr(X_i \mid \hat{s}_i)$'


    methodDict = {}
    methodDict['autoreg'] = 'auto-\nregressive'
    #methodDict['autoreg'] = 'autoregressive'
    methodDict['localSolver'] = 'local search'
    methodDict['metropolas'] = 'Metropolis\nHastings'
    methodDict['GFlowReward'] = 'GFlowNet'
    methodDict['naiveReward'] = 'naive policy\nlearning'

    methodDict['GFlowReward_offPolicy'] = 'GFlowNet'
    methodDict['naiveReward_offPolicy'] = 'naive policy\nlearning'
    

    methodDict['ours'] = 'GReinSS'
    methodDict['ours_onPolicy'] = 'on policy GReinSS'
    methodDict['ours_offPolicy'] = 'GReinSS'# (off policy)'
    #methodDict['FlowMatch'] = 'flow matching (GFlowNet)'

    methodNameList = []
    for a in range(len(methodList)):
        modelType_name = methodList[a]
        if modelType_name in methodDict:
            modelType_name = methodDict[modelType_name]
        methodNameList.append( modelType_name )
    
    df = {}
    df[methodName1] = []
    df[errorName] = []
    df[probName] = []

    numGraphs = []


    for simIndex in range(1):



        simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(noise_level) + '_sim' + str(simIndex)
        
        for methodIndex in range(len(methodList)):
            modelType = methodList[methodIndex]

            print (modelType)
            #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_N10_P100_1.npz')

            modelType_name = modelType
            if modelType_name in methodDict:
                modelType_name = methodDict[modelType_name]
            

            latent = loadnpz('./data/sims/' + simType + '/input/' + simPart + '_latent.npz')
            observations_batch = loadnpz('./data/sims/' + simType + '/input/' + simPart + '_obs.npz')

            if np.max(latent) != 1:
                graph_true = processLatent(latent) 
            else:
                graph_true = latent
            
            
            if modelType == 'ground truth':
                graph_pred = np.copy(graph_true)
            else:
                graph_pred  =   loadnpz('./data/sims/' + simType + '/pred/graph_' + simPart + '_' + modelType + '.npz')

                if modelType == 'ours_offPolicy':
                    graph_pred =   loadnpz('./data/sims/weird_simpleSet/pred/graph_' + simPart + '_' + modelType + '_3.npz')



            num_graph_inverse = uniqueValMaker(graph_pred)
            _, num_graph_count = np.unique(num_graph_inverse, return_counts=True)
            num_graph = np.unique(num_graph_inverse).shape[0]
            numGraphs.append(num_graph)
            #print (modelType, 'num_graph', num_graph)
            #print (num_graph_count)

            #print (np.mean(graph_true))
            #quit()

            #sum1 = np.sum(graph_true, axis=0)
            #graph_pred[:, sum1 == 0] = 0 #TODO REMOVE!!!

            #plt.plot(  np.sum(graph_true, axis=0)  )
            #plt.show()

            



            #print (graph_true.shape)
            #print (graph_pred.shape)

            diff1 = graph_pred - graph_true

            cat1 = np.concatenate(( graph_true, graph_pred, diff1 ), axis=0)


            #sns.clustermap(graph_pred)
            #plt.show()

            
            

            prob_list = np.array([log_calculate_pr_x_given_g(graph_pred[a], observations_batch[a]) for a in range(observations_batch.shape[0])])



            TP = np.sum( graph_pred * graph_true, axis=1 )
            FN = np.sum( (1 - graph_pred)  * graph_true , axis=1 )
            FP = np.sum( graph_pred  * (1 - graph_true)  , axis=1 )
            Fscore = (2 * TP) / ( (2 * TP) + FN + FP )
            Fscore[TP == 0] = 0

            error1 = np.sum( np.abs(graph_pred - graph_true), axis=1 )


            predZero = np.sum(graph_pred, axis=0)
            trueZero = np.sum(graph_true, axis=0)
            #argZero = np.argwhere(predZero == 0)[:, 0]
            argZero = np.argwhere(np.logical_and(predZero == 0,  trueZero > 0 ) )[:, 0]

            

            print ('FMedian', np.median(Fscore))
            print ('Fmean', np.mean(Fscore))
            #print ('error', np.mean(error1))

            #print (np.mean(prob_list))

            #sns.clustermap(cat1, row_cluster=False)
            #plt.show()


            #quit()

            #plt.hist(observations_batch[:, argZero].reshape((-1,)) , bins=100 )
            #plt.show()


            for score_index in range(Fscore.shape[0]):
                df[methodName1].append(modelType_name)
                df[errorName].append(Fscore[score_index])
                df[probName].append(prob_list[score_index])


    #quit()
    #print (df)

    #palette = sns.color_palette("Set1")

    #palette = ['tab:red', 'tab:blue', 'lightblue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive']
    #palette = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive']
    palette = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]
    #print (palette)
    #quit()

    #removeYTrick = True
    removeYTrick = False


    sns.boxplot(data=df, x=errorName, y=methodName1, palette=palette)
    plt.ylabel('')
    #plt.xlim(-0.01, 1.05)
    
    plt.xticks([0.25, 0.5, 0.75, 1], ['.25', '.5', '.75', '1'])
    plt.gcf().set_size_inches(2.6, 3)
    #plt.gcf().set_size_inches(2.7, 3)

    if removeYTrick:
        plt.gcf().set_size_inches(1.6, 3)
        plt.yticks([], [])
    else:
        plt.xticks([0.0, 0.25, 0.5, 0.75, 1], ['0', '.25', '.5', '.75', '1'])
        plt.gcf().set_size_inches(2.6, 3)

    
    plt.tight_layout()
    #plt.savefig('./images/simpleSet/standard/F-score.pdf')
    #plt.savefig('./images/simpleSet/all/F-score_' + 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(noise_level) + '.pdf')
    plt.show()
    

    quit()

    sns.boxplot(data=df, x=probName, y=methodName1, palette=palette)
    plt.ylabel('')
    plt.xlim( np.min(df[probName]) * 1.05 , 0.0)
    plt.gcf().set_size_inches(2.6, 3)
    plt.tight_layout()
    #plt.savefig('./images/simpleSet/standard/logProbObserve.pdf')
    plt.show()

    for a in range(len(numGraphs)):
        plt.scatter( numGraphs[a], [-a], c=palette[a] )
    #plt.scatter( numGraphs, np.arange(len(numGraphs)) )
    print (methodNameList)
    plt.yticks( -np.arange(len(methodNameList)), methodNameList  )
    plt.xlabel('# of unique graphs')
    plt.xscale('log')
    plt.gcf().set_size_inches(2.6, 3)
    plt.tight_layout()
    plt.savefig('./images/simpleSet/standard/uniqueGraphs.pdf')
    plt.show()

    
    

#evaluateSet()
#quit()


def scaleSets():


    preAnalysis = False

    #scaleTypeList = ['graphs', 'paths', 'nodes']

    #scaleTypeList = ['sets', 'noise', 'elements']
    scaleTypeList = ['noise', 'elements']
    #scaleTypeList = ['elements']
    
    methodList = []
    methodList.append('ours_offPolicy')
    #methodList.append('ours_onPolicy')
    methodList.append('VAE')
    methodList.append('autoreg')
    methodList.append('diffusion')
    methodList.append('localSolver')
    methodList.append('naiveReward_offPolicy')
    #methodList.append('naiveReward_onPolicy')
    methodList.append('GFlowReward_offPolicy') #
    #methodList.append('GFlowReward_onPolicy')
    #methodList.append('zero')
    

    
    methodDict = {}

    if not preAnalysis:
        methodDict['autoreg'] = 'auto-\nregressive'
        #methodDict['localSolver'] = 'local search'
        #methodDict['metropolas'] = 'Metropolis\nHastings'
        #methodDict['GFlowReward'] = 'GFlowNet'
        #methodDict['naiveReward'] = 'naive policy\nlearning'
        #methodDict['ours'] = 'GReinSS'

        #methodDict['autoreg'] = 'autoregressive'
        methodDict['localSolver'] = 'local search'
        methodDict['metropolas'] = 'Metropolis Hastings'
        methodDict['GFlowReward'] = 'GFlowNet'
        methodDict['naiveReward'] = 'naive policy learning'
        methodDict['ours'] = 'GReinSS'

        methodDict['GFlowReward_offPolicy'] = 'GFlowNet'
        methodDict['naiveReward_offPolicy'] = 'naive policy\nlearning'
        methodDict['ours'] = 'GReinSS'
        methodDict['ours_onPolicy'] = 'on policy GReinSS'
        methodDict['ours_offPolicy'] = 'GReinSS'# (off policy)'


    #methodDict['FlowMatch'] = 'flow matching (GFlowNet)'

    

    methodNameList = []
    for a in range(len(methodList)):
        modelType_name = methodList[a]
        if modelType_name in methodDict:
            modelType_name = methodDict[modelType_name]
        methodNameList.append( modelType_name )


    #palette = ['tab:red', 'tab:blue', 'lightblue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]
    palette = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]
    #palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]

    if False:
        #legendList = []
        for modelType0 in range(len(methodList)):
            modelType = methodList[modelType0]
            plt.scatter([0], [0], c=palette[modelType0])
            #plt.plot([0], [0], c=palette[modelType0])
            #plt.bar([0], [0], color=palette[modelType0])
        plt.legend(methodNameList)
        plt.gcf().set_size_inches(6, 6)
        plt.tight_layout()
        plt.savefig('./images/startEnd/standard/legend.pdf')
        plt.show()
        quit()


    for scaleType in scaleTypeList:
        #methodList.append('metropolas')

        simList = [] #data points, nodes, paths

        #scaleType = 'nodes'
        #scaleType = 'paths'
        #scaleType = 'graphs'

        #Missing

        if scaleType == 'sets':
            xName = '# of sets'
            simList.append([1, 100, 0.5])  #modified number of data points 
            simList.append([10, 100, 0.5])
            simList.append([100, 100, 0.5])
            simList.append([1000, 100, 0.5])
            simList.append([10000, 100, 0.5])


        #All included, ours best
        if scaleType == 'elements':
            xName = 'universe size $|\mathcal{U}|$'
            
            #simList.append([100, 10, 0.5])  #This one 100, 10, 0.5 is old! incorrect latent structure 
            
            simList.append([100, 10, 0.3])  
            simList.append([100, 100, 0.3])  
            simList.append([100, 1000, 0.3]) 

            #simList.append([100, 100, 0.5])  
            #simList.append([100, 1000, 0.5]) 


        if scaleType == 'noise':
            xName = 'noise level $\sigma$'
            #simList.append([100, 100, 0.01])
            #simList.append([100, 100, 0.25])  
            #simList.append([100, 100, 0.5])
            #simList.append([100, 100, 0.75]) 
            #simList.append([100, 100, 1.0]) 

            #simList.append([1000, 100, 0.1])
            #simList.append([1000, 100, 0.2]) 
            #simList.append([1000, 100, 0.3])
            #simList.append([1000, 100, 0.4])
            #simList.append([1000, 100, 0.5])
            #simList.append([1000, 100, 0.6])

            simList.append([100, 100, 0.1])
            simList.append([100, 100, 0.2]) 
            simList.append([100, 100, 0.3])
            simList.append([100, 100, 0.4])
            simList.append([100, 100, 0.5])

            #simList.append([100, 100, 0.75])
            #simList.append([100, 100, 1.0])
            #simList.append([1000, 100, 0.6])


        

        errors = {}
        for modelType0 in methodList: 
            errors[modelType0] = []

        errorsList = []
        
        

        for modelType0 in range(len(methodList)):
            modelType = methodList[modelType0]
            Xvariable = []
            for simParamIndex in range(len(simList)):
                print ('simParamIndex', simParamIndex)
                for simIndex in range(0, 1):

                    #print (simIndex)

                    num_data_points = simList[simParamIndex][0]
                    num_nodes =  simList[simParamIndex][1]
                    num_paths_per_graph =  simList[simParamIndex][2]

                    if scaleType == 'sets':
                        Xvariable.append(num_data_points)
                    if scaleType == 'elements':
                        Xvariable.append(num_nodes)
                    if scaleType == 'noise':
                        Xvariable.append(num_paths_per_graph)

                    simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

                    #print (simPart)

                    
                    #np.savez_compressed('./data/sims/initial/' + simPart + '_obs.npz', observations_batch)
                    graph_true = loadnpz('./data/sims/simpleSet/input/' + simPart + '_latent.npz')
                    #graph_true = processLatent(graph_true)


                    #print (graph_true[0])
                    #quit()


                    

                    if modelType == 'zero':
                        graph_pred = np.zeros(graph_true.shape)
                    else:
                        graph_pred = loadnpz('./data/sims/simpleSet/pred/graph_' + simPart + '_' + modelType + '.npz')


                    #print (modelType)
                    #print (np.unique(graph_pred))


                    #print (np.sum(graph_pred) / graph_pred.shape[0] )
                    #print (np.sum(graph_true) / graph_pred.shape[0] )
                    #quit()

                    #error1 = np.sum(np.abs(graph_pred - graph_true), axis=(1, 2)  )
                    #error1 = np.mean(error1)

                    error1 = np.mean(np.abs(graph_pred - graph_true))

                    #print (graph_pred.shape, graph_true.shape)

                    TP = np.sum( graph_pred * graph_true , axis=1)
                    FN = np.sum( (1 - graph_pred)  * graph_true , axis=1 )
                    FP = np.sum( graph_pred  * (1 - graph_true), axis=1  )
                    Fscore = (2 * TP) / ( (2 * TP) + FN + FP ) 
                    Fscore[TP == 0] = 0

                    #Fscore = np.median(Fscore)
                    Fscore = np.median(Fscore)

                    if np.max(Fscore) > 1:
                        print ('issue')
                        print (np.sum( graph_pred  * (1 - graph_true) ))

                        print (np.min( graph_pred ))
                        print (np.min(1 - graph_true) )
                        #print (np.min(FP))
                        quit()

                    #print (np.max(Fscore))

                    error1 = Fscore


                    #print (np.mean(np.sum(graph_pred, axis= (1, 2) )))
                    #print (error1)
                    #quit()

                    errors[modelType].append(error1)
            errorsList.append(errors[modelType])


            #print (Xvariable, errors[modelType])
        

        #print (modelType0)
        
        

        for modelType0 in range(len(methodList)):
            modelType = methodList[modelType0]
            linestyle = ['solid', 'dashed'][modelType0 % 2]

            if modelType0 == 0:
                zorder = 100
            else:
                zorder = 1

            plt.plot(Xvariable, errors[modelType], c=palette[modelType0], zorder=zorder)
            #plt.plot(Xvariable, errors[modelType], c=palette[modelType0 // 2], linestyle = linestyle)
            
        for modelType0 in range(len(methodList)):
            modelType = methodList[modelType0]

            if modelType0 == 0:
                zorder = 100
            else:
                zorder = 1
            
            plt.scatter(Xvariable, errors[modelType], c=palette[modelType0], zorder=zorder)
            #plt.scatter(Xvariable, errors[modelType], c=palette[modelType0 // 2])#, linestyle = linestyle)
            

            #print (modelType)
            #print (Xvariable, errors[modelType] )


        print (Xvariable)
        print (errorsList)

        figFile = './images/simpleSet/scale/' + scaleType + '.pdf'
        #if scaleType == 'noise':
        #    makeBreakPlot(Xvariable, errorsList, -0.02, 0.85, 0.85, 1.01, hspace=0.0, palette=palette, figFile=figFile, xlabel='$\sigma$', vertline=0.3)
        #else:
        #    makeBreakPlot(Xvariable, errorsList, -0.02, 0.85, 0.85, 1.01, hspace=0.0, palette=palette, xscale='log', figFile=figFile, xlabel='vector size', vertline=100)

        
        if True:
            if scaleType == 'noise':
                #plt.xlim(-0.01, 1.01)
                #plt.xlim(0.09, 0.51)
                xLine = 0.3
                #plt.ylim(0.85, 1.01)
                plt.ylim(0.35, 1.02)

                plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


                

                #plt.xticks([.1, ])

            if scaleType == 'sets':
                xLine = 100

            if scaleType == 'elements':
                xLine = 100

                plt.ylim(0.0, 1.02)
                #plt.ylim(0.85, 1.01)
                #plt.ylim(-0.01, 1.03)
            
            plt.axvline(x=xLine, color='grey', linestyle=':', linewidth=2)

            if scaleType != 'noise':
                plt.xscale('log')
            #plt.yscale('symlog', linthresh=1e-2)
            plt.xlabel(xName)



            
            #print (Xvariable)

            if scaleType == 'noise':
                plt.xticks(Xvariable, Xvariable)
                #plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5],  ['.1', '.2', '.3', '.4', '.5'])
            else:
                plt.xticks(Xvariable)
            plt.gca().minorticks_off()
            
            plt.ylabel('$F_1$ score')
            if scaleType == 'noise':
                #plt.gcf().set_size_inches(2.3, 3)
                #plt.gcf().set_size_inches(2.3, 2.6)
                #plt.gcf().set_size_inches(2.1, 2.6) #Default
                plt.gcf().set_size_inches(2.0, 2.6) 
            else:
                #plt.gcf().set_size_inches(1.8, 3)
                plt.gcf().set_size_inches(1.8, 2.6)
                #plt.gcf().set_size_inches(2.1, 1.8) #Half Height

            
            #if preAnalysis:
            #    plt.gcf().set_size_inches(7, 7)
            #    plt.legend(methodNameList)


            #plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig('./images/simpleSet/scale/' + scaleType + '.pdf')
            #plt.legend(methodNameList)
            plt.show()

#scaleSets()
#quit()



def modifiedOffPolicySets():


    preAnalysis = False

    #scaleTypeList = ['graphs', 'paths', 'nodes']

    #scaleTypeList = ['sets', 'noise', 'elements']
    scaleTypeList = ['noise']#:, 'elements']
    #scaleTypeList = ['elements']
    
    methodList = []
    methodList.append('ours_offPolicy')

    methodList.append('ours_offPolicy_15x')
    methodList.append('ours_offPolicy_05x')
    #methodList.append('ours_onPolicy')
    methodList.append('VAE')
    methodList.append('autoreg')
    methodList.append('diffusion')
    methodList.append('localSolver')
    methodList.append('naiveReward_offPolicy')
    #methodList.append('naiveReward_onPolicy')
    methodList.append('GFlowReward_offPolicy') #
    #methodList.append('GFlowReward_onPolicy')
    #methodList.append('zero')
    

    
    methodDict = {}

    if not preAnalysis:
        methodDict['autoreg'] = 'auto-\nregressive'
        #methodDict['localSolver'] = 'local search'
        #methodDict['metropolas'] = 'Metropolis\nHastings'
        #methodDict['GFlowReward'] = 'GFlowNet'
        #methodDict['naiveReward'] = 'naive policy\nlearning'
        #methodDict['ours'] = 'GReinSS'

        #methodDict['autoreg'] = 'autoregressive'
        methodDict['localSolver'] = 'local search'
        methodDict['metropolas'] = 'Metropolis Hastings'
        methodDict['GFlowReward'] = 'GFlowNet'
        methodDict['naiveReward'] = 'naive policy learning'
        methodDict['ours'] = 'GReinSS'

        methodDict['GFlowReward_offPolicy'] = 'GFlowNet'
        methodDict['naiveReward_offPolicy'] = 'naive policy\nlearning'
        methodDict['ours'] = 'GReinSS'
        methodDict['ours_onPolicy'] = 'on policy GReinSS'
        methodDict['ours_offPolicy'] = 'GReinSS'# (off policy)'


        #methodDict['ours_offPolicy_2x'] = 'GReinSS (2 times off-policy bias)'
        methodDict['ours_offPolicy_15x'] = 'GReinSS (50% more off-policy bias)'
        methodDict['ours_offPolicy_05x'] = 'GReinSS (50% less off-policy bias)'


    methodNameList = []
    for a in range(len(methodList)):
        modelType_name = methodList[a]
        if modelType_name in methodDict:
            modelType_name = methodDict[modelType_name]
        methodNameList.append( modelType_name )


    #palette = ['tab:red', 'tab:blue', 'lightblue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]
    palette = ['tab:red', 'tab:blue', 'darkblue', 'lightblue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]


    for scaleType in scaleTypeList:
        #methodList.append('metropolas')

        simList = [] #data points, nodes, paths

        #scaleType = 'nodes'
        #scaleType = 'paths'
        #scaleType = 'graphs'

        #Missing

        if scaleType == 'sets':
            xName = '# of sets'
            simList.append([1, 100, 0.5])  #modified number of data points 
            simList.append([10, 100, 0.5])
            simList.append([100, 100, 0.5])
            simList.append([1000, 100, 0.5])
            simList.append([10000, 100, 0.5])


        #All included, ours best
        if scaleType == 'elements':
            xName = 'universe size $|\mathcal{U}|$'
            
            #simList.append([100, 10, 0.5])  #This one 100, 10, 0.5 is old! incorrect latent structure 
            
            simList.append([100, 10, 0.3])  
            simList.append([100, 100, 0.3])  
            simList.append([100, 1000, 0.3]) 

            #simList.append([100, 100, 0.5])  
            #simList.append([100, 1000, 0.5]) 


        if scaleType == 'noise':
            xName = 'noise level $\sigma$'
            simList.append([100, 100, 0.1])
            simList.append([100, 100, 0.2]) 
            simList.append([100, 100, 0.3])
            simList.append([100, 100, 0.4])
            simList.append([100, 100, 0.5])





        

        errors = {}
        for modelType0 in methodList: 
            errors[modelType0] = []

        errorsList = []
        
        

        for modelType0 in range(len(methodList)):
            modelType = methodList[modelType0]
            Xvariable = []
            for simParamIndex in range(len(simList)):
                print ('simParamIndex', simParamIndex)
                for simIndex in range(0, 1):

                    #print (simIndex)

                    num_data_points = simList[simParamIndex][0]
                    num_nodes =  simList[simParamIndex][1]
                    num_paths_per_graph =  simList[simParamIndex][2]

                    if scaleType == 'sets':
                        Xvariable.append(num_data_points)
                    if scaleType == 'elements':
                        Xvariable.append(num_nodes)
                    if scaleType == 'noise':
                        Xvariable.append(num_paths_per_graph)

                    simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

                    #print (simPart)

                    
                    #np.savez_compressed('./data/sims/initial/' + simPart + '_obs.npz', observations_batch)
                    graph_true = loadnpz('./data/sims/simpleSet/input/' + simPart + '_latent.npz')
                    #graph_true = processLatent(graph_true)


                    #print (graph_true[0])
                    #quit()


                    

                    if modelType == 'zero':
                        graph_pred = np.zeros(graph_true.shape)
                    else:
                        graph_pred = loadnpz('./data/sims/simpleSet/pred/graph_' + simPart + '_' + modelType + '.npz')


                    #print (modelType)
                    #print (np.unique(graph_pred))


                    #print (np.sum(graph_pred) / graph_pred.shape[0] )
                    #print (np.sum(graph_true) / graph_pred.shape[0] )
                    #quit()

                    #error1 = np.sum(np.abs(graph_pred - graph_true), axis=(1, 2)  )
                    #error1 = np.mean(error1)

                    error1 = np.mean(np.abs(graph_pred - graph_true))

                    #print (graph_pred.shape, graph_true.shape)

                    TP = np.sum( graph_pred * graph_true , axis=1)
                    FN = np.sum( (1 - graph_pred)  * graph_true , axis=1 )
                    FP = np.sum( graph_pred  * (1 - graph_true), axis=1  )
                    Fscore = (2 * TP) / ( (2 * TP) + FN + FP ) 
                    Fscore[TP == 0] = 0

                    #Fscore = np.median(Fscore)
                    Fscore = np.median(Fscore)

                    if np.max(Fscore) > 1:
                        print ('issue')
                        print (np.sum( graph_pred  * (1 - graph_true) ))

                        print (np.min( graph_pred ))
                        print (np.min(1 - graph_true) )
                        #print (np.min(FP))
                        quit()

                    #print (np.max(Fscore))

                    error1 = Fscore


                    #print (np.mean(np.sum(graph_pred, axis= (1, 2) )))
                    #print (error1)
                    #quit()

                    errors[modelType].append(error1)
            errorsList.append(errors[modelType])


            #print (Xvariable, errors[modelType])
        

        #print (modelType0)
        
        

        for modelType0 in range(len(methodList)):
            modelType = methodList[modelType0]
            linestyle = ['solid', 'dashed'][modelType0 % 2]

            if modelType0 == 0:
                zorder = 100
            else:
                zorder = 1

            plt.plot(Xvariable, errors[modelType], c=palette[modelType0], zorder=zorder)
            #plt.plot(Xvariable, errors[modelType], c=palette[modelType0 // 2], linestyle = linestyle)
            
        for modelType0 in range(len(methodList)):
            modelType = methodList[modelType0]

            print (modelType, errors[modelType])

            if modelType0 == 0:
                zorder = 100
            else:
                zorder = 1
            
            plt.scatter(Xvariable, errors[modelType], c=palette[modelType0], zorder=zorder)
            #plt.scatter(Xvariable, errors[modelType], c=palette[modelType0 // 2])#, linestyle = linestyle)
            

            #print (modelType)
            #print (Xvariable, errors[modelType] )


        #print (Xvariable)
        #rint (errorsList)

        figFile = './images/simpleSet/scale/' + scaleType + '.pdf'
        #if scaleType == 'noise':
        #    makeBreakPlot(Xvariable, errorsList, -0.02, 0.85, 0.85, 1.01, hspace=0.0, palette=palette, figFile=figFile, xlabel='$\sigma$', vertline=0.3)
        #else:
        #    makeBreakPlot(Xvariable, errorsList, -0.02, 0.85, 0.85, 1.01, hspace=0.0, palette=palette, xscale='log', figFile=figFile, xlabel='vector size', vertline=100)

        
        if True:
            if scaleType == 'noise':
                #plt.xlim(-0.01, 1.01)
                #plt.xlim(0.09, 0.51)
                xLine = 0.3
                #plt.ylim(0.85, 1.01)
                plt.ylim(0.35, 1.02)

                plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


                

                #plt.xticks([.1, ])

            if scaleType == 'sets':
                xLine = 100

            if scaleType == 'elements':
                xLine = 100

                plt.ylim(0.0, 1.02)
                #plt.ylim(0.85, 1.01)
                #plt.ylim(-0.01, 1.03)
            
            plt.axvline(x=xLine, color='grey', linestyle=':', linewidth=2)

            if scaleType != 'noise':
                plt.xscale('log')
            #plt.yscale('symlog', linthresh=1e-2)
            plt.xlabel(xName)



            
            #print (Xvariable)

            if scaleType == 'noise':
                plt.xticks(Xvariable, Xvariable)
                #plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5],  ['.1', '.2', '.3', '.4', '.5'])
            else:
                plt.xticks(Xvariable)
            plt.gca().minorticks_off()
            
            plt.ylabel('$F_1$ score')
            if scaleType == 'noise':
                #plt.gcf().set_size_inches(2.3, 3)
                #plt.gcf().set_size_inches(2.3, 2.6)
                #plt.gcf().set_size_inches(2.1, 2.6) #Default
                plt.gcf().set_size_inches(2.0, 2.6) 
            else:
                #plt.gcf().set_size_inches(1.8, 3)
                plt.gcf().set_size_inches(1.8, 2.6)
                #plt.gcf().set_size_inches(2.1, 1.8) #Half Height

            
            #if preAnalysis:
            #    plt.gcf().set_size_inches(7, 7)
            #    plt.legend(methodNameList)


            #plt.yscale('log')

            if True: #For Review:
                plt.legend(methodNameList)
                #plt.gcf().set_size_inches(5, 4)
                plt.gcf().set_size_inches(5, 6)
                
            
            plt.tight_layout()
            #plt.savefig('./images/simpleSet/scale/' + scaleType + '.pdf')
            plt.savefig('./images/review/off-policy/modifiedBias.png')
            #plt.legend(methodNameList)
            plt.show()

#modifiedOffPolicySets()
#quit()








def offPolicyScaleSets():


    preAnalysis = False

    

    comparePolicy = True

    #scaleTypeList = ['graphs', 'paths', 'nodes']

    #scaleTypeList = ['sets', 'noise', 'elements']
    scaleTypeList = ['noise', 'elements']
    #scaleTypeList = ['elements']

    for comparePolicy in [False, True]:

        if comparePolicy:

            methodList = []
            methodList.append('ours_offPolicy')
            methodList.append('ours_onPolicy')
            methodList.append('naiveReward_offPolicy')
            methodList.append('naiveReward_onPolicy')
            methodList.append('GFlowReward_offPolicy') #
            methodList.append('GFlowReward_onPolicy')


        else:
            methodList = []
            #methodList.append('ours_offPolicy')
            methodList.append('ours_onPolicy')
            methodList.append('VAE')
            methodList.append('autoreg')
            methodList.append('diffusion')
            #methodList.append('localSolver')
            #methodList.append('naiveReward_offPolicy')
            methodList.append('naiveReward_onPolicy')
            #methodList.append('GFlowReward_offPolicy') #
            methodList.append('GFlowReward_onPolicy')
            #methodList.append('zero')
        

        
        methodDict = {}

        if not preAnalysis:
            methodDict['autoreg'] = 'autoregressive'
            methodDict['localSolver'] = 'local search'
            methodDict['metropolas'] = 'Metropolis Hastings'
            methodDict['GFlowReward'] = 'GFlowNet'
            methodDict['naiveReward'] = 'naive policy learning'
            methodDict['ours'] = 'GReinSS'

            methodDict['GFlowReward_offPolicy'] = 'GFlowNet'
            methodDict['naiveReward_offPolicy'] = 'naive policy\gradients'
            methodDict['ours'] = 'GReinSS'
            methodDict['ours_offPolicy'] = 'GReinSS'# (off policy)'


            methodDict['GFlowReward_onPolicy'] = 'GFlowNet on policy'
            methodDict['naiveReward_onPolicy'] = 'naive policy gradients on policy'
            methodDict['ours_onPolicy'] = 'GReinSS on policy'


        #methodDict['FlowMatch'] = 'flow matching (GFlowNet)'

        

        methodNameList = []
        for a in range(len(methodList)):
            modelType_name = methodList[a]
            if modelType_name in methodDict:
                modelType_name = methodDict[modelType_name]
            methodNameList.append( modelType_name )


        if comparePolicy:
            palette = ['tab:red', 'tab:blue',  'tab:blue', 'tab:purple', 'tab:purple', 'tab:olive', 'tab:olive'][1:]
        else:
            #palette = ['tab:red', 'tab:blue', 'lightblue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]
            palette = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple', 'tab:olive'][1:]
            #palette = ['tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]
        


        if True:
            
            methodList = []
            methodList.append('ours_offPolicy')
            methodList.append('ours_onPolicy')
            methodList.append('VAE')
            methodList.append('autoreg')
            methodList.append('diffusion')
            methodList.append('naiveReward_offPolicy')
            methodList.append('naiveReward_onPolicy')
            methodList.append('GFlowReward_offPolicy') #
            methodList.append('GFlowReward_onPolicy')

            palette = ['tab:red', 'tab:blue',  'tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple', 'tab:purple', 'tab:olive', 'tab:olive'][1:]

            #print (len(methodList))
            #print (len(palette))
            #quit()

            if True:
                methodDict['autoreg'] = 'autoregressive'
                methodDict['localSolver'] = 'local search'
                methodDict['metropolas'] = 'Metropolis Hastings'
                methodDict['GFlowReward'] = 'GFlowNet'
                methodDict['naiveReward'] = 'naive policy learning'
                methodDict['ours'] = 'GReinSS'
                methodDict['GFlowReward_offPolicy'] = 'GFlowNet'
                methodDict['naiveReward_offPolicy'] = 'naive policy gradients'
                methodDict['ours'] = 'GReinSS'
                methodDict['ours_offPolicy'] = 'GReinSS'# (off policy)'
                methodDict['GFlowReward_onPolicy'] = 'GFlowNet on policy'
                methodDict['naiveReward_onPolicy'] = 'naive policy\ngradients on policy'
                methodDict['ours_onPolicy'] = 'GReinSS on policy'

            methodNameList = []
            for a in range(len(methodList)):
                modelType_name = methodList[a]
                if modelType_name in methodDict:
                    modelType_name = methodDict[modelType_name]
                methodNameList.append( modelType_name )

            for modelType0 in range(len(methodList)):
                modelType = methodList[modelType0]
                linestyle = 'solid'
                if 'onPolicy' in modelType:
                    linestyle = 'dashed'
                
                #plt.scatter([0], [0], c=palette[modelType0])
                plt.plot([0], [0], c=palette[modelType0], linestyle=linestyle)
                #plt.bar([0], [0], color=palette[modelType0])
            plt.legend(methodNameList)
            plt.gcf().set_size_inches(6, 6)
            plt.tight_layout()
            plt.savefig('./images/simpleSet/onPolicy/legend.pdf')
            plt.show()
            quit()

        
        for scaleType in scaleTypeList:
            #methodList.append('metropolas')

            simList = [] #data points, nodes, paths
            #All included, ours best
            if scaleType == 'elements':
                xName = '# of elements'
                simList.append([100, 10, 0.3])  
                simList.append([100, 100, 0.3])  
                simList.append([100, 1000, 0.3])


            if scaleType == 'noise':
                xName = 'standard deviation'
                simList.append([100, 100, 0.1])
                simList.append([100, 100, 0.2]) 
                simList.append([100, 100, 0.3])
                simList.append([100, 100, 0.4])
                simList.append([100, 100, 0.5])
            
            errors = {}
            for modelType0 in methodList: 
                errors[modelType0] = []

            
            

            for modelType0 in range(len(methodList)):
                modelType = methodList[modelType0]
                Xvariable = []
                for simParamIndex in range(len(simList)):
                    print ('simParamIndex', simParamIndex)
                    for simIndex in range(0, 1):

                        #print (simIndex)

                        num_data_points = simList[simParamIndex][0]
                        num_nodes =  simList[simParamIndex][1]
                        num_paths_per_graph =  simList[simParamIndex][2]

                        if scaleType == 'sets':
                            Xvariable.append(num_data_points)
                        if scaleType == 'elements':
                            Xvariable.append(num_nodes)
                        if scaleType == 'noise':
                            Xvariable.append(num_paths_per_graph)

                        simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)
                        graph_true = loadnpz('./data/sims/simpleSet/input/' + simPart + '_latent.npz')
                        graph_pred = loadnpz('./data/sims/simpleSet/pred/graph_' + simPart + '_' + modelType + '.npz')

                        error1 = np.mean(np.abs(graph_pred - graph_true))

                        #print (graph_pred.shape, graph_true.shape)

                        TP = np.sum( graph_pred * graph_true , axis=1)
                        FN = np.sum( (1 - graph_pred)  * graph_true , axis=1 )
                        FP = np.sum( graph_pred  * (1 - graph_true), axis=1  )
                        Fscore = (2 * TP) / ( (2 * TP) + FN + FP ) 
                        Fscore[TP == 0] = 0

                        #Fscore = np.median(Fscore)
                        Fscore = np.median(Fscore)

                        if np.max(Fscore) > 1:
                            print ('issue')
                            print (np.sum( graph_pred  * (1 - graph_true) ))

                            print (np.min( graph_pred ))
                            print (np.min(1 - graph_true) )
                            #print (np.min(FP))
                            quit()

                        error1 = Fscore

                        errors[modelType].append(error1)


            for modelType0 in range(len(methodList)):
                modelType = methodList[modelType0]
                linestyle = 'solid'
                if 'onPolicy' in modelType:
                    linestyle = 'dashed'
                plt.plot(Xvariable, errors[modelType], c=palette[modelType0], linestyle=linestyle)
            
            for modelType0 in range(len(methodList)):
                modelType = methodList[modelType0]
                plt.scatter(Xvariable, errors[modelType], c=palette[modelType0])

            if scaleType == 'noise':
                #plt.xlim(-0.01, 1.01)
                #plt.xlim(0.09, 0.51)
                xLine = 0.3
                #plt.ylim(0.85, 1.01)
            if scaleType == 'sets':
                xLine = 100

            if scaleType == 'elements':
                xLine = 100

                #plt.ylim(0.6, 1.01)
            
            plt.axvline(x=xLine, color='grey', linestyle=':', linewidth=2)

            if scaleType != 'noise':
                plt.xscale('log')
            #plt.yscale('symlog', linthresh=1e-2)
            plt.xlabel(xName)
            
            #print (Xvariable)

            if scaleType == 'noise':
                plt.xticks(Xvariable, Xvariable)
                #plt.ylim(0.85, 1.01)
            else:
                plt.xticks(Xvariable)
            
            plt.ylabel('$F_1$ score')
            if scaleType == 'noise':
                plt.gcf().set_size_inches(2.3, 3)
            else:
                plt.gcf().set_size_inches(2.3, 3)

            
            if preAnalysis:
                plt.gcf().set_size_inches(7, 7)
                plt.legend(methodNameList)

            comparePolicyString = ''
            if comparePolicy:
                comparePolicyString = 'comparePolicy_'

            plt.tight_layout()
            plt.savefig('./images/simpleSet/onPolicy/' + comparePolicyString +  scaleType + '.pdf')
            #plt.legend(methodNameList)
            plt.show()

#offPolicyScaleSets()
#quit()




def plotGraphDist():

    #simList.append([1000, 10, 10])  

    num_data_points = 1000
    num_nodes = 10
    #num_paths_per_graph = 10
    
    
    num_paths_per_graph = 1000
    #num_paths_per_graph = 1000
    
    #num_data_points = 1000
    


    #num_paths_per_graph = 10 #Works
    #num_data_points = 1000
    #num_nodes = 10 


    learning_rate = 1e-3



    methodList = []
    #methodList.append('ground truth')
    methodList.append('ours')
    methodList.append('VAE')
    methodList.append('autoreg')
    methodList.append('diffusion')
    methodList.append('localSolver')
    ########methodList.append('metropolas')
    methodList.append('naiveReward')
    methodList.append('GFlowReward')
    #methodList.append('zero')

    #methodList.append('FlowMatch')
    


    #modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'FlowMatch'
    #modelType = 'localSolver'
    #modelType = 'metropolas'

    methodName1 = 'method'
    #errorName = 'error'
    #errorName = 'graph edit distance'
    errorName = '$F_1$ score'
    probName = '$\log(\Pr(X_i \mid \hat{s}_i)$'


    methodDict = {}
    methodDict['autoreg'] = 'auto-\nregressive'
    methodDict['localSolver'] = 'local search'
    methodDict['metropolas'] = 'Metropolis\nHastings'
    methodDict['GFlowReward'] = 'GFlowNet'
    methodDict['naiveReward'] = 'naive policy\nlearning'
    methodDict['ours'] = 'GReinSS'
    #methodDict['FlowMatch'] = 'flow matching (GFlowNet)'

    methodNameList = []
    for a in range(len(methodList)):
        modelType_name = methodList[a]
        if modelType_name in methodDict:
            modelType_name = methodDict[modelType_name]
        methodNameList.append( modelType_name )
    
    df = {}
    df[methodName1] = []
    df[errorName] = []
    df[probName] = []

    numGraphs = []


    for simIndex in range(1):

        

        simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)
        
        for methodIndex in range(len(methodList)):
            modelType = methodList[methodIndex]
            #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_N10_P100_1.npz')

            modelType_name = modelType
            if modelType_name in methodDict:
                modelType_name = methodDict[modelType_name]
            

            graph_true = loadnpz('./data/sims/new/' + simPart + '_graphs.npz')

            #num_node = int(np.floor( graph_pred0.shape[1] ** 0.5 )) + 1
            num_node = graph_true.shape[1]
            #assert graph_pred0.shape[1] == (num_node - 1) * num_node
            eye1 = np.eye(num_node)
            arg1 = np.argwhere(eye1 == 0)
            graph_true0 = graph_true[:, arg1[:, 0], arg1[:, 1]]

            if modelType == 'ground truth':
                graph_pred0 = np.copy(graph_true0)
            else:
                graph_pred0 = loadnpz('./data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '.npz')

            num_graph_inverse = uniqueValMaker(graph_pred0)
            _, num_graph_count = np.unique(num_graph_inverse, return_counts=True)
            num_graph = np.unique(num_graph_inverse).shape[0]
            numGraphs.append(num_graph)
            #print (modelType, 'num_graph', num_graph)
            #print (num_graph_count)

            graph_pred = np.zeros((graph_pred0.shape[0], num_node, num_node), dtype=int)
            graph_pred[:, arg1[:, 0], arg1[:, 1]] = graph_pred0



            observations_batch = loadnpz('./data/sims/new/' + simPart + '_obs.npz')
            log_calculate_pr_x_given_g = sim1_log_calculate_pr_x_given_g

            prob_list = np.array([log_calculate_pr_x_given_g(graph_pred0[a], observations_batch[a]) for a in range(observations_batch.shape[0])])



            


            TP = np.sum( graph_pred * graph_true, axis=(1, 2) )
            FN = np.sum( (1 - graph_pred)  * graph_true , axis=(1, 2) )
            FP = np.sum( graph_pred  * (1 - graph_true)  , axis=(1, 2) )
            Fscore = (2 * TP) / ( (2 * TP) + FN + FP )
            Fscore[TP == 0] = 0

            error1 = np.sum( np.abs(graph_pred - graph_true), axis=(1, 2) )


            print (modelType)
            #print ('F mean', np.mean(Fscore))
            print ('F median', np.median(Fscore))
            

            #print (np.mean(prob_list))

            for score_index in range(Fscore.shape[0]):
                df[methodName1].append(modelType_name)
                df[errorName].append(Fscore[score_index])
                df[probName].append(prob_list[score_index])

            
            #print (graph_true.shape)
            sum1 = np.sum(graph_true, axis=(1, 2))
            argsort1 = np.argsort(sum1)
            sum2 = np.sum(graph_true, axis=0).reshape(( graph_true.shape[1]*graph_true.shape[2], ))
            argsort2 = np.argsort(sum2)
            

            graph_true = graph_true.reshape((graph_true.shape[0], graph_true.shape[1]*graph_true.shape[2]))
            graph_pred = graph_pred.reshape(graph_true.shape)

            cat1 = np.concatenate((  graph_true[argsort1][:, argsort2], graph_pred[argsort1][:, argsort2] ))

            #sns.heatmap(cat1  )
            #plt.show()


    #quit()
    #print (df)

    #palette = sns.color_palette("Set1")

    palette = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]
    #print (palette)
    #quit()

    #removeYTrick = True
    removeYTrick = False

    #sns.boxplot(data=df, x=errorName, y=methodName1, palette=palette)
    sns.boxplot(data=df, x=errorName, y=methodName1, palette=palette)
    plt.ylabel('')
    plt.xlim(-0.05, 1.05)
    plt.xlabel("$F_1$ score")
    #plt.yticks( np.arange(len(methodNameList)), methodNameList  )

    #plt.xticks([0, 0.5, 1], [0, 0.5, 1])
    plt.xticks([0, 0.25, 0.5, 0.75, 1], ['0', '.25', '.5', '.75', '1'])
    #plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
    #plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '.2', '.4', '.6', '.8', '1'])
    plt.gcf().set_size_inches(2.6, 3)
    
    if removeYTrick:
        plt.gcf().set_size_inches(1.6, 3)
        plt.yticks([], [])
    else:
        plt.gcf().set_size_inches(2.6, 3)

    plt.tight_layout()
    plt.savefig('./images/startEnd/all/F-score_' + 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '.pdf')
    plt.show()

    quit()

    sns.boxplot(data=df, x=probName, y=methodName1, palette=palette)
    plt.ylabel('')
    plt.xlim( np.min(df[probName]) * 1.05 , 0.0)
    plt.gcf().set_size_inches(2.6, 3)
    plt.tight_layout()
    #plt.savefig('./images/startEnd/standard/logProbObserve.pdf')
    plt.show()

    for a in range(len(numGraphs)):
        plt.scatter( numGraphs[a], [-a], c=palette[a] )
    #plt.scatter( numGraphs, np.arange(len(numGraphs)) )
    print (methodNameList)
    plt.yticks( -np.arange(len(methodNameList)), methodNameList  )
    plt.xlabel('# of unique graphs')
    #plt.xscale('log')
    plt.gcf().set_size_inches(2.6, 3)
    plt.tight_layout()
    #plt.savefig('./images/startEnd/standard/uniqueGraphs.pdf')
    plt.show()

    
    


#plotGraphDist()
#quit()




def saveScaleGraph():


    #scaleTypeList = ['graphs', 'paths', 'nodes']
    scaleTypeList = ['paths']
    
    methodList = []
    methodList.append('ours')
    methodList.append('VAE')
    methodList.append('autoreg')
    methodList.append('diffusion')
    methodList.append('localSolver')
    methodList.append('naiveReward')
    methodList.append('GFlowReward') #
    #methodList.append('zero')
    
    methodDict = {}
    #methodDict['autoreg'] = 'auto-\nregressive'
    #methodDict['localSolver'] = 'local search'
    #methodDict['metropolas'] = 'Metropolis\nHastings'
    #methodDict['GFlowReward'] = 'GFlowNet'
    #methodDict['naiveReward'] = 'naive policy\nlearning'
    #methodDict['ours'] = 'GReinSS'

    methodDict['autoreg'] = 'autoregressive'
    methodDict['localSolver'] = 'local search'
    methodDict['metropolas'] = 'Metropolis Hastings'
    methodDict['GFlowReward'] = 'GFlowNet'
    methodDict['naiveReward'] = 'naive policy learning'
    methodDict['ours'] = 'GReinSS'


    #methodDict['FlowMatch'] = 'flow matching (GFlowNet)'

    methodNameList = []
    for a in range(len(methodList)):
        modelType_name = methodList[a]
        if modelType_name in methodDict:
            modelType_name = methodDict[modelType_name]
        methodNameList.append( modelType_name )


    palette = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'][1:]

    if False:
        #legendList = []
        for modelType0 in range(len(methodList)):
            modelType = methodList[modelType0]
            plt.scatter([0], [0], c=palette[modelType0])
            #plt.plot([0], [0], c=palette[modelType0])
            #plt.bar([0], [0], color=palette[modelType0])
        plt.legend(methodNameList)
        plt.gcf().set_size_inches(6, 6)
        plt.tight_layout()
        plt.savefig('./images/startEnd/standard/legend.pdf')
        plt.show()
        quit()


    for scaleType in scaleTypeList:
        #methodList.append('metropolas')

        simList = [] #data points, nodes, paths

        #scaleType = 'nodes'
        #scaleType = 'paths'
        #scaleType = 'graphs'

        #Missing

        if scaleType == 'graphs':
            xName = '# of graphs'
            #simList.append([1, 10, 100])  #modified number of data points 
            simList.append([10, 10, 100]) 
            #simList.append([25, 10, 100])  #
            simList.append([100, 10, 100])
            #simList.append([250, 10, 100])
            simList.append([1000, 10, 100])


        #All included, ours best
        if scaleType == 'nodes':
            xName = '# of nodes'
            simList.append([1000, 5, 100])  #modified number of nodes
            simList.append([1000, 10, 100])  
            simList.append([1000, 15, 100])  
            simList.append([1000, 20, 100])  #


        if scaleType == 'paths':
            xName = '# walks $k$'
            #simList.append([1000, 10, 1])  #modified number of paths
            simList.append([1000, 10, 10])  #modified number of paths
            simList.append([1000, 10, 100])  #modified number of paths
            simList.append([1000, 10, 1000])  #modified number of paths
            #simList.append([1000, 10, 10000])  #modified number of paths

        doWide = False 
        

        errors = {}
        for modelType in methodList: 
            errors[modelType] = []
        errorsList = []
        
        

        for modelType0 in range(len(methodList)):
            modelType = methodList[modelType0]
            Xvariable = []
            for simParamIndex in range(len(simList)):
                print ('simParamIndex', simParamIndex)
                for simIndex in range(0, 1):

                    #print (simIndex)

                    num_data_points = simList[simParamIndex][0]
                    num_nodes =  simList[simParamIndex][1]
                    num_paths_per_graph =  simList[simParamIndex][2]

                    if scaleType == 'graphs':
                        Xvariable.append(num_data_points)
                    if scaleType == 'nodes':
                        Xvariable.append(num_nodes)
                    if scaleType == 'paths':
                        Xvariable.append(num_paths_per_graph)

                    simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

                    #print (simPart)

                    
                    #np.savez_compressed('./data/sims/initial/' + simPart + '_obs.npz', observations_batch)
                    graph_true = loadnpz('./data/sims/new/' + simPart + '_graphs.npz')

                    if modelType == 'zero':
                        graph_pred = np.zeros(graph_true.shape)
                    else:
                        graph_pred0 = loadnpz('./data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '.npz')

                        #print (graph_pred0.shape)

                        num_node = int(np.floor( graph_pred0.shape[1] ** 0.5 )) + 1
                        assert graph_pred0.shape[1] == (num_node - 1) * num_node
                        eye1 = np.eye(num_node)
                        arg1 = np.argwhere(eye1 == 0)
                        graph_pred = np.zeros((graph_pred0.shape[0], num_node, num_node), dtype=int)

                        graph_pred[:, arg1[:, 0], arg1[:, 1]] = graph_pred0




                    #print (np.sum(graph_pred) / graph_pred.shape[0] )
                    #print (np.sum(graph_true) / graph_pred.shape[0] )
                    #quit()

                    #error1 = np.sum(np.abs(graph_pred - graph_true), axis=(1, 2)  )
                    #error1 = np.mean(error1)

                    error1 = np.mean(np.abs(graph_pred - graph_true))

                    TP = np.sum( graph_pred * graph_true, axis=(1, 2) )
                    FN = np.sum( (1 - graph_pred)  * graph_true , axis=(1, 2) )
                    FP = np.sum( graph_pred  * (1 - graph_true)  , axis=(1, 2) )


                    assert np.min(TP) >= 0
                    assert np.min(FN) >= 0
                    assert np.min(FP) >= 0

                    Fscore = (2 * TP) / ( (2 * TP) + FN + FP ) 

                    

                    Fscore[TP == 0] = 0



                    #Fscore = np.median(Fscore)
                    Fscore = np.median(Fscore)

                    print (Fscore)

                    error1 = Fscore


                    #print (np.mean(np.sum(graph_pred, axis= (1, 2) )))
                    #print (error1)
                    #quit()

                    errors[modelType].append(error1)
            
            errorsList.append(errors[modelType])

            if True:#doWide:
                if modelType0 == 0:
                    zorder = 100
                else:
                    zorder = 1
                
                plt.scatter(Xvariable, errors[modelType], c=palette[modelType0], zorder=zorder)
                plt.plot(Xvariable, errors[modelType], c=palette[modelType0], zorder=zorder)

            print (modelType)
            print (Xvariable, errors[modelType] )





        figFile = './images/startEnd/scale/' + scaleType + '.pdf'

        
        
        #if not doWide:
        #    makeBreakPlot(Xvariable, errorsList, -0.02, 0.85, 0.85, 1.01, hspace=0.0, palette=palette, figFile=figFile, xlabel='# of paths', xscale='log', vertline=100)
        #    quit()

        #print (errors['diffusion'])
        #plt.plot(errors['diffusion'])
        #plt.legend(methodNameList)

        if scaleType == 'nodes':
            xLine = 10
        if scaleType == 'paths':
            xLine = 100
            plt.ylim(-0.02, 1.02)
            #plt.ylim(0.85, 1.01)


        if scaleType == 'graphs':
            xLine = 1000
        
        #plt.axvline(x=xLine, color='grey', linestyle=':', linewidth=2)

        if scaleType != 'nodes':
            plt.xscale('log')
        #plt.yscale('symlog', linthresh=1e-2)
        plt.xlabel(xName)
        print (Xvariable)
        plt.xticks(Xvariable)
        plt.gca().minorticks_off()
        plt.ylabel('$F_1$-score')
        #plt.gcf().set_size_inches(2.3, 3)
        #plt.gcf().set_size_inches(1.8, 3)
        plt.gcf().set_size_inches(1.8, 2.6)
        #plt.gcf().set_size_inches(2, 2.6)
        plt.tight_layout()
        plt.savefig('./images/startEnd/scale/' + scaleType + '.pdf')
        plt.show()

#saveScaleGraph()
#quit()


def plotConvergence():


    if True:
        NumPaths = 1000

        simPart = 'D' + str(1000) +  '_N' + str(10) + '_P' + str(NumPaths) + '_sim' + str(0)
        saveTrainingLoss = './data/sims/startEnd/loss/TrainingLoss_' + simPart + '_' + 'ours' + '.npz'
        loss = loadnpz(saveTrainingLoss)

    
    #geneNow = 'ENSG00000159173'
    #geneNow = 'ENSG00000165948'
    #geneNow = 'ENSG00000196247'
    #geneNow = 'ENSG00000058453'
    #saveTrainingLoss = './data/real/splicing/geneFiles/loss/TrainingLoss_' + geneNow + '_median.npz'
    #loss = loadnpz(saveTrainingLoss)

    if True:
        plt.plot(loss)
        plt.xlabel("Training epoch")
        plt.ylabel('Mean observation log likelihood')
        plt.gcf().set_size_inches(5, 4)
        plt.tight_layout()
        #plt.savefig('./images/review/converge/genes_' + geneNow + '.png')

        plt.savefig('./images/review/converge/graph_' + simPart + '.png')

        plt.show()





#plotConvergence()
#quit()


def plotAlignedGraphDist():

    from scipy.optimize import linear_sum_assignment

    methodList = []
    methodList.append('ours')
    
    if False:
        methodList.append('zero')
        methodList.append('VAE')
        methodList.append('autoreg')
        methodList.append('diffusion')
        methodList.append('localSolver')
        methodList.append('metropolas')
    #methodList.append('naiveReward')
    #methodList.append('GFlowReward')

    methodName1 = 'method'
    errorName = 'graph edit distance'

    methodDict = {}
    methodDict['autoreg'] = 'autoregressive'
    methodDict['localSolver'] = 'local search'
    methodDict['metropolas'] = 'Metropolis\nHastings'
    methodDict['GFlowReward'] = 'GFlowNet'
    methodDict['naiveReward'] = 'naive policy\nlearning'
    methodDict['ours'] = 'GReinSS'
    #methodDict['FlowMatch'] = 'flow matching (GFlowNet)'

    if True:
        for key1 in methodDict.keys():
            value = methodDict[key1]
            value = value.replace('\n', ' ')
            methodDict[key1] = value


    df = {}
    df[methodName1] = []
    df[errorName] = []

    for simIndex in [0]:#range(0, 4):

        print (simIndex)

        #simPart = f'N10_P10_{simIndex}'
        #simPart = f'N10_P100_{simIndex}'
        #simPart = f'causal_N10_D100_M5_{simIndex}'
        #simPart = f'D5000_N10_P1_{simIndex}'
        simPart = f'D1000_N10_P1_{simIndex}'
        #simPart = f'D10_N10_P100_{simIndex}'


        for methodIndex in range(len(methodList)):
            modelType = methodList[methodIndex]
            #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_N10_P100_1.npz')

            modelType_name = modelType
            if modelType_name in methodDict:
                modelType_name = methodDict[modelType_name]
            

            adjacency_matrices = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')

            if modelType == 'zero':
                predicted_graphs = np.zeros(adjacency_matrices.shape)
            else:
                #predicted_graphs =   loadnpz('./data/sims/causal/pred/graph_' + simPart + '_' + modelType + '.npz')
                #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_' + simPart + '.npz')
                if 'causal' in simPart:
                    predicted_graphs0 =   loadnpz('./data/sims/causal/pred/graph_' + simPart + '_' + modelType + '.npz')
                else:

                    if 'ours' in  modelType:
                        predicted_graphs0 =   loadnpz('./data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '.npz')
                    else:
                        predicted_graphs0 =   loadnpz('./data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '.npz')


                if 'causal' in simPart:
                    num_node = int(np.floor( predicted_graphs0.shape[1] ** 0.5 ))
                    predicted_graphs = predicted_graphs0.reshape(( predicted_graphs0.shape[0], num_node, num_node ))
                else:
                    num_node = int(np.floor( predicted_graphs0.shape[1] ** 0.5 )) + 1
                    assert predicted_graphs0.shape[1] == (num_node - 1) * num_node
                    eye1 = np.eye(num_node)
                    arg1 = np.argwhere(eye1 == 0)
                    predicted_graphs = np.zeros((predicted_graphs0.shape[0], num_node, num_node), dtype=int)

                    predicted_graphs[:, arg1[:, 0], arg1[:, 1]] = predicted_graphs0
            
            

            adjacency_matrices = adjacency_matrices.reshape((adjacency_matrices.shape[0]  , adjacency_matrices.size // adjacency_matrices.shape[0] ))
            predicted_graphs = predicted_graphs.reshape((predicted_graphs.shape[0]  , predicted_graphs.size // predicted_graphs.shape[0] ))



            if modelType == 'ours':
                plt.plot(  np.sum(predicted_graphs, axis=0) )
                plt.plot( np.sum(adjacency_matrices, axis=0) )
                plt.show()
            #quit()

            
            error1 = np.sum(np.abs(  adjacency_matrices.reshape((adjacency_matrices.shape[0], 1, adjacency_matrices.shape[1]))  - predicted_graphs.reshape((1, predicted_graphs.shape[0], predicted_graphs.shape[1]))  ), axis=2)
            
            if False:
                row_ind, col_ind = linear_sum_assignment(error1)
            else:
                row_ind, col_ind = np.arange(error1.shape[0]), np.arange(error1.shape[0])

            
            error1 = error1[row_ind, col_ind]

            print (np.mean(error1))
            


            df[methodName1].append(modelType_name)
            df[errorName].append(np.mean(error1))

            if False:
                for a in range(error1.shape[0]):            
                    df[methodName1].append(modelType_name)
                    df[errorName].append(error1[a])

        


    #sns.boxplot(data=df, x=methodName1, y=errorName)
    sns.boxplot(data=df, x=errorName, y=methodName1)
    plt.ylabel('')
    #plt.gcf().set_size_inches(4, 4)
    #plt.gcf().set_size_inches(6, 3)
    plt.gcf().set_size_inches(6, 6)
    plt.tight_layout()
    if 'causal' in simPart:
        plt.savefig('./images/causal/unorderGraphEdit.pdf')
    else:
        if 'P100' in simPart:
            plt.savefig('./images/startEnd/unorderGraphEdit_P100.pdf')
        else:
            plt.savefig('./images/startEnd/unorderGraphEdit.pdf')
    plt.show()

    

#plotAlignedGraphDist()
#quit()

def plotObserveFit():


    methodList = []
    methodList.append('ours')
    methodList.append('ground truth')
    methodList.append('VAE')
    methodList.append('autoreg')
    methodList.append('localSolver')
    methodList.append('metropolas')
    methodList.append('naiveReward')
    methodList.append('GFlowReward')
    #methodList.append('FlowMatch')
    
    

    methodName1 = 'method'
    errorName = 'log probability'

    methodDict = {}
    methodDict['autoreg'] = 'autoregressive'
    methodDict['localSolver'] = 'local search'
    methodDict['metropolas'] = 'Metropolis\nHastings'
    methodDict['GFlowReward'] = 'GFlowNet'
    methodDict['naiveReward'] = 'naive policy\nlearning'
    methodDict['FlowMatch'] = 'flow matching (GFlowNets)'

    df = {}
    df[methodName1] = []
    df[errorName] = []


    #simPart = 'N10_P100'
    #simPart = 'N10_P10'
    min1 = 0

    for simIndex in range(0, 10):

        #simPart = 'N10_P100'
        #simPart = 'N10_P10'

        #simPart = 'causal_1'

        #simPart = f'N10_P10_{simIndex}'
        #simPart = f'causal_N10_D100_M5_{simIndex}'

        observations_batch = loadnpz('./data/sims/initial/' + simPart + '_obs.npz')

        

        for methodIndex in range(len(methodList)):
            modelType = methodList[methodIndex]

            modelType_name = modelType
            if modelType_name in methodDict:
                modelType_name = methodDict[modelType_name]

            #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_N10_P100_1.npz')
            if modelType == 'ground truth':
                predicted_graphs = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')
            else:
                #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_' + simPart + '.npz')
                if 'causal' in simPart:
                    predicted_graphs =   loadnpz('./data/sims/causal/pred/graph_' + simPart + '_' + modelType + '.npz')
                    log_calculate_pr_x_given_g = causal_x_given_g
                else:
                    predicted_graphs =   loadnpz('./data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '.npz')
                    log_calculate_pr_x_given_g = sim1_log_calculate_pr_x_given_g


            prob_list = np.array([log_calculate_pr_x_given_g(predicted_graphs[a], observations_batch[a]) for a in range(observations_batch.shape[0])])

            #min1 = min(min1,  np.min(prob_list[prob_list != -np.inf]))

            prob_list[prob_list == -np.inf] = -1000

            prob_med = np.median(prob_list)

            df[methodName1].append(modelType_name)
            df[errorName].append(prob_med)

            #if modelType == 'ours':
            #    print (prob_med)

            min1 = min(min1,  prob_med)

            #print (min1)
            

            if False:
                for a in range(prob_list.shape[0]):
                    
                    df[methodName1].append(modelType_name)
                    df[errorName].append(prob_list[a])


    #print (min1)

    sns.boxplot(data=df, x=errorName, y=methodName1)
    plt.xlim((min1 * 1.05,  np.abs(min1) * 0.05 ))
    
    plt.ylabel('')
    plt.gcf().set_size_inches(4, 4)
    plt.tight_layout()
    if 'causal' in simPart:
        plt.savefig('./images/causal/xProb.pdf')
    else:
        if 'P10' in simPart:
            plt.savefig('./images/startEnd/graphEdit_P10.pdf')
        else:
            plt.savefig('./images/startEnd/xProb.pdf')
    
    plt.show()
    quit()

    predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_' + simPart + '.npz')

    adjacency_matrices = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')
    



#plotObserveFit()
#quit()



def plotNumberGraphs():

    methodList = []
    methodList.append('ours')
    methodList.append('ground truth')
    methodList.append('VAE')
    methodList.append('autoreg')
    methodList.append('localSolver')
    methodList.append('metropolas')
    methodList.append('naiveReward')
    methodList.append('GFlowReward')
    #methodList.append('FlowMatch')
    
    

    methodName1 = 'method'
    errorName = 'number of graphs'

    methodDict = {}
    methodDict['autoreg'] = 'autoregressive'
    methodDict['localSolver'] = 'local search'
    methodDict['metropolas'] = 'Metropolis\nHastings'
    #methodDict['GFlowReward'] = 'trajectory balance (GFlowNet)'
    methodDict['GFlowReward'] = 'GFlowNet'
    methodDict['naiveReward'] = 'naive policy\nlearning'
    methodDict['FlowMatch'] = 'flow matching (GFlowNets)'

    df = {}
    df[methodName1] = []
    df[errorName] = []


    for simIndex in range(0, 10):

        #simPart = 'N10_P100'
        #simPart = 'N10_P10'

        #simPart = 'causal_1'

        #simPart = f'N10_P10_{simIndex}'
        simPart = f'causal_N10_D100_M5_{simIndex}'


        observations_batch = loadnpz('./data/sims/initial/' + simPart + '_obs.npz')

        

        for methodIndex in range(len(methodList)):
            modelType = methodList[methodIndex]

            modelType_name = modelType
            if modelType_name in methodDict:
                modelType_name = methodDict[modelType_name]

            true_graphs = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')
            true_graphs = true_graphs.reshape((100, 100))

            #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_N10_P100_1.npz')
            if modelType == 'ground truth':
                #predicted_graphs = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')
                predicted_graphs = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')
            else:
                #predicted_graphs =   loadnpz('./data/sims/causal/pred/graph_' + simPart + '_' + modelType + '.npz')
                #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_' + simPart + '.npz')
                if 'causal' in simPart:
                    predicted_graphs =   loadnpz('./data/sims/causal/pred/graph_' + simPart + '_' + modelType + '.npz')
                else:
                    predicted_graphs =   loadnpz('./data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '.npz')

            if len(predicted_graphs.shape) == 3:
                predicted_graphs = predicted_graphs.reshape((  predicted_graphs.shape[0], predicted_graphs.shape[1]*predicted_graphs.shape[2] ))


            print (modelType)
            inverse1 = uniqueValMaker(predicted_graphs)
            _, counts = np.unique(inverse1, return_counts=True)

            print (counts.shape)

            df[methodName1].append(modelType_name)
            df[errorName].append(counts.shape[0])


            if False:
                entropy1 = np.sum(counts   *  np.log(counts / inverse1.shape[0] )  ) / inverse1.shape[0]

                #print (np.unique(counts, return_counts=True))
                print (entropy1)

                

                for index1 in range(predicted_graphs.shape[0]):
                    #for index2 in range(predicted_graphs.shape[0]):
                    #    if index2 > index1:       
                    
                    #error1 = np.sum(np.abs( predicted_graphs[index1] - predicted_graphs[index2] ))

                    error1 = predicted_graphs[index1:index1+1] - predicted_graphs[np.arange(predicted_graphs.shape[0]) != index1]

                    #error1 = predicted_graphs[index1:index1+1] - predicted_graphs[np.arange(predicted_graphs.shape[0]) != index1]
                    #error1 = predicted_graphs[index1:index1+1] - true_graphs

                    error1 = np.sum(np.abs(error1), axis=1)
                    error1 = np.min(error1)

                    df[methodName1].append(modelType_name)
                    df[errorName].append(error1)

    #quit()
    sns.boxplot(data=df, x=errorName, y=methodName1)
    #plt.xlim((min1 * 1.05,  np.abs(min1) * 0.05 ))
    
    plt.ylabel('')
    plt.gcf().set_size_inches(4, 4)
    plt.tight_layout()

    if 'causal' in simPart:
        plt.savefig('./images/causal/numGraph.pdf')
    else:
        plt.savefig('./images/startEnd/numGraph.pdf')
    
    plt.show()
    
    quit()


#plotNumberGraphs()
#quit()


def plotRNAruntime():

    runTimesList = loadnpz('./data/temp/runtimes_March.npz' )
    #runTimesList = loadnpz('./data/temp/runtimes_Naive.npz' )
    runTimesList = runTimesList[:100]

    print (np.median(runTimesList))
    print (np.mean(runTimesList))

    sns.boxplot(runTimesList)
    plt.xlabel('')
    plt.xticks([])
    #plt.xlim(1 - 0.2 , 1 + 0.2)
    plt.ylabel('runtime (seconds)')
    plt.gcf().set_size_inches(2.5, 3)
    plt.tight_layout()
    plt.savefig('./images/review/times/runtime.png')
    plt.show()




#plotRNAruntime()
#quit()



def plotSpiceError():

    scoreList = loadnpz('./data/real/splicing/eval/fast_scoreList.npz')
    usedList = loadnpz('./data/real/splicing/eval/fast_usedList.npz')


    scoreList = scoreList[usedList != 0]


    print (scoreList.shape)

    #scoreList = scoreList[:500]

    #print (scoreList.shape)
    #quit()

    #print (np.mean(scoreList[:, 1]), np.mean(scoreList[:, 0]))
    #print (np.median(scoreList[:, 1]), np.median(scoreList[:, 0]))
    #quit()
    if False:
        #plt.scatter(scoreList[:, 1], scoreList[:, 0], s=20, alpha=0.2)
        #plt.scatter(scoreList[:, 1], scoreList[:, 0], s=20, alpha=0.03)
        plt.scatter(scoreList[:, 1], scoreList[:, 0], s=20, alpha=0.02)
        plt.plot([0, 1], [0, 1], linestyle='dashed', color='black')
        plt.xlabel('RSEM error')
        plt.ylabel('GReinSS error')
        plt.gcf().set_size_inches(3.2, 3)
        plt.tight_layout()
        plt.savefig('./images/splicing/error.pdf')
        plt.show()

    #from matplotlib.colors import LogNorm

    #plt.hist2d(scoreList[:, 1], scoreList[:, 0], bins=20, norm=LogNorm(), cmap='Reds')
    #plt.show()


    improve = scoreList[:, 0] - scoreList[:, 1]

    #print (np.argwhere(improve >= 0.05).shape[0] / 500 )
    #print (np.argwhere(improve <= -0.05).shape[0] / 500 )

    print (np.median(improve))

    max1 = np.max(np.abs(improve))
    t_stat, p_value = scipy.stats.ttest_1samp(improve, popmean=0)
    print (t_stat, p_value)

    #plt.boxplot(improve)#, color='tab:blue')


    showAll = True

    maxAbs = np.max(np.abs(improve))
    maxAbs = maxAbs * 1.05
    plt.axhline(y=0, c='red', linestyle='dashed', linewidth=2)
    sns.boxplot(improve, 
                flierprops=dict(
        marker='o',
        markersize=2,      # smaller points
        alpha=0.05          # transparency
    ))
    #plt.axvline(x=0, c='black')
    #plt.axvline(x=  np.mean(improve) , c='red')#, linestyle='dashed')
    #plt.xlabel('GReinSS error - RSEM error')
    plt.ylabel('GReinSS error - RSEM error')
    plt.xticks([], [])
    #plt.ylim(-maxAbs, maxAbs)
    if not showAll:
        plt.ylim(-0.4, 0.4)
    plt.gcf().set_size_inches(2, 2.9) #original width
    #plt.gcf().set_size_inches(1.5, 3)
    plt.tight_layout()
    if not showAll:
        plt.savefig('./images/splicing/errorBox.pdf')
    else:
        plt.savefig('./images/splicing/errorBox_all.pdf')
    #plt.savefig('./images/splicing/errorBox_small.pdf')
    plt.show()

    #quit()

    plt.hist(improve, bins=50, range=(-max1, max1))
    plt.axvline(x=0, c='black')
    plt.axvline(x=  np.mean(improve) , c='red')#, linestyle='dashed')
    plt.xlabel('GReinSS error - RSEM error')
    plt.ylabel('number of genes')
    plt.gcf().set_size_inches(3.2, 3)
    plt.tight_layout()
    #plt.savefig('./images/splicing/errorHist.pdf')
    plt.show()


#plotSpiceError()
#quit()


def baselineSpliceCompare():

    scoreList_N = loadnpz('./data/real/splicing/eval/scoreList_Naive.npz')
    scoreList_G = loadnpz('./data/real/splicing/eval/scoreList_GFlow.npz')
    scoreList = loadnpz('./data/real/splicing/eval/fast_scoreList.npz')
    usedList = loadnpz('./data/real/splicing/eval/fast_usedList.npz')

    scoreList_N = scoreList_N[usedList != 0]
    scoreList_G = scoreList_G[usedList != 0]
    scoreList = scoreList[usedList != 0]

    scoreList_N = scoreList_N[:100]
    scoreList_G = scoreList_G[:100]
    scoreList = scoreList[:100]

    scoresAll = []
    namesAll = []
    for a in range(len(scoreList)):
        scoresAll.append(scoreList[a, 0])
        scoresAll.append(scoreList[a, 1])
        scoresAll.append(scoreList_N[a, 0])
        scoresAll.append(scoreList_G[a, 0])

        namesAll.append("GReinSS")
        namesAll.append("RSEM")
        namesAll.append("GFlowNet")
        namesAll.append("naive\n policy gradients")

    df = {}
    df['isoform prediction error'] = scoresAll
    df['names'] = namesAll
    print (np.median(scoreList[:, 0]))
    print (np.median(scoreList[:, 1]))
    print (np.median(scoreList_N[:, 0]))
    print (np.median(scoreList_G[:, 0]))

    #'tab:blue', 'lightblue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive'

    sns.boxplot(df,  x='names', y='isoform prediction error' , hue='names', palette=['tab:blue', 'red', 'tab:olive', 'tab:purple' ])
    #sns.boxplot(scoreList)
    #sns.boxplot(scoreList)

    plt.xlabel('')
    #plt.xticks([])
    plt.gcf().set_size_inches(5, 3)
    plt.tight_layout()
    plt.savefig('./images/review/RNAbaselines/baselines.png')
    plt.show()





#baselineSpliceCompare()
#quit()


def exampleIsoforms():

    from matplotlib.ticker import FixedLocator

    isoform1 = [[100, 200], [300, 400], [500, 550]]
    isoform2 = [[100, 200], [350, 450]]
    isoforms = [isoform1, isoform2]

    max1 = max( np.max(np.array(isoform1)), np.max(np.array(isoform2)) )

    for isoformIndex in range(len(isoforms)):
        isoform_now = isoforms[isoformIndex]
        posNow = isoformIndex

        for exon_index in range(len(isoform_now)):
            plt.plot(isoform_now[exon_index], [posNow, posNow], color='black')
        for exon_index in range(len(isoform_now)-1):
            start1 = isoform_now[exon_index][1]
            end1 = isoform_now[exon_index+1][0]
            length1 = end1 - start1
            arange1 = np.arange(101) / 100 
            yVals = 1 - (((arange1 * 2) - 1) ** 2)
            yVals = yVals * 0.7
            yVals = yVals + posNow
            xVals = start1 + (arange1 * length1)

            plt.plot( xVals , yVals, color='black', linestyle='dashed' )

    plt.gcf().set_size_inches(3.5, 1.5)
    plt.gca().xaxis.set_minor_locator(FixedLocator( 50 + (np.arange(6) * 100)  ))
    plt.ylim(-0.5, 2)
    plt.xlim(50, max1+50)
    plt.tight_layout()
    plt.savefig('./images/diagram/isoform_Jaccard.pdf')
    plt.show()


    intersection1 = [[100, 200], [350, 400]]
    union1 = [[100, 200], [300, 450], [500, 550]]
    linePlots = [intersection1, union1]

    for isoformIndex in range(len(isoforms)):
        isoform_now = linePlots[isoformIndex]
        posNow = isoformIndex
        for exon_index in range(len(isoform_now)):
            plt.plot(isoform_now[exon_index], [posNow, posNow], color='black')
    plt.gcf().set_size_inches(3.5, 1.5)
    plt.gca().xaxis.set_minor_locator(FixedLocator( 50 + (np.arange(6) * 100)  ))
    plt.ylim(-0.5, 2)
    plt.xlim(50, max1+50)
    plt.tight_layout()
    plt.savefig('./images/diagram/overlap_Jaccard.pdf')
    plt.show()


    

    #plt.plot(  start1 + (arange1 * length1) , yPos   , c=color, linestyle=':')
    #plt.scatter(  [start1, end1], [posLevel, posLevel] , c=color, edgecolors='black', s=10) #, marker="^"
    #plt.show()


#exampleIsoforms()
#quit()

def plotReadCount():

    readCountShort = loadnpz(  './data/real/splicing/eval/readCountShort.npz') #/ 61 
    readCountLong = loadnpz(  './data/real/splicing/eval/readCountLong.npz') #/ 61

    #bins = np.logspace(np.log10(10), np.log10(1e7), 50)

    #plt.hist(readCountShort, bins=bins, alpha=0.5)#, range=(100, 1e6))#, range=(-max1, max1))
    #plt.hist(readCountLong, bins=bins, alpha=0.5)
    #plt.xscale('log')
    #plt.show()


    df = pd.DataFrame({
    'read count': np.concatenate([readCountShort, readCountLong]),
    'type': (['junction covering\nshort reads'] * len(readCountShort) +
             ['isoform covering\nlong reads'] * len(readCountLong))
    })

    ax = sns.boxplot(x='type', y='read count', hue='type', data=df, legend=True)
    plt.yscale('log')
    plt.xlabel('')

    plt.xticks([], [])
    plt.ylim(1, np.max( readCountShort ) * 1.2)
    ax.legend(title=None)
    #plt.gcf().set_size_inches(2, 3)
    plt.gcf().set_size_inches(2.7, 3)
    plt.tight_layout()
    plt.savefig('./images/splicing/extra/readCount.pdf')
    plt.show()


    #sns.boxplot(readCountShort)
    #plt.yscale('log')
    #plt.ylabel('# junction covering reads')
    #plt.gcf().set_size_inches(2, 3)
    #plt.tight_layout()
    #plt.savefig('./images/splicing/extra/readShort.pdf')
    #plt.show()

    #sns.boxplot(readCountLong)
    #plt.yscale('log')
    #plt.ylabel('# isoform covering reads')
    #plt.gcf().set_size_inches(2, 3)
    #plt.tight_layout()
    #plt.savefig('./images/splicing/extra/readLong.pdf')
    #plt.show()

    quit()


#plotReadCount()
#quit()

################################################
###                                          ###
###                  OLD                     ###
###                                          ###
################################################




def OLD_plotGraphDist():


    
    num_paths_per_graph = 100
    num_data_points = 100
    num_nodes = 10


    doTrain = False
    #doTrain = True

    learning_rate = 1e-3

    #simPart = 'N10_P100'
    #simPart = 'N10_P10'
    #simPart = 'causal_1'
    #simPart = 'causal_2'
    #simPart = 'causal_3'


    methodList = []
    methodList.append('ours')
    #methodList.append('VAE')
    #methodList.append('autoreg')
    methodList.append('diffusion')
    #methodList.append('localSolver')
    #methodList.append('metropolas')
    #methodList.append('naiveReward')
    #methodList.append('GFlowReward')
    #methodList.append('zero')

    #methodList.append('FlowMatch')
    


    #modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'FlowMatch'
    #modelType = 'localSolver'
    #modelType = 'metropolas'

    methodName1 = 'method'
    #errorName = 'error'
    errorName = 'graph edit distance'

    methodDict = {}
    methodDict['autoreg'] = 'autoregressive'
    methodDict['localSolver'] = 'local search'
    methodDict['metropolas'] = 'Metropolis\nHastings'
    methodDict['GFlowReward'] = 'GFlowNet'
    methodDict['naiveReward'] = 'naive policy\nlearning'
    methodDict['ours'] = 'GReinSS'
    #methodDict['FlowMatch'] = 'flow matching (GFlowNet)'

    if True:
        for key1 in methodDict.keys():
            value = methodDict[key1]
            value = value.replace('\n', ' ')
            methodDict[key1] = value
            


    


    df = {}
    df[methodName1] = []
    df[errorName] = []

    for simIndex in range(0, 3):

        #simPart = 'N10_P100'
        #simPart = 'N10_P10'

        #simPart = 'causal_1'

        #simPart = f'N10_P10_{simIndex}'
        simPart = f'N10_P100_{simIndex}'
        #simPart = f'causal_N10_D100_M5_{simIndex}'


        for methodIndex in range(len(methodList)):
            modelType = methodList[methodIndex]
            #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_N10_P100_1.npz')

            modelType_name = modelType
            if modelType_name in methodDict:
                modelType_name = methodDict[modelType_name]
            

            adjacency_matrices = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')

            if modelType == 'zero':
                predicted_graphs = np.zeros(adjacency_matrices.shape)
            else:
                #predicted_graphs =   loadnpz('./data/sims/causal/pred/graph_' + simPart + '_' + modelType + '.npz')
                #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_' + simPart + '.npz')
                if 'causal' in simPart:
                    predicted_graphs0 =   loadnpz('./data/sims/causal/pred/graph_' + simPart + '_' + modelType + '.npz')
                else:
                    #if 
                    predicted_graphs0 =   loadnpz('./data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '.npz')


                if 'causal' in simPart:
                    num_node = int(np.floor( predicted_graphs0.shape[1] ** 0.5 ))
                    predicted_graphs = predicted_graphs0.reshape(( predicted_graphs0.shape[0], num_node, num_node ))
                else:
                    num_node = int(np.floor( predicted_graphs0.shape[1] ** 0.5 )) + 1
                    assert predicted_graphs0.shape[1] == (num_node - 1) * num_node
                    eye1 = np.eye(num_node)
                    arg1 = np.argwhere(eye1 == 0)
                    predicted_graphs = np.zeros((predicted_graphs0.shape[0], num_node, num_node), dtype=int)

                    predicted_graphs[:, arg1[:, 0], arg1[:, 1]] = predicted_graphs0
            
            

            if False:#modelType == 'ours':
                print (np.sum(predicted_graphs[:10], axis=(1, 2)))
                print (np.sum(adjacency_matrices[:10], axis=(1, 2)))
                #print (adjacency_matrices[0])
                quit()
            
            #np.savez_compressed('./data/sims/initial/N10_P100_obs.npz', observations_batch)


            #error1 = checkFscore(predicted_graphs, adjacency_matrices)
            #print (np.mean(Fscore))
            error1 = np.sum(np.abs(predicted_graphs - adjacency_matrices), axis=(1, 2)  )

            print (error1)
            #quit()

            df[methodName1].append(modelType_name)
            df[errorName].append(np.mean(error1))

            if False:
                for a in range(error1.shape[0]):            
                    df[methodName1].append(modelType_name)
                    df[errorName].append(error1[a])

        
    print (df)

    #sns.boxplot(data=df, x=methodName1, y=errorName)
    sns.boxplot(data=df, x=errorName, y=methodName1)
    plt.ylabel('')
    #plt.gcf().set_size_inches(4, 4)
    #plt.gcf().set_size_inches(6, 3)
    plt.gcf().set_size_inches(6, 6)
    plt.tight_layout()
    if 'causal' in simPart:
        plt.savefig('./images/causal/graphEdit.pdf')
    else:
        if 'P100' in simPart:
            #plt.savefig('./images/startEnd/graphEdit_P100.pdf')
            True
        else:
            #plt.savefig('./images/startEnd/graphEdit.pdf')
            True 
    plt.show()
    





def plotKL():


    
    num_paths_per_graph = 100
    num_data_points = 100
    num_nodes = 10


    doTrain = False
    #doTrain = True

    learning_rate = 1e-3

    #simPart = 'N10_P100'
    #simPart = 'N10_P10'
    #simPart = 'causal_1'
    #simPart = 'causal_2'
    #simPart = 'causal_3'


    methodList = []
    methodList.append('ground truth')
    methodList.append('ours')
    methodList.append('VAE')
    methodList.append('autoreg')
    #methodList.append('diffusion')
    #methodList.append('localSolver')
    #methodList.append('metropolas')
    methodList.append('naiveReward')
    methodList.append('GFlowReward')
    #methodList.append('zero')

    #methodList.append('FlowMatch')
    


    #modelType = 'ours'
    #modelType = 'naiveReward'
    #modelType = 'GFlowReward'
    #modelType = 'autoreg'
    #modelType = 'VAE'
    #modelType = 'FlowMatch'
    #modelType = 'localSolver'
    #modelType = 'metropolas'

    methodName1 = 'method'
    #errorName = 'error'
    #errorName = 'graph edit distance'
    errorName = 'F score'
    probName = '$\log(\Pr(X_i \mid \hat{s}_i)$'


    methodDict = {}
    methodDict['autoreg'] = 'auto-\nregressive'
    methodDict['localSolver'] = 'local search'
    methodDict['metropolas'] = 'Metropolis\nHastings'
    methodDict['GFlowReward'] = 'GFlowNet'
    methodDict['naiveReward'] = 'naive policy\nlearning'
    methodDict['ours'] = 'GReinSS'
    #methodDict['FlowMatch'] = 'flow matching (GFlowNet)'

    methodNameList = []
    for a in range(len(methodList)):
        modelType_name = methodList[a]
        if modelType_name in methodDict:
            modelType_name = methodDict[modelType_name]
        methodNameList.append( modelType_name )
    
    df = {}
    df[methodName1] = []
    df[errorName] = []
    df[probName] = []

    numGraphs = []


    for simIndex in range(1):

        num_data_points = 100
        num_nodes = 10
        num_paths_per_graph = 100

        simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)
        
        for methodIndex in range(len(methodList)):
            modelType = methodList[methodIndex]

            print (modelType)

            
            #predicted_graphs = loadnpz('./data/pred/graphs/' + modelType + '_N10_P100_1.npz')

            modelType_name = modelType
            if modelType_name in methodDict:
                modelType_name = methodDict[modelType_name]
            

            graph_true = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')

            

            #num_node = int(np.floor( graph_pred0.shape[1] ** 0.5 )) + 1
            num_node = graph_true.shape[1]
            #assert graph_pred0.shape[1] == (num_node - 1) * num_node
            eye1 = np.eye(num_node)
            arg1 = np.argwhere(eye1 == 0)
            graph_true0 = graph_true[:, arg1[:, 0], arg1[:, 1]]

            if modelType == 'ground truth':
                sampled_graphs0 = np.copy(graph_true0)[np.arange(10000) % 100]
            else:
                sampled_graphs0 = loadnpz('./data/sims/startEnd/model/sampledGraphs_' + simPart + '_' + modelType + '.npz')

            graphs_cat = np.concatenate(( sampled_graphs0, graph_true0 ), axis=0)
            graphs_cat_inverse = uniqueValMaker(graphs_cat)

            sampled_inverse = graphs_cat_inverse[:10000]
            true_inverse = graphs_cat_inverse[10000:]

            print (np.unique(sampled_inverse).shape)

            prob_dist = np.zeros(10000 *2, dtype=float)
            sampled_unique, sampled_count = np.unique(sampled_inverse, return_counts=True)
            prob_dist[sampled_unique] = sampled_count
            prob_dist = prob_dist / np.sum(prob_dist)

            prob_vals = prob_dist[true_inverse]
            prob_vals_log = np.log(prob_vals + 0.0001)


            
            print (np.mean(prob_vals_log))

            #KL_value = np.mean(prob_vals_log)
            
            for score_index in range(prob_vals_log.shape[0]):
                df[methodName1].append(modelType_name)
                df[errorName].append(prob_vals_log[score_index])


    #print (df)

    sns.boxplot(data=df, x=errorName, y=methodName1)
    plt.ylabel('')
    #plt.xlim(-0.01, 1.05)
    #plt.xticks([0, 0.5, 1.0], [0, 0.5, 1.0])
    plt.gcf().set_size_inches(2.6, 3)
    plt.tight_layout()
    plt.show()


    
    


#plotKL()
#quit()



def showScaleGraph():


    methodList = []
    methodList.append('ours')
    methodList.append('VAE')
    methodList.append('autoreg')
    #methodList.append('diffusion')
    methodList.append('localSolver')
    methodList.append('naiveReward')
    methodList.append('GFlowReward') #
    #methodList.append('zero')


    #methodList.append('metropolas')

    simList = [] #data points, nodes, paths




    #simList.append([100, 10, 10])  #default 

    #simList.append([5, 10, 100]) 


    #Missing
    simList.append([1, 10, 100])  #modified number of data points 
    simList.append([10, 10, 100]) 
    #simList.append([25, 10, 100])  #
    simList.append([100, 10, 100])
    #simList.append([250, 10, 100])
    simList.append([1000, 10, 100])


    #All included, ours best
    #simList.append([100, 5, 100])  #modified number of nodes
    #simList.append([100, 10, 100])  
    #simList.append([100, 15, 100])  
    #simList.append([100, 20, 100])  #


    #Missing GFlow
    #simList.append([100, 10, 1])  #modified number of paths
    #simList.append([100, 10, 10])  #modified number of paths
    #simList.append([100, 10, 100])  #modified number of paths
    #simList.append([100, 10, 1000])  #modified number of paths
    #simList.append([100, 10, 10000])  #modified number of paths


    

    errors = {}
    for modelType in methodList: 
        errors[modelType] = []

    

    if False:
        simParamIndex = 1
        for simIndex in range(10):

            #print (simIndex)

            num_data_points = simList[simParamIndex][0]
            num_nodes =  simList[simParamIndex][1]
            num_paths_per_graph =  simList[simParamIndex][2]

            simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)
            graph_true = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')
            #print (np.sum(graph_true) / graph_true.shape[0] )

    #quit()


    for modelType in methodList:
        for simParamIndex in range(len(simList)):
            print ('simParamIndex', simParamIndex)
            for simIndex in range(0, 1):

                print (simIndex)

                num_data_points = simList[simParamIndex][0]
                num_nodes =  simList[simParamIndex][1]
                num_paths_per_graph =  simList[simParamIndex][2]

                simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

                print (simPart)

                
                #np.savez_compressed('./data/sims/initial/' + simPart + '_obs.npz', observations_batch)
                graph_true = loadnpz('./data/sims/initial/' + simPart + '_graphs.npz')

                if modelType == 'zero':
                    graph_pred = np.zeros(graph_true.shape)
                else:
                    graph_pred0 = loadnpz('./data/sims/startEnd/pred/graph_' + simPart + '_' + modelType + '.npz')

                    #print (graph_pred0.shape)

                    num_node = int(np.floor( graph_pred0.shape[1] ** 0.5 )) + 1
                    assert graph_pred0.shape[1] == (num_node - 1) * num_node
                    eye1 = np.eye(num_node)
                    arg1 = np.argwhere(eye1 == 0)
                    graph_pred = np.zeros((graph_pred0.shape[0], num_node, num_node), dtype=int)

                    graph_pred[:, arg1[:, 0], arg1[:, 1]] = graph_pred0




                print (np.sum(graph_pred) / graph_pred.shape[0] )
                print (np.sum(graph_true) / graph_pred.shape[0] )
                #quit()

                #error1 = np.sum(np.abs(graph_pred - graph_true), axis=(1, 2)  )
                #error1 = np.mean(error1)

                error1 = np.mean(np.abs(graph_pred - graph_true))

                TP = np.sum( graph_pred * graph_true )
                FN = np.sum( (1 - graph_pred)  * graph_true )
                FP = np.sum( graph_pred  * (1 - graph_true) )
                Fscore = (2 * TP) / ( (2 * TP) + FN + FP )

                error1 = Fscore


                #print (np.mean(np.sum(graph_pred, axis= (1, 2) )))
                print (error1)
                #quit()

                errors[modelType].append(error1)

        plt.plot(errors[modelType])

    #print (errors['diffusion'])
    #plt.plot(errors['diffusion'])
    plt.legend(methodList)
    #plt.yscale('log')
    #plt.yscale('symlog', linthresh=1e-2)
    plt.show()

#showScaleGraph()
#quit()



def evaluateVector():


    #num_data_points = 1000

    num_data_points = 100
    num_nodes = 100
    num_paths_per_graph = 0.5


    methodList = []
    methodList.append('ours_onPolicy')
    methodList.append('ours_offPolicy')

    if False:
        methodList.append('VAE')
        methodList.append('autoreg')
        #methodList.append('diffusion')
        methodList.append('localSolver')
        ########methodList.append('metropolas')
        methodList.append('naiveReward')
        methodList.append('GFlowReward')





    simIndex = 0

    


    simType = 'vector'

    
    

    simList = [] 
    simList.append([100, 10, 0.5])
    simList.append([100, 25, 0.5])
    simList.append([100, 50, 0.5])
    simList.append([100, 100, 0.5])

    vectorSizes = []
    for a in range(len(simList)):
        vectorSizes.append(simList[a][1])

    df = {}
    for a in range(len(methodList)):
        df[methodList[a]] = []


    for simParamIndex in range(len(simList)):
        print ('simParamIndex', simParamIndex)

        simNow = simList[simParamIndex]

        num_data_points = simNow[0]
        num_nodes = simNow[1]
        num_paths_per_graph = simNow[2]


        simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)

        latent = loadnpz('./data/sims/' + simType + '/input/' + simPart + '_latent.npz')
        obs = loadnpz('./data/sims/' + simType + '/input/' + simPart + '_obs.npz')

        latent = processLatent(latent) 

        for modelIndex in range(len(methodList)):


            

            modelType = methodList[modelIndex]
            modelType_name = modelType
            print (modelType)
            pred  =   loadnpz('./data/sims/' + simType + '/pred/graph_' + simPart + '_' + modelType + '.npz')

            #pred2  =   loadnpz('./data/sims/' + simType + '/pred/graph_' + simPart + '_' + modelType + '_onPolicy.npz')

            
            error1 = np.mean(np.abs(pred - latent))
            #print (error1)

            TP = np.sum( pred * latent, axis=1 )
            FN = np.sum( (1 - pred)  * latent , axis=1 )
            FP = np.sum( pred  * (1 - latent)  , axis=1 )
            Fscore = (2 * TP) / ( (2 * TP) + FN + FP )

            Fscore_mean = np.mean(Fscore)

            #print (Fscore_mean)

            df[modelType].append(Fscore_mean)
            
            
            #for score_index in range(Fscore.shape[0]):
            #    df[methodName1].append(modelType_name)
            #    df[errorName].append(Fscore[score_index])

    palette = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive']


    plt.plot(vectorSizes, df['ours_onPolicy'])
    plt.plot(vectorSizes, df['ours_offPolicy'])
    plt.scatter(vectorSizes, df['ours_onPolicy'])
    plt.scatter(vectorSizes, df['ours_offPolicy'])
    plt.xscale('log')
    plt.xticks(vectorSizes, vectorSizes)
    plt.legend(['on-policy learning', 'off-policy learning'])
    plt.xlabel('vector size')
    plt.ylabel('$F_1$-score')
    plt.show()
    quit()


    sns.boxplot(data=df, x=errorName, y=methodName1, palette=palette)
    plt.ylabel('')
    plt.xlim(-0.01, 1.05)
    plt.xticks([0, 0.5, 1], [0, 0.5, 1])
    plt.gcf().set_size_inches(2.6, 3)
    plt.tight_layout()
    plt.show()


#evaluateVector()
#quit()



def checkTemporal():

    num_data_points = 1000
    #num_nodes = 10
    #num_paths_per_graph = 10

    num_nodes = 50
    num_paths_per_graph = 20

    simIndex = 0
    modelType = 'ours'
    simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)
    #simPart = 'fake'
    pred  =   loadnpz('./data/sims/temporal/pred/graph_' + simPart + '_' + modelType + '_mod.npz')

    #print (pred)

    #quit()
    lists = loadnpz('./data/sims/temporal/' + simPart + '_lists.npz')
    observations_batch = loadnpz('./data/sims/temporal/' + simPart + '_obs.npz')


    #index1 = 0

    for index1 in range(5):
        print ("A")

        obsNow = observations_batch[index1]
        listNow = lists[index1]

        sum1 = np.sum(obsNow)

        #print (obsNow)
        #quit()
        listNow = listNow[: sum1 ]

        #print (listNow)
        listNow = listNow[listNow!=-1]
        predNow = pred[index1]

        #print (predNow)

        predNow_argsort = np.argsort(predNow)
        predNow_argsort = predNow_argsort[predNow[predNow_argsort] != -1]
        
        print (listNow)
        print (predNow_argsort)
        #quit()

    quit()





def OLD_evaluateSet():


    #num_data_points = 1000
    num_data_points = 10
    #num_nodes = 1000
    num_nodes = 100
    num_paths_per_graph = 0.5
    #num_paths_per_graph = 1.0
    #num_paths_per_graph = 0.25


    methodList = []
    #methodList.append('ground truth')
    #methodList.append('ours_onPolicy')
    methodList.append('ours')
    #methodList.append('ours')

    if True:
        methodList.append('VAE')
        methodList.append('autoreg')
        #methodList.append('diffusion')
        methodList.append('localSolver')
        ########methodList.append('metropolas')
        methodList.append('naiveReward')
        methodList.append('GFlowReward')


    methodName1 = 'method'
    errorName = 'F score'
    df = {}
    df[methodName1] = []
    df[errorName] = []



    simIndex = 0

    simPart = 'D' + str(num_data_points) +  '_N' + str(num_nodes) + '_P' + str(num_paths_per_graph) + '_sim' + str(simIndex)


    simType = 'simpleSet'

    
    latent = loadnpz('./data/sims/' + simType + '/input/' + simPart + '_latent.npz')
    obs = loadnpz('./data/sims/' + simType + '/input/' + simPart + '_obs.npz')

    latent = processLatent(latent) 



    for modelIndex in range(len(methodList)):

        modelType = methodList[modelIndex]
        modelType_name = modelType
        print (modelType)
        pred  =   loadnpz('./data/sims/' + simType + '/pred/graph_' + simPart + '_' + modelType + '.npz')

        #pred2  =   loadnpz('./data/sims/' + simType + '/pred/graph_' + simPart + '_' + modelType + '_onPolicy.npz')

        #print (pred[0])
        #print (latent[0])
        #quit()

        
        error1 = np.mean(np.abs(pred - latent))
        print (error1)

        TP = np.sum( pred * latent, axis=1 )
        FN = np.sum( (1 - pred)  * latent , axis=1 )
        FP = np.sum( pred  * (1 - latent)  , axis=1 )
        Fscore = (2 * TP) / ( (2 * TP) + FN + FP )
        
        
        for score_index in range(Fscore.shape[0]):
            df[methodName1].append(modelType_name)
            df[errorName].append(Fscore[score_index])

    palette = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:pink', 'tab:purple', 'tab:olive']


    sns.boxplot(data=df, x=errorName, y=methodName1, palette=palette)
    plt.ylabel('')
    plt.xlim(-0.01, 1.05)
    plt.xticks([0, 0.5, 1], [0, 0.5, 1])
    plt.gcf().set_size_inches(2.6, 3)
    plt.tight_layout()
    plt.show()


#evaluateSet()
#quit()


