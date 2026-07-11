import numpy as np
import random
import math
import time
from collections import deque

####
####
#Helper functions
####
####

def Kendall_Tau_Dist(first, second):
    mappedrank = []
    for i in range(len(second)):
        mappedrank.append(first.index(second[i]))
    cost, _ = mergesort(mappedrank)
    return cost

#mergesort to compute distance in nlogn time
#input: A single ranking
#output: Kendall tau distance to the ranking 1, 2, ..., n
def mergesort(ranking):
    if len(ranking) <= 1:
        return 0, ranking
    leftsum, leftrank = mergesort(ranking[:len(ranking)//2])
    rightsum, rightrank = mergesort(ranking[len(ranking)//2:])
    csum = leftsum + rightsum
    leftindex = 0
    rightindex = 0
    outrank = []
    while leftindex < len(leftrank) and rightindex < len(rightrank):
        if leftrank[leftindex] < rightrank[rightindex]:
            outrank.append(leftrank[leftindex])
            leftindex += 1
        else:
            outrank.append(rightrank[rightindex])
            csum += len(leftrank) - leftindex
            rightindex += 1
    if leftindex < len(leftrank):
        outrank += leftrank[leftindex:]
    if rightindex < len(rightrank):
        outrank += rightrank[rightindex:]
    return csum, outrank
    
def Get_Objective_Value(query, rankings):
    median_cost = 0
    for rank in rankings:
        median_cost += Kendall_Tau_Dist(query, rank)
    return median_cost

#helper function to return the weighted tournament corresponding to the rank aggregation problem
def Get_Frac_Tournament(rankings):
    element_count = len(rankings[0])
    frac_tournament = np.ndarray((element_count, element_count))
    for i in range(element_count):
        for j in range(element_count):
            frac_tournament[i][j] = 0

    for ranking in rankings:
        for i in range(len(ranking)):
            for j in range(i+1, len(ranking)):
                frac_tournament[ranking[i]][ranking[j]] += 1
    for i in range(element_count):
        for j in range(element_count):
            frac_tournament[i][j] = frac_tournament[i][j] / len(rankings)

    return frac_tournament
    
#helper function to recover ordering from acyclic tournament
def Topological_Sort(adj):
    n = len(adj)  
    in_degree = [0] * n

    for i in range(n):
        for j in range(n):
            if adj[i][j] > 0.5:
                in_degree[j] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_sort = []

    while queue:
        node = queue.popleft()
        topo_sort.append(node)

        for j in range(n):
            if adj[node][j] > 0.5:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

    return topo_sort
    
####
####
# End of helper functions
####

#This implementation of our algorithm uses KWIKSORT to solve the standard rank aggregation problem
#For details on KWIKSORT, see Ailon, Newman, Charikar 2007
def OurAlgo_KS(alphas, betas, rankings, id_attribute, num_attributes):
    element_count = len(rankings[0])

    #STEP 1: determining top-k elements
    #Construct weighted tournament, and then sort by indegrees, and take it as following the algorithm in the paper

    start_time = time.time()
    frac_tournament = Get_Frac_Tournament(rankings)

    fract_time = time.time()

    #List of lists.
    #List i contains tuples of elements with attribute i
    #tuple is in the form (element id, indegree)
    indegree_attr = []
    
    for attribute in range(num_attributes):
        indegree_attr.append([])
    for i in range(element_count):
        i_attr = id_attribute[i]
        indeg = 0
        for j in range(element_count):
            indeg += frac_tournament[j][i]
        indegree_attr[i_attr].append((i, indeg))
    for attr in range(num_attributes):
        indegree_attr[attr].sort(key = lambda ituple : ituple[1])
        
    topk_elements = set()
    elements_taken = [0] * num_attributes
    num_taken = 0
    #now, we get top k elements following the algo
    #take lower bound first
    #form combined list at same time
    indegree_combined = []
    for attr in range(num_attributes):
        for j in range(math.floor(alphas[attr] * TOPK)):
            topk_elements.add(indegree_attr[attr][j][0])
            elements_taken[attr] += 1
        indegree_combined += indegree_attr[attr][math.floor(alphas[attr] * TOPK):]
    
    #sort combined list, then take while respecting beta upper bounds
    indegree_combined.sort(key = lambda ituple : ituple[1])
    for i in range(len(indegree_combined)):
        if len(topk_elements) >= TOPK:
            break
        element = indegree_combined[i]
        i_attr = id_attribute[element[0]]
        if elements_taken[i_attr] < math.ceil(betas[i_attr] * TOPK):
            elements_taken[i_attr] += 1
            topk_elements.add(element[0])


    #STEP 2, we need to order the top-k.
    #Following the paper, we need to run rank aggregation over the two partitions.
    
    #In this implementation, Kwiksort is used to solve approximately, runs fast and easy to implement

    #left is top k, the front part
    rankings_left = []
    rankings_right = []
    for rank in rankings:
        left_rank = []
        right_rank = []
        for i in rank:
            if i in topk_elements:
                left_rank.append(i)
            else:
                right_rank.append(i)
        rankings_left.append(left_rank)
        rankings_right.append(right_rank)

    #NOTE: Because the elements of the reduced rankings are not a continuous 1 ... k, we need to relabel the elements to be 1 ... k, and save the mapping
    #so we can map the result back to these elements

    left_forward_map = {}
    left_backward_map = {}
    mapped_rankings_left = []
    for i in range(len(rankings_left[0])):
        left_forward_map[rankings_left[0][i]] = i
        left_backward_map[i] = rankings_left[0][i]
    mapped_rankings_left.append([i for i in range(len(rankings_left[0]))])
    for i in range(1, len(rankings_left)):
        mapped_rank = []
        for j in rankings_left[i]:
            mapped_rank.append(left_forward_map[j])
        mapped_rankings_left.append(mapped_rank)

    right_forward_map = {}
    right_backward_map = {}
    mapped_rankings_right = []
    for i in range(len(rankings_right[0])):
        right_forward_map[rankings_right[0][i]] = i
        right_backward_map[i] = rankings_right[0][i]
    mapped_rankings_right.append([i for i in range(len(rankings_right[0]))])
    for i in range(1, len(rankings_right)):
        mapped_rank = []
        for j in rankings_right[i]:
            mapped_rank.append(right_forward_map[j])
        mapped_rankings_right.append(mapped_rank)

    #Using the better of kwiksort and best from input algorithms from Ailon, Newman, Charikar 2007 paper
    leftKwiksort = Kwiksort(mapped_rankings_left)
    rightKwiksort = Kwiksort(mapped_rankings_right)
    leftInput = Best_From_Input(mapped_rankings_left)
    rightInput = Best_From_Input(mapped_rankings_right)

    if Get_Objective_Value(leftKwiksort, mapped_rankings_left) < Get_Objective_Value(leftInput, mapped_rankings_left):
        left_topo_sorted = leftKwiksort
    else:
        left_topo_sorted = leftInput

    if Get_Objective_Value(rightKwiksort, mapped_rankings_right) < Get_Objective_Value(rightInput, mapped_rankings_right):
        right_topo_sorted = rightKwiksort
    else:
        right_topo_sorted = rightInput
    
    #Re-map the topologically sorted elements, to the original elements using the backward maps

    left_original = [left_backward_map[i] for i in left_topo_sorted]
    right_original = [right_backward_map[i] for i in right_topo_sorted]

    output_ranking = left_original + right_original

    end_time = time.time()
    print("Algo (with kwiksort) took time " + str(end_time - start_time) + " seconds.")

    return output_ranking

##Helper functions for KWIKSORT
##Need to take bettter of best from input, and the kwiksort algorithm
def Best_From_Input(rankings):
    best_rank = []
    obj_value = 1e9
    for rank in rankings:
        median_cost = Get_Objective_Value(rank, rankings)
        if median_cost < obj_value:
            obj_value = median_cost
            best_rank = rank
    return best_rank

def Kwiksort(rankings):
    frac_tournament = Get_Frac_Tournament(rankings)
    initial = [i for i in range(len(rankings[0]))]
    rank = DoKwiksort(initial, frac_tournament)
    return rank

def DoKwiksort(elements, frac_tournament):
    if len(elements) <= 1:
        return elements
    pivot = rng.choice(elements)

    left = []
    right = []
    for element in elements:
        if element != pivot:
            if frac_tournament[element][pivot] >= 0.5:
                left.append(element)
            else:
                right.append(element)
    return DoKwiksort(left, frac_tournament) + [pivot] + DoKwiksort(right, frac_tournament)
    

def getObjCost(sol, costRank):
    ccost = 0
    assigned = set()
    for i in range(len(sol)):
        ccost += costRank[sol[i]][i]
        assigned.add(sol[i])
    assert(len(assigned) == len(sol)), "Error: Output is not a ranking"
    return ccost

####
#Setup to run algorithms here

#The code expects that the input rankings are over the elements 0 ... d-1.
#If it is not 0 indexed, it will give an error.

#This shows an example of how to run the algorithm on one of the input datasets.

parameters = input().rstrip().split(" ")

NUM_RANKINGS = int(parameters[0])
TOPK = int(parameters[1])
NUM_GROUPS = int(parameters[2])

alphas = []
betas = []
rankings = []
group_count = [0] * NUM_GROUPS

for i in range(NUM_GROUPS):
    fairparams = input().rstrip().split(" ")
    alphas.append(float(fairparams[0]))
    betas.append(float(fairparams[1]))

for i in range(NUM_RANKINGS):
    ranking = input().rstrip().split(" ")
    ranking = [int(j) for j in ranking]
    rankings.append(ranking)

DIM = len(rankings[0])

#Read group mapping
item_to_group = {}
group_to_item = [[] for _ in range(NUM_GROUPS)]
for i in range(DIM):
    inline = input().rstrip().split(" ")
    inline = [int(j) for j in inline]
    item_to_group[inline[0]] = inline[1]
    group_to_item[inline[1]].append(inline[0])
    group_count[inline[1]] += 1

#This sets alpha/beta to proportion of input
for g in range(NUM_GROUPS):
    alphas[g] = group_count[g] / DIM
    betas[g] = group_count[g] / DIM


print("Our algorithm + KS:")

#This code is to set seed for kwiksort, to allow reproducibility
seed_val = 1
for alpha in alphas:
    seed_val *= alpha * 100
rng = np.random.default_rng([9, TOPK, len(rankings[0]), int(seed_val)])
sol = OurAlgo_KS(alphas, betas, rankings, item_to_group, NUM_GROUPS)

#compute cost-rank matrix
costRank = np.zeros((DIM, DIM))

for i in range(NUM_RANKINGS):
    ranking = rankings[i]
    for j in range(DIM):
        elem = ranking[j]
        for k in range(DIM):
            costRank[elem][k] += abs(j - k)

print(getObjCost(sol, costRank))