import math
import sys
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

#Input: costRank: spearman's footrule cost matrix
#       K: fairness parameter
#       group_to_item: hashmap from group to list of items in that group
#       ALPHA, BETA: fairness parameters
#Computes and returns optimal fair aggregate ranking via ILP.
def optimalILP(costRank, K, group_to_item, ALPHA, BETA):
    m = gp.Model("fair-ra")
    m.params.OutputFlag = 0

    # Create variables
    assignmentVars = m.addMVar(shape=(DIM, DIM), vtype = GRB.BINARY)

    m.addConstr(assignmentVars @ np.ones(DIM) == np.ones(DIM))

    m.addConstr(assignmentVars.T @ np.ones(DIM) == np.ones(DIM))

    # Add fairness constraints
    fairness_ones = np.concatenate((np.ones(K), np.zeros(DIM - K)))
    for i in range(NUM_GROUPS):
        LB = math.floor(ALPHA[i] * K)
        UB = math.ceil(BETA[i] * K)
        m.addConstr(gp.quicksum([assignmentVars[j] @ fairness_ones for j in group_to_item[i]]) >= LB)
        m.addConstr(gp.quicksum([assignmentVars[j] @ fairness_ones for j in group_to_item[i]]) <= UB)

    # Create objective function
    m.setObjective(gp.quicksum([costRank[i] @ assignmentVars[i] for i in range(DIM)]), GRB.MINIMIZE)

    m.optimize()

    #Retrieve solution from the assignment variables
    output_rank = [0] * DIM
    vars = assignmentVars.X

    for i in range(DIM):
        for j in range(DIM):
            if vars[i][j] > 0.99:
                output_rank[j] = i
                break

    return output_rank

#Input: costRank: spearman's footrule cost matrix
#       K: fairness parameter
#       group_to_item: hashmap from group to list of items in that group
#       ALPHA, BETA: fairness parameters
# Runs Algorithm 1 of our paper, and returns the output ranking.
def ourAlgo(costRank, K, LB, UB, group_to_item):

    objval = 0
    #solve the top K
    m = gp.Model("topk")
    m.params.OutputFlag = 0

    #use simplex method
    m.params.Method = 0

    # Create variables
    assignmentVars = m.addMVar(shape=(DIM, K))

    m.addConstr(assignmentVars @ np.ones(K) <= np.ones(DIM))

    #Each position needs at least one item 
    m.addConstr(assignmentVars.T @ np.ones(DIM) == np.ones(K))
    # Add fairness constraints
    
    for i in range(NUM_GROUPS):
        m.addConstr(gp.quicksum([assignmentVars[j] @ np.ones(K) for j in group_to_item[i]]) >= LB[i])
        m.addConstr(gp.quicksum([assignmentVars[j] @ np.ones(K) for j in group_to_item[i]]) <= UB[i])

    m.setObjective(gp.quicksum([costRank[i][:K] @ assignmentVars[i] for i in range(DIM)]), GRB.MINIMIZE)
    m.optimize()
    
    #Retrieve solution from the assignment variables
    output_rank = [0] * DIM
    assigned = set()

    assignvars = assignmentVars.X

    for i in range(DIM):
        for j in range(K):
            if assignvars[i][j] > 0.99:
                output_rank[j] = i
                assigned.add(i)
                break

    objval += m.ObjVal

    #Fill rest of ranking
    m = gp.Model("topk")
    m.params.OutputFlag = 0
    #use simplex method
    m.params.Method = 0

    assignmentVars = m.addMVar(shape=(DIM, DIM - K))

    isOne = np.ones(DIM)
    for i in assigned:
        isOne[i] = 0

    m.addConstr(assignmentVars @ np.ones(DIM - K) == isOne)

    m.addConstr(assignmentVars.T @ np.ones(DIM) == np.ones(DIM - K))
    m.setObjective(gp.quicksum([costRank[i][K:] @ assignmentVars[i] for i in range(DIM)]), GRB.MINIMIZE)
    m.optimize()

    assignvars = assignmentVars.X
    for i in range(DIM):
        for j in range(DIM - K):
            if assignvars[i][j] > 0.99:
                output_rank[j + K] = i
                assigned.add(i)
                break
    
    return output_rank

#Input: costRank: spearman's footrule cost matrix
#       K: fairness parameter
#       group_to_item: hashmap from group to list of items in that group
#       ALPHA, BETA: fairness parameters
# Runs Algorithm 3 of our paper, and returns the output ranking.
# To run Algorithm 1, simply remove the second call to ourAlgo.
def ourAlgoWrapper(costRank, K, group_to_item, ALPHA, BETA):
    #compute top K bounds
    LB = [math.floor(ALPHA[i] * K) for i in range(NUM_GROUPS)]
    UB = [math.ceil(BETA[i] * K) for i in range(NUM_GROUPS)]

    #print("Solving first step...")
    sigma1 = ourAlgo(costRank, K, LB, UB, group_to_item)
    sigma1val = getObjCost(sigma1, trueCostRank)

    new_LB = [len(group_to_item[i]) - UB[i] for i in range(NUM_GROUPS)]
    new_UB = [len(group_to_item[i]) - LB[i] for i in range(NUM_GROUPS)]

    newCostRank = np.zeros((DIM, DIM), dtype = np.int16)
    for i in range(NUM_RANKINGS):
        ranking = rankings[i]
        for j in range(DIM):
            elem = ranking[j]
            for k in range(j+1, DIM):
                newCostRank[elem][k] += k - j
    newCostRank = np.flip(newCostRank, axis = 1)

    #print("Solving second step...")
    sigma2= ourAlgo(newCostRank, DIM - K, new_LB, new_UB, group_to_item)
    sigma2 = sigma2[::-1]
    sigma2val = getObjCost(sigma2, trueCostRank)

    if sigma1val < sigma2val:
        return sigma1
    return sigma2

#Input: rankings: set of input rankings
#       costRank: spearman's footrule cost matrix
#       K: fairness parameter
#       item_to_group: hashmap from group to list of items in that group
#       ALPHA, BETA: fairness parameters
# Runs the best from input 3-approximation algorithm, and returns the best fair ranking among them along with its Spearman's footrule objective cost.
def bestFromInput(rankings, costRank, K, item_to_group, ALPHA, BETA):
    bestObj = 1e9
    bestRanking = None
    for i in range(len(rankings)):
        ranking = rankings[i]
        fair_ranking = getClosestRanking(ranking, item_to_group, ALPHA, BETA, K)
        objval = getObjCost(fair_ranking, costRank)
        if objval < bestObj:
            bestObj = objval
            bestRanking = fair_ranking
    return bestRanking

#Helper function, finds closest fair ranking to the function input ranking.
def getClosestRanking(ranking, item_to_group, ALPHA, BETA, K):
    group_taken = []
    for i in range(NUM_GROUPS):
        group_taken.append([])

    LB = [math.floor(ALPHA[g] * K) for g in range(NUM_GROUPS)]
    UB = [math.ceil(BETA[g] * K) for g in range(NUM_GROUPS)]
    
    new_ranking = []
    total_taken = 0
    for i in range(DIM):
        item = ranking[i]
        group = item_to_group[item]
        if len(group_taken[group]) < LB[group]:
            group_taken[group].append(item)
            total_taken += 1

    for i in range(DIM):
        if total_taken >= K:
            break
        item = ranking[i]
        group = item_to_group[item]
        if item in group_taken[group]:
            continue
        if len(group_taken[group]) < UB[group]:
            group_taken[group].append(item)
            total_taken += 1

    new_ranking = []
    for i in range(DIM):
        item = ranking[i]
        group = item_to_group[item]
        if item in group_taken[group]:
            new_ranking.append(item)
    
    for i in range(DIM):
        item = ranking[i]
        group = item_to_group[item]
        if item not in group_taken[group]:
            new_ranking.append(item)
    
    return new_ranking

#Input: sol: ranking
#       costRank: spearman's footrule cost matrix
#Computes and returns the spearman's footrule objective of sol.
def getObjCost(sol, costRank):
    ccost = 0
    assigned = set()
    for i in range(len(sol)):
        ccost += int(costRank[sol[i]][i])
        assigned.add(sol[i])
    assert(len(assigned) == len(sol)), "Error: Output is not a ranking"
    return ccost

#Helper function to check if a ranking is indeed fair
#(not actually used, but useful for checking)
def isFairRanking(ranking, item_to_group, ALPHA, BETA, K):
    group_count = [0] * NUM_GROUPS
    for i in range(K):
        item = ranking[i]
        group = item_to_group[item]
        group_count[group] += 1
    for g in range(NUM_GROUPS):
        if group_count[g] < math.floor(ALPHA[g] * K) or group_count[g] > math.ceil(BETA[g] * K):
            return False
    return True

# Input: sol: ranking
#       rankings: set of input rankings
# Computes and returns the kendall-tau objective of sol.
def getKTObjCost(sol, rankings):
    median_cost = 0
    for rank in rankings:
        median_cost += Kendall_Tau_Dist(sol, rank)
    return median_cost

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


#MAIN
parameters = input().rstrip().split(" ")

NUM_RANKINGS = int(parameters[0])
NUM_GROUPS = int(parameters[2])

ALPHA = []
BETA = []
rankings = []
group_count = [0] * NUM_GROUPS

for i in range(NUM_GROUPS):
    fairparams = input().rstrip().split(" ")
    ALPHA.append(float(fairparams[0]))
    BETA.append(float(fairparams[1]))

for i in range(NUM_RANKINGS):
    ranking = input().rstrip().split(" ")
    ranking = [int(j) for j in ranking]
    rankings.append(ranking)

DIM = len(rankings[0])

#K should be varied as desired for test case.
K = DIM//2

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
    ALPHA[g] = group_count[g] / DIM
    BETA[g] = group_count[g] / DIM

#compute cost-rank matrix
leftCostRank = np.zeros((DIM, DIM), dtype = np.int16)

for i in range(NUM_RANKINGS):
    ranking = rankings[i]
    for j in range(DIM):
        elem = ranking[j]
        for k in range(j):
            leftCostRank[elem][k] += abs(j - k)
trueCostRank = np.zeros((DIM, DIM), dtype = np.int16)

for i in range(NUM_RANKINGS):
    ranking = rankings[i]
    for j in range(DIM):
        elem = ranking[j]
        for k in range(DIM):
            trueCostRank[elem][k] += abs(j - k)


sol1 = optimalILP(trueCostRank, K, group_to_item, ALPHA, BETA)
sol2= ourAlgoWrapper(leftCostRank, K, group_to_item, ALPHA, BETA)
sol3 = bestFromInput(rankings[:NUM_RANKINGS], trueCostRank, K, item_to_group, ALPHA, BETA)

# If running experiments on kendall-tau objective
objOPT = getObjCost(sol1, trueCostRank)
objours = getObjCost(sol2, trueCostRank)
obj3 = getObjCost(sol3, trueCostRank)

print("ILP")
#print(sol1)
print(objOPT)

print("Ours")
#print(sol2)
print(objours)

print("BFI")
#print(sol3)
print(obj3)