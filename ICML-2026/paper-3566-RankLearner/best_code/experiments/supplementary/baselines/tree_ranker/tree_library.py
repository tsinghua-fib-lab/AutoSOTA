"""Author: Fahad Kamran 
Creates custom random forest
Options to split by maximizing AUTOC or minimizing MSE 
Training splits are done using build DR-proxies, as CATEs are not available in training 
Script saves: 
1. Learned forests 
2. Results across 30 seeds """

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
from causalml.optimize.policylearner import * 
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys 
from random import seed
from math import sqrt
from scipy.stats import rankdata


def bootstrap_sample(X, y, to_choose = 1):
    """Bootstraps data for different samples that will train each treee"""
    n_samples = X.shape[0]
    to_choose = int(to_choose * n_samples)
    idxs = np.random.choice(n_samples, to_choose, replace=True)
    return X[idxs], y[idxs]

def gen_data(n, seed = 0):
    """Generates training data based on random seed"""
    np.random.seed(seed)
    
    p = 10 
    
    X = np.random.multivariate_normal(np.zeros(p), np.eye(p), size = n)
    
    e = 1 / (1 + np.exp(-X[:, 2]))
    
    Z = np.random.binomial(1, 1 / (1 + np.exp(-X[:, 2])))
    eps = np.random.normal(size = n)
    tau = 1 + 2 * np.abs(X[:,3]) + (X[:, 9]) ** 2
    Y = (5 * (2 + 0.5 * np.sin(np.pi * X[:,0]) - 0.25 * X[:, 1] *+ 2 + 0.75 * X[:, 2] * X[:, 8])) + Z * tau + eps
    
    return (X, Y, Z, tau, e)


def gen_data_val(n, seed = 0):
    """Generates validation data based on random seed
    Validation data does not need anything beyond covariates and true CATE"""
    np.random.seed(seed + 50)
    
    p = 10 
    
    X = np.random.multivariate_normal(np.zeros(p), np.eye(p), size = n)
    tau = 1 + 2 * np.abs(X[:,3]) + (X[:, 9]) ** 2
    
    return (X, tau)


def gen_data_test(n, seed = 0):
    """Generates test data based on random seed
    Test data does not need anything beyond covariates and true CATE"""

    np.random.seed(seed + 100)
    
    p = 10 
    
    X = np.random.multivariate_normal(np.zeros(p), np.eye(p), size = n)
    tau = 1 + 2 * np.abs(X[:,3]) + (X[:, 9]) ** 2
    
    return (X, tau)


    
def AUTOC(dr_scores, priorities, sample_weights = None, query = 'AUTOC'): 
    """Calculate AUTOC for evaluation. 
    Code is largely adapted from original authors implementation
    Break ties using average ordering"""
    if not sample_weights: 
        sample_weights = np.ones(len(dr_scores))
        
    priorities = rankdata(priorities).astype(int)
    sort_idx = np.argsort(priorities)[::-1]
    num_ties = np.bincount(priorities)
    num_ties = num_ties[num_ties != 0]
    df = pd.DataFrame(np.array([dr_scores, sample_weights, priorities]).T[sort_idx])
    grp_sum = df.groupby(2, sort = False).sum()
    dr_avg = grp_sum[0].values / grp_sum[1].values
    dr_scores_sorted = np.repeat(dr_avg, num_ties[::-1])
    sample_weights = sample_weights[sort_idx]
    sample_weights_cumsum = np.cumsum(sample_weights)
    sample_weights_sum = sample_weights_cumsum[len(sample_weights) - 1]
    ATE = sum(dr_scores_sorted * sample_weights) / sample_weights_sum
    TOC = np.cumsum(dr_scores_sorted * sample_weights) / sample_weights_cumsum - ATE
    # plt.plot(TOC)
    if query == 'AUTOC': 
        RATE = sum(TOC * sample_weights) / sum(sample_weights)
    elif query == 'QINI': 
        RATE = np.sum(np.cumsum(sample_weights) / sum(sample_weights) * sample_weights * TOC) / sum(sample_weights)
    else: 
        RATE = np.nan
    return RATE, TOC


class Node:
    """Data class that specifies the data in current node, used to build up tree"""
    def __init__(self, examples, depth, value = 0):
        self.depth = depth
        self.examples = examples
        self.value = value
        
    def update(self, feature=None, threshold=None, left=None, right=None, value=None): 
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    

    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTreeAUTOC:
    """Custom tree that builds multiple nodes from a root based on method of building"""
    def __init__(self, min_samples_split=2, min_samples_leaf=1,
                 max_depth=np.inf, n_feats=None, method = 0, min_impurity = -1000):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self.all_nodes = []
        self.method = method
        self.min_impurity = min_impurity


    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)
    

    def predict(self, X):
        #To predict, traverse the tree for every example in the data
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):        

        #Root node starts with all examples and a value equal to the average outcome
        root = Node(examples = (X, y), depth = 0, value = np.mean(y))
        self.all_nodes.append(root)
        to_process = [root]
        #Process until we are out of nodes to process
        while len(to_process) > 0: 
            currNode = to_process[0]
            (feats, outcomes, depth, currVal) = currNode.examples[0], currNode.examples[1], currNode.depth, currNode.value
            
            to_process = to_process[1:]
            
            n_samples, n_features = feats.shape
            n_labels = len(np.unique(outcomes))
            
            if (depth >= self.max_depth
                    or n_labels == 1
                    or n_samples < self.min_samples_split):
                #Do not process this branch any further
                leaf_value = np.mean(outcomes)
                currNode.update(value=leaf_value)
                
            else:
                
                all_leaves = self._curr_leaves()
                
                if len(all_leaves) == 1: 
                    curr_vals = []
                    curr_outcomes = []
                    
                    curr_gain = 0
                else: 

                    #Find all the leaves without the current node
                    all_leaves_wo_curr = [x for x in all_leaves if not ((len(x.examples[0]) == len(feats)) and (x.value == currVal) and (x.depth == depth))]
                                        
                    assert len(all_leaves) == (len(all_leaves_wo_curr) + 1)
                                        
                    curr_vals = np.concatenate([np.ones(len(m.examples[1])) * m.value for m in all_leaves_wo_curr]).ravel()
                    curr_outcomes = np.concatenate([m.examples[1] for m in all_leaves_wo_curr]).ravel()
                    
                    #Find all the leaves with the current node 
                    all_leaves_w_curr = [x for x in all_leaves]
                    
                    w_curr_vals = np.concatenate([np.ones(len(m.examples[1])) * m.value for m in all_leaves_w_curr]).ravel()
                    w_curr_outcomes = np.concatenate([m.examples[1] for m in all_leaves_w_curr]).ravel()
                    
                    #Our current AUTOC for a given tree
                    curr_gain = self._splitting_val(w_curr_outcomes, w_curr_vals, query = 'AUTOC')
                
                
                feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

                #greedily select the best split according to information gain
                best_feat, best_thresh = self._best_criteria(feats, outcomes, feat_idxs, curr_vals, curr_outcomes, curr_gain, depth, self.min_impurity)

                if best_thresh is None: 
                    #If no improvements over our current tree, we are done. 
                    leaf_value = np.mean(outcomes)
                    currNode.update(value=leaf_value)
    
                else:
                    #Grow the children that result from the split
                    left_idxs, right_idxs = self._split(feats[:, best_feat], best_thresh)

                    leftNode = Node(examples = (feats[left_idxs, :], outcomes[left_idxs]), depth = depth+1, value = np.mean(outcomes[left_idxs]))
                    
                    rightNode = Node(examples = (feats[right_idxs, :], outcomes[right_idxs]), depth = depth+1, value = np.mean(outcomes[right_idxs]))

                    self.all_nodes.append(leftNode)
                    self.all_nodes.append(rightNode)

                    #Update current tree for traversal
                    currNode.update(left = leftNode, right = rightNode, feature = best_feat, threshold = best_thresh) 

                    #Breadth first, rather than depth first
                    to_process.append(leftNode)
                    to_process.append(rightNode)
        
        for node in self.all_nodes: 
            #Remove examples to reduce model size when saving
            node.examples = None
            
        return root

    def _best_criteria(self, X, y, feat_idxs, curr_vals, curr_outcomes, curr_gain, curr_depth, min_impurity):
        """Calculates the best split id and threshold for a given method of building the tree. 
        Returns None, None if we can not improve over the current tree (based on min_impurity value)"""

        best_gain = (curr_gain + min_impurity)
        curr_mse = np.mean(np.square(y - np.mean(y)))
        best_gain_mse = curr_mse + min_impurity

        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                if self.method == 1:
                    gain = self._information_gain(y, X_column, threshold, curr_vals, curr_outcomes)
                    if gain > (best_gain):
                        best_gain = gain
                        split_idx = feat_idx
                        split_thresh = threshold
                else:
                    gain = self._information_gain_mse(y, X_column, threshold)
                    if gain < best_gain_mse:
                        best_gain_mse = gain
                        split_idx = feat_idx
                        split_thresh = threshold                        
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh, curr_vals, curr_outcomes):
        #generate split using AUTOC
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            #If we have too little samples in leaf, we do not consider this split
            return -np.inf

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        
        left_outcomes = y[left_idxs]
        right_outcomes = y[right_idxs]
        
        left_value = np.mean(left_outcomes)
        right_value = np.mean(right_outcomes)
        
        all_outcomes = np.concatenate([curr_outcomes, left_outcomes, right_outcomes]).ravel()
        all_vals = np.concatenate((curr_vals, np.ones(len(left_outcomes)) * left_value, np.ones(len(right_outcomes)) * right_value)).ravel()
    
        #Calculate AUTOC given the existing outcomes and values in the trees, and a proposed new split. 
        ig = self._splitting_val(all_outcomes, all_vals, query = 'AUTOC')

        return ig
    
    def _information_gain_mse(self, y, X_column, split_thresh):
        #generate split using MSE
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            #If we have too little samples in leaf, we do not consider this split
            return np.inf

        #compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = np.mean(np.square(y[left_idxs] - np.mean(y[left_idxs]))), np.mean(np.square(y[right_idxs] - np.mean(y[right_idxs])))
        child_var = (n_l / n) * e_l + (n_r / n) * e_r

        ig = child_var
        return ig
    
    def _splitting_val(self, dr_scores, priorities, sample_weights = None, query = 'AUTOC'): 
        """AUTOC calculation, similar to evaluation implementation above, but used to grow tree"""
        try: 
            if not sample_weights: 
                sample_weights = np.ones(len(dr_scores))
        except: 
            1+1
        
        priorities = rankdata(priorities).astype(int)
        sort_idx = np.argsort(priorities)[::-1]
        num_ties = np.bincount(priorities)
        num_ties = num_ties[num_ties != 0]
        df = pd.DataFrame(np.array([dr_scores, sample_weights, priorities]).T[sort_idx])
        grp_sum = df.groupby(2, sort = False).sum()
        dr_avg = grp_sum[0].values / grp_sum[1].values
        dr_scores_sorted = np.repeat(dr_avg, num_ties[::-1])
        sample_weights = sample_weights[sort_idx]
        sample_weights_cumsum = np.cumsum(sample_weights)
        sample_weights_sum = sample_weights_cumsum[len(sample_weights) - 1]
        ATE = sum(dr_scores_sorted * sample_weights) / sample_weights_sum
        TOC = np.cumsum(dr_scores_sorted * sample_weights) / sample_weights_cumsum - ATE

        if query == 'AUTOC': 
            RATE = sum(TOC * sample_weights) / sum(sample_weights)
        elif query == 'QINI': 
            RATE = np.sum(np.cumsum(sample_weights) / sum(sample_weights) * sample_weights * TOC) / sum(sample_weights)
        else: 
            RATE = np.nan
            
        return RATE
    
    def _curr_leaves(self): 
        return [n for n in self.all_nodes if n.is_leaf_node()]
    
    def _split(self, X_column, split_thresh):
        #Split tree based on feature value to get left and right children 
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        #Traverse tree recurisvely to find estimated value of tree
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    
class RandomForestAUTOC:
    
    def __init__(self, n_estimators=10, min_samples_split=2, min_samples_leaf=1, max_samples=1,
                 max_depth=np.inf, n_feats=None, method = 0, seed = 0, n_jobs = 0, min_impurity = -1000):
        self.n_trees = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        self.method = method
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        
        np.random.seed(self.seed) #Seed seed for reproducability for fitting
        
        def new_tree():
            """Helper function to build a new tree"""
            tree = DecisionTreeAUTOC(min_samples_split=self.min_samples_split,
                max_depth=self.max_depth, n_feats=self.n_feats, method=self.method, min_samples_leaf=self.min_samples_leaf, min_impurity = self.min_impurity)
            X_samp, y_samp = bootstrap_sample(X, y, to_choose = self.max_samples)
            tree.fit(X_samp, y_samp)
            return tree
         
        #Either build forest using paralleilization or lazy for loop 
        if not self.n_jobs: 
            for i in tqdm(range(self.n_trees)): 
                self.trees.append(new_tree())
        
        else: 
            self.trees = Parallel(n_jobs=self.n_jobs)(delayed(new_tree)() for s in range(self.n_trees))

    def predict(self, X, n_jobs = 0):
        """Create predictions to be used for evaluation"""
        def preds(tree): 
            return tree.predict(X)
        
        #Either generate predictions using paralleilization or lazy for loop 
        if not n_jobs: 
            tree_preds = []
            for tree in tqdm(self.trees): 
                tree_preds.append(preds(tree))
            tree_preds = np.array(tree_preds)
        else: 
            tree_preds = np.array(Parallel(n_jobs=n_jobs)(delayed(preds)(tree) for tree in self.trees))
        #Estimated value is average across all trees
        y_pred = tree_preds.mean(axis = 0)
        return np.array(y_pred)
    

    
    
def run(n_estimators = 100, max_samples = 1, min_samples_leaf=1, min_samples_split=2, method = 0, max_depth = np.inf, min_impurity = -100, use_ground_truth_prop = True):


    #Can change to any amount of training data
    N = 250

    dct_val, dct_test = {}, {}

    rfs_val = []
    rfs_test = []

    def train_on_DR(seq, N): 

        #Generate train, val, or test data
        (X, Y, Z, tau, e)  = gen_data(n = N, seed = seq)
        (Xte, taute) = gen_data_test(n = 5000, seed = seq)
        (Xval, tauval) = gen_data_val(n = 1000, seed = seq)

        #Create DR estimates to be used only for training, as we do not have access to CATEs in training 
        #Either use ground-truth training propensity scores or learn them using some model
        if use_ground_truth_prop: 
            pl = PolicyLearner(policy_learner = LogisticRegressionCV(), treatment_learner = LogisticRegressionCV(), random_state = 0).fit(X,Z,Y, p = e)
        else: 
            pl = PolicyLearner(policy_learner = LogisticRegressionCV(), treatment_learner = LogisticRegressionCV(), random_state = 0).fit(X,Z,Y)

        y = pl._dr_score

        #Initialize and fit random forest
        est = RandomForestAUTOC(n_estimators = n_estimators, max_samples = max_samples, 
                                min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, 
                                method = method, n_jobs = 100, min_impurity = min_impurity, max_depth = max_depth)
        est.fit(X, y)  

        #Save random forest
        with open('models/autoc_forest_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(seq, N, n_estimators, max_samples, min_samples_leaf, min_samples_split, min_impurity, max_depth, method), 'wb') as f: 
            pickle.dump(est, f)

        #Generate predictions
        curr_est_rf_val = est.predict(Xval, n_jobs = 100)
        curr_est_rf = est.predict(Xte, n_jobs = 100)

        return [(AUTOC(taute.reshape(-1), res.reshape(-1))) for res in [curr_est_rf_val, curr_est_rf]]

    #Across all seeds, train and evaluate
    for s in tqdm(range(30)):
        rfs_curr_val, rfs_curr_test = train_on_DR(s, N)
        rfs_val.append(rfs_curr_val)
        rfs_test.append(rfs_curr_test)
    
    #Save out model performance across all seeds
    rfs_val = np.array(rfs_val)
    rfs_test = np.array(rfs_test)

    dct_val[N] = rfs_val
    dct_test[N] = rfs_test


    with open('val_results/results_{}_{}_{}_{}_{}_{}_{}.pkl'.format(n_estimators, max_samples, min_samples_leaf, min_samples_split, min_impurity, max_depth, method), 'wb') as f: 
        pickle.dump(dct_val, f)
    with open('test_results/results_{}_{}_{}_{}_{}_{}_{}.pkl'.format(n_estimators, max_samples, min_samples_leaf, min_samples_split, min_impurity, max_depth, method), 'wb') as f: 
        pickle.dump(dct_test, f)

if __name__ == '__main__': 
    n_estimators = int(sys.argv[1])
    max_samples = float(sys.argv[2])
    min_samples_leaf = int(sys.argv[3])
    min_samples_split = int(sys.argv[4])
    method = int(sys.argv[5])
    min_impurity = float(sys.argv[6])
    max_depth = int(float(sys.argv[7]))
    run(n_estimators = n_estimators, max_samples = max_samples, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, method = method, min_impurity = min_impurity, max_depth = max_depth)
    
    
####################################################################################################################################

def approximate_autoc_df(df):
    """ 
        Approximates AUTOC cf. causal ML book of Kallus and Chernozhukov.
        Expects data loader that yield (x, dr);
        Scoring model assigns higher score to those that should be treated first.
    """

    # init numerical stability constant
    eps = 1e-12
            
    # concate scores and drs
    y_dr = df['DR'].to_numpy(dtype=np.float64)
    scores = df['score'].to_numpy(dtype=np.float64)
    N = y_dr.shape[0]
    
    # init toc collector
    toc_vals = []
    
    # first covariance component
    A = y_dr - y_dr.mean()
    
    # define grid of quantiles to consider
    q_grid = np.linspace(0.05, 1.0, 21)
    for q in q_grid:
    
        # target number to instances to treat
        target = int(np.round(q * N))
        thr = np.quantile(scores, 1.0 - q)
    
        # quantify how many match or exceed threshold
        above = scores > thr
        equal = scores == thr
        n_above = int(above.sum())
        n_equal = int(equal.sum())
    
        # handle ties
        need = max(target - n_above, 0)
        frac = 0.0 if n_equal == 0 else min(need / n_equal, 1.0)
    
         # treatment indicator (with tie handling)
        Iq = above.astype(np.float64) + frac * equal.astype(np.float64)
    
        # probability of treatment
        pi = max(Iq.mean(), eps)
    
        # second covariance component
        B = (Iq / pi)
        B = B - B.mean()
    
        # approximate toc at q
        toc_q = np.mean(A * B)  # Cov_n(A, B)
        toc_vals.append(toc_q)
    
    # AUTOC discrete difference approximation
    dq = np.diff(q_grid)                
    autoc = np.sum(toc_vals[:-1] * dq)   

    return autoc