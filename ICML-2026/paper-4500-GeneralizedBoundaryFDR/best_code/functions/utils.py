import numpy as np
import scipy.stats as stats
import pandas as pd
import random
from scipy.stats import norm, truncnorm
from scipy.linalg import toeplitz
import requests
import json
import time
from sklearn.model_selection import train_test_split
# --------------------Functions in Simulation--------------------------

def generate_correlated_data_evalues(m=100, pi1=0.1, muc=3, sigma=1, rho=0, seed=2026):
    """
    Generate correlated data and then return e-values
    """
    if seed is not None:
        np.random.seed(seed)

    cov_val = rho * sigma
    Sigma = (sigma - cov_val) * np.eye(m) + cov_val * np.ones((m, m))
    
    out_loc = np.random.choice([0, 1], size=m, replace=True, p=[1-pi1, pi1])
    
    mu = np.zeros(m)
    idx_h1 = np.where(out_loc == 1)[0]
    
    if len(idx_h1) > 0:
        sd = np.sqrt(sigma)
        # truncated Gaussian with lower bound 0
        a, b = (0 - muc) / sd, np.inf  
        mu[idx_h1] = truncnorm.rvs(a, b, loc=muc, scale=sd, size=len(idx_h1))

    if np.isclose(rho, 1.0):
        common_noise = np.random.normal(0, np.sqrt(sigma))
        X = mu + common_noise
    else:
        X = np.random.multivariate_normal(mean=mu, cov=Sigma)
    
   
    log_e = (X * muc - 0.5 * muc**2) / sigma
    e_values = np.exp(log_e)

    null_e = e_values[out_loc == 0]
    h1_e = e_values[out_loc == 1]
    
    return null_e, h1_e


def generate_correlated_data(m=500, pi1=0.1, muc=3, sigma=1, rho=0.5, structure='cs', seed=2026):
    if seed is not None:
        np.random.seed(seed)
    
    if structure == 'independent':
        Sigma = sigma * np.eye(m)
        
    elif structure == 'cs': 
        cov_val = rho * sigma 
        Sigma = (sigma - cov_val) * np.eye(m) + cov_val * np.ones((m, m))
        
    out_loc = np.random.choice([0, 1], size=m, replace=True, p=[1-pi1, pi1])

    mu = np.zeros(m)
    idx_h1 = np.where(out_loc == 1)[0]
    if len(idx_h1) > 0:
        sd = np.sqrt(sigma)
        # truncated Gaussian with lower bound 0
        a, b = (0 - muc) / sd, np.inf  
        mu[idx_h1] = truncnorm.rvs(a, b, loc=muc, scale=sd, size=len(idx_h1))
    
    X = np.random.multivariate_normal(mean=mu, cov=Sigma)
    
    sd = np.sqrt(sigma)
    p_values = norm.cdf(-X, scale=sd)

    null_p = p_values[out_loc == 0]
    h1_p = p_values[out_loc == 1]
    
    return null_p, h1_p, Sigma


#---------------------Functions in Gene Data--------------------------
def get_threshold(p_data, indices):
    if len(indices) == 0:
        return 0.0 
    return np.max(p_data[indices])


def evaluate_boundary_k_errors(reject_indices, p_vals, true_labels, k=10):
    """
    Compute the number of false discoveries among the top k p-values in the rejection set.
    """
    if len(reject_indices) == 0:
        return 0, np.array([])
    
    rej_p = p_vals[reject_indices]
    rej_labels = true_labels[reject_indices]
    
    
    sort_order = np.argsort(rej_p)[::-1] 
    
    actual_k = min(k, len(reject_indices))
    top_k_indices = sort_order[:actual_k]
    
    boundary_labels = rej_labels[top_k_indices]
    
    n_errors = (boundary_labels == 0).sum()
    
    
    
    
def add_true_labels(p_values, ceg_file='CEGv2.txt', neg_file='NEGv1.txt'):
    """
    add true labels
    """
    try:
        ceg_genes = pd.read_csv(ceg_file, header=None, sep='\t')[0].values
        neg_genes = pd.read_csv(neg_file, header=None, sep='\t')[0].values
  
        ceg_set = set(ceg_genes)
        neg_set = set(neg_genes)
    except FileNotFoundError:
        return None


    df = pd.DataFrame({'P_Value': p_values})
    df['Gene_Symbol'] = [x.split(' ')[0] for x in df.index]

    
    df['Label'] = -1 
    df.loc[df['Gene_Symbol'].isin(ceg_set), 'Label'] = 1
    
    df.loc[df['Gene_Symbol'].isin(neg_set), 'Label'] = 0
    
    n_pos = (df['Label'] == 1).sum()
    n_neg = (df['Label'] == 0).sum()

    
    return df

def load_single_cell_line(file_path, cell_line_index=0):
    """
    Read one cell_line's gene effect scores from a large CSV file.
    """
    df_chunk = pd.read_csv(file_path, skiprows=lambda x: x > 0 and x != (cell_line_index + 1), nrows=1, header=0)

    header = pd.read_csv(file_path, nrows=0)
    df_chunk.columns = header.columns

    gene_scores = df_chunk.iloc[0, 1:] 

    return gene_scores

def get_p_values(scores, neg_file_path='NEGv1.txt'):
    """
    Transform DepMap Scores into P-values using the NEG reference set.
    """
    scores = scores.astype(float)

    try:
       
        neg_genes = pd.read_csv(neg_file_path, header=None, sep='\t')[0].values
    except FileNotFoundError:
        return None

    valid_neg_genes = [g for g in neg_genes if g in scores.index]
    
    if len(valid_neg_genes) < 50:
        print(f"Not enough NEG genes found in scores. Found {len(valid_neg_genes)} genes.")
    
    null_scores = scores[valid_neg_genes]
    
    mu_null = null_scores.mean()
    sigma_null = null_scores.std()
    z_scores = (scores - mu_null) / sigma_null

    p_values = norm.cdf(z_scores)

    p_series = pd.Series(p_values, index=scores.index)
    
    return p_series


def BH(pvals, q):
    """
    Given a list of p-values and nominal FDR level q, apply BH procedure to get a rejection set.
    """
    ntest = len(pvals)
         
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return np.array([])
    else:
        k = np.max(idx_smaller)
        idx_sel = df_test['id'].iloc[range(k+1)].values
        return idx_sel
    
# ------------------------kbfdr Control Methods---------------------------------
def kfwer_eclosure_k_localtest(e_values, k=2, alpha=0.05):
    """
    k-local test using e-closure
    """
    e_values = np.array(e_values)
    m = len(e_values)
    
    if m < k:
        return 0

    sorted_e = np.sort(e_values)[::-1]
    
    candidate_R = sorted_e[:k]
    sum_R = np.sum(candidate_R)

    tail_values = sorted_e[k:]
    tail_values_asc = tail_values[::-1]

    cum_tail = np.cumsum(tail_values_asc)
    cum_tail = np.insert(cum_tail, 0, 0.0) 
    all_S_sums = sum_R + cum_tail
    all_S_sizes = k + np.arange(len(cum_tail))
    all_S_means = all_S_sums / all_S_sizes
    min_S_mean = np.min(all_S_means)
    
    threshold = 1.0 / alpha
    if min_S_mean >= threshold:
        return k
    else:
        return 0
    
def convert_p_to_e_integrated(p_values_array):
    """
    convert p-values to integrated e-values。
    Formula: e = (1 - p + p * ln(p)) / (p * (ln(p))^2)
    """
    p = np.array(p_values_array, dtype=float)
    e_values = np.zeros_like(p)
    mask_valid = (p > 0) & (p < 1)
    if np.any(mask_valid):
        p_valid = p[mask_valid]
        ln_p = np.log(p_valid)
        
        numerator = 1 - p_valid + (p_valid * ln_p)
        denominator = p_valid * (ln_p ** 2)
        
        e_values[mask_valid] = numerator / denominator
    e_values[p <= 0] = np.inf
    e_values[p >= 1] = 0.5
    return e_values


def bfdr_k_eclosure(e_values, k=2, alpha=0.05):
    """
    Using e-closure to control k-bFDR directly.
    """
    e_values = np.array(e_values)
    m = len(e_values)
    sorted_indices = np.argsort(e_values)[::-1]
    sorted_e = e_values[sorted_indices]
    if m < k:
        return sorted_indices, sorted_indices

    threshold = 1.0 / alpha
    reject_count = k - 1 
    reversed_e = sorted_e[::-1]
    cumulative_tail_sum = np.cumsum(reversed_e) 
    cumulative_tail_sum = np.insert(cumulative_tail_sum, 0, 0.0)
    for j in range(m - k, -1, -1):
        head_sum = np.sum(sorted_e[j : j+k])
        head_size = k

        max_tail_len = m - (j + k)
        
        t_values = np.arange(max_tail_len + 1)

        current_tail_sums = cumulative_tail_sum[:max_tail_len+1]
        
        all_means = (head_sum + current_tail_sums) / (head_size + t_values)
        min_mean = np.min(all_means)
        if min_mean >= threshold:
            reject_count = j + k
            break

    rejected_indices = sorted_indices[:reject_count]
    start_bound = max(0, reject_count - k)
    boundary_indices = sorted_indices[start_bound : reject_count]
    
    return rejected_indices, boundary_indices



def bfdr_k_domino(p_values, k=2, alpha=0.1):
    """
    Using kholms(generalized k-Bonferroni) as local test
    """
    p_values = np.array(p_values)
    m = len(p_values)

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    if m < k:
        return sorted_indices, sorted_indices

    reject_count = k - 1

    for j in range(k, m + 1):
        threshold = (k * alpha) / (m - j + k)
        if sorted_p[j-1] <= threshold:
            reject_count = j
        else:
            break 

    rejected_indices = sorted_indices[:reject_count]

    start_index = max(0, reject_count - k)
    boundary_indices = sorted_indices[start_index : reject_count]
    
    return rejected_indices, boundary_indices


def bfdr_k_edomino(e_values, k=2, alpha=0.05):
    """
    Using e-closure as k-local test
    Equivalent to e-closure kbfdr if using average e
    """
    e_values = np.array(e_values)
    m = len(e_values)

    if m < k:
        sorted_indices = np.argsort(e_values)[::-1]
        return sorted_indices, sorted_indices

    sorted_indices = np.argsort(e_values)[::-1]
    sorted_e = e_values[sorted_indices]
    threshold = 1.0 / alpha
    ascending_e = sorted_e[::-1]
    s_min = np.insert(np.cumsum(ascending_e), 0, 0.0)
    t_values = np.arange(m + 1)
    v_values = threshold * t_values - s_min
    max_penalty = np.maximum.accumulate(v_values)
    reject_count = k - 1
    sorted_e_cumsum = np.insert(np.cumsum(sorted_e), 0, 0.0)

    for i in range(m - k, -1, -1):
        head_sum = sorted_e_cumsum[i+k] - sorted_e_cumsum[i]
        tail_len = m - (i + k)
        required_val = threshold * k + max_penalty[tail_len]   
        if head_sum >= required_val:
            reject_count = i + k
            break

    rejected_indices = sorted_indices[:reject_count]

    start_bound = max(0, reject_count - k)
    boundary_indices = sorted_indices[start_bound : reject_count]
    
    return rejected_indices, boundary_indices


def kbfdr_evaluate(groundtruth, boundary_indices,k):
    """
    evaluate k-bfdr
    """
    groundtruth = np.array(groundtruth)
    boundary_indices = np.atleast_1d(np.array(boundary_indices))

    if len(boundary_indices) == 0:
        return 0

    # Check for -1 sentinel (no valid boundary)
    if np.any(boundary_indices < 0):
        return 0

    boundary_truth = groundtruth[boundary_indices]

    if np.sum(boundary_truth) == 0 and len(boundary_truth) >= k:
        return 1
    else:
        return 0
    
    
# -------------------------bfdr Control Methods---------------------------------
def cauchy_combination(p_values, weights=None):
    """
    cauchy combination of p-values
    """
    p_values = np.array(p_values)

    p_values = np.clip(p_values, 1e-16, 1 - 1e-16)

    if weights is None:
        w = np.ones(len(p_values)) / len(p_values)
    else:
        w = np.array(weights)
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative.")
        w = w / np.sum(w)

    t_stat = np.sum(w * np.tan((0.5 - p_values) * np.pi))

    p_combined = 0.5 - (np.arctan(t_stat) / np.pi)

    return max(0.0, min(1.0, p_combined))

def cauchypbfdr(data, alpha):
    m = len(data)
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    
    for j in range(m - 1, -1, -1):
        all_below_threshold = True
        if sorted_data[j] > alpha:
            all_checks_passed = False
            continue 
        current_ps = [sorted_data[j]]
        for k in range(1, m - j):
            current_ps.append(sorted_data[m - k])
            combined_p = cauchy_combination(current_ps)
            if combined_p > alpha:
                all_below_threshold = False
                break
        
        if all_below_threshold:
            result_indices = sorted_indices[:j + 1]
            threshold_value = sorted_data[j]
            boundary_index = sorted_indices[j]
            return result_indices, boundary_index, threshold_value
    
    return np.empty(0, dtype=np.int64), -1, 0.0


def averagepbfdr(data, alpha):
    """
    using average p-value to control bFDR
    """
    m = len(data)
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    
    alpha_half = alpha / 2.0
    
    for j in range(m - 1, -1, -1):
        all_below_threshold = True
        if sorted_data[j] > alpha:
            all_checks_passed = False
            continue 
        cumsum = 0.0
        for k in range(1, m - j):
            cumsum += sorted_data[m - k]
            total_sum = cumsum + sorted_data[j]
            avg_val = total_sum / (k + 1)

        
            if avg_val > alpha_half:
                all_below_threshold = False
                break
        
        if all_below_threshold:
            result_indices = sorted_indices[:j + 1]
            threshold_value = sorted_data[j]
            boundary_index = sorted_indices[j]
            return result_indices, boundary_index, threshold_value
    
    return np.empty(0, dtype=np.int64), -1, 0.0





def harmonicpbfdr(data, alpha):
    m = len(data)
   
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]

    for j in range(m - 1, -1, -1):
        all_below_threshold = True
        if sorted_data[j] > alpha:
            all_below_threshold = False
            continue 

        cumsum = 0.0
        for k in range(1, m - j):
        
            cumsum = cumsum + 1 / sorted_data[m - k]
            
            total_sum = cumsum + (1 / sorted_data[j])
            avg_val = np.exp(1) * np.log(k+1)* ((k + 1) / total_sum)

            if avg_val > alpha:
                all_below_threshold = False
                break
       
    
        if all_below_threshold:
            result_indices = sorted_indices[:j + 1]
            threshold_value = sorted_data[j]
            boundary_index = sorted_indices[j]
            return result_indices, boundary_index, threshold_value
    
    return np.empty(0, dtype=np.int64), -1, 0.0


def dual_threshold_bfdr(p_values, alpha):
    """
    Dual-threshold bFDR control: intersect harmonic mean rejection set
    with k=1 Bonferroni rejection set. Harmonic mean provides power for
    bulk rejection, Bonferroni filters false discoveries at the boundary.

    This directly targets the bFDR metric by removing boundary genes
    that fail the stricter Bonferroni test, while preserving most of
    the harmonic mean's power advantage.
    """
    # First pass: harmonic mean (powerful bulk rejection)
    rej_hmp, boundary_hmp, _ = harmonicpbfdr(p_values, alpha)
    
    if len(rej_hmp) == 0:
        return np.empty(0, dtype=np.int64), -1, 0.0
    
    # Second pass: k=1 Bonferroni (stricter boundary control)
    rej_bonf, boundary_bonf = bfdr_k_domino(p_values, k=1, alpha=alpha)
    
    # Intersection: keep only genes passing both tests
    rej_set_hmp = set(rej_hmp)
    rej_set_bonf = set(rej_bonf)
    rej_intersection = rej_set_hmp & rej_set_bonf
    
    if len(rej_intersection) == 0:
        return np.empty(0, dtype=np.int64), -1, 0.0
    
    # Sort intersection indices by p-value to maintain ordering
    rej_list = sorted(rej_intersection, key=lambda i: p_values[i])
    rej_array = np.array(rej_list, dtype=np.int64)
    
    # Boundary is the least significant rejection (largest p-value in intersection)
    boundary_index = rej_list[-1]
    threshold_value = p_values[boundary_index]
    
    return rej_array, boundary_index, threshold_value


def sl_procedure(p_values, alpha):
    
    p = np.array(p_values)
    m = len(p)
    
    if m == 0:
        return np.empty(0, dtype=np.int64), -1, 0.0

    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]

    p_padded = np.concatenate(([0.0], sorted_p))

    k_vec = np.arange(m + 1)
    

    slope = alpha / m

    objective = p_padded - (k_vec * slope)
    min_val = np.min(objective)
    candidates = np.flatnonzero(objective <= min_val + 1e-12)
    k_hat = candidates[-1]
    if k_hat > 0:
        result_indices = sorted_indices[:k_hat]
        boundary_idx_in_sorted = k_hat - 1
        boundary_index = sorted_indices[boundary_idx_in_sorted]
        threshold_value = sorted_p[boundary_idx_in_sorted]
        
        return result_indices, boundary_index, threshold_value
    
    else:

        return np.empty(0, dtype=np.int64), -1, 0.0


def simesdomino(data, alpha):
    """
    use Simes as 1-local test
    """
    m = len(data)
    if m == 0:
        return np.empty(0, dtype=np.int64), -1, 0.0

    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
   
    all_ranks = np.arange(1, m + 1, dtype=float)

    for j in range(m - 1, -1, -1):
        current_p = sorted_data[j]

        is_j_valid = True
        
   
        num_larger = m - 1 - j

        max_q_small = m - j
        q_vals_small = np.arange(1, max_q_small + 1)
        
        failed_mask = (current_p * q_vals_small) > alpha
        failed_q_indices = np.where(failed_mask)[0] 
        
        if len(failed_q_indices) > 0:
            for idx in failed_q_indices:
                q = idx + 1
                if q == 1:
                    is_j_valid = False
                    break
                k = q - 1
                tail_vals = sorted_data[m-k:]
                tail_ranks = all_ranks[1:q] 
                
                threshold = alpha / q
                if not np.any(tail_vals <= tail_ranks * threshold):
                    is_j_valid = False
                    break
            
            if not is_j_valid:
                continue 
        if max_q_small < m:
            for q in range(max_q_small + 1, m + 1):
               
                
                start_idx = m - q
                subset_vals = sorted_data[start_idx:]
                subset_ranks = all_ranks[:q]
                
                threshold = alpha / q
                
                if not np.any(subset_vals <= subset_ranks * threshold):
                    is_j_valid = False
                    break
        
        if is_j_valid:
            result_indices = sorted_indices[:j + 1]
            boundary_index = sorted_indices[j]
            threshold_value = sorted_data[j]
            return result_indices, boundary_index, threshold_value

    return np.empty(0, dtype=np.int64), -1, 0.0



    
def evaluate_procedure(rejection_indices, p_values, labels):
    """
    Evaluation function.
    """
    rejection_indices = np.array(rejection_indices, dtype=int)
    labels = np.array(labels)
    m = len(p_values)
    num_rejections = len(rejection_indices)
    if m > 0:
        rejection_prop = num_rejections / m
    else:
        rejection_prop = 0.0
    num_false_discoveries = 0
    power = 0
    if num_rejections > 0:
        rejected_labels = labels[rejection_indices]
        num_false_discoveries = int(np.sum(rejected_labels == 0))/len(rejection_indices)
        # power = int(np.sum(rejected_labels == 1)) / np.sum(labels == 1) if np.sum(labels == 1) > 0 else 0.0
    return num_rejections, rejection_prop, num_false_discoveries    
    
    
# -------------------------Gene Data Process---------------------------------
def load_single_cell_line(file_path, cell_line_index=0):
    """
    We only take one cell line's data to do the analysisfor memory-efficieny.
    """
    df_chunk = pd.read_csv(file_path, skiprows=lambda x: x > 0 and x != (cell_line_index + 1), nrows=1, header=0)

    header = pd.read_csv(file_path, nrows=0)
    df_chunk.columns = header.columns

    gene_scores = df_chunk.iloc[0, 1:]
    
    print(f"成功提取: {df_chunk.iloc[0, 0]} (包含 {len(gene_scores)} 个基因)")
    return gene_scores

FILE_CEG = 'CEGv2.txt'                # True positive(Hart Lab)
FILE_NEG = 'NEGv1.txt'                # True negative (Hart Lab)
def build_labeled_dataframe(scores, ceg_file, neg_file):
    """
    Build a DataFrame with gene scores and true labels based on reference sets.
    """
    try:
        ceg_genes = pd.read_csv(ceg_file, header=None, sep='\t')[0].values
        neg_genes = pd.read_csv(neg_file, header=None, sep='\t')[0].values
        ceg_set = set(ceg_genes)
        neg_set = set(neg_genes)
        print(f"Success: CEG (Essential)={len(ceg_set)} , NEG (Non-Essential)={len(neg_set)} ")
    except FileNotFoundError:
        print("Wrong: no such file")
        return None

    df = pd.DataFrame({'Score': scores.astype(float)})

    df['Label'] = -1

    df.loc[df.index.isin(ceg_set), 'Label'] = 1
    df.loc[df.index.isin(neg_set), 'Label'] = 0

    df_labeled = df[df['Label'] != -1].copy()

    n_pos = (df_labeled['Label'] == 1).sum()
    n_neg = (df_labeled['Label'] == 0).sum()
    print(f"Data matching completed! Extracted labeled genes: {len(df_labeled)}")
    print(f"  - True positives (Label=1): {n_pos} genes")
    print(f"  - True negatives (Label=0): {n_neg} genes")

    return df_labeled


def sample_split_testing(scores, df_labels,test_size, seed):
    merged = pd.concat([scores.rename('Score'), df_labels['Label']], axis=1).dropna()
    df_labeled = merged[merged['Label'].isin([0, 1])].copy()

    train_df, test_df = train_test_split(
        df_labeled, 
        test_size=test_size, 
        stratify=df_labeled['Label'], 
        random_state=seed 
    )
    
    train_h0 = train_df[train_df['Label'] == 0]['Score']
    train_h1 = train_df[train_df['Label'] == 1]['Score']
    
    mu_0, std_0 = train_h0.mean(), train_h0.std()
    mu_1, std_1 = train_h1.mean(), train_h1.std()
    
    test_scores = test_df['Score'].values
    test_labels = test_df['Label'].values 
    
    z_scores = (test_scores - mu_0) / std_0
    p_values = norm.cdf(z_scores)
    
    f0_x = norm.pdf(test_scores, loc=mu_0, scale=std_0)
    f1_x = norm.pdf(test_scores, loc=mu_1, scale=std_1)
    
    eps = 1e-15 
    e_values = (f1_x + eps) / (f0_x + eps)
    
    return p_values, e_values, test_labels

def evaluate_fdrpower(rej_indices, test_labels):
    m1 = np.sum(test_labels == 1) 
    R = len(rej_indices)
    if R > 0:
        V = np.sum(test_labels[rej_indices] == 0) # False Discoveries
        S = np.sum(test_labels[rej_indices] == 1) # True Discoveries
    else:
        V, S = 0, 0
        
    fdp = V / max(R, 1)
    power = S / m1 if m1 > 0 else 0.0
    return fdp, power


def evaluate_boundary_k_errors(reject_indices, p_vals, true_labels, k=10):
    """
    Compute how many of the top k rejected hypotheses (with the largest P values) are false discoveries (Label=0).
    """
    if len(reject_indices) == 0:
        return 0, np.array([])
    
    rej_p = p_vals[reject_indices]
    rej_labels = true_labels[reject_indices]
    
    sort_order = np.argsort(rej_p)[::-1] 

    actual_k = min(k, len(reject_indices))
    top_k_indices = sort_order[:actual_k]

    boundary_labels = rej_labels[top_k_indices]
    n_errors = (boundary_labels == 0).sum()
    
    return n_errors, boundary_labels

# -------------------------Stock Selection---------------------------------
# No additional functions.

# -------------------------Pathway Enrichment Analysis---------------------------------


ENRICHR_URL = 'https://maayanlab.cloud/Enrichr'

def get_enrichr_scores(gene_list, library_name='KEGG_2021_Human'):
    """
    Send the gene list to Enrichr and retrieve the enrichment scores for the specified pathway library
    """
    if not gene_list or len(gene_list) < 5:
        # If the number of genes is too small, the enrichment analysis has no statistical significance
        return None

    genes_str = '\n'.join(gene_list)
    payload = {
        'list': (None, genes_str),
        'description': (None, 'BFDR_Test_List')
    }
    
    try:
        response = requests.post(f'{ENRICHR_URL}/addList', files=payload, timeout=15)
        if not response.ok:
            print(f"\n❌ API loading failed")
            print(f"HTTP: {response.status_code}")
            print(f"{response.text}")
            print(f"Current number of genes to upload: {len(gene_list)}")
            return None
    except Exception as e:
        print(f"\n❌ Network connection error: {e}")
        return None
    
    data = json.loads(response.text)
    user_list_id = data['userListId']
    time.sleep(1) 
    query_string = f'?userListId={user_list_id}&backgroundType={library_name}'
    response = requests.get(f'{ENRICHR_URL}/enrich' + query_string)
    if not response.ok:
        print("API query failed")
        return None
    data = json.loads(response.text)
    results = data.get(library_name, [])
    if not results:
        return None

    df_res = pd.DataFrame(results, columns=[
        'Rank', 'Term', 'P-value', 'Z-score', 'Combined_Score', 
        'Overlapping_Genes', 'Adj_P-value', 'Old_P-value', 'Old_Adj_P-value'
    ])
    
    return df_res