import numpy as np
from numba import jit

@jit(nopython=True)
def extend_postive_range(labels, window):
    extended_labels = np.copy(labels)
    n = len(labels)
    for i in range(n):
        if labels[i] == 1:
            start = max(0, i - window)
            end = min(n, i + window + 1)
            extended_labels[start:end] = 1
    return extended_labels

@jit(nopython=True)
def range_convers_new(labels):
    ranges = []
    in_range = False
    start = 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_range:
            start = i
            in_range = True
        elif labels[i] == 0 and in_range:
            end = i - 1
            ranges.append((start, end))
            in_range = False
    if in_range:
        ranges.append((start, len(labels) - 1))
    return np.array(ranges)

@jit(nopython=True)
def TPR_FPR_RangeAUC(labels, pred, P, L):
    product = labels * pred
    
    TP = np.sum(product)
    
    P_new = (P + np.sum(labels)) / 2
    recall = min(TP / P_new, 1)
    
    existence = 0
    for seg in L:
        if np.sum(product[seg[0]:(seg[1] + 1)]) > 0:
            existence += 1
    existence_ratio = existence / len(L)
    
    TPR_RangeAUC = recall * existence_ratio
    
    FP = np.sum(pred) - TP
    N_new = len(labels) - P_new
    FPR_RangeAUC = FP / N_new
    
    Precision_RangeAUC = TP / np.sum(pred)
    
    return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC

@jit(nopython=True)
def RangeAUC_volume(labels_original, score, windowSize):
    score_sorted = -np.sort(-score)
    P = np.sum(labels_original)
    tpr_3d = []
    fpr_3d = []
    prec_3d = []
    
    auc_3d = []
    ap_3d = []
    
    window_3d = np.arange(0, windowSize + 1, 1)
    
    for window in window_3d:
        labels = extend_postive_range(labels_original, window)
        L = range_convers_new(labels)
        
        TPR_list = [0]
        FPR_list = [0]
        Precision_list = [1]
        
        for i in np.linspace(0, len(score) - 1, 250).astype(np.int32):
            threshold = score_sorted[i]
            pred = score >= threshold
            TPR, FPR, Precision = TPR_FPR_RangeAUC(labels, pred, P, L)
            
            TPR_list.append(TPR)
            FPR_list.append(FPR)
            Precision_list.append(Precision)
        
        TPR_list.append(1)
        FPR_list.append(1)
        
        tpr = np.array(TPR_list)
        fpr = np.array(FPR_list)
        prec = np.array(Precision_list)
        
        tpr_3d.append(tpr)
        fpr_3d.append(fpr)
        prec_3d.append(prec)
        
        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1]) / 2
        AUC_range = np.sum(width * height)
        auc_3d.append(AUC_range)
        
        width_PR = tpr[1:-1] - tpr[:-2]
        height_PR = (prec[1:] + prec[:-1]) / 2
        AP_range = np.sum(width_PR * height_PR)
        ap_3d.append(AP_range)
    
    return tpr_3d, fpr_3d, prec_3d, window_3d, np.sum(np.array(auc_3d)) / len(window_3d), np.sum(np.array(ap_3d)) / len(window_3d)

# Example usage:
# labels_original = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.int32)
# score = np.array([0.5, 0.4, 0.6, 0.8, 0.3, 0.1, 0.7], dtype=np.float64)
# windowSize = 3
# P = np.sum(labels_original)

# tpr_3d, fpr_3d, prec_3d, window_3d, auc_mean, ap_mean = RangeAUC_volume(labels_original, score, windowSize, P)
# print("TPR 3D:", tpr_3d)
# print("FPR 3D:", fpr_3d)
# print("Prec 3D:", prec_3d)
# print("Window 3D:", window_3d)
# print("Mean AUC:", auc_mean)
# print("Mean AP:", ap_mean)
