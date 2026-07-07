from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
from utils.tools import adjustment


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def adjusted_metric(pred,gt):

    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)
    # (4) detection adjustment
    gt, pred = adjustment(gt, pred)

    pred = np.array(pred)
    gt = np.array(gt)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))

    return precision, recall, f_score

def affiliation_metric(pred, gt):
    events_pred = convert_vector_to_events(pred)
    events_gt = convert_vector_to_events(gt)
    Trange = (0, len(pred))

    results = pr_from_events(events_pred, events_gt, Trange)

    # (5) 计算指标
    Aff_precision, Aff_recall = results['precision'], results['recall']
    Aff_F_score = 2 * Aff_precision * Aff_recall / (Aff_precision + Aff_recall)
    print("Aff_precision : {:0.4f}, Aff_recall : {:0.4f}, Aff_F_score : {:0.4f} ".format(
        Aff_precision, Aff_recall,Aff_F_score))

    return Aff_precision, Aff_recall,Aff_F_score


import numpy as np
from sklearn import metrics


class metricor:
    def metric_AUC(self, label, score):
        return metrics.roc_auc_score(label, score)

    def metric_PR(self, label, score):
        return metrics.average_precision_score(label, score)

    def range_convers_new(self, label):
        '''
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        L = []
        i = 0
        j = 0
        while j < len(label):
            # print(i)
            while label[i] == 0:
                i += 1
                if i >= len(label):  # ?
                    break  # ?
            j = i + 1
            # print('j'+str(j))
            if j >= len(label):
                if j == len(label):
                    L.append((i, j - 1))

                break
            while label[j] != 0:
                j += 1
                if j >= len(label):
                    L.append((i, j - 1))
                    break
            if j >= len(label):
                break
            L.append((i, j - 1))
            i = j
        return L

    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)  # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e, min(e + window // 2, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label

    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)  # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            l0 = int((e - s + 1) * percentage)

            x1 = np.arange(e, min(e + l0, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (2 * l0))

            x2 = np.arange(max(s - l0, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (2 * l0))

        label = np.minimum(np.ones(length), label)
        return label

    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        product = labels * pred

        TP = np.sum(product)

        # recall = min(TP/P,1)
        P_new = (P + np.sum(labels)) / 2  # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP / P_new, 1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))

        existence = 0
        for seg in L:
            if np.sum(product[seg[0]:(seg[1] + 1)]) > 0:
                existence += 1

        existence_ratio = existence / len(L)
        # print(existence_ratio)

        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall * existence_ratio

        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))

        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP / N_new

        Precision_RangeAUC = TP / np.sum(pred)

        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC

    def RangeAUC(self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type='window'):
        # AUC_type='window'/'percentage'
        score_sorted = -np.sort(-score)

        P = np.sum(labels)
        # print(np.sum(labels))
        if AUC_type == 'window':
            labels = self.extend_postive_range(labels, window=window)
        else:
            labels = self.extend_postive_range_individual(labels, percentage=percentage)

        # print(np.sum(labels))
        L = self.range_convers_new(labels)
        TF_list = np.zeros((252, 2))
        Precision_list = np.ones(251)
        j = 0
        for i in np.linspace(0, len(score) - 1, 250).astype(int):
            threshold = score_sorted[i]
            # print('thre='+str(threshold))
            pred = score >= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P, L)
            j += 1
            TF_list[j] = [TPR, FPR]
            Precision_list[j] = (Precision)

        TF_list[j + 1] = [1, 1]

        width = TF_list[1:, 1] - TF_list[:-1, 1]
        height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
        AUC_range = np.dot(width, height)

        width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
        height_PR = (Precision_list[1:] + Precision_list[:-1]) / 2
        AP_range = np.dot(width_PR, height_PR)

        if plot_ROC:
            return AUC_range, AP_range, TF_list[:, 1], TF_list[:, 0], Precision_list

        return AUC_range

    def new_sequence(self, label, sequence_original, window):
        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new = []
        for i in range(len(sequence_original) - 1):
            if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2
        sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
        return sequence_new

    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)
        length = len(label)

        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label

    # TPR_FPR_window
    def RangeAUC_volume_opt(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)

        for window in window_3d:

            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score >= threshold
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                    if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                TP = 0
                N_labels = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]

            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)

    def RangeAUC_volume_opt_mem(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre, len(score)))

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            p[k] = pred
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * p[j][seg[0]:seg[1] + 1]
                    if (p[j][seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                N_labels = 0
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], p[j][seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio

                N_new = len(labels) - P_new
                FPR = FP / N_new
                Precision = TP / N_pred[j]
                j += 1

                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]
            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = (AP_range)
        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)




def auc_vus_metric(results_storage, metrics, labels, score, version='opt_mem', thre=250, **args):
    if "auc" in metrics:
        grader = metricor()
        # result = {}
        AUC_ROC = grader.metric_AUC(labels, score)
        AUC_PR = grader.metric_PR(labels, score)
        results_storage['AUC_ROC'] = AUC_ROC
        results_storage['AUC_PR'] = AUC_PR
        # results_storage['auc'] = pd.DataFrame([result])
    if "r_auc" in metrics:
        slidingWindow = args.get('slidingWindow', 100)  # default set 100
        grader = metricor()
        # result = {}
        R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
        results_storage['R_AUC_ROC'] = R_AUC_ROC
        results_storage['R_AUC_PR'] = R_AUC_PR
        # results_storage['r_auc'] = pd.DataFrame([result])
    if "vus" in metrics:
        slidingWindow = args.get('slidingWindow', 100)  # default set 100
        # result = {}
        if version == 'opt_mem':
            tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt_mem(
                labels_original=labels, score=score, windowSize=slidingWindow, thre=thre
            )
        else:
            tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt(
                labels_original=labels, score=score, windowSize=slidingWindow, thre=thre
            )

        results_storage['VUS_ROC'] = avg_auc_3d
        results_storage['VUS_PR'] = avg_ap_3d
        # results_storage['vus'] = pd.DataFrame([result])


def UCR_evaluate(results_storage, labels, score):

    result = {}
    sorted_indexs = np.argsort(-score)  # DESC
    # find the first overlap index
    topk = 0
    for i, (index) in enumerate(sorted_indexs):
        if labels[index] == 1:
            topk = i + 1
            break

    results_storage['topk'] = topk
    results_storage['total_len'] = len(score)
    aplha_quantile = topk / len(score)
    results_storage['aplha_quantile'] = aplha_quantile
    results_storage['3_alpha'] = aplha_quantile < 0.03
    results_storage['10_alpha'] = aplha_quantile < 0.1
    # results_storage['accomplish_UCR'] = pd.DataFrame([result])
