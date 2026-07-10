from sklearn import metrics
import numpy as np
import math
import copy

def generate_curve(label, score, slidingWindow, version='opt', thre=250):
    if version =='opt_mem':
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt_mem(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
    else:
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume_opt(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)


    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)

    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d

class basic_metricor():
    def __init__(self, a = 1, probability = True, bias = 'flat', ):
        self.a = a
        self.probability = probability
        self.bias = bias
        self.eps = 1e-15

    def detect_model(self, model, label, contamination = 0.1, window = 100, is_A = False, is_threshold = True):
        if is_threshold:
            score = self.scale_threshold(model.decision_scores_, model._mu, model._sigma)
        else:
            score = self.scale_contamination(model.decision_scores_, contamination = contamination)
        if is_A is False:
            scoreX = np.zeros(len(score)+window)
            scoreX[math.ceil(window/2): len(score)+window - math.floor(window/2)] = score
        else:
            scoreX = score

        self.score_=scoreX
        L = self.metric(label, scoreX)
        return L

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start +AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue +=  bi
            if i in p:
                MyValue += bi
        return MyValue/MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score +=1
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1
        if score == 0:
            return 0
        else:
            return 1/score

    def b(self, i, length):
        bias = self.bias
        if bias == 'flat':
            return 1
        elif bias == 'front-end bias':
            return length - i + 1
        elif bias == 'back-end bias':
            return i
        else:
            if i <= length/2:
                return i
            else:
                return length - i + 1

    def scale_threshold(self, score, score_mu, score_sigma):
        return (score >= (score_mu + 3*score_sigma)).astype(int)

    def _adjust_predicts(self, score, label, threshold=None, pred=None, calc_latency=False):
        """
        Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

        Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                A point is labeled as "anomaly" if its score is higher than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):

        Returns:
            np.ndarray: predict labels
        """
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        score = np.asarray(score)
        label = np.asarray(label)
        latency = 0
        if pred is None:
            predict = score > threshold
        else:
            predict = copy.deepcopy(pred)
        actual = label > 0.1
        anomaly_state = False
        anomaly_count = 0
        for i in range(len(score)):
            if actual[i] and predict[i] and not anomaly_state:
                    anomaly_state = True
                    anomaly_count += 1
                    for j in range(i, 0, -1):
                        if not actual[j]:
                            break
                        else:
                            if not predict[j]:
                                predict[j] = True
                                latency += 1
            elif not actual[i]:
                anomaly_state = False
            if anomaly_state:
                predict[i] = True
        if calc_latency:
            return predict, latency / (anomaly_count + 1e-4)
        else:
            return predict

    def metric_new(self, label, score, preds, plot_ROC=False, alpha=0.2):
        '''input:
               Real labels and anomaly score in prediction

           output:
               AUC,
               Precision,
               Recall,
               F-score,
               Range-precision,
               Range-recall,
               Range-Fscore,
               Precison@k,

            k is chosen to be # of outliers in real labels
        '''
        if np.sum(label) == 0:
            print('All labels are 0. Label must have groud truth value for calculating AUC score.')
            return None

        if np.isnan(score).any() or score is None:
            print('Score must not be none.')
            return None

        #area under curve
        auc = metrics.roc_auc_score(label, score)
        # plor ROC curve
        if plot_ROC:
            fpr, tpr, thresholds  = metrics.roc_curve(label, score)
            # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            # display.plot()

        #precision, recall, F
        if preds is None:
            preds = score > (np.mean(score)+3*np.std(score))
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        precision = Precision[1]
        recall = Recall[1]
        f = F[1]

        #point-adjust
        adjust_preds = self._adjust_predicts(score, label, pred=preds)
        PointF1PA = metrics.f1_score(label, adjust_preds)

        #range anomaly
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha)
        Rprecision = self.range_recall_new(preds, label, 0)[0]

        if Rprecision + Rrecall==0:
            Rf=0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1-k/len(label)))

        # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
        p_at_k = np.where(preds > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k/k

        L = [auc, precision, recall, f, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, Rf, precision_at_k]
        if plot_ROC:
            return L, fpr, tpr
        return L

    def metric_ROC(self, label, score):
        return metrics.roc_auc_score(label, score)

    def metric_PR(self, label, score):
        return metrics.average_precision_score(label, score)

    def metric_PointF1(self, label, score, preds=None):
        if preds is None:
            precision, recall, thresholds = metrics.precision_recall_curve(label, score)
            f1_scores = 2 * (precision * recall) / (precision + recall + 0.00001)
            F1 = np.max(f1_scores)
            Precision, Recall = precision[np.argmax(f1_scores)], recall[np.argmax(f1_scores)]
            threshold = thresholds[np.argmax(f1_scores)]
        else:
            Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
            F1 = F[1]
        return F1, Precision, Recall

    def metric_Affiliation(self, label, score, preds=None):
        from .affiliation.generics import convert_vector_to_events
        from .affiliation.metrics import pr_from_events

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            Affiliation_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                events_pred = convert_vector_to_events(preds)
                events_gt = convert_vector_to_events(label)
                Trange = (0, len(preds))
                affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)
                Affiliation_Precision = affiliation_metrics['Affiliation_Precision']
                Affiliation_Recall = affiliation_metrics['Affiliation_Recall']
                Affiliation_F = 2*Affiliation_Precision*Affiliation_Recall / (Affiliation_Precision+Affiliation_Recall+self.eps)

                Affiliation_scores.append(Affiliation_F)

            Affiliation_F1_Threshold = thresholds[np.argmax(Affiliation_scores)]
            Affiliation_F1 = max(Affiliation_scores)

        else:
            events_pred = convert_vector_to_events(preds)
            events_gt = convert_vector_to_events(label)
            Trange = (0, len(preds))
            affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)
            Affiliation_Precision = affiliation_metrics['Affiliation_Precision']
            Affiliation_Recall = affiliation_metrics['Affiliation_Recall']
            Affiliation_F1 = 2*Affiliation_Precision*Affiliation_Recall / (Affiliation_Precision+Affiliation_Recall+self.eps)

        return Affiliation_F1

    def metric_RF1(self, label, score, preds=None):

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            Rf1_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha=0.2)
                Rprecision = self.range_recall_new(preds, label, 0)[0]
                if Rprecision + Rrecall==0:
                    Rf=0
                else:
                    Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

                Rf1_scores.append(Rf)

            RF1_Threshold = thresholds[np.argmax(Rf1_scores)]
            RF1 = max(Rf1_scores)
        else:
            Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha=0.2)
            Rprecision = self.range_recall_new(preds, label, 0)[0]
            if Rprecision + Rrecall==0:
                RF1=0
            else:
                RF1 = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)
        return RF1

    def metric_PointF1PA(self, label, score, preds=None):

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            PointF1PA_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                adjust_preds = self._adjust_predicts(score, label, pred=preds)
                PointF1PA = metrics.f1_score(label, adjust_preds)

                PointF1PA_scores.append(PointF1PA)

            PointF1PA_Threshold = thresholds[np.argmax(PointF1PA_scores)]
            PointF1PA1 = max(PointF1PA_scores)

        else:
            adjust_preds = self._adjust_predicts(score, label, pred=preds)
            PointF1PA1 = metrics.f1_score(label, adjust_preds)

        return PointF1PA1

    def _get_events(self, y_test, outlier=1, normal=0):
        events = dict()
        label_prev = normal
        event = 0  # corresponds to no event
        event_start = 0
        for tim, label in enumerate(y_test):
            if label == outlier:
                if label_prev == normal:
                    event += 1
                    event_start = tim
            else:
                if label_prev == outlier:
                    event_end = tim - 1
                    events[event] = (event_start, event_end)
            label_prev = label

        if label_prev == outlier:
            event_end = tim - 1
            events[event] = (event_start, event_end)
        return events

    def metric_EventF1PA(self, label, score, preds=None):
        from sklearn.metrics import precision_score
        true_events = self._get_events(label)

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            EventF1PA_scores = []

            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
                fn = len(true_events) - tp
                rec_e = tp/(tp + fn)
                prec_t = precision_score(label, preds, zero_division=0)
                EventF1PA = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

                EventF1PA_scores.append(EventF1PA)

            EventF1PA_Threshold = thresholds[np.argmax(EventF1PA_scores)]
            EventF1PA1 = max(EventF1PA_scores)

        else:

            tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
            fn = len(true_events) - tp
            rec_e = tp/(tp + fn)
            prec_t = precision_score(label, preds, zero_division=0)
            EventF1PA1 = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

        return EventF1PA1

    def range_recall_new(self, labels, preds, alpha):
        p = np.where(preds == 1)[0]    # positions of predicted label==1
        range_pred = self.range_convers_new(preds)
        range_label = self.range_convers_new(labels)

        Nr = len(range_label)    # total # of real anomaly segments

        ExistenceReward = self.existence_reward(range_label, preds)


        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)


        score = alpha * ExistenceReward + (1-alpha) * OverlapReward
        if Nr != 0:
            return score/Nr, ExistenceReward/Nr, OverlapReward/Nr
        else:
            return 0,0,0

    def range_convers_new(self, label):
        '''
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        anomaly_ends, = np.where(np.diff(label) == -1)
        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                # we started with an anomaly, so the start of the first anomaly is the start of the labels
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                # we ended on an anomaly, so the end of the last anomaly is the end of the labels
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        return list(zip(anomaly_starts, anomaly_ends))

    def existence_reward(self, labels, preds):
        '''
        labels: list of ordered pair
        preds predicted data
        '''

        score = 0
        for i in labels:
            if preds[i[0]:i[1]+1].any():
                score += 1
        return score

    def num_nonzero_segments(self, x):
        count=0
        if x[0]>0:
            count+=1
        for i in range(1, len(x)):
            if x[i]>0 and x[i-1]==0:
                count+=1
        return count

    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]


            x1 = np.arange(e,min(e+window//2,length))
            label[x1] += np.sqrt(1 - (x1-e)/(window))

            x2 = np.arange(max(s-window//2,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(window))

        label = np.minimum(np.ones(length), label)
        return label

    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            l0 = int((e-s+1)*percentage)

            x1 = np.arange(e,min(e+l0,length))
            label[x1] += np.sqrt(1 - (x1-e)/(2*l0))

            x2 = np.arange(max(s-l0,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(2*l0))

        label = np.minimum(np.ones(length), label)
        return label

    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        indices = np.where(labels == 1)[0]
        product = labels * pred
        TP = np.sum(product)
        newlabels = product.copy()
        newlabels[indices] = 1

        # recall = min(TP/P,1)
        P_new = (P + np.sum(newlabels)) / 2  # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP / P_new, 1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))

        existence = 0
        for seg in L:
            if np.sum(product[seg[0]:(seg[1] + 1)]) > 0:  # if newlabels>0, that segment must contained
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
        TPR_list = [0]
        FPR_list = [0]
        Precision_list = [1]

        for i in np.linspace(0, len(score) - 1, 250).astype(int):
            threshold = score_sorted[i]
            # print('thre='+str(threshold))
            pred = score >= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P, L)

            TPR_list.append(TPR)
            FPR_list.append(FPR)
            Precision_list.append(Precision)

        TPR_list.append(1)
        FPR_list.append(1)  # otherwise, range-AUC will stop earlier than (1,1)

        tpr = np.array(TPR_list)
        fpr = np.array(FPR_list)
        prec = np.array(Precision_list)

        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1]) / 2
        AUC_range = np.sum(width * height)

        width_PR = tpr[1:-1] - tpr[:-2]
        height_PR = prec[1:]
        AP_range = np.sum(width_PR * height_PR)

        if plot_ROC:
            return AUC_range, AP_range, fpr, tpr, prec

        return AUC_range

    def range_convers_new(self, label):
        '''
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        anomaly_ends, = np.where(np.diff(label) == -1)
        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                # we started with an anomaly, so the start of the first anomaly is the start of the labels
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                # we ended on an anomaly, so the end of the last anomaly is the end of the labels
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        return list(zip(anomaly_starts, anomaly_ends))

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