from .basic_metrics import basic_metricor, generate_curve
import numpy as np

def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):

    score = np.nan_to_num(score)
    grader = basic_metricor()
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)
    PointF1, Precision, Recall = grader.metric_PointF1(labels, score, preds=pred)

    return AUC_ROC, AUC_PR, PointF1, Precision, Recall

def point_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    score = np.nan_to_num(score)
    grader = basic_metricor()
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, slidingWindow, version, thre)

    PointF1, Precision, Recall = grader.metric_PointF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)

    return AUC_ROC, AUC_PR, PointF1, Precision, Recall, Affiliation_F, VUS_ROC, VUS_PR