#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/10/17 16:36
 @File    : eva_classifier.py
 """

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import numpy as np


def eva_classifier(data, target, MB):
    cum = 0
    number, kVar = np.shape(data)
    # use classifier to compare phase
    t_size = int(number * 4 / 5)
    sourcedata = np.array(data, dtype=np.int_)
    lable_all = sourcedata[:, target]

    # use MB and data  to train classifer clf
    X_train = sourcedata[0:t_size, MB]
    X_label = np.array(lable_all[0:t_size])
    clf = BernoulliNB()
    clf.fit(X_train, X_label)
    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    Y_lable = clf.predict(sourcedata[t_size:number, MB])
    # print("Y_label is: " + str(Y_lable))
    true_label = lable_all[t_size: number]
    acc_score = accuracy_score(true_label, Y_lable)
    # print("score is: " + str(acc_score))
    return acc_score
