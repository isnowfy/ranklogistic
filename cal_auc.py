# -*- coding: utf-8 -*-

import os
import sys


def cal_auc(labels, scores):
    ss = zip(scores, labels)
    ss = sorted(ss, key=lambda x: x[0], reverse=True)
    #ss = sorted(ss, key=lambda x: x[0])
    num_pos = 0
    num_neg = 0
    for s in ss:
        if s[1] == '1' or s[1] == 1:
            num_pos += 1
        else:
            num_neg += 1
    if num_pos == 0 or num_neg == 0:
        return 0.0
    print 'num_pos', num_pos, 'num_neg', num_neg
    tp = 0
    fp = 0
    prev_tpr = 0
    prev_fpr = 0
    prev_score = 0
    auc = 0
    for s in ss:
        if s[1] == '1' or s[1] == 1:
            tp += 1
        else:
            fp += 1
        #if ((fp + 0.0) / num_neg != prev_fpr):
        if s[0] != prev_score:
            now_tpr = (tp + 0.0) / num_pos
            #auc += prev_tpr * ((fp + 0.0) / num_neg - prev_fpr)
            auc += (prev_tpr + now_tpr) / 2 * ((fp + 0.0) / num_neg - prev_fpr)
            prev_tpr = now_tpr
            prev_fpr = (fp + 0.0) / num_neg
            prev_score = s[0]
    now_tpr = (tp + 0.0) / num_pos
    auc += (prev_tpr + now_tpr) / 2 * ((fp + 0.0) / num_neg - prev_fpr)
    return auc


if __name__ == '__main__':
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    print f1
    print f2
    labels = []
    scores = []
    for rline in open(f1):
        labels.append(rline[0])
    for rline in open(f2):
        scores.append(float(rline))
    print cal_auc(labels, scores)
