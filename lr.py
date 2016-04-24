# -*- coding: utf-8 -*-

import math
import os
import sys

import cal_auc

EPS = 1e-15
MAX_EXP_NUM = 50
MIN_SIGMOID = 1e-15
MAX_ITER = 10


def sign(x):
    return (x > EPS) - (x < -EPS)


def safe_exp(x):
    return math.exp(max(min(x, MAX_EXP_NUM), -MAX_EXP_NUM))


def sigmoid(z):
    return 1.0 / (1.0 + safe_exp(-1 * z))


def predict(w, x):
    z = 0
    for k, v in x.items():
        z += v * w[k]
    return sigmoid(z)


def loss(label, w, x, c=0):
    pred = predict(w, x)
    pred = max(min(pred, 1. - MIN_SIGMOID), MIN_SIGMOID)
    cost = 0
    if sign(c) != 0:
        for k, v in w.items():
            cost += v * v
        cost *= c / 2.0
    if sign(label, 1) == 0:
        return cost - math.log(pred)
    return cost - math.log(1 - pred)


def train(label, w, x, alpha, c=0):
    pred = predict(w, x)
    gred = pred - label
    for k, v in x.items():
        old = w[k]
        w[k] -= alpha * gred * v
        if sign(c) != 0:
            w[k] -= alpha * c  * old


def parse_file(fname):
    for rline in open(fname):
        if rline[0] == '#':
            continue
        line = rline.split()
        label = int(line[0])
        if label < 0:
            label = 0
        x = {}
        for item in line[1:]:
            item = item.split(':')
            x[item[0]] = float(item[1])
        x['bias'] = 1.
        yield label, x


def cal_acc(labels, scores):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(labels)):
        y = labels[i]
        p = scores[i]
        if y == '1' or y == 1:
            if p > 0.5:
                tp += 1
            else:
                fn += 1
        else:
            if p > 0.5:
                fp += 1
            else:
                tn += 1
    al = tp + tn + fp + fn
    t = tp + tn
    f = fp + fn
    p1 = tp / (tp + fp + 0.0)
    r1 = tp / (tp + fn + 0.0)
    f1 = 2 / (1 / p1 + 1 / r1)
    p2 = tn / (tn + fn + 0.0)
    r2 = tn / (tn + fp + 0.0)
    f2 = 2 / (1 / p2 + 1/ r2)
    print 'tp:', tp, 'tn:', tn, 'fp:', fp, 'fn:', fn
    print 'acc:', (t + 0.0) / al, 'true:', t, 'all:', al
    print 'precision:', p1, 'recall:', r1
    print 'postive f score:', f1
    print 'precision:', p2, 'recall:', r2
    print 'negative f score:', f2


def train_file(fname, mname, max_iter=MAX_ITER, alpha=1, c=0):
    w = {}
    for _ in range(max_iter):
        for y, x in parse_file(fname):
            for k, v in x.items():
                if k not in w:
                    w[k] = 0
            train(y, w, x, alpha, c)
    with open(mname, 'w') as fw:
        for k, v in sorted(w.items()):
            fw.write('%s\t%s\n' % (k, v))


def test_file(fname, mname, oname):
    labels = []
    scores = []
    w = {}
    for rline in open(mname):
        line = rline.split()
        w[line[0]] = float(line[1])
    for y, x in parse_file(fname):
        pred = predict(w, x)
        labels.append(y)
        scores.append(pred)
    with open(oname, 'w') as fw:
        for k in scores:
            fw.write('%s\n' % k)
    cal_acc(labels, scores)
    print 'auc:', cal_auc.cal_auc(labels, scores)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: python lr.py train/test'
        exit(0)
    if sys.argv[1] == 'train':
        if len(sys.argv) < 4:
            print 'usage: python lr.py train file model [alpha] [cost] [iter]'
            exit(0)
        if len(sys.argv) == 4:
            train_file(sys.argv[2], sys.argv[3])
        if len(sys.argv) == 5:
            train_file(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        if len(sys.argv) == 6:
            train_file(sys.argv[2], sys.argv[3], int(sys.argv[4]),
                       float(sys.argv[5]))
        if len(sys.argv) == 7:
            train_file(sys.argv[2], sys.argv[3], int(sys.argv[4]),
                       float(sys.argv[5]), float(sys.argv[6]))
    if sys.argv[1] == 'test':
        if len(sys.argv) < 5:
            print 'usage: python lr.py test file model out'
            exit(0)
        test_file(sys.argv[2], sys.argv[3], sys.argv[4])
