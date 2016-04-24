# -*- coding: utf-8 -*-

import math
import os
import sys

import lr

MAX_ITER = 50


def parse_file(fname):
    for rline in open(fname):
        if rline[0] == '#':
            continue
        line = rline.split()
        label = float(line[0])
        if label < 0:
            label = 0
        x = {}
        for item in line[1:]:
            item = item.split(':')
            x[item[0]] = float(item[1])
        qid = x.pop('qid', 'other')
        yield label, qid, x


def train_file(fname, mname, max_iter=MAX_ITER, alpha=1, c=0):
    w = {}
    qids = {}
    for y, qid, x in parse_file(fname):
        for k, v in x.items():
            if k not in w:
                w[k] = 0
        if qid not in qids:
            qids[qid] = []
        qids[qid].append((y, x))
    for _ in range(max_iter):
        for qid, vs in qids.items():
            for i in range(len(vs)):
                for j in range(len(vs)):
                    if vs[i][0] <= vs[j][0]:
                        continue
                    x = {}
                    for xk, xv in vs[i][1].items():
                        x[xk] = xv
                    for xk, xv in vs[j][1].items():
                        if xk not in x:
                            x[xk] = 0
                        x[xk] -= xv
                    lr.train(1, w, x, alpha, c)
    with open(mname, 'w') as fw:
        for k, v in sorted(w.items()):
            fw.write('%s\t%s\n' % (k, v))


def test_file(fname, mname, oname):
    labels = []
    scores = []
    qids = {}
    w = {}
    for rline in open(mname):
        line = rline.split()
        w[line[0]] = float(line[1])
    for y, qid, x in parse_file(fname):
        pred = lr.predict(w, x)
        labels.append(y)
        scores.append(pred)
        if qid not in qids:
            qids[qid] = []
        qids[qid].append((y, pred))
    with open(oname, 'w') as fw:
        for k in scores:
            fw.write('%s\n' % k)
    cal_acc(qids)


def cal_acc(qids):
    total = 0
    l = len(qids)
    now = 0
    for qid, vs in qids.items():
        cnt = 0
        tmp = 0
        for i in range(len(vs)):
            for j in range(i, len(vs)):
                if vs[i][0] == vs[j][0]:
                    continue
                cnt += 1
                if (vs[i][0] - vs[j][0]) * (vs[i][1] - vs[j][1]) <= 0:
                    tmp += 1
                    total += 1
        now += (tmp + 0.0) / cnt
    print 'Total Num Swappedpairs:', total
    print 'Avg Swappedpairs Percent:', now / l


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: python rank.py train/test'
        exit(0)
    if sys.argv[1] == 'train':
        if len(sys.argv) < 4:
            print 'usage: python rank.py train file model [alpha] [cost] [iter]'
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
            print 'usage: python rank.py test file model out'
            exit(0)
        test_file(sys.argv[2], sys.argv[3], sys.argv[4])
