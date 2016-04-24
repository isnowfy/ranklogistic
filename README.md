# Ranklogistics

This project is implemented to do the pairwise learning to rank with logistics regression like ranksvm.

## Logistics Regression

### Format

[libsvm format](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

### Usage

#### Train
~~~~
python lr.py train svmguide1.scale m1 50 0.001
~~~~

#### Predict
~~~~
python lr.py test svmguide1.t.scale m1 out.txt
~~~~

This will show precision, recall, auc, accuracy.

## Ranklogistics

pairwise learning to rank algorithm like svmrank

### Format

[ranksvm format](http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)

### Usage

#### Train

~~~~
python rank.py train train.dat m1 50 0.1
~~~~

#### Predict

~~~~
python rank.py test test.dat m1 out.txt
~~~~

This will show 
1  Total number of swapped pairs summed over all queries.
2  Fraction of swapped pairs averaged over all queries.
