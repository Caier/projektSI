import numpy as np

from RegressionTree import RegressionTree
import datasets.wine.prepareXY as wine
import datasets.automobile.prepareXY as automobile
import random

for dataset in (automobile, wine):
    (X, Y) = dataset.prepareXY()
    train_idx = random.sample(range(len(Y)), int(0.8 * len(Y)))
    X_train = [row for i, row in enumerate(X) if i in train_idx]
    Y_train = [row for i, row in enumerate(Y) if i in train_idx]
    X_test = [row for i, row in enumerate(X) if i not in train_idx]
    Y_test = [row for i, row in enumerate(Y) if i not in train_idx]

    rt = RegressionTree(X_train, Y_train)

    SE = 0 #Standard error of regression
    for x, y in zip(X_test, Y_test):
        #print(rt.predict(x), y)
        SE += np.absolute(y - rt.predict(x))
    SE /= len(Y_test)

    print(dataset.__name__)
    print(SE)
   # print(rt.get_tree_logic())